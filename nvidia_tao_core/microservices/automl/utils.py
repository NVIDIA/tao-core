# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for AutoML"""
import os
import math
import glob
import datetime
import time
from kubernetes import client, config
import logging

from nvidia_tao_core.microservices.handlers.stateless_handlers import BACKEND
from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_input_dimension(dimension_value, factor=32):
    """Return dimension as a multiple of factor"""
    if int(dimension_value) % factor == 0:
        return dimension_value
    return (int(dimension_value / factor) + 1) * factor


def fix_power_of_factor(value, factor=2):
    """Return the nearest power of factor that is >= value"""
    if value <= 0:
        return factor  # Return the base factor for non-positive values
    # Calculate the power needed: factor^power >= value
    power = math.ceil(math.log(value) / math.log(factor))
    return int(factor ** power)


def clamp_value(value, v_min, v_max):
    """Clamps value within the given range"""
    if value >= v_max:
        epsilon = v_max / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_max - epsilon
    if value <= v_min:
        epsilon = v_min / 10
        if epsilon == 0.0:
            epsilon = 0.0000001
        value = v_min + epsilon
    return value


def get_valid_range(parameter_config, parent_params):
    """Compute the clamp range for the given parameter"""
    v_min = float(parameter_config.get("valid_min"))
    v_max = float(parameter_config.get("valid_max"))
    default_value = float(parameter_config.get("default_value"))
    if math.isinf(v_min):
        v_min = default_value
    if math.isinf(v_max):
        v_max = default_value

    dependent_on_param = parameter_config.get("depends_on", None)
    if type(dependent_on_param) is str and dependent_on_param:
        dependent_on_param_op = dependent_on_param.split(" ")[0]
        dependent_on_param_name = dependent_on_param.split(" ")[1]
        if dependent_on_param_name in parent_params.keys():
            limit_value = parent_params[dependent_on_param_name]
        else:
            limit_value = default_value

        epsilon = 0.000001
        if limit_value == epsilon:
            epsilon /= 10

        if dependent_on_param_op == ">":
            v_min = limit_value + epsilon
        elif dependent_on_param_op == ">=":
            v_min = limit_value
        elif dependent_on_param_op == "<":
            v_max = limit_value - epsilon
        elif dependent_on_param_op == "<=":
            v_max = limit_value

    return v_min, v_max


def report_healthy(path, message, clear=False):
    """Write health message to the provided file"""
    mode = "w" if clear else "a"
    with open(path, mode, encoding='utf-8') as f:
        f.write(f"Healthy at {datetime.datetime.now().isoformat()}\n")
        if message:
            f.write(str(message) + "\n")


def wait_for_job_completion(job_id):
    """Check if the provided job_id is actively running and wait until completion"""
    if BACKEND == "local-docker":
        while True:
            handler = DockerHandler.get_handler_for_container(job_id)
            if not handler:
                return
            time.sleep(5)

    config.load_incluster_config()
    while True:
        dgx_active_jobs = []
        if BACKEND == "NVCF":
            custom_api = client.CustomObjectsApi()
            crd_group = 'nvcf-job-manager.nvidia.io'
            crd_version = 'v1alpha1'
            crd_plural = 'nvcfjobs'
            # List all instances of the Custom Resource across all namespaces
            custom_resources = custom_api.list_cluster_custom_object(crd_group, crd_version, crd_plural)
            dgx_active_jobs = [dgx_cr["spec"].get("job_id") for dgx_cr in custom_resources['items']]

        ret = client.BatchV1Api().list_job_for_all_namespaces()
        active_jobs = dgx_active_jobs + [job.metadata.name for job in ret.items]
        active_jobs = list(set(active_jobs))
        if job_id not in active_jobs:
            break
        time.sleep(5)


def delete_lingering_checkpoints(epoch_number, path):
    """Delete checkpoints which are present even after job deletion"""
    trained_files = (
        glob.glob(path + "/**/*.tlt", recursive=True) +
        glob.glob(path + "/**/*.hdf5", recursive=True) +
        glob.glob(path + "/**/*.pth", recursive=True) +
        glob.glob(path + "/**/*.ckzip", recursive=True)
    )
    for file_name in trained_files:
        if os.path.isfile(file_name):
            if not (f"{epoch_number}.tlt" in file_name or
                    f"{epoch_number}.hdf5" in file_name or
                    f"{epoch_number}.pth" in file_name or
                    f"{epoch_number}.ckzip" in file_name):
                os.remove(file_name)


class Recommendation:
    """Recommendation class for AutoML recommendations"""

    def __init__(self, identifier, specs, metric):
        """Initialize the Recommendation class

        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
        """
        assert type(identifier) is int, f"Recommendation identifier must be an integer, got {type(identifier)}"
        self.id = identifier

        assert type(specs) is dict, f"Recommendation specs must be a dictionary, got {type(specs)}"
        self.specs = specs

        self.job_id = None
        self.status = JobStates.pending
        self.result = 0.0
        self.best_epoch_number = ""
        self.metric = metric

    def items(self):
        """Returns specs.items"""
        return self.specs.items()

    def get(self, key):
        """Returns value of requested key in the spec"""
        return self.specs.get(key, None)

    def assign_job_id(self, job_id):
        """Associates provided job id to the class objects job id"""
        assert type(job_id) is str, f"Job ID must be a string, got {type(job_id)}"
        self.job_id = job_id

    def update_result(self, result):
        """Update the result value"""
        result = float(result)
        assert type(result) is float, f"Result must be a float value, got {type(result)}"
        self.result = result

    def update_status(self, status):
        """Update the status value"""
        assert type(status) is str, f"Status must be a string, got {type(status)}"
        self.status = status

    def __repr__(self):
        """Constructs a dictionary with the class members and returns them"""
        return f"id: {self.id}\njob_id: {self.job_id}\nresult: {self.result}\nstatus: {self.status}"


class ResumeRecommendation:
    """Recommendation class for Hyperband resume experiments"""

    def __init__(self, identity, specs):
        """Initialize the ResumeRecommendation class

        Args:
            identity: the id of the recommendation
            specs: the specs/config of the recommendation
        """
        self.id = identity
        self.specs = specs


class JobStates():
    """Various states of an automl job"""

    pending = "pending"
    started = "started"
    running = "running"
    success = "success"
    failure = "failure"
    error = "error"  # alias for failure
    done = "done"  # alias for success
    canceled = "canceled"
    canceling = "canceling"
