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

"""Templates for Python files."""

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TEMPLATE_TIS_MODEL = """
import json
import os
import shutil

import numpy as np
import triton_python_backend_utils as pb_utils
from monai.bundle import ConfigWorkflow
import torch
import base64
import ast
import sys

def unlink(path):
    if os.path.islink(path):
        os.unlink(path)

def find_inference_file(path, ext):
    for e in ext:
        name = f"inference{{e}}"
        if os.path.exists(os.path.join(path, name)):
            return name
    return None

class TritonPythonModel:

    def initialize(self, args):
        self.model_config = json.loads(args["model_config"])
        self.model_version = args["model_version"]
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        bundle_root = os.path.join(parent_dir, "{bundle_name}")
        sys.path = [bundle_root] + sys.path

        ext = [".json", ".yaml", ".yml"]
        inference_config_name = find_inference_file(os.path.join(bundle_root, "configs"), ext)

        self.workflow = ConfigWorkflow(
            workflow_type="infer",
            bundle_root=bundle_root,
            config_file=os.path.join(bundle_root, f"configs/{{inference_config_name}}"),
            logging_file=os.path.join(bundle_root, "configs/logging.conf"),
            meta_file=os.path.join(bundle_root, "configs/metadata.json"),
        )
        # update workflow
        override = {override}
        self.workflow.parser.update(override)
        self.workflow.initialize()
        logger.info("model initialized!!!")

    def execute(self, requests):
        responses = []
        for request in requests:
            # input path is a folder of dicom series, and can contain multiple files
            input_path = pb_utils.get_input_tensor_by_name(request, "INPUT_PATH").as_numpy().tolist()
            if len(input_path) > 1:
                input_path = [p.decode().rstrip("/") for p in input_path]
            else:
                input_path = input_path[0].decode().rstrip("/")
            output_dir = pb_utils.get_input_tensor_by_name(request, "OUTPUT_DIR").as_numpy()[0].decode()
            encode_prompts = pb_utils.get_input_tensor_by_name(request, "PROMPTS").as_numpy()[0].decode()
            prompts = ast.literal_eval(base64.b64decode(encode_prompts).decode("utf-8"))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            unlink("{output_dir}")
            os.symlink(output_dir, "{output_dir}")

            input_data = {{"{image_key}": input_path}}
            input_data.update(prompts)
            self.workflow.parser.ref_resolver.items["dataset"].config["data"][0] = input_data
            logger.info("model version: %s", self.model_version)

            # remove existing files/dirs in output_dir
            dir_contents = os.listdir(output_dir)
            for item in dir_contents:
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

            output0_tensor = pb_utils.Tensor("OUTPUT_PATH", np.array([output_dir], dtype=np.object_))

            error_msg = ""
            try:
                # TODO: better to call self.workflow.run() (it requires the enhancement of monai.bundle code)
                self.workflow.evaluator.run()
                if len(os.listdir(output_dir)) == 0:
                    error_msg = "Cannot find output file"
            except Exception as e:
                error_msg = str(e)
                torch.cuda.empty_cache()
                logger.error(f"Got an unexcepted error: {error_msg}")
                # reinitialize if meet exceptions
                self.workflow.initialize()
            output1_tensor = pb_utils.Tensor("ERROR_MESSAGE", np.array([error_msg], dtype=np.object_))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output0_tensor, output1_tensor],
            )
            responses.append(inference_response)

            # unlink
            unlink("{output_dir}")

        return responses

    def finalize(self):
        self.workflow.finalize()

"""

TEMPLATE_TIS_CONFIG = """
name: "{tis_model}"
backend: "python"

input [
  {{
    name: "INPUT_PATH"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }}
]

input [
  {{
    name: "OUTPUT_DIR"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

input [
  {{
    name: "PROMPTS"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "OUTPUT_PATH"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

output [
  {{
    name: "ERROR_MESSAGE"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
    gpus: [ 0 ]
  }}
]

version_policy: {{
    latest: {{ num_versions: 1}}
}}
"""


TEMPLATE_CONTINUAL_LEARNING = """
import os
import shutil
import sys
import traceback
from time import sleep

sys.path.append("/opt/api")

from nvidia_tao_core.microservices.handlers.monai.helpers import DynamicSorter
from nvidia_tao_core.microservices.handlers.stateless_handlers import printc, update_job_status
from nvidia_tao_core.microservices.utils import safe_load_file
from nvidia_tao_core.microservices.job_utils.annotation_job_utils import (check_for_cancelation,
                                            handle_job_updates,
                                            initialize_cl_tracker,
                                            load_initial_state,
                                            process_notification_record)

notify_record = "{notify_record}"
job_context_dict = {job_context_dict}
logfile = "{logfile}"
handler_root = "{handler_root}"
logs_from_toolkit = "{logs_from_toolkit}"


if __name__ == "__main__":
    try:
        (
            org_name,
            experiment_id,
            cl_job_id,
            train_spec,
            round_size,
            stop_criteria,
            job_metadata_file,
            latest_mod_time,
            latest_record
        ) = load_initial_state(job_context_dict, handler_root, notify_record)

        cl_tracker, cl_state = initialize_cl_tracker(stop_criteria)

        metric_sorter = DynamicSorter()
        jobs_trigger = []
        jobs_done = []

        while (not cl_tracker.should_stop(cl_state)):
            metadata = safe_load_file(job_metadata_file)
            check_for_cancelation(metadata, jobs_trigger, jobs_done, job_context_dict)
            update_job_status(experiment_id, cl_job_id, status="Running")
            # Process notification record updates and potentially trigger a new training job
            (
                latest_mod_time,
                latest_record,
            ) = process_notification_record(
                notify_record,
                latest_mod_time,
                latest_record,
                train_spec,
                round_size,
                cl_state,
                job_context_dict,
                jobs_trigger
            )

            # Check the status of all training jobs and handle triton model updates
            handle_job_updates(cl_state, jobs_trigger, jobs_done, metric_sorter, job_context_dict)
            sleep(15)  # Check every 15 seconds
    except Exception as e:
        # Something went wrong inside...
        logger.error(traceback.format_exc())
        shutil.copy(logfile, logs_from_toolkit)
        update_job_status(experiment_id, cl_job_id, status="Error")
        sys.exit(1)

    update_job_status(experiment_id, cl_job_id, status="Done")
    printc("Continual Learning job completed.", context=job_context_dict, keys="handler_id", file=sys.stderr)

"""
