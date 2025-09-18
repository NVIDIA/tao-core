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

"""AutoML handler modules"""
import os
import json
import time
from copy import deepcopy
from datetime import datetime, timezone
import sysconfig
import logging

from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    get_handler_metadata,
    get_handler_type,
    get_jobs_root,
    save_automl_controller_info,
    serialize_object,
    write_job_metadata,
    update_handler_with_jobs_info,
    get_automl_controller_info
)
from nvidia_tao_core.microservices.handlers.utilities import Code, decrypt_handler_metadata
from nvidia_tao_core.microservices.handlers.docker_images import DOCKER_IMAGE_MAPPER
from nvidia_tao_core.microservices.job_utils import executor as jobDriver

# TODO Make sure the image name is current docker tag of the API
image = DOCKER_IMAGE_MAPPER["API"]

logger = logging.getLogger(__name__)


class AutoMLHandler:
    """Handles AutoML job operations including starting, stopping, resuming, deleting, and retrieving job metadata.

    - **Start**: Launches an AutoML job as a Kubernetes job.
    - **Stop**: Terminates an ongoing AutoML job and cancels any pending recommendations.
    - **Resume**: Restarts a previously stopped AutoML job with restored settings.
    - **Delete**: Deletes an AutoML job (same as AppHandler behavior).
    - **Download**: Downloads artifacts from an AutoML job (same as AppHandler behavior).
    - **Retrieve**: Constructs and returns job metadata based on the job's status.
    """

    @staticmethod
    def start(user_id, org_name, experiment_id, job_id, handler_metadata, name="", platform_id=""):
        """Starts an AutoML job by executing `automl_start.py` with the provided parameters.

        Args:
            user_id (str): ID of the user initiating the job.
            org_name (str): Name of the organization.
            experiment_id (str): ID of the associated experiment.
            job_id (str): Unique identifier for the AutoML job.
            handler_metadata (dict): Metadata containing AutoML configuration settings.
            name (str, optional): Name of the job. Defaults to "automl train job".
            platform_id (str, optional): Platform identifier for execution. Defaults to "".
        """
        job_metadata = {
            "name": name,
            "id": job_id,
            "org_name": org_name,
            "parent_id": None,
            "platform_id": platform_id,
            "action": "train",
            "created_on": datetime.now(tz=timezone.utc),
            "experiment_id": experiment_id,
            "status": "Pending",
            "job_details": {}
        }
        write_job_metadata(job_id, job_metadata)
        update_handler_with_jobs_info(job_metadata, experiment_id, job_id, "experiments")

        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            os.makedirs(root)

        if not name:
            name = "automl train job"
        network = get_handler_type(handler_metadata)
        metric = handler_metadata.get("metric", "map")
        automl_settings = handler_metadata.get("automl_settings", {})
        automl_algorithm = automl_settings.get("automl_algorithm", "Bayesian")
        automl_max_recommendations = automl_settings.get("automl_max_recommendations", 20)
        automl_delete_intermediate_ckpt = automl_settings.get("automl_delete_intermediate_ckpt", True)
        automl_R = automl_settings.get("automl_R", 27)
        automl_nu = automl_settings.get("automl_nu", 3)
        epoch_multiplier = automl_settings.get("epoch_multiplier", 1)
        automl_hyperparameters = automl_settings.get("automl_hyperparameters", "[]")
        override_automl_disabled_params = automl_settings.get("override_automl_disabled_params", False)

        workspace_id = handler_metadata.get("workspace")
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        decrypted_workspace_metadata = deepcopy(workspace_metadata)
        decrypt_handler_metadata(decrypted_workspace_metadata)
        decrypted_workspace_metadata.pop('_id', None)

        # Call the script
        logger.info("Starting automl %s", job_id)
        python_lib_path = sysconfig.get_path("purelib")
        automl_script = os.path.join(python_lib_path, "nvidia_tao_core/microservices/automl_start.py")

        run_command = (
            f'umask 0 && python3 {automl_script} '
            f'--user_id={user_id} '
            f'--org_name={org_name} '
            f'--name="{name}" '
            f'--root={root} '
            f'--automl_job_id={job_id} '
            f'--network={network} '
            f'--experiment_id={experiment_id} '
            f'--resume=False '
            f'--automl_algorithm={automl_algorithm} '
            f'--automl_max_recommendations={automl_max_recommendations} '
            f'--automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} '
            f'--automl_R={automl_R} '
            f'--automl_nu={automl_nu} '
            f'--metric={metric} '
            f'--epoch_multiplier={epoch_multiplier} '
            f'--automl_hyperparameters="{automl_hyperparameters}" '
            f'--override_automl_disabled_params={override_automl_disabled_params} '
            f"--decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata, default=str)}'"
        )
        if platform_id:
            run_command = f"{run_command} --platform_id={platform_id}"

        jobDriver.create(
            org_name,
            job_id,
            image,
            run_command,
            num_gpu=0,
            automl_brain=True,
            automl_exp_job=False
        )

    @staticmethod
    def stop(user_id, org_name, experiment_id, job_id):
        """Stops a running AutoML job and cancels any active recommendations.

        Args:
            user_id (str): ID of the user.
            org_name (str): Name of the organization.
            experiment_id (str): ID of the associated experiment.
            job_id (str): Unique identifier for the AutoML job.

        Returns:
            Code: Status code and message indicating job cancellation success or failure.
        """
        logger.info("Stopping automl")

        try:
            jobDriver.delete(job_id, use_ngc=False)
            k8s_status = jobDriver.status(
                org_name,
                experiment_id,
                job_id,
                "experiments",
                use_ngc=False,
                automl_exp_job=False
            )
            while k8s_status in ("Done", "Error", "Running", "Pending"):
                if k8s_status in ("Done", "Error"):
                    break
                k8s_status = jobDriver.status(
                    org_name,
                    experiment_id,
                    job_id,
                    "experiments",
                    use_ngc=False,
                    automl_exp_job=False
                )
                time.sleep(5)
            recommendations = get_automl_controller_info(job_id)
            for recommendation in recommendations:
                recommendation_job_id = recommendation.get("job_id")
                if recommendation_job_id:
                    if recommendation.get("status") in ("pending", "running", "started"):
                        recommendation["status"] = "canceling"
                        save_automl_controller_info(job_id, recommendations)
                    jobDriver.delete(recommendation_job_id)
                    rec_k8s_status = jobDriver.status(
                        org_name,
                        experiment_id,
                        recommendation_job_id,
                        "experiments",
                        automl_exp_job=True
                    )
                    while rec_k8s_status in ("Done", "Error", "Running", "Pending"):
                        if rec_k8s_status in ("Done", "Error"):
                            break
                        rec_k8s_status = jobDriver.status(
                            org_name,
                            experiment_id,
                            recommendation_job_id,
                            "experiments",
                            automl_exp_job=True
                        )
                        time.sleep(5)
                    if recommendation.get("status") in ("pending", "running", "started", "canceling"):
                        recommendation["status"] = "canceled"
                        save_automl_controller_info(job_id, recommendations)
        except Exception as e:
            logger.error("Exception thrown in AutomlHandler stop is %s", str(e))
            return Code(404, [], "job cannot be stopped in platform")

        return Code(200, {"message": f"job {job_id} cancelled"})

    @staticmethod
    def resume(user_id, org_name, experiment_id, job_id, handler_metadata, name="", platform_id=""):
        """Resumes a previously stopped AutoML job by re-running `automl_start.py` with the resume flag.

        Args:
            user_id (str): ID of the user.
            org_name (str): Name of the organization.
            experiment_id (str): ID of the associated experiment.
            job_id (str): Unique identifier for the AutoML job.
            handler_metadata (dict): Metadata containing AutoML configuration settings.
            name (str, optional): Name of the job. Defaults to "automl train job".
            platform_id (str, optional): Platform identifier for execution. Defaults to "".
        """
        logger.info("Resuming automl %s", job_id)

        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            os.makedirs(root)

        if not name:
            name = "automl train job"
        network = get_handler_type(handler_metadata)
        metric = handler_metadata.get("metric", "map")
        automl_settings = handler_metadata.get("automl_settings", {})
        automl_algorithm = automl_settings.get("automl_algorithm", "Bayesian")
        automl_max_recommendations = automl_settings.get("automl_max_recommendations", 20)
        automl_delete_intermediate_ckpt = automl_settings.get("automl_delete_intermediate_ckpt", True)
        automl_R = automl_settings.get("automl_R", 27)
        automl_nu = automl_settings.get("automl_nu", 3)
        epoch_multiplier = automl_settings.get("epoch_multiplier", 1)
        automl_hyperparameters = automl_settings.get("automl_hyperparameters", "[]")
        override_automl_disabled_params = automl_settings.get("override_automl_disabled_params", False)

        workspace_id = handler_metadata.get("workspace")
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        decrypted_workspace_metadata = deepcopy(workspace_metadata)
        decrypt_handler_metadata(decrypted_workspace_metadata)
        decrypted_workspace_metadata.pop('_id', None)

        # Call the script
        python_lib_path = sysconfig.get_path("purelib")
        automl_script = os.path.join(python_lib_path, "nvidia_tao_core/microservices/automl_start.py")
        run_command = (
            f"umask 0 && python3 {automl_script} "
            f'--user_id={user_id} '
            f'--org_name={org_name} '
            f'--name="{name}" '
            f'--root={root} '
            f'--automl_job_id={job_id} '
            f'--network={network} '
            f'--experiment_id={experiment_id} '
            f'--resume=True '
            f'--automl_algorithm={automl_algorithm} '
            f'--automl_max_recommendations={automl_max_recommendations} '
            f'--automl_delete_intermediate_ckpt={automl_delete_intermediate_ckpt} '
            f'--automl_R={automl_R} '
            f'--automl_nu={automl_nu} '
            f'--metric={metric} '
            f'--epoch_multiplier={epoch_multiplier} '
            f'--automl_hyperparameters="{automl_hyperparameters}" '
            f'--override_automl_disabled_params={override_automl_disabled_params} '
            f"--decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata, default=serialize_object)}'"
        )
        if platform_id:
            run_command = f"{run_command} --platform_id={platform_id}"
        jobDriver.create(org_name, job_id, image, run_command, num_gpu=0, automl_brain=True, automl_exp_job=False)
