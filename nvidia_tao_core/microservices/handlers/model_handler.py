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

"""Model publishing and management handler module"""
import logging
import traceback

from nvidia_tao_core.microservices.constants import MAXINE_NETWORKS
from nvidia_tao_core.microservices.utils import ngc_utils
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    check_read_access,
    get_handler_job_metadata,
    resolve_metadata
)
from nvidia_tao_core.microservices.utils.handler_utils import (
    Code,
    resolve_checkpoint_root_and_search
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelHandler:
    """Handles model publishing and removal operations."""

    @staticmethod
    def publish_model(org_name, team_name, experiment_id, job_id, display_name, description):
        """Publish a model with the specified details after validating the job status.

        Parameters:
        org_name (str): The name of the organization.
        team_name (str): The name of the team.
        experiment_id (str): UUID corresponding to the experiment.
        job_id (str): UUID corresponding to the job.
        display_name (str): Display name for the model.
        description (str): Description for the model.

        Returns:
        Code: A response code (200 if the model is successfully published, 404 or 403 for errors).
              - 200: Model successfully created and uploaded.
              - 404: If experiment, job, or relevant files are not found.
              - 403: If the user does not have permission to publish the model.
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {"message": "Experiment not found"})

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {"message": "Experiment cant be read"})

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {"message": "Job trying to retrieve not found"})

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(404, {"message": "Job is not in success or Done state"})
        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(
                404,
                {"message": "Publish model is available only for train, distill, quantize, prune, retrain, export, "
                 "gen_trt_engine actions"}
            )

        try:
            network_arch = handler_metadata.get('network_arch')
            source_files = []
            if job_action == 'gen_trt_engine' and network_arch in MAXINE_NETWORKS:
                encoder_regex = r'.*encoder.*\.(engine|engine\.trtpkg)$'
                encoder_file = resolve_checkpoint_root_and_search(handler_metadata, job_id, regex=encoder_regex)
                if encoder_file:
                    source_files.append(encoder_file)
                decoder_regex = r'.*decoder.*\.(engine|engine\.trtpkg)$'
                decoder_file = resolve_checkpoint_root_and_search(handler_metadata, job_id, regex=decoder_regex)
                if decoder_file:
                    source_files.append(decoder_file)
            else:
                source_file = resolve_checkpoint_root_and_search(handler_metadata, job_id)
                source_files.append(source_file)
            if not source_files:
                return Code(404, {"message": "Unable to find a model for the given job"})

            # Create NGC model
            ngc_key = ngc_utils.get_user_key(user_id, org_name)
            if not ngc_key:
                return Code(403, {"message": "User does not have access to publish model"})

            code, message = ngc_utils.create_model(
                org_name, team_name, handler_metadata, source_files[0], ngc_key, display_name, description
            )
            if code not in [200, 201]:
                logger.error("Error while creating NGC model")
                return Code(code, {"message": message})

            # Upload model version
            response_code, response_message = ngc_utils.upload_model(
                org_name, team_name, handler_metadata, source_files, ngc_key, job_id, job_action
            )
            if "already exists" in response_message:
                response_message = (
                    "Version trying to upload already exists, use remove_published_model endpoint to reupload the model"
                )
            return Code(response_code, {"message": response_message})
        except Exception as e:
            logger.error("Exception thrown in publish_model is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, {"message": "Unable to publish model"})

    @staticmethod
    def remove_published_model(org_name, team_name, experiment_id, job_id):
        """Remove a previously published model.

        Parameters:
        org_name (str): The name of the organization.
        team_name (str): The name of the team.
        experiment_id (str): UUID corresponding to the experiment.
        job_id (str): UUID corresponding to the job.

        Returns:
        Code: A response code (200 if the model is successfully removed, 404 for errors).
              - 200: Successfully deleted the model.
              - 404: If experiment, job, or the published model is not found.
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {"message": "Experiment not found"})

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {"message": "Experiment cant be read"})

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {"message": "Job trying to retrieve not found"})

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(404, {"message": "Job is not in success or Done state"})
        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(
                404,
                {},
                "Delete published model is available only for train, distill, ",
                "quantize, prune, retrain, export, gen_trt_engine actions"
            )

        try:
            ngc_key = ngc_utils.get_user_key(user_id, org_name)
            if not ngc_key:
                return Code(403, {"message": "User does not have access to remove published model"})

            response = ngc_utils.delete_model(
                org_name, team_name, handler_metadata, ngc_key, job_id, job_action
            )
            if response.ok:
                return Code(response.status_code, {"message": "Successfully deleted model"})
            return Code(response.status_code, {"message": "Unable to delete published model"})
        except Exception as e:
            logger.error("Exception thrown in remove_published_model is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, {"message": "Unable to delete published model"})
