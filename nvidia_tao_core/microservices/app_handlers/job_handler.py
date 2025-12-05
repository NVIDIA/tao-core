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

"""Job handler module for managing job operations"""
import re
import os
import glob
import tarfile
import time
import traceback
import uuid
import logging
from datetime import datetime, timezone

from nvidia_tao_core.microservices.constants import (
    MISSING_EPOCH_FORMAT_NETWORKS
)
from nvidia_tao_core.microservices.handlers import stateless_handlers
from nvidia_tao_core.microservices.handlers.nvcf_handler import get_available_nvcf_instances
from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    check_read_access,
    check_write_access,
    get_handler_job_metadata,
    get_handler_log_root,
    get_jobs_root,
    get_jobs_for_handler,
    is_request_automl,
    update_job_status,
    save_dnn_status,
    save_job_specs,
    resolve_metadata,
    resolve_existence
)
from nvidia_tao_core.microservices.handlers.utilities import (
    Code,
    download_log_from_cloud,
    get_files_from_cloud,
    get_num_gpus_from_spec
)
from nvidia_tao_core.microservices.job_utils.executor import (
    JobExecutor,
    StatefulSetExecutor
)
from nvidia_tao_core.microservices.job_utils.workflow_driver import create_job_context, on_delete_job, on_new_job
from nvidia_tao_core.microservices.job_utils.automl_job_utils import on_delete_automl_job
from nvidia_tao_core.microservices.utils import (
    check_and_convert
)

if os.getenv("BACKEND"):
    from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler

from nvidia_tao_core.microservices.app_handlers.utils import (
    get_job,
    resolve_metadata_with_jobs,
    get_job_logs as get_job_logs_util
)

# Configure logging
logger = logging.getLogger(__name__)

# Identify if workflow is on NGC
BACKEND = os.getenv("BACKEND", "local-k8s")


class JobHandler:
    """Handles job operations including running, canceling, pausing, and retrieving job information."""

    @staticmethod
    def job_run(
        org_name,
        handler_id,
        parent_job_id,
        action,
        kind,
        specs=None,
        name=None,
        description=None,
        num_gpu=-1,
        platform_id=None,
        from_ui=False
    ):
        """Runs a job based on the specified parameters.

        This method initiates a job based on the given organization, experiment,
        dataset details, and other job specifications. It handles different job
        scenarios including AutoML and regular jobs, and validates all necessary
        conditions before scheduling the job.

        Args:
            org_name (str): The organization name.
            handler_id (str): UUID corresponding to experiment or dataset.
            parent_job_id (str): UUID of the parent job.
            action (str): The action to be performed.
            kind (str): The type of resource ("experiment" or "dataset").
            specs (dict, optional): Specifications for the job.
            name (str, optional): The job's name.
            description (str, optional): The job's description.
            num_gpu (int, optional): The number of GPUs to allocate.
            platform_id (str, optional): The platform ID for job execution.
            from_ui (bool, optional): Indicates whether the job call is from the UI.

        Returns:
            Code: A response code object containing the status and job ID or error details:
                  - 200: Job successfully queued.
                  - 400: If job execution was unsuccessful.
                  - 404: If dataset/experiment/action not found or access is denied.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{handler_id} {kind} doesn't exist")

        user_id = handler_metadata.get("user_id")
        network_arch = handler_metadata.get("network_arch", "")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{org_name} has no write access to {handler_id}")

        if not action:
            return Code(404, [], "action not sent")

        if action not in handler_metadata.get("actions", []):
            if not (kind == "dataset" and action == "validate_images"):
                return Code(
                    404, {},
                    f"Action {action} requested not in {','.join(handler_metadata.get('actions', []))}"
                )

        if not user_id:
            return Code(
                404, [],
                "User ID couldn't be found in the experiment metadata. "
                "Try creating the experiment again"
            )

        if parent_job_id:
            parent_job_metadata = get_handler_job_metadata(parent_job_id)
            if not parent_job_metadata:
                return Code(404, [], f"Parent job {parent_job_id} not found")

            if kind != "dataset":
                parent_handler_id = parent_job_metadata.get("dataset_id")
                parent_kind = "dataset"
                if not parent_handler_id:
                    parent_kind = "experiment"
                    parent_handler_id = parent_job_metadata.get("experiment_id")
                if not parent_handler_id:
                    return Code(404, [], f"Unable to identify {parent_kind} id for parent job {parent_job_id}")

                if parent_kind == "experiment":
                    if parent_handler_id != handler_id:
                        return Code(
                            404, [],
                            f"Parent job {parent_job_id} trying to assign doesn't belong to current experiment "
                            f"{handler_id}, it belongs to experiment {parent_handler_id}"
                        )

        if BACKEND == "NVCF":
            available_nvcf_instances = get_available_nvcf_instances(user_id, org_name)
            if not available_nvcf_instances:
                platform_id = "052fc221-ffaa-5c15-8d22-b663e7339349"
            else:
                if not platform_id:
                    def get_powers_of_2(start: int):
                        power = 1
                        powers = []
                        while power <= 8:
                            if power >= start:
                                powers.append(power)
                            power *= 2
                        return powers

                    num_gpu = 1
                    if specs:
                        num_gpu = get_num_gpus_from_spec(specs, action, network=network_arch, default=1)
                    gpu_based_subset = {}
                    valid_gpu_counts = get_powers_of_2(num_gpu)
                    for nvcf_instance_id, nvcf_instance_info in available_nvcf_instances.items():
                        cluster = available_nvcf_instances[nvcf_instance_id].get('cluster', '')
                        instance_type = available_nvcf_instances[nvcf_instance_id].get('instance_type', '')
                        if cluster == 'GFN':
                            instance_type = instance_type.replace("2x", "1x").replace("4x", "2x")
                        for valid_gpu_count in valid_gpu_counts:
                            if f"{valid_gpu_count}x" in instance_type:
                                gpu_based_subset[nvcf_instance_id] = nvcf_instance_info

                    if gpu_based_subset:
                        sorted_platform_ids = sorted(
                            gpu_based_subset,
                            key=lambda x: gpu_based_subset[x]['current_available'],
                            reverse=True
                        )
                    else:
                        sorted_platform_ids = sorted(
                            available_nvcf_instances,
                            key=lambda x: available_nvcf_instances[x]['current_available'],
                            reverse=True
                        )
                    platform_id = sorted_platform_ids[0]
                if platform_id not in available_nvcf_instances:
                    return Code(
                        404, [],
                        f"Requested NVCF resource {platform_id} not available. "
                        f"Valid platform_id options are {str(available_nvcf_instances.keys())}"
                    )
                if available_nvcf_instances[platform_id]["current_available"] == 0:
                    return Code(
                        404, [],
                        f"Requested NVCF resource {platform_id} maxed out. Choose other platform_id options, "
                        f"valid options are: {str(available_nvcf_instances.keys())}"
                    )

        try:
            job_id = str(uuid.uuid4())
            if specs:
                from nvidia_tao_core.microservices.app_handlers.spec_handler import SpecHandler
                spec_schema_response = SpecHandler.get_spec_schema(user_id, org_name, handler_id, action, kind)
                if spec_schema_response.code == 200:
                    spec_schema = spec_schema_response.data
                    default_spec = spec_schema["default"]
                    check_and_convert(specs, default_spec)
            msg = ""
            if is_request_automl(handler_id, action, kind):
                logger.info("Creating AutoML job %s", job_id)
                AutoMLHandler.start(
                    user_id,
                    org_name,
                    handler_id,
                    job_id,
                    handler_metadata,
                    name=name,
                    platform_id=platform_id
                )
                msg = "AutoML "
            else:
                logger.info("Creating job %s", job_id)
                job_context = create_job_context(
                    parent_job_id,
                    action,
                    job_id,
                    handler_id,
                    user_id,
                    org_name,
                    kind,
                    handler_metadata=handler_metadata,
                    specs=specs,
                    name=name,
                    description=description,
                    num_gpu=num_gpu,
                    platform_id=platform_id
                )
                on_new_job(job_context)
            if specs:
                save_job_specs(job_id, specs)
            return Code(200, job_id, f"{msg}Job scheduled")
        except Exception as e:
            logger.error("Exception thrown in job_run is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(500, [], "Exception in job_run fn")

    @staticmethod
    def job_retry(org_name, handler_id, kind, job_id, from_ui=False):
        """Retries a job based on its status and metadata.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            job_id (str): UUID of the job to retry.
            from_ui (bool): Whether the retry is triggered from the UI (default is False).

        Returns:
            Code: A response indicating the result of the retry action:
                - 200 with the new job UUID if successfully queued.
                - 400 if the job status prevents retrying.
                - 404 if the job or related resources are not found.
        """
        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], f"Job {job_id} doesn't exist")
        status = job_metadata.get('status', 'Unknown')
        if status in ("Done", "Running", "Pending", "Canceling", "Resuming", "Pausing", "Paused"):
            return Code(400, [], f"Unable to retry job with {status} status")

        parent_job_id = job_metadata.get('parent_id')
        action = job_metadata.get('action')
        specs = job_metadata.get('specs')
        name = job_metadata.get('name', "Job") + " Retry"
        description = job_metadata.get('description')
        platform_id = job_metadata.get('platform_id')
        job_response = JobHandler.job_run(org_name, handler_id, parent_job_id, action, kind, specs, name, description,
                                          platform_id=platform_id, from_ui=from_ui)
        return job_response

    @staticmethod
    def job_get_epoch_numbers(user_id, org_name, handler_id, job_id, kind):
        """Retrieves the epoch numbers associated with a given job.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 with a list of epoch numbers if found.
                - 404 if no epoch numbers are found or an error occurs.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        handler_metadata = stateless_handlers.get_handler_metadata(handler_id, kind + "s")
        if not handler_metadata or "user_id" not in handler_metadata:
            return Code(404, [], "job trying to update not found")

        job = get_job(job_id)
        if not job:
            return Code(404, [], "job trying to update not found")
        try:
            job_files, _, _, _ = get_files_from_cloud(handler_metadata, job_id)
            epoch_numbers = []
            for job_file in job_files:
                # Extract numbers before the extension using regex
                match = re.search(r'(\d+)(?=\.(pth|hdf5|tlt)$)', job_file)
                if match:
                    epoch_numbers.append(match.group(1))
            return Code(200, {"data": epoch_numbers}, "Job status updated")
        except Exception:
            logger.error(traceback.format_exc())
            return Code(404, [], "Exception caught during getting epoch numbers")

    @staticmethod
    def job_status_update(org_name, handler_id, job_id, kind, callback_data):
        """Updates the status of a given job.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            callback_data (dict): Data containing the new status information.

        Returns:
            Code: A response indicating the result:
                - 200 if the job status was successfully updated.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        action = job.get("action", "")
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        automl = False
        if is_request_automl(handler_id, action, kind) and action == "train":
            automl = True
        save_dnn_status(job_id, automl, callback_data, handler_id, kind)
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_log_update(org_name, handler_id, job_id, kind, callback_data):
        """Appends log contents to the job's log file.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            callback_data (dict): Data containing the log contents to append.

        Returns:
            Code: A response indicating the result:
                - 200 if the log was successfully updated.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        action = job.get("action", "")
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        handler_log_root = get_handler_log_root(user_id, org_name, handler_id)
        log_file = os.path.join(handler_log_root, job_id + ".txt")
        if is_request_automl(handler_id, action, kind):
            job_root = os.path.join(get_jobs_root(user_id, org_name), job_id)
            experiment_number = callback_data.get("experiment_number", "0")
            handler_log_root = f"{job_root}/experiment_{experiment_number}"
            log_file = f"{job_root}/experiment_{experiment_number}/log.txt"
        if not os.path.exists(handler_log_root):
            os.makedirs(handler_log_root, exist_ok=True)
        with open(log_file, "a", encoding='utf-8') as file_ptr:
            file_ptr.write(callback_data["log_contents"])
            file_ptr.write("\n")
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_list(user_id, org_name, handler_id, kind):
        """Retrieves a list of jobs associated with a given handler.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 with a list of jobs if found.
                - 404 if the handler or jobs are not found.
        """
        if handler_id not in ("*", "all") and not resolve_existence(kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if handler_id not in ("*", "all") and not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        return_metadata = resolve_metadata_with_jobs(user_id, org_name, kind, handler_id).get("jobs", [])

        return Code(200, return_metadata, "Jobs retrieved")

    @staticmethod
    def job_retrieve(org_name, handler_id, job_id, kind, return_specs=False):
        """Retrieve the specified job based on its ID and kind (experiment or dataset).

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job to be retrieved.
        kind (str): The type of job, either "experiment" or "dataset".
        return_specs (bool): Flag indicating whether to return specs.

        Returns:
        Code: A response code (200 if the job is found, 404 if not found).
              - 200: Returns a dictionary following the JobResultSchema.
              - 404: Returns an empty dictionary if the job or handler is not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, f"{kind} not found")

        job_meta = get_job(job_id)
        if not job_meta:
            return Code(404, {}, "Job trying to retrieve not found")
        job_meta.pop('num_gpu', None)
        if not return_specs:
            if "specs" in job_meta:
                _ = job_meta.pop("specs")
        return Code(200, job_meta, "Job retrieved")

    @staticmethod
    def job_cancel(org_name, handler_id, job_id, kind):
        """Cancels a job based on its current status.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job to cancel.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if the job was successfully canceled or if the job cannot be canceled due to its current status.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if job_id not in handler_metadata.get("jobs", {}).keys():
            return Code(404, [], f"Job to cancel not found in the {kind}.")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to cancel not found")
        action = job_metadata.get("action", "")

        if is_request_automl(handler_id, action, kind):
            update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            automl_response = AutoMLHandler.stop(user_id, org_name, handler_id, job_id)
            # Remove any pending jobs from Workflow queue
            try:
                on_delete_automl_job(job_id)
            except Exception as e:
                logger.error("Exception thrown in automl_job_cancel is %s", str(e))
                return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})
            update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            return automl_response

        # If job is error / done, then cancel is NoOp
        job_status = job_metadata.get("status", "Error")

        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            return Code(
                200,
                {
                    f"Job {job_id} with current status {job_status} can't be attemped to cancel. "
                    "Current status should be one of Running, Pending, Resuming"
                }
            )
        specs = job_metadata.get("specs", None)
        use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")

        if job_status == "Pending":
            update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            on_delete_job(job_id)
            StatefulSetExecutor().delete_statefulset(job_id, use_ngc=use_ngc)
            update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            return Code(200, {"message": f"Pending job {job_id} cancelled"})

        if job_status == "Running":
            try:
                # Delete K8s job
                update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
                StatefulSetExecutor().delete_statefulset(job_id, use_ngc=use_ngc)
                k8s_status = JobExecutor().get_job_status(
                    org_name,
                    handler_id,
                    job_id,
                    kind + "s",
                    use_ngc=use_ngc,
                    automl_exp_job=False
                )
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        break
                    k8s_status = JobExecutor().get_job_status(
                        org_name,
                        handler_id,
                        job_id,
                        kind + "s",
                        use_ngc=use_ngc,
                        automl_exp_job=False
                    )
                    time.sleep(5)
                update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
                return Code(200, {"message": f"Running job {job_id} cancelled"})
            except Exception as e:
                logger.error("Exception thrown in job_cancel is %s", str(e))
                logger.error("Cancel traceback: %s", traceback.format_exc())
                return Code(404, [], "job not found in platform")
        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def job_pause(org_name, handler_id, job_id, kind):
        """Pauses a job based on its current status.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job to pause.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if the job was successfully paused or if the job cannot be paused due to its current status.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to pause not found")
        action = job_metadata.get("action", "")

        if is_request_automl(handler_id, action, kind):
            update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            automl_response = AutoMLHandler.stop(user_id, org_name, handler_id, job_id)
            # Remove any pending jobs from Workflow queue
            try:
                on_delete_automl_job(job_id)
            except Exception as e:
                logger.error("Exception thrown in automl job_pause is %s", str(e))
                return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})
            update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            return automl_response

        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "retrain"):
            return Code(
                404, [],
                f"Only train, distill, quantize or retrain jobs can be paused. The current action is {job_action}"
            )
        job_status = job_metadata.get("status", "Error")

        # If job is error / done, or one of cancel or pause states then pause is NoOp
        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            return Code(
                200,
                {
                    f"Job {job_id} with current status {job_status} can't be attemped to pause. "
                    "Current status should be one of Running, Pending, Resuming"
                }
            )
        specs = job_metadata.get("specs", None)
        use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")

        if job_status == "Pending":
            update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            on_delete_job(job_id)
            StatefulSetExecutor().delete_statefulset(job_id, use_ngc=use_ngc)
            update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            return Code(200, {"message": f"Pending job {job_id} paused"})

        if job_status == "Running":
            try:
                # Delete K8s job
                update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
                StatefulSetExecutor().delete_statefulset(job_id, use_ngc=use_ngc)
                k8s_status = JobExecutor().get_job_status(
                    org_name,
                    handler_id,
                    job_id,
                    kind + "s",
                    use_ngc=use_ngc,
                    automl_exp_job=False
                )
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        break
                    k8s_status = JobExecutor().get_job_status(
                        org_name,
                        handler_id,
                        job_id,
                        kind + "s",
                        use_ngc=use_ngc,
                        automl_exp_job=False
                    )
                    time.sleep(5)
                update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
                return Code(200, {"message": f"Running job {job_id} paused"})
            except Exception as e:
                logger.error("Exception thrown in job_pause is %s", str(e))
                logger.error("Pause traceback: %s", traceback.format_exc())
                return Code(404, [], "job not found in platform")

        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def all_job_cancel(user_id, org_name, handler_id, kind):
        """Cancels all jobs associated with a given handler.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if all jobs within the experiment can be canceled.
                - 404 if the handler or jobs are not found.
        """
        from nvidia_tao_core.microservices.handlers.stateless_handlers import write_handler_metadata
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        def cancel_jobs_within_handler(cancel_handler_id, cancel_kind):
            cancel_success = True
            cancel_message = ""
            jobs = get_jobs_for_handler(cancel_handler_id, cancel_kind)
            for job_metadata in jobs:
                job_id = job_metadata.get("id")
                job_status = job_metadata.get("status", "Error")
                if job_status not in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"] and job_id:
                    cancel_response = JobHandler.job_cancel(org_name, cancel_handler_id, job_id, cancel_kind)
                    if cancel_response.code != 200:
                        if (type(cancel_response.data) is dict and
                                cancel_response.data.get("error_desc", "") != "incomplete job not found"):
                            cancel_success = False
                            cancel_message += f"Cancelation for job {job_id} failed due to {str(cancel_response.data)} "
            return cancel_success, cancel_message

        if handler_metadata.get("all_jobs_cancel_status") == "Canceling":
            return Code(200, {"message": "Canceling all jobs is already triggered"})

        try:
            handler_metadata["all_jobs_cancel_status"] = "Canceling"
            write_handler_metadata(handler_id, handler_metadata, kind)

            appended_message = ""
            for train_dataset in handler_metadata.get("train_datasets", []):
                jobs_cancel_sucess, message = cancel_jobs_within_handler(train_dataset, "dataset")
                appended_message += message

            eval_dataset = handler_metadata.get("eval_dataset", None)
            if eval_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(eval_dataset, "dataset")
                appended_message += message

            inference_dataset = handler_metadata.get("inference_dataset", None)
            if inference_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(inference_dataset, "dataset")
                appended_message += message

            jobs_cancel_sucess, message = cancel_jobs_within_handler(handler_id, kind)
            appended_message += message

            handler_metadata = resolve_metadata(kind, handler_id)
            if jobs_cancel_sucess:
                handler_metadata["all_jobs_cancel_status"] = "Canceled"
                write_handler_metadata(handler_id, handler_metadata, kind)
                return Code(200, {"message": "All jobs within experiment canceled"})
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], appended_message)
        except Exception as e:
            logger.error("Exception thrown in all_job_cancel is %s", str(e))
            logger.error(traceback.format_exc())
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], "Runtime exception caught during deleting a job")

    @staticmethod
    def job_delete(org_name, handler_id, job_id, kind):
        """Delete the specified job.

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job to be deleted.
        kind (str): The type of job, either "experiment" or "dataset".

        Returns:
        Code: A response code:
                 - 200 if the job is successfully deleted
                 - 404 if not found
                 - 400 if deletion is not allowed
        """
        from nvidia_tao_core.microservices.handlers.stateless_handlers import (
            write_handler_metadata,
            get_handler_status
        )
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to delete not found")

        try:
            # If job is running, cannot delete
            if job_metadata.get("status", "Error") in ["Running", "Pending"]:
                return Code(400, [], "job cannot be deleted")
            # Delete job metadata
            mongo_jobs = MongoHandler("tao", "jobs")
            mongo_jobs.delete_one({'id': job_id})
            # Delete job from handler metadata
            if "jobs" in handler_metadata and job_id in handler_metadata["jobs"]:
                del handler_metadata["jobs"][job_id]
                handler_metadata["status"] = get_handler_status(handler_metadata)
                handler_metadata["last_modified"] = datetime.now(tz=timezone.utc)
                write_handler_metadata(handler_id, handler_metadata, kind)
            # Delete job logs
            job_log_path = os.path.join(
                stateless_handlers.get_handler_log_root(user_id, org_name, handler_id),
                job_id + ".txt"
            )
            if os.path.exists(job_log_path):
                os.remove(job_log_path)
            return Code(200, [job_id], "job deleted")
        except Exception as e:
            logger.error("Exception thrown in job_delete is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, [], "job cannot be deleted")

    @staticmethod
    def job_download(
        org_name,
        handler_id,
        job_id,
        kind,
        file_lists=None,
        best_model=None,
        latest_model=None,
        tar_files=True,
        export_type="tao"
    ):
        """Download files associated with the specified job.

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job.
        kind (str): The type of job, either "experiment" or "dataset".
        file_lists (list, optional): List of files to download (defaults to None).
        best_model (bool, optional): Whether to download the best model (defaults to None).
        latest_model (bool, optional): Whether to download the latest model (defaults to None).
        tar_files (bool, optional): Whether to compress the downloaded files into a tarball (defaults to True).
        export_type (str, optional): The type of export (defaults to "tao").

        Returns:
        Code: A response code (200 if the files are successfully downloaded, 404 if not found, 400 for errors).
              - 200: Successfully downloaded the files.
              - 404: If the job or handler is not found.
              - 400: If there are errors during file download.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, None, f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, None, f"{kind} not found")

        handler_job_metadata = get_handler_job_metadata(job_id)
        if not handler_job_metadata:
            return Code(404, None, "job trying to download not found")

        try:
            root = get_jobs_root(user_id, org_name)

            # Following is for `if export_type == "tao":`
            # Copy job logs from root/logs/<job_id>.txt to root/<job_id>/logs_from_toolkit.txt
            out_tar = os.path.join(root, job_id + ".tar.gz")
            files = [os.path.join(root, job_id)]
            if file_lists or best_model or latest_model:
                files = []
                for file in file_lists:
                    if os.path.exists(os.path.join(root, file)):
                        files.append(os.path.join(root, file))
                action = handler_job_metadata.get("action", "")
                epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
                best_checkpoint_epoch_number = epoch_number_dictionary.get(f"best_model_{job_id}", 0)
                latest_checkpoint_epoch_number = epoch_number_dictionary.get(f"latest_model_{job_id}", 0)
                if (not best_model) and latest_model:
                    best_checkpoint_epoch_number = latest_checkpoint_epoch_number
                network = handler_metadata.get("network_arch", "")
                if network in MISSING_EPOCH_FORMAT_NETWORKS:
                    format_epoch_number = str(best_checkpoint_epoch_number)
                else:
                    format_epoch_number = f"{best_checkpoint_epoch_number:03}"
                if best_model or latest_model:
                    job_root = os.path.join(root, job_id)
                    if (handler_metadata.get("automl_settings", {}).get("automl_enabled") is True and
                       action in ("train", "distill", "quantize")):
                        job_root = os.path.join(job_root, "best_model")
                    find_trained_tlt = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.tlt") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.tlt") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.tlt")
                    )
                    find_trained_pth = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.pth") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.pth") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.pth")
                    )
                    find_trained_hdf5 = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.hdf5") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.hdf5") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.hdf5")
                    )
                    if find_trained_tlt:
                        files.append(find_trained_tlt[0])
                    if find_trained_pth:
                        files.append(find_trained_pth[0])
                    if find_trained_hdf5:
                        files.append(find_trained_hdf5[0])
                if not files:
                    return Code(404, None, "Atleast one of the requested files not present")

            log_root = get_handler_log_root(user_id, org_name, handler_id)
            log_file = glob.glob(f"{log_root}/**/*{job_id}.txt", recursive=True)
            files = list(set(files))

            if tar_files or (not tar_files and len(files) > 1) or files == [os.path.join(root, job_id)]:
                if files == [os.path.join(root, job_id)]:
                    files += log_file
                    files = list(set(files))

                def get_files_recursively(directory):
                    return [
                        file for file in glob.glob(os.path.join(directory, '**'), recursive=True)
                        if os.path.isfile(file) and not file.endswith(".lock")
                    ]
                all_files = []
                for file in files:
                    if os.path.isdir(file):
                        all_files.extend(get_files_recursively(file))
                    elif os.path.isfile(file):
                        all_files.append(file)

                # Appending UUID to not overwrite the tar file created at end of job complete
                out_tar = out_tar.replace(
                    ".tar.gz",
                    str(uuid.uuid4()) + ".tar.gz"
                )
                with tarfile.open(out_tar, "w:gz") as tar:
                    for file_path in all_files:
                        tar.add(file_path, arcname=file_path.replace(root, "", 1).replace(log_root, "", 1))
                return Code(200, out_tar, "selective files of job downloaded")

            if files and os.path.exists(os.path.join(root, files[0])):
                return Code(200, os.path.join(root, files[0]), "single file of job downloaded")
            return Code(404, None, "job output not found")

        except Exception as e:
            logger.error("Exception thrown in job_download is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, None, "job output not found")

    @staticmethod
    def job_list_files(org_name, handler_id, job_id, kind):
        """Lists the files associated with a specific job.

        Args:
            org_name (str): The name of the organization.
            handler_id (str): The UUID corresponding to the experiment or dataset.
            job_id (str): The UUID of the job whose files need to be listed.
            kind (str): The type of handler, either 'experiment' or 'dataset'.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 with a list of file paths if files are found.
                - 404 with an error message if no files are found or access is denied.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        if not job:
            return Code(404, None, "job trying to view not found")

        files, _, _, _ = get_files_from_cloud(handler_metadata, job_id)
        if files:
            return Code(200, files, "Job files retrieved")
        return Code(200, files, "No downloadable files for this job is found")

    @staticmethod
    def get_job_logs(org_name, handler_id, job_id, kind, automl_experiment_index=None):
        """Retrieves real-time logs for a specific job.

        Args:
            org_name (str): The name of the organization.
            handler_id (str): The UUID corresponding to the experiment or dataset.
            job_id (str): The UUID of the job for which logs are being retrieved.
            kind (str): The type of handler, either 'experiment' or 'dataset'.
            automl_experiment_index (int, optional): The index of the AutoML experiment, if applicable.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 with the log content or a detailed message if logs are not yet available.
                - 404 if the log file is not found.
        """
        from nvidia_tao_core.microservices.handlers.stateless_handlers import (
            get_automl_current_rec
        )
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, f"{kind} not found.")

        # Get log file path
        # Normal action log is saved at /orgs/<org_name>/users/<user_id>/logs/<job_id>.txt
        # AutoML train  log is saved at:
        # /orgs/<org_name>/users/<user_id>/jobs/<job_id>/experiment_<recommendation_index>/log.txt
        user_id = handler_metadata.get("user_id")
        log_file_path = os.path.join(get_handler_log_root(user_id, org_name, handler_id), str(job_id) + ".txt")
        job_metadata = get_handler_job_metadata(job_id)
        automl_index = None
        if (handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and
                job_metadata.get("action", "") == "train"):
            root = os.path.join(get_jobs_root(user_id, org_name), job_id)
            automl_index = get_automl_current_rec(job_id)
            if automl_experiment_index is not None:
                automl_index = int(automl_experiment_index)
            log_file_path = os.path.join(root, f"experiment_{automl_index}", "log.txt")

        if (job_metadata.get("status", "") not in ("Done", "Error", "Canceled", "Paused") or
                not os.path.exists(log_file_path)):
            workspace_id = handler_metadata.get("workspace", "")
            if not workspace_id:
                return Code(404, {}, "Handler doesn't have workspace assigned, can't download logs.")
            download_log_from_cloud(handler_metadata, job_id, log_file_path, automl_index)

        # File not present - Use detailed message or job status
        if not os.path.exists(log_file_path):
            detailed_result_msg = (
                job_metadata.get("job_details", {})
                .get(job_id, {})
                .get("detailed_status", {})
                .get("message", "")
            )
            if detailed_result_msg:
                return Code(200, detailed_result_msg)

            if (handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and
                    job_metadata.get("action", "") == "train"):
                if handler_metadata.get("status") in ["Canceled", "Canceling"]:
                    return Code(200, "AutoML training has been canceled.")
                if handler_metadata.get("status") in ["Paused", "Pausing"]:
                    return Code(200, "AutoML training has been paused.")
                if handler_metadata.get("status") == "Resuming":
                    return Code(200, "AutoML training is resuming.")
                if handler_metadata.get("status") == "Running":
                    return Code(200, "Generating new recommendation for AutoML experiment.")
            return Code(404, {}, "Logs for the job are not available yet.")
        return Code(200, get_job_logs_util(log_file_path))
