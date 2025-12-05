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
import ast
import os
import json
import traceback
from copy import deepcopy
from datetime import datetime, timezone
import sysconfig
import logging

from nvidia_tao_core.microservices.utils.automl_utils import update_automl_details_metadata
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_metadata,
    get_handler_type,
    get_jobs_root,
    save_automl_controller_info,
    serialize_object,
    write_job_metadata,
    get_handler_job_metadata,
    update_handler_with_jobs_info,
    get_automl_controller_info
)
from nvidia_tao_core.microservices.utils.handler_utils import Code, decrypt_handler_metadata
from .docker_images import DOCKER_IMAGE_MAPPER
from nvidia_tao_core.microservices.utils.job_utils.executor import (
    JobExecutor,
    StatefulSetExecutor
)
from nvidia_tao_core.microservices.utils.log_monitor_service import start_monitoring_job, stop_monitoring_job

# TODO Make sure the image name is current docker tag of the API
image = DOCKER_IMAGE_MAPPER["API"]

logger = logging.getLogger(__name__)


def _normalize_automl_hyperparameters(automl_hyperparameters):
    """Normalize automl_hyperparameters to JSON format for shell-safe passing.

    Handles both SDK format "['param1', 'param2']" and CLI format "[param1, param2]".
    Returns a JSON string that can be safely passed through shell and parsed with json.loads().
    """
    if not isinstance(automl_hyperparameters, str):
        return json.dumps(automl_hyperparameters)

    try:
        # Try ast.literal_eval first (works for SDK format with quoted elements)
        params_list = ast.literal_eval(automl_hyperparameters)
        return json.dumps(params_list)
    except (ValueError, SyntaxError):
        # Fallback for CLI format (unquoted elements due to shell processing)
        params_str = automl_hyperparameters.strip('[]').strip()
        if params_str:
            params_list = [p.strip() for p in params_str.split(',')]
            return json.dumps(params_list)
        return "[]"


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
    def start(user_id, org_name, experiment_id, job_id, handler_metadata, name="",
              platform_id="", retain_checkpoints_for_resume=False, timeout_minutes=60):
        """Starts an AutoML job by executing `automl_start.py` with the provided parameters.

        Args:
            user_id (str): ID of the user initiating the job.
            org_name (str): Name of the organization.
            experiment_id (str): ID of the associated experiment.
            job_id (str): Unique identifier for the AutoML job.
            handler_metadata (dict): Metadata containing AutoML configuration settings.
            name (str, optional): Name of the job. Defaults to "automl train job".
            platform_id (str, optional): Platform identifier for execution. Defaults to "".
            retain_checkpoints_for_resume (bool, optional): Whether to retain .pth
                checkpoints for training resume. Defaults to False.
            timeout_minutes (int, optional): The job-specific timeout in minutes. If not specified, uses global timeout.
        """
        network = get_handler_type(handler_metadata)
        metric = handler_metadata.get("metric", "map")
        automl_settings = handler_metadata.get("automl_settings", {})
        automl_algorithm = automl_settings.get("automl_algorithm", "Bayesian")
        if automl_algorithm.lower() == "hyperband":
            retain_checkpoints_for_resume = True
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
            "job_details": {},
            "retain_checkpoints_for_resume": retain_checkpoints_for_resume,
            "timeout_minutes": timeout_minutes
        }
        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            os.makedirs(root)

        if not name:
            name = "automl train job"
        automl_max_recommendations = automl_settings.get("automl_max_recommendations", 20)
        automl_delete_intermediate_ckpt = automl_settings.get("automl_delete_intermediate_ckpt", True)
        automl_R = automl_settings.get("automl_R", 27)
        automl_nu = automl_settings.get("automl_nu", 3)
        epoch_multiplier = automl_settings.get("epoch_multiplier", 1)
        automl_hyperparameters = automl_settings.get("automl_hyperparameters", "[]")
        override_automl_disabled_params = automl_settings.get("override_automl_disabled_params", False)

        write_job_metadata(job_id, job_metadata)
        update_handler_with_jobs_info(job_metadata, experiment_id, job_id, "experiments")

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
            f"--automl_hyperparameters='{_normalize_automl_hyperparameters(automl_hyperparameters)}' "
            f'--override_automl_disabled_params={override_automl_disabled_params} '
            f'--retain_checkpoints_for_resume={retain_checkpoints_for_resume} '
            f'--timeout_minutes={timeout_minutes} '
            f"--decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata, default=str)}'"
        )
        if platform_id:
            run_command = f"{run_command} --platform_id={platform_id}"

        cluster_num_gpus = int(os.getenv('NUM_GPU_PER_NODE', default='0'))
        docker_env_vars = {
            "NUM_GPU_PER_NODE": str(cluster_num_gpus),
            "TAO_LOG_LEVEL": os.getenv('TAO_LOG_LEVEL', default='DEBUG')
        }
        logger.debug(
            f"[AUTOML-START] Creating brain job {job_id}: "
            f"Setting NUM_GPU_PER_NODE={cluster_num_gpus} in brain container env (no actual GPUs assigned)"
        )

        JobExecutor().create_job(
            org_name,
            job_id,
            image,
            run_command,
            num_gpu=0,  # Brain job gets NO actual GPUs (runs in app pod)
            docker_env_vars=docker_env_vars,  # But knows cluster GPU count via env var
            automl_brain=True,
            automl_exp_job=False
        )

        # Start log monitoring for AutoML brain job (server-side)
        backend = os.getenv("BACKEND", "local-k8s")
        logger.debug(f"[AUTOML] Checking if log monitoring should start for brain job {job_id}, backend={backend}")
        if backend in ("local-k8s", "local-docker"):
            try:
                logger.debug(f"[AUTOML] Starting log monitoring setup for brain job {job_id}")
                # Get namespace for K8s
                namespace = None
                if backend == "local-k8s":
                    namespace = os.getenv("NAMESPACE")
                    logger.debug(f"[AUTOML] K8s namespace from env: {namespace}")
                    if not namespace:
                        try:
                            namespace_file = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
                            logger.debug(f"[AUTOML] Reading namespace from {namespace_file}")
                            with open(namespace_file, 'r', encoding='utf-8') as f:
                                namespace = f.read().strip()
                            logger.debug(f"[AUTOML] Got namespace from service account: {namespace}")
                        except Exception as e:
                            namespace = "default"
                            logger.debug(f"[AUTOML] Using default namespace (error: {e})")

                # Start monitoring for brain job
                logger.info(f"[AUTOML] Starting log monitoring for AutoML brain job {job_id}, namespace={namespace}")
                start_monitoring_job(
                    job_id,
                    callback_url=None,  # Brain job doesn't need callbacks (server-side only)
                    namespace=namespace,
                    metadata={
                        'handler_id': experiment_id,
                        'handler_kind': 'experiment',
                        'action': 'automl_brain',
                        'network': network,
                        'is_brain_job': True
                    }
                )
                logger.info(f"[AUTOML] Successfully started log monitoring for AutoML brain job {job_id}")
            except Exception as e:
                logger.warning(
                    f"[AUTOML] Failed to start log monitoring for AutoML brain job {job_id}: "
                    f"{type(e).__name__}: {e}"
                )
                logger.debug("[AUTOML] Exception details:", exc_info=True)
        else:
            logger.debug(f"[AUTOML] Skipping log monitoring for backend {backend}")

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
        logger.debug(
            f"[AUTOML-STOP] Starting AutoML stop operation: job_id={job_id}, experiment_id={experiment_id}, "
            f"org_name={org_name}, user_id={user_id}"
        )

        try:
            logger.debug(f"[AUTOML-STOP] Deleting AutoML brain K8s job: job_id={job_id}")

            # Stop log monitoring for brain job
            backend = os.getenv("BACKEND", "local-k8s")
            logger.debug(f"[AUTOML] Stopping brain job {job_id}, checking if should stop log monitoring")
            if backend in ("local-k8s", "local-docker"):
                try:
                    logger.info(f"[AUTOML] Stopping log monitoring for AutoML brain job {job_id}")
                    stop_monitoring_job(job_id)
                    logger.info(f"[AUTOML] Successfully stopped log monitoring for AutoML brain job {job_id}")
                except Exception as e:
                    logger.warning(
                        f"[AUTOML] Failed to stop log monitoring for AutoML brain job {job_id}: "
                        f"{type(e).__name__}: {e}"
                    )

            JobExecutor().delete_job(job_id, use_ngc=False)
            logger.debug(f"[AUTOML-STOP] Brain K8s job deleted, waiting for pod termination: job_id={job_id}")

            # Wait for actual K8s Job/Pod to terminate (synchronous)
            job_terminated = JobExecutor().wait_for_job_termination(job_id, timeout_seconds=120)
            if not job_terminated:
                logger.warning(f"[AUTOML-STOP] Timeout waiting for brain termination: job_id={job_id}")
            else:
                logger.debug(f"[AUTOML-STOP] Brain termination confirmed: job_id={job_id}")

            recommendations = get_automl_controller_info(job_id)
            logger.debug(
                f"[AUTOML-STOP] Retrieved recommendations: job_id={job_id}, "
                f"num_recommendations={len(recommendations) if recommendations else 0}"
            )
            for idx, recommendation in enumerate(recommendations):
                recommendation_job_id = recommendation.get("job_id")
                rec_status = recommendation.get("status")
                logger.debug(
                    f"[AUTOML-STOP] Processing recommendation {idx}: "
                    f"rec_job_id={recommendation_job_id}, status={rec_status}"
                )

                # Update status to canceling for all non-terminal states
                if rec_status not in ("done", "success", "failure", "error", "canceled", "canceling"):
                    logger.debug(
                        f"[AUTOML-STOP] Updating recommendation status to canceling: "
                        f"rec_job_id={recommendation_job_id}, old_status={rec_status}"
                    )
                    recommendation["status"] = "canceling"
                    save_automl_controller_info(job_id, recommendations)
                    logger.debug(f"[AUTOML-STOP] Saved controller info to MongoDB: job_id={job_id}")
                    # Verify the save worked
                    verification = get_automl_controller_info(job_id)
                    verified_status = (
                        verification[idx]["status"]
                        if verification and idx < len(verification) else "NOT_FOUND"
                    )
                    logger.debug(
                        f"[AUTOML-STOP] Verified status in MongoDB: "
                        f"rec_job_id={recommendation_job_id}, status={verified_status}"
                    )
                    update_automl_details_metadata(job_id, experiment_id, "experiments")

                if recommendation_job_id:
                    logger.debug(
                        f"[AUTOML-STOP] Deleting recommendation statefulset: "
                        f"rec_job_id={recommendation_job_id}"
                    )
                    StatefulSetExecutor().delete_statefulset(recommendation_job_id)
                    logger.debug(
                        f"[AUTOML-STOP] Waiting for recommendation pod termination: "
                        f"rec_job_id={recommendation_job_id}"
                    )

                    # Wait for actual K8s StatefulSet/Pod to terminate (synchronous)
                    sts_terminated = StatefulSetExecutor().wait_for_statefulset_termination(
                        recommendation_job_id, timeout_seconds=120
                    )
                    if not sts_terminated:
                        logger.warning(
                            f"[AUTOML-STOP] Timeout waiting for recommendation termination: "
                            f"rec_job_id={recommendation_job_id}"
                        )
                    else:
                        logger.debug(
                            f"[AUTOML-STOP] Recommendation termination confirmed: "
                            f"rec_job_id={recommendation_job_id}"
                        )

                # Update ALL non-terminal recommendations to canceled (check current status after potential updates)
                current_status = recommendation.get("status")
                if current_status not in ("done", "success", "failure", "error", "canceled"):
                    logger.debug(
                        f"[AUTOML-STOP] Updating recommendation status to canceled: "
                        f"rec_job_id={recommendation_job_id}, old_status={current_status}"
                    )
                    recommendation["status"] = "canceled"
                    save_automl_controller_info(job_id, recommendations)
                    logger.debug(f"[AUTOML-STOP] Saved controller info to MongoDB: job_id={job_id}")
                    # Verify the save worked
                    verification = get_automl_controller_info(job_id)
                    verified_status = (
                        verification[idx]["status"]
                        if verification and idx < len(verification) else "NOT_FOUND"
                    )
                    logger.debug(
                        f"[AUTOML-STOP] Verified final status in MongoDB: "
                        f"rec_job_id={recommendation_job_id}, status={verified_status}"
                    )
                    update_automl_details_metadata(job_id, experiment_id, "experiments")
                else:
                    logger.debug(
                        f"[AUTOML-STOP] Skipping status update for terminal state: "
                        f"rec_job_id={recommendation_job_id}, status={current_status}"
                    )
            logger.debug(
                f"[AUTOML-STOP] AutoML stop operation completed successfully: job_id={job_id}"
            )
        except Exception as e:
            logger.error(f"[AUTOML-STOP] Exception thrown in AutoMLHandler stop: job_id={job_id}, error={str(e)}")
            logger.error(f"[AUTOML-STOP] Traceback: {traceback.format_exc()}")
            return Code(404, [], "job cannot be stopped in platform")

        return Code(200, {"message": f"job {job_id} cancelled"})

    @staticmethod
    def resume(user_id, org_name, experiment_id, job_id, handler_metadata, name="", platform_id="", timeout_minutes=60):
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
        logger.debug(
            f"[AUTOML-RESUME] Starting AutoML resume operation: job_id={job_id}, "
            f"experiment_id={experiment_id}, org_name={org_name}, user_id={user_id}"
        )

        root = os.path.join(get_jobs_root(user_id, org_name), job_id)
        if not os.path.exists(root):
            logger.debug(f"[AUTOML-RESUME] Creating job root directory: path={root}")
            os.makedirs(root)
        else:
            logger.debug(f"[AUTOML-RESUME] Job root directory exists: path={root}")

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

        logger.debug(
            f"[AUTOML-RESUME] AutoML settings: job_id={job_id}, network={network}, "
            f"algorithm={automl_algorithm}, metric={metric}, "
            f"max_recommendations={automl_max_recommendations}, R={automl_R}, nu={automl_nu}, "
            f"epoch_multiplier={epoch_multiplier}, timeout_minutes={timeout_minutes}"
        )

        workspace_id = handler_metadata.get("workspace")
        logger.debug(f"[AUTOML-RESUME] Loading workspace metadata: job_id={job_id}, workspace_id={workspace_id}")
        workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
        decrypted_workspace_metadata = deepcopy(workspace_metadata)
        decrypt_handler_metadata(decrypted_workspace_metadata)
        decrypted_workspace_metadata.pop('_id', None)

        job_metadata = get_handler_job_metadata(job_id)
        retain_checkpoints_for_resume = (
            job_metadata.get("retain_checkpoints_for_resume", False) if job_metadata else False
        )
        if automl_algorithm.lower() == "hyperband":
            retain_checkpoints_for_resume = True
            logger.debug(
                f"[AUTOML-RESUME] Hyperband algorithm detected, "
                f"forcing retain_checkpoints_for_resume=True: job_id={job_id}"
            )

        logger.debug(
            f"[AUTOML-RESUME] Job settings: job_id={job_id}, "
            f"retain_checkpoints_for_resume={retain_checkpoints_for_resume}"
        )

        # Call the script
        python_lib_path = sysconfig.get_path("purelib")
        automl_script = os.path.join(python_lib_path, "nvidia_tao_core/microservices/automl_start.py")
        logger.debug(f"[AUTOML-RESUME] AutoML script path: {automl_script}")
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
            f"--automl_hyperparameters='{_normalize_automl_hyperparameters(automl_hyperparameters)}' "
            f'--override_automl_disabled_params={override_automl_disabled_params} '
            f'--retain_checkpoints_for_resume={retain_checkpoints_for_resume} '
            f'--timeout_minutes={timeout_minutes} '
            f"--decrypted_workspace_metadata='{json.dumps(decrypted_workspace_metadata, default=serialize_object)}'"
        )
        if platform_id:
            run_command = f"{run_command} --platform_id={platform_id}"
            logger.debug(f"[AUTOML-RESUME] Platform ID specified: job_id={job_id}, platform_id={platform_id}")

        # CRITICAL: Pass NUM_GPU_PER_NODE to resumed brain job for GPU validation
        cluster_num_gpus = int(os.getenv('NUM_GPU_PER_NODE', default='0'))
        docker_env_vars = {
            "NUM_GPU_PER_NODE": str(cluster_num_gpus),
            "TAO_LOG_LEVEL": os.getenv('TAO_LOG_LEVEL', default='INFO')
        }
        logger.debug(
            f"[AUTOML-RESUME] Creating K8s job for AutoML brain: job_id={job_id}, num_gpu=0, "
            f"NUM_GPU_PER_NODE={cluster_num_gpus} in env"
        )
        JobExecutor().create_job(
            org_name, job_id, image, run_command, num_gpu=0,
            docker_env_vars=docker_env_vars,
            automl_brain=True, automl_exp_job=False
        )
        logger.debug(f"[AUTOML-RESUME] AutoML resume operation completed: job_id={job_id}")
