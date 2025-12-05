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

"""Util functions for AutoML jobs"""
import logging
import os

from .stateless_handler_utils import (
    get_public_experiments,
    get_handler_job_metadata,
    write_job_metadata,
    get_job_specs,
    get_job
)
# StatefulSetExecutor import moved to function level to avoid circular imports

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


def get_base_experiment_id_from_recommendation(specs, network_arch):
    """Dynamicaly obtain base_experiment id based on the backbone and num_layers chosen"""
    #  TODO : generalize this for all models
    backbone_arch = specs.get("backbone", "resnet")
    num_layers = specs.get("num_layers", 34)
    match_string = f":{backbone_arch}{num_layers}"
    base_experiment_id = None
    for metadata in get_public_experiments():
        ngc_path_exists = metadata.get("ngc_path", None) is not None
        correct_arch = metadata.get("network_arch", "") == network_arch
        base_experiment_string_match = match_string in metadata.get("ngc_path", "")
        if ngc_path_exists and correct_arch and base_experiment_string_match:
            base_experiment_id = metadata.get("id", None)

    return base_experiment_id


def on_new_automl_job(automl_context, recommendation):
    """Assigns dependencies for the automl recommendation job"""
    # Controller interacts with this
    # TODO: uncomment below block after fixing get_base_experiment_id_from_recommendation
    # # Download NGC pretrained model as a background process
    # base_experiment_id = get_base_experiment_id_from_recommendation(recommendation.specs, automl_context.network)
    # # Background process to download this base_experiment
    # if base_experiment_id:
    #     ptm_download_thread = threading.Thread(target=download_base_experiment, args=(base_experiment_id,))
    #     ptm_download_thread.start()

    # automl_context is same as JobContext that was created for AutoML job
    from .job_utils.workflow import Workflow, Job, Dependency

    recommendation_id = recommendation.id
    automl_brain_job_id = automl_context.id  # Brain job ID
    experiment_job_id = recommendation.job_id  # Individual experiment job ID

    deps = []
    deps.append(Dependency(type="automl", name=str(recommendation_id)))
    deps.append(Dependency(type="specs"))
    deps.append(Dependency(type="model"))
    deps.append(Dependency(type="dataset"))

    num_gpu = automl_context.num_gpu
    logger.debug(f"AutoML experiment {experiment_job_id}: Creating GPU dependency with {num_gpu} GPU(s)")

    if num_gpu > 0:
        deps.append(Dependency(type="gpu", name=automl_context.platform_id, num=num_gpu))

    job = {
        'user_id': automl_context.user_id,
        'org_name': automl_context.org_name,
        'num_gpu': automl_context.num_gpu,
        'platform_id': automl_context.platform_id,
        'kind': "experiment",
        'id': experiment_job_id,  # Use experiment's unique job ID
        'parent_id': automl_brain_job_id,  # Reference to brain job
        'priority': 2,
        'action': "train",
        'network': automl_context.network,
        'handler_id': automl_context.handler_id,
        'created_on': automl_context.created_on,
        'last_modified': automl_context.last_modified,
        'specs': get_job_specs(automl_brain_job_id),  # Specs are stored under brain job ID
        'dependencies': deps,
        'retain_checkpoints_for_resume': automl_context.retain_checkpoints_for_resume,
        'early_stop_epoch': automl_context.early_stop_epoch,
        'timeout_minutes': automl_context.timeout_minutes
    }
    j = Job(**job)
    Workflow.enqueue(j)
    logger.debug("Experiment job %s (recommendation %s) submitted to workflow for brain job %s",
                 experiment_job_id, recommendation_id, automl_brain_job_id)
    metadata = get_handler_job_metadata(automl_brain_job_id)
    job_details = metadata.get("job_details", {}).get(experiment_job_id, {})
    if job_details:
        metadata["job_details"][experiment_job_id] = {}
        write_job_metadata(automl_brain_job_id, metadata)


def on_delete_automl_job(job_id):
    """Dequeue the automl experiment job"""
    # AutoML handler stop would handle this
    # job_id can be either brain job ID or experiment job ID
    # Brain jobs are NOT in the workflow queue, so we skip dequeuing for them
    job_metadata = get_job(job_id)

    # Check if this is a brain job (missing workflow-specific fields)
    # Brain jobs don't have user_id, num_gpu, network, handler_id, workflow_status
    if not job_metadata or 'user_id' not in job_metadata or 'handler_id' not in job_metadata:
        logger.debug(f"Skipping dequeue for job {job_id} - appears to be a brain job or missing required fields")
        return

    job_dict = {
        'user_id': job_metadata.get("user_id"),
        'org_name': job_metadata.get("org_name"),
        'num_gpu': job_metadata.get("num_gpu", 0),
        'platform_id': job_metadata.get("platform_id"),
        'kind': "experiment",
        'id': job_metadata.get("id"),  # Experiment job ID
        'parent_id': job_metadata.get("parent_id", None),  # Brain job ID
        'priority': 2,
        'action': job_metadata.get("action", "train"),
        'network': job_metadata.get("network"),
        'handler_id': job_metadata.get("handler_id"),
        'created_on': job_metadata.get("created_on"),
        'last_modified': job_metadata.get("last_modified"),
        'specs': job_metadata.get("specs", {}),
        'workflow_status': job_metadata.get("workflow_status", "Pending"),
        'retain_checkpoints_for_resume': job_metadata.get("retain_checkpoints_for_resume", False),
        'early_stop_epoch': job_metadata.get("early_stop_epoch"),
        'timeout_minutes': job_metadata.get("timeout_minutes", 60)
    }
    from .job_utils.workflow import Workflow, Job
    job = Job(**job_dict)
    Workflow.dequeue(job)


def on_cancel_automl_job(job_id):
    """Delete the job from k8's jobs"""
    logger.debug(
        f"{'-' * 80}\n"
        f"CANCELLING AUTOML EXPERIMENT JOB\n"
        f"Job ID: {job_id}\n"
        f"Reason: Explicit cancellation via on_cancel_automl_job\n"
        f"Action: Deleting StatefulSet\n"
        f"{'-' * 80}"
    )

    from .job_utils.executor import StatefulSetExecutor
    result = StatefulSetExecutor().delete_statefulset(job_id)

    if result:
        logger.info(f"Successfully deleted StatefulSet for AutoML job {job_id}")
    else:
        logger.error(f"Failed to delete StatefulSet for AutoML job {job_id}")

    return result
