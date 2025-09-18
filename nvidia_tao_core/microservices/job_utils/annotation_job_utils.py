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

"""MONAI annotation job utils module"""
import os
import sys
import numpy as np
import logging

from nvidia_tao_core.microservices.handlers.app_handler import AppHandler
from nvidia_tao_core.microservices.handlers.monai.helpers import ImageLabelRecord
from nvidia_tao_core.microservices.handlers.stateless_handlers import (get_handler_metadata,
                                                                       get_jobs_for_handler,
                                                                       printc,
                                                                       safe_load_file)
from nvidia_tao_core.microservices.handlers.tis_handler import TISHandler
from nvidia_tao_core.microservices.handlers.utilities import prep_tis_model_repository
from nvidia_tao_core.microservices.utils import safe_get_file_modified_time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def update_inference_model(job_context_dict, job_id):
    """Trigger inference job for the current model."""
    model_id = job_context_dict.get("handler_id")
    user_id = job_context_dict.get("user_id")
    org_name = job_context_dict.get("org_name")
    # Read handler metadata
    handler_metadata = get_handler_metadata(model_id, "experiments")
    if not handler_metadata["realtime_infer"]:
        raise ValueError(f"User {org_name} model {model_id} is not enabled for Realtime Inference")
    model_params = handler_metadata["model_params"]
    success, tis_model, msg, _ = prep_tis_model_repository(
        model_params,
        model_id,
        org_name,
        user_id,
        model_id,
        job_id=job_id,
        update_model=True
    )
    if not success:
        raise RuntimeError(f"Inference job failed with message {msg}")

    response = TISHandler.update(model_id, tis_model)
    if response.code != 201:
        msg = response.data["error_desc"]
        raise RuntimeError(f"Triton Inference failed with message {msg}")


def trigger_train(train_spec, job_context_dict, current_record, latest_record):
    """Trigger training job for the current model.

    Args:
        train_spec: Training spec dict for the monai training.
        job_context_dict: Job context for the current annotation job.
        current_record: Current record obtained from the `notify`. Not used with the current implementation.
        latest_record: latest record generated from the previous `notify`. Not used with the current implementation.
    """
    model_id = job_context_dict.get("handler_id")
    org_name = job_context_dict.get("org_name")
    job_id = job_context_dict.get("id")

    if not os.getenv("CI_PROJECT_DIR", None):
        logger.info("Starting training with all the labeled image from DICOM endpoint")
    train_spec_copy = train_spec.copy()
    if "finetune" not in train_spec_copy:
        # By default, use the model.pt as the pretrained weight
        train_spec_copy["finetune"] = True
    train_spec_copy["cluster"] = "local"  # For the CL train job to use local cluster resource
    # Start training for all the labeled images
    description = f"Train Job for CL {job_id} with experiment {model_id}"
    response = AppHandler.job_run(
        org_name,
        model_id,
        job_id,
        "train",
        "experiment",
        specs=train_spec_copy,
        name="Train Job for CL",
        description=description
    )
    if response.code != 201:
        raise RuntimeError(f"Training job failed with status code {response.code}")
    return response.data  # job_id


def load_initial_state(job_context_dict, handler_root, notify_record):
    """Load Initial State for the Continual Learning Job."""
    printc("Continual Learning started", context=job_context_dict, keys="handler_id")
    org_name = job_context_dict.get("org_name")
    experiment_id = job_context_dict.get("handler_id")
    cl_job_id = job_context_dict.get("id")
    specs = job_context_dict.get("specs")
    train_spec = specs.get("train_spec")
    round_size = specs.get("round_size", None)
    stop_criteria = specs.get("stop_criteria", None)
    job_metadata_file = os.path.join(handler_root, "jobs_metadata", cl_job_id + ".json")

    latest_mod_time, latest_record = None, {}
    if os.path.isfile(notify_record):
        latest_mod_time = safe_get_file_modified_time(notify_record)
        latest_record = safe_load_file(notify_record)

    printc(f"Initial record: {latest_record}", context=job_context_dict, keys="handler_id")

    return (org_name, experiment_id, cl_job_id, train_spec, round_size, stop_criteria,
            job_metadata_file, latest_mod_time, latest_record)


def initialize_cl_tracker(stop_criteria):
    """Initialize the CL tracker and state."""
    cl_tracker = CLCriteriaTracker(stop_criteria)
    cl_state = cl_tracker.schema()
    cl_state["round"] = 0
    cl_state["key_metric"] = -1.0
    return cl_tracker, cl_state


def cancel_trigger_jobs(org_name, experiment_id, jobs_trigger, jobs_done):
    """Cancel all the triggered jobs for the current model."""
    job_metadatas = get_jobs_for_handler(experiment_id, "experiment")
    for job_metadata in job_metadatas:
        if job_metadata.get("id", "") in jobs_trigger and job_metadata["id"] not in jobs_done:
            response = AppHandler.job_cancel(org_name, experiment_id, job_metadata.get("id"), "experiment")
            if response.code != 200:
                raise RuntimeError(f"Cancel job failed with status code {response.code} with {response.data}")


def check_for_cancelation(metadata, jobs_trigger, jobs_done, job_context_dict):
    """Check for the cancelation of the Continual Learning job."""
    org_name = job_context_dict.get("org_name")
    experiment_id = job_context_dict.get("handler_id")
    metadata_status = metadata.get("status", "Error")
    if metadata_status == "Canceled":
        cancel_trigger_jobs(org_name, experiment_id, jobs_trigger, jobs_done)
        printc("Continual Learning job cancelled", context=job_context_dict, keys="handler_id")
        sys.exit(0)


def handle_first_round_specifics(cl_state, train_spec_copy):
    """Handle the first round specifics for the Continual Learning Job."""
    if cl_state["round"] == 0:
        if train_spec_copy.get("val_at_start", None) is None:
            train_spec_copy["val_at_start"] = True


def process_notification_record(
    notify_record,
    latest_mod_time,
    latest_record,
    train_spec,
    round_size,
    cl_state,
    job_context_dict,
    jobs_trigger
):
    """Process the notification record and trigger training if the round size is met."""
    current_mod_time = safe_get_file_modified_time(notify_record) if os.path.isfile(notify_record) else None
    if current_mod_time != latest_mod_time:
        # If notify record file is modified, load the current record
        latest_mod_time = current_mod_time
        current_record = safe_load_file(notify_record)
        printc(f"Current record: {current_record}", context=job_context_dict, keys="handler_id")

        # Start training if the number of updated labels is greater than the round size
        num_updated = ImageLabelRecord.count_added_labels(current_record, latest_record)
        if round_size is not None and num_updated >= round_size:
            printc(
                f"Number of updated labels {num_updated} is greater than round size {round_size}",
                context=job_context_dict,
                keys="handler_id",
            )
            train_spec_copy = train_spec.copy()
            handle_first_round_specifics(cl_state, train_spec_copy)
            job_id = trigger_train(train_spec_copy, job_context_dict, current_record, latest_record)
            jobs_trigger.append(job_id)
            printc(f"Triggered training job {job_id}", context=job_context_dict, keys="handler_id")
            # Update the notify record file only when training is successfully submitted
            latest_record = current_record

    return latest_mod_time, latest_record


def handle_job_updates(cl_state, jobs_trigger, jobs_done, metric_sorter, job_context_dict):
    """Handle the job updates for the Continual Learning Job."""
    handler_id = job_context_dict.get("handler_id")
    job_metadatas = get_jobs_for_handler(handler_id, "experiment")
    for job_metadata in job_metadatas:
        job_id = job_metadata.get("id", "")
        if job_id in jobs_trigger and job_id not in jobs_done:
            handle_single_job_update(job_metadata, cl_state, jobs_done, metric_sorter, job_context_dict)


def handle_single_job_update(job_metadata, cl_state, jobs_done, metric_sorter, job_context_dict):
    """Handle the single job update for the Continual Learning Job."""
    job_status = job_metadata["status"]
    job_id = job_metadata["id"]
    if job_status == "Done":
        handle_successful_job(job_metadata, cl_state, jobs_done, metric_sorter, job_context_dict)
    elif job_status == "Error":
        printc(f"Training job {job_id} failed", context=job_context_dict, keys="handler_id")
        jobs_done.append(job_id)


def handle_successful_job(job_metadata, cl_state, jobs_done, metric_sorter, job_context_dict):
    """Handle the successful job for the Continual Learning Job."""
    job_id = job_metadata["id"]
    metric = job_metadata["result"].get("key_metric", -1.0)
    metric_sorter.add_pair(job_id, metric)
    jobs_done.append(job_id)
    cl_state["round"] += 1
    update_state_with_metric(cl_state, job_metadata, metric, job_context_dict)


def update_state_with_metric(cl_state, job_metadata, metric, job_context_dict):
    """Update the state with the metric for the Continual Learning Job."""
    best_epoch = job_metadata["result"].get("epoch", -1)
    job_id = job_metadata["id"]
    if best_epoch == -1:
        printc(
            f"Job {job_id} did not provide a valid result. No model update will be made.",
            context=job_context_dict,
            keys="handler_id",
        )
    elif best_epoch == 0 and cl_state["round"] > 0:
        printc(
            f"Job {job_id} did not perform better than the pre-trained. "
            f"Saving the pre-trained metric for record. No model update will be made.",
            context=job_context_dict,
            keys="handler_id",
        )
    elif best_epoch == 0 and cl_state["round"] == 0:
        cl_state["key_metric"] = metric
        printc(
            f"Job {job_id} did not perform better than the pre-trained. "
            f"Saving the pre-trained metric for record. No model update will be made.",
            context=job_context_dict,
            keys="handler_id",
        )
    elif metric > cl_state["key_metric"]:
        printc(
            f"Job {job_id} with {metric} is being used to update inference model",
            context=job_context_dict,
            keys="handler_id",
        )
        cl_state["key_metric"] = metric
        update_inference_model(job_context_dict, job_metadata["id"])
    printc(f"Continual Learning state {cl_state}", context=job_context_dict, keys="handler_id")


class CLCriteriaTracker:
    """Class to track the criteria for CL."""

    def __init__(self, criteria):
        """Initialize CLCriteriaTracker class

        Args:
            criteria: Criteria dict for Continual Learning
        """
        if criteria is None:
            criteria = {}
        criteria_copy = criteria.copy()
        self.max_rounds = criteria_copy.pop("max_rounds", np.nan)
        self.key_metric = criteria_copy.pop("key_metric", np.nan)
        if len(criteria_copy) > 0:
            raise RuntimeError(f"Unknown criteria: {criteria_copy}")

    def schema(self):
        """Creates schema dict based on the member variables"""
        _schema = {
            "round": None,
            "key_metric": None,
        }
        return _schema

    def check_max_rounds(self, current_state):
        """Check if the max round is met."""
        current_round = current_state.get("round", None)
        if np.isnan(self.max_rounds) or current_round is None:
            # max_rounds is nan: Not a valid criteria.
            # current_round is None: Not updated.
            return False
        return current_round >= self.max_rounds

    def check_key_metric(self, current_state):
        """Check if the key metric is met."""
        current_metric = current_state.get("key_metric", None)
        if np.isnan(self.key_metric) or current_metric is None:
            # key_metric is nan: Not a valid criteria.
            # current_metric is None: Not updated.
            return False
        return current_metric >= self.key_metric

    def should_stop(self, current_state):
        """Check if the criteria are met."""
        if self.check_max_rounds(current_state):
            logger.info("Max rounds reached with %s. Stopping Continual Learning Job.", current_state)
            return True

        if self.check_key_metric(current_state):
            logger.info("Key metric reached with %s. Stopping Continual Learning Job.", current_state)
            return True

        # Other criteria can be added similarly
        # If none of the criteria are met
        return False
