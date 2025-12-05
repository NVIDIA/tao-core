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

"""Job Workflow modules"""
import os
import threading
import functools
import time
import uuid
import logging

from queue import PriorityQueue

from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timezone

from nvidia_tao_core.microservices.handlers.utilities import JobContext
from nvidia_tao_core.microservices.handlers.actions import ACTIONS_TO_FUNCTIONS, AutoMLPipeline
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    get_dnn_status,
    get_handler_job_metadata,
    update_job_status,
    save_automl_controller_info,
    get_all_pending_jobs,
    get_all_running_jobs,
    get_all_running_automl_experiments,
    update_job_message,
    get_handler_type,
    get_handler_metadata,
    get_automl_controller_info,
    delete_dnn_status
)
from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler
from nvidia_tao_core.microservices.utils import read_network_config
from nvidia_tao_core.microservices.job_utils.dependencies import dependency_type_map, dependency_check_default
from nvidia_tao_core.microservices.constants import JOB_STATUS_TIMEOUT_MINUTES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def synchronized(wrapped):
    """Decorator function for synchronizing threaded functions"""
    lock = threading.Lock()

    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            return wrapped(*args, **kwargs)
    return _wrap


@dataclass
class IdedItem:
    """Base class for representing id's in uuid"""

    id: uuid.UUID = field(default=uuid.uuid4())


@dataclass(order=True)
class PrioritizedItem:
    """Base class for prioritizing items"""

    priority: int = field(default=1)
    created_on: str = field(default=datetime.now(tz=timezone.utc))


@dataclass
class Dependency:
    """Base class for representing dependecies"""

    type: str = field(default=None)
    name: str = field(default=None)
    num: int = field(default=1)


@dataclass
class Job(PrioritizedItem, IdedItem):
    """Class for representing jobs"""

    last_modified: str = field(compare=False, default=datetime.now(tz=timezone.utc))
    action: str = field(compare=False, default=None)
    dependencies: list = field(compare=False, default=None)
    # More parameters for Job from JobContext
    parent_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    network: str = field(compare=False, default=None)
    handler_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    user_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    org_name: str = field(compare=False, default=None)
    kind: str = field(compare=False, default=None)
    specs: dict = field(compare=False, default=None)
    num_gpu: int = field(compare=False, default=0)
    platform_id: uuid.UUID = field(compare=False, default=uuid.uuid4())
    workflow_status: str = field(compare=False, default=None)


def dependency_check(job_context, dependency):
    """Checks if depencies for the job are met"""
    dependency_check_fn = dependency_type_map.get(dependency.type, dependency_check_default)
    dependency_met = dependency_check_fn(job_context, dependency)
    return dependency_met


def execute_job(job_context):
    """Starts a thread on pipelines present in actions.py"""
    isautoml = False
    for dep in job_context.dependencies:
        if dep.type == "automl":
            isautoml = True
            break

    if not isautoml:
        # Get action, network
        action = job_context.action
        network = job_context.network
        # Get the correct ActionPipeline:
        # - build specs
        # - build run command
        # - launch K8s job
        # - monitor status
        # - run post-job steps
        network_config = read_network_config(network)
        action_pipeline_name = network_config["api_params"]["actions_pipe"].get(action, "")
        if action == "validate_images":
            action_pipeline_name = "data_services"
        if action_pipeline_name:
            action_pipeline = ACTIONS_TO_FUNCTIONS[action_pipeline_name]
            _Actionpipeline = action_pipeline(job_context)
            # Thread this!
            job_run_thread = threading.Thread(
                target=_Actionpipeline.run,
                args=(),
                name=f'tao-job-thread-{job_context.id}'
            )
            job_run_thread.start()
        else:
            logger.error("Action pipeline couldn't be found: %s %s %s", network_config, network, job_context)
            return False
    else:
        # AUTOML Job
        # TODO: At test time, sequentially run it and not as a thread to catch errors
        _AutoMLPipeline = AutoMLPipeline(job_context)
        job_run_thread = threading.Thread(target=_AutoMLPipeline.run, args=(), name=f'tao-job-thread-{job_context.id}')
        job_run_thread.start()
        # AutoMLPipeline(job_context)
    return True


@synchronized
def still_exists(job_to_check):
    """Checks if the the job is yet to be executed/queued or not"""
    job_metadata = get_handler_metadata(job_to_check.id, "jobs")
    if job_metadata and job_metadata.get('workflow_status', 'unknown') == 'enqueued':
        return True
    return False


@synchronized
def report_healthy(message, clear=False):
    """Writes healthy message with timestamp"""
    mongo_health = MongoHandler("tao", "health")
    now = datetime.now(tz=timezone.utc)
    health_id = str(uuid.uuid4())
    health_message = f"Healthy at {now.isoformat()}"
    if message:
        health_message += f"\n{message}"
    logger.info(health_message)
    mongo_health.upsert({'id': health_id}, {'id': health_id, 'created_on': now,
                        'message': f"Healthy at {now.isoformat()}"})
    mongo_health.create_ttl_index('created_on', 86400)


def write_job(job):
    """Writes job into DB"""
    mongo_jobs = MongoHandler("tao", "jobs")
    mongo_jobs.upsert({'id': job.id}, asdict(job))


def get_last_status_timestamp(job_id, automl=False, experiment_number="0"):
    """Get the timestamp of the last status update for a job"""
    try:
        status_data = get_dnn_status(job_id, automl=automl, experiment_number=experiment_number)
        if not status_data:
            logger.debug(f"No status data found for job {job_id}")
            return None

        # Find the most recent timestamp in the status data
        latest_timestamp = None
        for status_entry in status_data:
            if isinstance(status_entry, dict) and 'timestamp' in status_entry:
                try:
                    timestamp_str = status_entry['timestamp']
                    if isinstance(timestamp_str, str):
                        # Try different timestamp formats
                        for fmt in ['%Y-%m-%dT%H:%M:%S.%fZ', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%f']:
                            try:
                                timestamp = datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format matches, try parsing as ISO format
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        continue

                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse timestamp {status_entry.get('timestamp')} for job {job_id}: {e}")
                    continue

        return latest_timestamp

    except Exception as e:
        logger.error(f"Error getting last status timestamp for job {job_id}: {e}")
        return None


def check_job_timeout(job_info):
    """Check if a job has timed out based on last status update"""
    job_id = job_info.get('job_id')
    is_automl = job_info.get('is_automl', False)
    experiment_number = job_info.get('experiment_number', '0')

    if not job_id:
        return False

    try:
        # For AutoML experiments, we check the experiment status differently
        if is_automl:
            # For AutoML experiments, check if the specific experiment is still running
            # by looking at the controller info
            controller_info = get_automl_controller_info(job_id)
            experiment_found = False
            experiment_status = ""

            if isinstance(controller_info, list):
                for recommendation in controller_info:
                    if isinstance(recommendation, dict) and str(recommendation.get("id", "")) == experiment_number:
                        experiment_status = recommendation.get("status", "")
                        experiment_found = True
                        break

            if not experiment_found:
                logger.debug(f"AutoML experiment {experiment_number} for job {job_id} not found")
                return False

            # Only check timeout for running AutoML experiments
            if experiment_status not in ("pending", "running", "started"):
                logger.debug(
                    f"AutoML experiment {experiment_number} for job {job_id} status is {experiment_status}, "
                    "skipping timeout check"
                )
                return False
        else:
            # Regular job - get job metadata to check current status
            job_metadata = get_handler_job_metadata(job_id)
            if not job_metadata:
                logger.debug(f"No metadata found for job {job_id}")
                return False

            current_status = job_metadata.get("status", "")

            # Only check timeout for running jobs
            if current_status not in ["Running", "Pending"]:
                logger.debug(f"Job {job_id} status is {current_status}, skipping timeout check")
                return False

        timeout_seconds = JOB_STATUS_TIMEOUT_MINUTES * 60
        last_timestamp = get_last_status_timestamp(job_id, automl=is_automl, experiment_number=experiment_number)

        if last_timestamp is None:
            # If no status updates found, check job creation time
            logger.debug(
                f"No status timestamp found for job {job_id} (AutoML: {is_automl}, exp: {experiment_number}), "
                "using job metadata last_modified"
            )

            if not is_automl:
                job_metadata = get_handler_job_metadata(job_id)
                last_modified = job_metadata.get("last_modified") if job_metadata else None
            else:
                # For AutoML experiments, use the main job's last_modified
                last_modified = job_info.get('last_modified')

            if last_modified:
                if isinstance(last_modified, str):
                    try:
                        last_timestamp = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                    except ValueError:
                        last_timestamp = datetime.now(tz=timezone.utc)
                elif isinstance(last_modified, datetime):
                    last_timestamp = last_modified
                else:
                    last_timestamp = datetime.now(tz=timezone.utc)
            else:
                # No timestamp available, assume job just started
                logger.debug(
                    f"No timestamp available for job {job_id} (AutoML: {is_automl}), assuming recently started"
                )
                return False

        # Check if the time since last update exceeds timeout
        current_time = datetime.now(tz=timezone.utc)
        time_since_update = current_time - last_timestamp

        is_timed_out = time_since_update.total_seconds() > timeout_seconds

        job_description = f"AutoML experiment {experiment_number} for job {job_id}" if is_automl else f"Job {job_id}"

        if is_timed_out:
            logger.warning(
                f"{job_description} timed out: last update {time_since_update.total_seconds():.0f}s ago "
                f"(timeout: {timeout_seconds}s)"
            )
        else:
            logger.debug(
                f"{job_description} last update: {time_since_update.total_seconds():.0f}s ago "
                f"(timeout: {timeout_seconds}s)"
            )

        return is_timed_out

    except Exception as e:
        logger.error(f"Error checking timeout for job {job_id} (AutoML: {is_automl}): {e}")
        return False


def terminate_timed_out_job(job_info):
    """Terminate a timed out job"""
    from nvidia_tao_core.microservices.job_utils.executor.statefulset_executor import StatefulSetExecutor

    job_id = job_info.get('job_id')
    handler_id = job_info.get('handler_id')
    kind = job_info.get('kind', '')
    is_automl = job_info.get('is_automl', False)
    experiment_number = job_info.get('experiment_number', '0')

    if not job_id or not handler_id:
        logger.error(f"Cannot terminate job: missing job_id or handler_id in {job_info}")
        return False

    try:
        if is_automl:
            logger.info(f"Terminating timed out AutoML experiment {experiment_number} for job {job_id}")

            # For AutoML experiments, we need to update the specific experiment status
            # and potentially terminate the StatefulSet for that experiment
            controller_info = get_automl_controller_info(job_id)

            if isinstance(controller_info, list):
                updated_controller = []
                experiment_found = False

                for recommendation in controller_info:
                    if isinstance(recommendation, dict):
                        if str(recommendation.get("id", "")) == experiment_number:
                            # Mark this experiment as failed due to timeout
                            recommendation["status"] = "error"
                            recommendation["message"] = "Terminated due to timeout - no status updates received"
                            experiment_found = True
                            logger.info(f"Marked AutoML experiment {experiment_number} as error due to timeout")
                        updated_controller.append(recommendation)

                if experiment_found:
                    # Save the updated controller info
                    save_automl_controller_info(job_id, updated_controller)

                    # Try to terminate the StatefulSet for this specific experiment
                    # The StatefulSet name for AutoML experiments typically includes the experiment number
                    experiment_job_id = f"{job_id}-{experiment_number}"
                    statefulset_executor = StatefulSetExecutor()
                    success = statefulset_executor.delete_statefulset(experiment_job_id, use_ngc=True)

                    if not success:
                        # If the specific experiment StatefulSet doesn't exist, try the main job ID
                        success = statefulset_executor.delete_statefulset(job_id, use_ngc=True)

                    if success:
                        logger.info(
                            f"Successfully terminated timed out AutoML experiment {experiment_number} for job {job_id}"
                        )
                    else:
                        logger.error(
                            f"Failed to terminate StatefulSet for AutoML experiment {experiment_number} "
                            f"for job {job_id}"
                        )

                    return success
                logger.warning(f"AutoML experiment {experiment_number} not found in controller info for job {job_id}")
                return False
            logger.error(f"Invalid controller info format for AutoML job {job_id}")
            return False
        logger.info(f"Terminating timed out job {job_id}")

        # Update job status to indicate timeout
        update_job_status(handler_id, job_id, status="Error", kind=kind)

        # Delete the StatefulSet
        statefulset_executor = StatefulSetExecutor()
        success = statefulset_executor.delete_statefulset(job_id, use_ngc=True)

        if success:
            logger.info(f"Successfully terminated timed out job {job_id}")
        else:
            logger.error(f"Failed to terminate timed out job {job_id}")

        return success

    except Exception as e:
        logger.error(f"Error terminating timed out job {job_id} (AutoML: {is_automl}): {e}")
        return False


def check_for_timed_out_jobs():
    """Check all running jobs and AutoML experiments for timeouts and terminate timed out ones"""
    # Check if timeout monitoring is enabled
    timeout_monitoring_enabled = os.getenv("JOB_TIMEOUT_MONITORING_ENABLED", "true").lower() in ("true", "1")
    if not timeout_monitoring_enabled:
        return []

    terminated_jobs = []

    try:
        # Get all running jobs (regular jobs)
        running_jobs = get_all_running_jobs()

        # Get all running AutoML experiments
        running_automl_experiments = get_all_running_automl_experiments()

        # Combine both lists
        all_jobs_to_check = running_jobs + running_automl_experiments

        logger.debug(
            f"Checking {len(running_jobs)} regular jobs and {len(running_automl_experiments)} AutoML experiments "
            "for timeouts"
        )

        for job_info in all_jobs_to_check:
            try:
                job_id = job_info.get("job_id")
                is_automl = job_info.get("is_automl", False)
                experiment_number = job_info.get("experiment_number", "0")

                if not job_id:
                    logger.warning(f"Skipping job with missing ID: {job_info}")
                    continue

                # Check if job is timed out
                if check_job_timeout(job_info):
                    # Terminate the job
                    if terminate_timed_out_job(job_info):
                        job_description = (
                            f"AutoML experiment {experiment_number} for job {job_id}" if is_automl else job_id
                        )
                        terminated_jobs.append(job_description)

            except Exception as e:
                logger.error(f"Error processing job {job_info}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error checking running jobs for timeouts: {e}")

    if terminated_jobs:
        logger.info(f"Terminated {len(terminated_jobs)} timed out jobs/experiments: {terminated_jobs}")
    else:
        logger.debug("No jobs or AutoML experiments terminated due to timeout")

    return terminated_jobs


def get_all_queued_jobs():
    """Gets all jobs with workflow_status of enqueued"""
    mongo_jobs = MongoHandler("tao", "jobs")
    jobs = []
    jobs_raw = mongo_jobs.find({'workflow_status': 'enqueued'})
    for job in jobs_raw:
        properties = {field.name for field in fields(Job)}
        job_keys = list(job.keys())
        for key in job_keys:
            if key not in properties:
                job.pop(key, None)
        j = Job(**job)
        j.dependencies = []
        for d in job.get('dependencies'):
            j.dependencies.append(Dependency(**d))
        jobs.append(j)

    return jobs


@synchronized
def scan_for_jobs():
    """Scans for new jobs and queues them if dependencies are met"""
    while True:
        report_healthy("Workflow has waken up", clear=True)

        # Check for inference microservice auto-deletion requests
        process_inference_microservice_auto_deletions()

        # Create global queue
        queue = PriorityQueue()
        for j in get_all_queued_jobs():
            queue.put(j)
        len_q = len(queue.queue)
        report_healthy(f"Found {len_q} pending jobs")
        # Parse to dequeue
        jobs_to_dequeue = []
        list.sort(queue.queue)
        for i in range(len(queue.queue)):
            # check dependencies
            job = queue.queue[i]
            report_healthy(f"{job.id} with action {job.action}: Checking dependencies")
            report_healthy(f"Total dependencies: {len(job.dependencies)}")
            all_met = True
            pending_reason_message = ""
            for dep in job.dependencies:
                dependency_met, message = dependency_check(job, dep)
                if not dependency_met:
                    pending_reason_message += f"{message} and, "
                    report_healthy(f"Unmet dependency: {dep.type} {pending_reason_message}")
                    all_met = False
                if "Parent job " in message and "errored out" in message:
                    jobs_to_dequeue.append(job)
                    break

            # Update detailed status message in response when appropriate message is available
            pending_reason_message = ''.join(pending_reason_message.rsplit(" and, ", 1))
            update_job_message(job.handler_id, job.id, kind=job.kind + "s", message=pending_reason_message)

            # if all dependencies are met
            if all_met and still_exists(job):
                # execute job
                # check if job is still enqueued in the DB
                report_healthy(f"{job.id} with action {job.action}: All dependencies met")
                if execute_job(job):
                    # dequeue job
                    jobs_to_dequeue.append(job)
                else:
                    report_healthy(f"{job.id} with action {job.action}: Job execution failed")
        for job in jobs_to_dequeue:
            Workflow.dequeue(job)

        # Check for timed out jobs and terminate them
        try:
            terminated_jobs = check_for_timed_out_jobs()
            if terminated_jobs:
                report_healthy(f"Terminated {len(terminated_jobs)} timed out jobs")
        except Exception as e:
            logger.error(f"Error checking for timed out jobs: {e}")

        report_healthy("Workflow going to sleep")
        time.sleep(15)


def process_inference_microservice_auto_deletions():
    """Process pending inference microservice auto-deletion requests

    This function monitors job_statuses for AUTO_DELETION_REQUESTED status from
    inference microservice pods and handles the actual deletion (requires DB access).
    """
    try:
        # Check job_statuses collection for AUTO_DELETION_REQUESTED status
        mongo_status_table = MongoHandler("tao", "job_statuses")

        # Find all jobs with AUTO_DELETION_REQUESTED in their status array
        all_status_records = mongo_status_table.find({})

        # Import here to avoid circular dependency
        from nvidia_tao_core.microservices.handlers.inference_microservice_handler import (
            InferenceMicroserviceHandler
        )

        jobs_to_delete = []

        for status_record in all_status_records:
            job_id = status_record.get("id", "")
            status_list = status_record.get("status", [])

            # Check if any status entry has AUTO_DELETION_REQUESTED
            for status_entry in status_list:
                if isinstance(status_entry, dict) and status_entry.get("status") == "AUTO_DELETION_REQUESTED":
                    # Check if we haven't processed this yet
                    if not status_entry.get("auto_deletion_processed", False):
                        jobs_to_delete.append({
                            "job_id": job_id,
                            "idle_time_minutes": status_entry.get("idle_time_minutes", 0),
                            "reason": status_entry.get("reason", "unknown")
                        })
                        break

        if not jobs_to_delete:
            return

        logger.info(f"Found {len(jobs_to_delete)} inference microservices requesting auto-deletion")

        for job_info in jobs_to_delete:
            job_id = job_info["job_id"]
            try:
                logger.info(
                    f"Processing auto-deletion for inference microservice job {job_id}. "
                    f"Reason: {job_info.get('reason', 'unknown')}, "
                    f"Idle time: {job_info.get('idle_time_minutes', 0):.2f} minutes"
                )

                # Call the actual deletion (has DB access here in workflow)
                result = InferenceMicroserviceHandler.stop_inference_microservice(
                    job_id, auto_deletion=True
                )

                if result.code == 200:
                    logger.info(f"Auto-deletion successful for job {job_id}")
                    # Note: Job status is already updated to Done in stop_inference_microservice

                    # Mark status as processed to avoid re-processing
                    status_record = mongo_status_table.find_one({"id": job_id})
                    if status_record:
                        status_list = status_record.get("status", [])
                        for status_entry in status_list:
                            if (isinstance(status_entry, dict) and
                                    status_entry.get("status") == "AUTO_DELETION_REQUESTED"):
                                status_entry["auto_deletion_processed"] = True
                                status_entry["processed_at"] = datetime.now(tz=timezone.utc).isoformat()
                        mongo_status_table.upsert({"id": job_id}, {"id": job_id, "status": status_list})
                else:
                    logger.error(f"Auto-deletion failed for job {job_id}: {result.data}")
                    # Mark as processed even on failure to avoid retry loops
                    status_record = mongo_status_table.find_one({"id": job_id})
                    if status_record:
                        status_list = status_record.get("status", [])
                        for status_entry in status_list:
                            if (isinstance(status_entry, dict) and
                                    status_entry.get("status") == "AUTO_DELETION_REQUESTED"):
                                status_entry["auto_deletion_processed"] = True
                                status_entry["failed"] = True
                                status_entry["error"] = str(result.data)
                        mongo_status_table.upsert({"id": job_id}, {"id": job_id, "status": status_list})

            except Exception as e:
                logger.error(f"Error processing auto-deletion for job {job_id}: {e}")
                # Mark as processed with error to avoid retry loops
                status_record = mongo_status_table.find_one({"id": job_id})
                if status_record:
                    status_list = status_record.get("status", [])
                    for status_entry in status_list:
                        if isinstance(status_entry, dict) and status_entry.get("status") == "AUTO_DELETION_REQUESTED":
                            status_entry["auto_deletion_processed"] = True
                            status_entry["failed"] = True
                            status_entry["error"] = str(e)
                    mongo_status_table.upsert({"id": job_id}, {"id": job_id, "status": status_list})

    except Exception as e:
        logger.error(f"Error in process_inference_microservice_auto_deletions: {e}")


class Workflow:
    """Workflow is an abstraction that can run on multiple threads.

    Its use is to be able to perform dependency checks and spawn off K8s jobs
    Currently, jobs are packaged inside the ActionPipeline that runs as a thread.
    On application restart, it will check if there were any pending job
    monitoring threads that were interrupted and restart them.
    """

    @staticmethod
    def restart_threads():
        """Method used to restart unfinished job monitoring threads"""
        jobs = get_all_pending_jobs()
        automl_brain_restarted = False
        for job_dict in jobs:
            parent_job_id = job_dict.get("parent_id")
            action = job_dict.get("action")
            job_id = job_dict.get("id")
            name = job_dict.get("name")
            org_name = job_dict.get("org_name")
            specs = job_dict.get("specs")
            platform_id = job_dict.get("platform_id")
            if 'experiment_id' in job_dict:
                kind = 'experiment'
                handler_id = job_dict['experiment_id']
            elif 'dataset_id' in job_dict:
                kind = 'dataset'
                handler_id = job_dict['dataset_id']
            elif 'workspace_id' in job_dict:
                kind = 'workspace'
                handler_id = job_dict['workspace_id']
            else:
                logger.error(
                    "Warning: Job %s monitoring unable to be restarted, "
                    "cannot determine handler kind",
                    job_id
                )
                continue

            handler_metadata = get_handler_metadata(handler_id, kind)
            if not handler_metadata:
                logger.error(
                    "Warning: Job %s monitoring unable to be restarted, "
                    "cannot find %s %s",
                    job_id,
                    kind,
                    handler_id
                )
                continue
            network = get_handler_type(handler_metadata)
            user_id = handler_metadata.get("user_id")
            if not org_name:
                if "org_name" not in handler_metadata:
                    logger.error(
                        "Warning: Job %s monitoring unable to be restarted, "
                        "cannot determine org name",
                        job_id
                    )
                    continue
                org_name = handler_metadata.get("org_name")
            num_gpu = handler_metadata.get("num_gpu", -1)
            isautoml = handler_metadata.get("automl_settings", {}).get("automl_enabled", False)

            job_context = JobContext(
                job_id,
                parent_job_id,
                network,
                action,
                handler_id,
                user_id,
                org_name,
                kind,
                name=name,
                num_gpu=num_gpu,
                specs=specs,
                platform_id=platform_id
            )
            # If job has yet to be executed, skip monitoring
            if still_exists(job_context):
                continue
            logger.error(
                "Found unfinished monitoring thread for job %s, restarting job thread now",
                job_id
            )

            if not isautoml:
                # Reset timeout timer by clearing old status history for restarted jobs
                delete_dnn_status(job_id, automl=False)
                logger.info(f"Cleared status history for restarted job {job_id} to reset timeout timer")
                # Get the correct ActionPipeline and monitor status
                network_config = read_network_config(network)
                action_pipeline_name = network_config["api_params"]["actions_pipe"].get(action, "")
                if action_pipeline_name:
                    action_pipeline = ACTIONS_TO_FUNCTIONS[action_pipeline_name]

                    _Actionpipeline = action_pipeline(job_context)
                    # Thread this!
                    job_run_thread = threading.Thread(
                        target=_Actionpipeline.monitor_job,
                        args=(),
                        name=f'tao-monitor-job-thread-{job_context.id}'
                    )
                    job_run_thread.start()
                    logger.info("Monitoring thread for job %s restarted", job_id)
                else:
                    logger.error("Action pipeline couldn't be found: %s %s %s", network_config, network, job_dict)
            else:
                # Restart AutoML job monitoring threads
                recommendations = get_automl_controller_info(job_id)
                handler_metadata = get_handler_metadata(handler_id, kind + "s")
                if handler_metadata:
                    if not automl_brain_restarted:
                        AutoMLHandler.resume(user_id, org_name, handler_id, job_id, handler_metadata, name=name)
                        automl_brain_restarted = True
                    for recommendation in recommendations:
                        if (recommendation.get("status", None) in ("pending", "running", "started") and
                                recommendation.get("id", None)):
                            rec_id = recommendation["id"]
                            deps = [Dependency(type="automl", name=str(rec_id))]
                            automl_context = JobContext(
                                job_id,
                                parent_job_id,
                                network,
                                action,
                                handler_id,
                                user_id,
                                org_name,
                                kind,
                                name=name,
                                num_gpu=num_gpu,
                                platform_id=platform_id
                            )
                            automl_context.dependencies = deps
                            _AutoMLPipeline = AutoMLPipeline(automl_context)
                            job_run_thread = threading.Thread(
                                target=_AutoMLPipeline.monitor_job,
                                args=(),
                                name=f'tao-monitor-job-thread-{automl_context.id}'
                            )
                            job_run_thread.start()
                            logger.info(
                                f"Restarted AutoML monitoring thread for job {job_id} "
                                f"and recommendation {rec_id}"
                            )

    @staticmethod
    def start():
        """Method used to initialize the workflow. Starts a thread if thread is not there from before"""
        # Make sure there is no other Workflow thread
        for thread in threading.enumerate():
            if thread.name == "WorkflowThreadTAO":
                return False
        # Restart unfinished monitoring threads, if any
        Workflow.restart_threads()
        t = threading.Thread(target=scan_for_jobs)
        t.name = 'WorkflowThreadTAO'
        t.daemon = True
        t.start()
        return True

    @staticmethod
    def enqueue(job):
        """Method used from outside to put a job into the workflow"""
        # Called only by on_new_job()
        job.workflow_status = 'enqueued'
        write_job(job)

    @staticmethod
    def dequeue(job):
        """Method used from outside to remove a job from the workflow"""
        # Simply remove the job from the filename
        # Read all jobs
        job.workflow_status = 'dequeued'
        write_job(job)

    @staticmethod
    def healthy():
        """Method used to see if the workflow thread is running"""
        try:
            mongo_health = MongoHandler("tao", "health")
            health_record = mongo_health.find_latest()
            if not health_record.get("created_on"):
                return False
            last_updated_time = datetime.now(tz=timezone.utc) - health_record.get("created_on")
            total_seconds = last_updated_time.total_seconds()
            if total_seconds > 3600:
                logger.error("Health file was updated %s ago which is > 3600", total_seconds)
            return total_seconds <= 3600
        except Exception as e:
            logger.error(str(e))
            return False
