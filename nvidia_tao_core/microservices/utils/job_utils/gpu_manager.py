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
"""Simple GPU manager that stores the GPU state in DB."""

import logging
import os
from pymongo import ReturnDocument

logger = logging.getLogger(__name__)

if os.getenv("BACKEND") in ("local-k8s", "local-docker"):
    from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler

if os.getenv("BACKEND") == "local-docker":
    import docker
    try:
        docker_client = docker.from_env() if os.getenv("DOCKER_HOST") else None
    except Exception as e:
        logger.error(f"Failed to initialize docker client: {e}")
        docker_client = None


class GPUManager:
    """Simple GPU manager that stores the GPU state in DB."""

    # Sentinel values for tracking GPU release sources (for debugging)
    RELEASE_INIT = -1              # Released during initialization
    RELEASE_NO_JOB_ID = -2         # Released because GPU had no job_id
    RELEASE_NO_STATUS_NO_CONTAINER = -3  # Released because no job status and container not running
    RELEASE_TERMINAL_STATE = -4    # Released because job in terminal state and container stopped
    RELEASE_PARTIAL_ASSIGN_FAIL = -5  # Released due to partial assignment failure
    RELEASE_MANUAL = -6            # Released via manual release_gpus() call

    def __init__(self, num_gpus=-1):
        """Initialize the GPU manager."""
        if num_gpus == -1:
            num_gpus = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
        self.num_gpus = num_gpus
        self.mongo_handler = MongoHandler("tao", "gpus")

        # CRITICAL FIX: Only initialize GPUs that don't exist yet
        # Don't blindly reset all GPUs on every instantiation!
        existing_gpus = list(self.mongo_handler.find({"id": {"$exists": True}}))
        existing_gpu_ids = {g.get("id") for g in existing_gpus}

        for i in range(num_gpus):
            if i not in existing_gpu_ids:
                # Only create GPU entry if it doesn't exist
                logger.debug(f"[GPU_INIT] Initializing GPU {i} (first time)")
                self.mongo_handler.upsert(
                    {"id": i},
                    {"id": i, "status": "available", "job_id": self.RELEASE_INIT, "assigned_at": None}
                )
            else:
                logger.debug(f"[GPU_INIT] GPU {i} already exists in DB, skipping initialization")

        # Verify no duplicate GPU IDs exist
        all_gpus = list(self.mongo_handler.find({"id": {"$exists": True}}))
        gpu_ids = [g.get("id") for g in all_gpus]
        if len(gpu_ids) != len(set(gpu_ids)):
            logger.error(
                f"DUPLICATE GPU IDs detected in MongoDB! "
                f"Total docs: {len(all_gpus)}, Unique IDs: {len(set(gpu_ids))}"
            )
            for gpu in all_gpus:
                logger.error(
                    f"  GPU doc: _id={gpu.get('_id')}, id={gpu.get('id')}, "
                    f"status={gpu.get('status')}, job_id={gpu.get('job_id')}"
                )

        logger.info(
            f"GPU Manager initialized with {num_gpus} GPUs "
            f"(existing: {len(existing_gpu_ids)}, new: {num_gpus - len(existing_gpu_ids)})"
        )

    def _is_container_running(self, container_name):
        """Check if a Docker container is currently running.

        Args:
            container_name: Name of the container to check

        Returns:
            bool: True if container is running, False otherwise
        """
        if os.getenv("BACKEND") != "local-docker":
            logger.warning(f"Container check not supported for backend {os.getenv('BACKEND')}")
            return True  # Assume running for non-docker backends

        if not docker_client:
            logger.error("Docker client not available, cannot check container status")
            return False

        try:
            container = docker_client.containers.get(container_name)
            is_running = container.status.lower() == 'running'
            logger.debug(f"Container {container_name} status: {container.status}, is_running: {is_running}")
            return is_running
        except docker.errors.NotFound:
            logger.debug(f"Container {container_name} not found (likely completed or failed)")
            return False
        except Exception as e:
            logger.error(f"Error checking container {container_name} status: {e}")
            return False

    def _reclaim_stale_gpus(self):
        """Reclaim GPUs assigned to jobs that have completed.

        This is the lazy garbage collection mechanism - checks all assigned GPUs
        and frees those where the job is in a terminal state.

        Terminal states for normal jobs: Done, Error, Canceled, Paused
        Terminal states for AutoML: success, failure, error, done, canceled
        Non-terminal states: Running, Pending, Creating, Canceling, Pausing, running, pending, canceling

        Returns:
            int: Number of GPUs reclaimed
        """
        logger.debug("Checking for stale GPU assignments (jobs in terminal states)...")
        assigned_gpus = self.mongo_handler.find({"status": "assigned", "id": {"$exists": True}})

        # Import here to avoid circular dependency
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_job_metadata
        from nvidia_tao_core.microservices.utils.mongo_utils import MongoHandler

        # Terminal states for both normal jobs and AutoML
        TERMINAL_STATES = {
            "Done", "Error", "Canceled", "Paused", "success", "failure", "error", "done", "canceled"
        }

        reclaimed_count = 0
        for gpu in assigned_gpus:
            gpu_id = gpu.get("id")
            assigned_job_id = gpu.get("job_id")

            if not assigned_job_id:
                logger.warning(
                    f"[GPU_RELEASE] GPU {gpu_id} marked as assigned but has no job_id, "
                    f"marking as available. Reason: No job_id found (sentinel: {self.RELEASE_NO_JOB_ID})"
                )
                self.mongo_handler.upsert(
                    {"id": gpu_id},
                    {"id": gpu_id, "status": "available", "job_id": self.RELEASE_NO_JOB_ID}
                )
                logger.debug(
                    f"[GPU_RELEASE] GPU {gpu_id} → AVAILABLE "
                    f"(was: assigned to None, sentinel: {self.RELEASE_NO_JOB_ID})"
                )
                reclaimed_count += 1
                continue

            # Get job status - handle both normal jobs and AutoML experiments
            job_status = None

            # First try as normal job
            job_metadata = get_handler_job_metadata(assigned_job_id)
            logger.debug(f"Job metadata (gpu_manager) for {assigned_job_id}: {job_metadata}")
            if job_metadata:
                job_status = job_metadata.get("status", "")
                logger.debug(f"Normal Job status (gpu_manager) for {assigned_job_id}: {job_status}")

            # If no direct status, check if it's an AutoML experiment job
            if not job_status or job_status == "":
                # Check if this job has a parent_id
                mongo_jobs = MongoHandler("tao", "jobs")
                job_doc = mongo_jobs.find_one({"id": assigned_job_id})
                logger.debug(
                    f"Job doc (gpu_manager) for {assigned_job_id}: {job_doc}"
                )
                if job_doc and "parent_id" in job_doc:
                    parent_id = job_doc.get("parent_id")
                    logger.debug(f"Parent ID (gpu_manager) for {assigned_job_id}: {parent_id}")
                    # Verify parent is an AutoML brain job (not just a regular parent like train->eval)
                    mongo_automl_jobs = MongoHandler("tao", "automl_jobs")
                    automl_parent = mongo_automl_jobs.find_one({"id": parent_id})
                    logger.debug(f"Automl parent (gpu_manager) for {assigned_job_id}: {automl_parent}")
                    if automl_parent:
                        # This is an AutoML experiment! Get status from controller
                        from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                            get_automl_controller_info
                        )

                        controller_info = get_automl_controller_info(parent_id)
                        logger.debug(f"Controller info (gpu_manager) for {assigned_job_id}: {controller_info}")
                        if controller_info and isinstance(controller_info, list):
                            # Find the recommendation matching this experiment job_id
                            for recommendation in controller_info:
                                logger.debug(f"Recommendation (gpu_manager) for {assigned_job_id}: {recommendation}")
                                if recommendation.get("job_id") == assigned_job_id:
                                    job_status = recommendation.get("status", "")
                                    logger.debug(f"Job status (gpu_manager) for {assigned_job_id}: {job_status}")
                                    logger.debug(
                                        f"GPU {gpu_id}: Job {assigned_job_id} is AutoML experiment "
                                        f"(recommendation {recommendation.get('id')}), "
                                        f"brain job: {parent_id}, status: {job_status}"
                                    )
                                    break

                            if not job_status:
                                logger.warning(
                                    f"GPU {gpu_id}: Job {assigned_job_id} is AutoML experiment but "
                                    f"not found in controller recommendations for brain job {parent_id}"
                                )
                    else:
                        logger.debug(
                            f"GPU {gpu_id}: Job {assigned_job_id} has parent {parent_id} but "
                            f"parent is not an AutoML brain job"
                        )

            if not job_status:
                logger.warning(
                    f"GPU {gpu_id} assigned to job {assigned_job_id} but no job status found. "
                    f"Checking if container exists..."
                )
                # Fallback: check if container is running
                if not self._is_container_running(assigned_job_id):
                    logger.debug(
                        f"[GPU_RELEASE] GPU {gpu_id} assigned to job {assigned_job_id}: "
                        f"no status found and container not running. "
                        f"Reason: No job metadata + container not running "
                        f"(sentinel: {self.RELEASE_NO_STATUS_NO_CONTAINER})"
                    )
                    self.mongo_handler.upsert(
                        {"id": gpu_id},
                        {"id": gpu_id, "status": "available", "job_id": self.RELEASE_NO_STATUS_NO_CONTAINER}
                    )
                    logger.debug(
                        f"[GPU_RELEASE] GPU {gpu_id} → AVAILABLE "
                        f"(was: assigned to {assigned_job_id}, sentinel: {self.RELEASE_NO_STATUS_NO_CONTAINER})"
                    )
                    reclaimed_count += 1
                continue

            # Only reclaim GPU if BOTH conditions are met:
            # 1. Job is in terminal state
            # 2. Container is not running (to ensure cleanup is complete)
            if job_status in TERMINAL_STATES:
                container_running = self._is_container_running(assigned_job_id)
                if not container_running:
                    logger.debug(
                        f"[GPU_RELEASE] GPU {gpu_id} assigned to job {assigned_job_id}: "
                        f"terminal state '{job_status}' AND container not running. "
                        f"Reason: Terminal state + container stopped (sentinel: {self.RELEASE_TERMINAL_STATE})"
                    )
                    self.mongo_handler.upsert(
                        {"id": gpu_id},
                        {"id": gpu_id, "status": "available", "job_id": self.RELEASE_TERMINAL_STATE}
                    )
                    logger.debug(
                        f"[GPU_RELEASE] GPU {gpu_id} → AVAILABLE "
                        f"(was: assigned to {assigned_job_id}, sentinel: {self.RELEASE_TERMINAL_STATE})"
                    )
                    reclaimed_count += 1
                else:
                    logger.debug(
                        f"[GPU_RELEASE_SKIP] GPU {gpu_id} assigned to job {assigned_job_id}: "
                        f"terminal state '{job_status}' BUT container still running, "
                        f"keeping GPU assignment (cleanup in progress)"
                    )
            else:
                logger.debug(
                    f"GPU {gpu_id} assigned to job {assigned_job_id} "
                    f"in non-terminal state '{job_status}', keeping assignment"
                )

        if reclaimed_count > 0:
            logger.debug(f"Reclaimed {reclaimed_count} GPU(s) from completed/failed containers")
        else:
            logger.debug("No stale GPU assignments found")

        return reclaimed_count

    @classmethod
    def decode_sentinel(cls, job_id):
        """Decode sentinel value to human-readable string.

        Args:
            job_id: The job_id value (could be actual job_id or sentinel)

        Returns:
            String describing the release source if sentinel, otherwise the job_id
        """
        sentinel_map = {
            cls.RELEASE_INIT: "INIT (first time initialization)",
            cls.RELEASE_NO_JOB_ID: "NO_JOB_ID (GPU had no job_id)",
            cls.RELEASE_NO_STATUS_NO_CONTAINER: "NO_STATUS_NO_CONTAINER (no job status + container not running)",
            cls.RELEASE_TERMINAL_STATE: "TERMINAL_STATE (job terminal + container stopped)",
            cls.RELEASE_PARTIAL_ASSIGN_FAIL: "PARTIAL_ASSIGN_FAIL (could not assign all requested GPUs)",
            cls.RELEASE_MANUAL: "MANUAL (manual release_gpus() call)"
        }
        return sentinel_map.get(job_id, f"JOB_ID: {job_id}")

    def get_available_gpus(self):
        """Get available GPUs."""
        return self.mongo_handler.find({"status": "available", "id": {"$exists": True}})

    def assign_gpus(self, job_id, num_gpus=-1):
        """Assign GPUs to a job.

        This method now implements lazy garbage collection:
        1. First checks all assigned GPUs to see if their containers are still running
        2. Frees GPUs from containers that have stopped
        3. Then assigns requested number of GPUs to the new job

        Args:
            job_id: Unique identifier for the job (typically container name)
            num_gpus: Number of GPUs to assign (-1 for all available)

        Returns:
            list: List of assigned GPU IDs as strings, empty list if insufficient GPUs
        """
        logger.debug(f"GPU assignment request for job {job_id}: requesting {num_gpus} GPU(s)")

        # Step 1: Reclaim GPUs from completed/failed containers
        reclaimed = self._reclaim_stale_gpus()
        if reclaimed > 0:
            logger.debug(f"After reclaiming {reclaimed} GPU(s), checking availability again")

        # Step 2: Atomically assign GPUs one by one
        # CRITICAL FIX: Use find_one_and_update to avoid read-after-write consistency issues
        # This ensures we atomically claim a GPU AND see the updated state

        if num_gpus == -1:
            available_count = len(self.get_available_gpus())
            num_gpus = available_count
            logger.debug(f"Assigning all available GPUs ({num_gpus}) to job {job_id}")

        assigned_gpus = []
        attempts = 0
        max_attempts = num_gpus * 2  # Allow some retries in case of race conditions

        while len(assigned_gpus) < num_gpus and attempts < max_attempts:
            attempts += 1

            # Atomically find and claim an available GPU
            # Sort by GPU ID to ensure consistent ordering and prevent starvation
            # This avoids stale reads - we get the document we actually modified
            claimed_gpu = self.mongo_handler.collection.find_one_and_update(
                {"status": "available", "id": {"$exists": True}},
                {"$set": {
                    "status": "assigned",
                    "job_id": job_id
                }},
                sort=[("id", 1)],  # Always try lowest ID first for consistency
                return_document=ReturnDocument.AFTER  # Return the document AFTER update
            )

            if claimed_gpu:
                gpu_id = claimed_gpu.get("id")
                mongo_id = claimed_gpu.get("_id")
                assigned_gpus.append(str(gpu_id))
                logger.debug(
                    f"Atomically claimed GPU {gpu_id} (MongoDB _id: {mongo_id}) "
                    f"for job {job_id}"
                )
            else:
                # No available GPUs found
                logger.warning(
                    f"No available GPU found for job {job_id} (attempt {attempts}/{max_attempts}). "
                    f"Assigned so far: {len(assigned_gpus)}/{num_gpus}"
                )
                break

        # Validate we got all requested GPUs
        if len(assigned_gpus) < num_gpus:
            logger.error(
                f"Could only assign {len(assigned_gpus)} GPU(s) to job {job_id}, "
                f"requested {num_gpus}. Insufficient GPUs available."
            )
            # Log which jobs are holding GPUs
            all_assigned = self.mongo_handler.find({"status": "assigned", "id": {"$exists": True}})
            for gpu in all_assigned:
                logger.debug(f"  GPU {gpu.get('id')} is assigned to job {gpu.get('job_id')}")

            # Release the partially assigned GPUs back to available pool
            logger.warning(
                f"[GPU_RELEASE] Partial assignment failure for job {job_id}: "
                f"releasing {len(assigned_gpus)} GPU(s) back. "
                f"Reason: Could not assign all requested {num_gpus} GPUs "
                f"(sentinel: {self.RELEASE_PARTIAL_ASSIGN_FAIL})"
            )
            for gpu_id in assigned_gpus:
                self.mongo_handler.collection.update_one(
                    {"id": int(gpu_id), "job_id": job_id},
                    {"$set": {"status": "available", "job_id": self.RELEASE_PARTIAL_ASSIGN_FAIL}}
                )
                logger.debug(
                    f"[GPU_RELEASE] GPU {gpu_id} → AVAILABLE "
                    f"(was: partially assigned to {job_id}, sentinel: {self.RELEASE_PARTIAL_ASSIGN_FAIL})"
                )
            return []

        logger.debug(f"Successfully assigned {len(assigned_gpus)} GPU(s) to job {job_id}: {assigned_gpus}")

        # DEBUG: Verify MongoDB state after assignment
        all_gpus_after = list(self.mongo_handler.find({"id": {"$exists": True}}))
        logger.debug(f"[GPU_TABLE_AFTER_ASSIGN] GPU table state AFTER assigning to job {job_id}:")
        for gpu in sorted(all_gpus_after, key=lambda x: x.get('id', -1)):
            logger.debug(f"  GPU {gpu.get('id')}: status={gpu.get('status')}, job_id={gpu.get('job_id')}")

        return assigned_gpus

    def release_gpus(self, job_id):
        """Release all GPUs assigned to a job.

        Note: This method is NOT automatically called by the system.
        GPUs are reclaimed via lazy garbage collection in assign_gpus().
        This method is provided for:
        - Manual cleanup/debugging
        - Testing purposes
        - Force-releasing GPUs if needed

        The system is designed to work WITHOUT this method being called.
        All GPU cleanup happens automatically when containers stop and
        new jobs request GPU assignments.

        Args:
            job_id: Job identifier whose GPUs should be released
        """
        assigned_gpus = self.mongo_handler.find({"job_id": job_id})
        gpu_ids = [gpu.get("id") for gpu in assigned_gpus if "id" in gpu]

        if not gpu_ids:
            logger.debug(f"No GPUs were assigned to job {job_id} (already released or never assigned)")
            return

        logger.debug(
            f"[GPU_RELEASE] Manual release requested for job {job_id}: "
            f"freeing {len(gpu_ids)} GPU(s): {gpu_ids}. "
            f"Reason: Manual release_gpus() call (sentinel: {self.RELEASE_MANUAL})"
        )
        self.mongo_handler.update_many(
            {"job_id": job_id},
            {"job_id": self.RELEASE_MANUAL, "status": "available"}
        )
        for gpu_id in gpu_ids:
            logger.debug(
                f"[GPU_RELEASE] GPU {gpu_id} → AVAILABLE "
                f"(was: manually released from {job_id}, sentinel: {self.RELEASE_MANUAL})"
            )

    def get_assigned_gpu_ids(self, job_id):
        """Get all GPUs assigned to a job."""
        assigned_gpus = self.mongo_handler.find({"job_id": job_id})
        gpu_ids = [str(gpu["id"]) for gpu in assigned_gpus if "id" in gpu]
        logger.debug(f"Query for GPUs assigned to job {job_id}: {gpu_ids if gpu_ids else 'none'}")
        return gpu_ids


if os.getenv("BACKEND") == "local-docker":
    gpu_manager = GPUManager()
