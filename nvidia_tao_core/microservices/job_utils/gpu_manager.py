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

logger = logging.getLogger(__name__)

if os.getenv("BACKEND") in ("local-k8s", "local-docker"):
    from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler


class GPUManager:
    """Simple GPU manager that stores the GPU state in DB."""

    def __init__(self, num_gpus=-1):
        """Initialize the GPU manager."""
        if num_gpus == -1:
            num_gpus = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
        self.num_gpus = num_gpus
        self.mongo_handler = MongoHandler("tao", "gpus")

        for i in range(num_gpus):
            self.mongo_handler.upsert({"id": i}, {"id": i, "status": "available", "job_id": None})

        logger.info(f"GPU Manager initialized with {num_gpus} GPUs.")

    def get_available_gpus(self):
        """Get available GPUs."""
        return self.mongo_handler.find({"status": "available", "id": {"$exists": True}})

    def assign_gpus(self, job_id, num_gpus=-1):
        """Assign GPUs to a job."""
        available_gpus = self.get_available_gpus()
        if num_gpus == -1:
            num_gpus = len(available_gpus)
        if num_gpus > len(available_gpus):
            logger.error(f"Requested {num_gpus} GPUs, but only {len(available_gpus)} are available.")
            return []

        assigned_gpus = []
        for i in range(num_gpus):
            if "id" in available_gpus[i]:
                gpu_id = available_gpus[i]["id"]
                self.mongo_handler.upsert({"id": gpu_id}, {"id": gpu_id, "status": "assigned", "job_id": job_id})
                assigned_gpus.append(str(gpu_id))
        logger.info(f"Assigned GPUs for job {job_id}: {assigned_gpus}")
        return assigned_gpus

    def release_gpus(self, job_id):
        """Release all GPUs assigned to a job."""
        self.mongo_handler.update_many({"job_id": job_id}, {"job_id": None, "status": "available"})
        logger.info(f"Released GPUs for job {job_id}.")

    def get_assigned_gpu_ids(self, job_id):
        """Get all GPUs assigned to a job."""
        assigned_gpus = self.mongo_handler.find({"job_id": job_id})
        logger.info(f"Assigned GPUs for job {job_id}: {assigned_gpus}")
        return [str(gpu["id"]) for gpu in assigned_gpus]


if os.getenv("BACKEND") == "local-docker":
    gpu_manager = GPUManager()
