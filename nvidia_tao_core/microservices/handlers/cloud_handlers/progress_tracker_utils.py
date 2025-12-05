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

"""Utility functions for progress tracking."""

import os
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


def format_size(size_mb):
    """Format size for display (MB or GB)."""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.1f} GB"
    return f"{size_mb:.1f} MB"


def format_eta(seconds):
    """Format ETA as timedelta string."""
    if seconds <= 0:
        return "0:00:00"
    return str(timedelta(seconds=int(seconds)))


def get_file_size_mb(file_path):
    """Get file size in MB.

    Args:
        file_path (str): Path to the file

    Returns:
        float: File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)  # Convert to MB
    except (OSError, IOError):
        return 0.0


def get_folder_stats(folder_path):
    """Get folder statistics (file count and total size).

    Args:
        folder_path (str): Path to the folder

    Returns:
        tuple: (file_count, total_size_mb)
    """
    file_count = 0
    total_size = 0

    try:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except (OSError, IOError):
                    continue

        return file_count, total_size / (1024 * 1024)  # Convert to MB
    except Exception:
        return 0, 0.0


def send_progress_status_callback(message):
    """Send progress status callback using HTTP (job pods don't have MongoDB access).

    Also notifies inference microservice progress bridge for real-time progress updates.

    Args:
        message (str): Progress message to send
    """
    try:
        # Notify inference microservice progress bridge (if registered)
        try:
            job_id = os.getenv("TAO_API_JOB_ID")
            if job_id:
                from nvidia_tao_core.microservices.handlers.inference_progress_bridge import notify_progress
                notify_progress(job_id, message)
        except Exception as e:
            logger.debug(f"Could not notify inference progress bridge: {e}")

        if os.getenv("CLOUD_BASED") == "True":
            # Use status_callback (HTTP) instead of internal_job_status_update (MongoDB)
            # Job pods don't have MongoDB access, so we send via HTTP to the server
            from nvidia_tao_core.microservices.handlers.cloud_handlers.utils import status_callback
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_internal_job_status_update_data

            # Create status update data with RUNNING status for progress updates
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=message,
                status="RUNNING"  # Progress updates use RUNNING status
            )

            # Send via HTTP callback (server will save to MongoDB)
            status_callback(callback_data)

    except Exception as e:
        # Don't let callback failures break the main operation
        logger.warning(f"Error sending progress status callback: {e}")
