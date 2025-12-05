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

"""Executor module with clean class-based architecture.

This module provides specialized executor classes for different Kubernetes operations.
Each class handles a specific type of operation for better code organization and maintainability.

Usage:
    from nvidia_tao_core.microservices.job_utils.executor import JobExecutor

    job_executor = JobExecutor()
    job_executor.create_job(...)
"""

# Import all executor classes
from nvidia_tao_core.microservices.job_utils.executor.base_executor import BaseExecutor
from nvidia_tao_core.microservices.job_utils.executor.service_executor import ServiceExecutor
from nvidia_tao_core.microservices.job_utils.executor.job_executor import JobExecutor
from nvidia_tao_core.microservices.job_utils.executor.statefulset_executor import StatefulSetExecutor
from nvidia_tao_core.microservices.job_utils.executor.microservice_executor import MicroserviceExecutor
from nvidia_tao_core.microservices.job_utils.executor.deployment_executor import DeploymentExecutor

# Import essential utilities
from nvidia_tao_core.microservices.job_utils.executor.utils import (
    release_name,
    logger
)

# Expose only the classes and essential utilities
__all__ = [
    # Executor Classes - The main interface for this module
    'BaseExecutor',
    'ServiceExecutor',
    'JobExecutor',
    'StatefulSetExecutor',
    'MicroserviceExecutor',
    'DeploymentExecutor',

    # Essential utilities
    'release_name',
    'logger'
]
