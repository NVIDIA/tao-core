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

"""Base executor class with common functionality"""
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND
from nvidia_tao_core.microservices.utils.executor_utils import (
    release_name,
    logger,
    get_namespace,
    get_service_in_cluster_ip,
    get_available_local_k8s_gpus,
    get_owner_reference,
    dependency_check,
    get_cluster_ip
)


class BaseExecutor:
    """Base class for all executor operations with common functionality"""

    def __init__(self):
        """Initialize base executor with common configuration."""
        self.release_name = release_name
        self.backend = BACKEND
        self.logger = logger

    def get_namespace(self):
        """Returns the namespace of the environment"""
        return get_namespace()

    def get_service_in_cluster_ip(self, service_name, namespace="default"):
        """Get the cluster IP of a service"""
        return get_service_in_cluster_ip(service_name, namespace)

    def get_available_local_k8s_gpus(self):
        """Construct a dictionary where key is a UUID and value contains the gpu type and node it belongs to"""
        return get_available_local_k8s_gpus()

    def get_owner_reference(self):
        """Get the owner reference for K8s resources"""
        return get_owner_reference()

    def dependency_check(self, num_gpu=-1, accelerator=None):
        """Checks for GPU dependency"""
        return dependency_check(num_gpu, accelerator)

    def get_cluster_ip(self, namespace='default'):
        """Get cluster IP of service"""
        return get_cluster_ip(namespace)
