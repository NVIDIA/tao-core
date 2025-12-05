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

"""Utility functions and shared components for executor module"""
import os
import uuid
import logging
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from nvidia_tao_core.microservices.handlers.stateless_handlers import get_toolkit_status

# Global constants
release_name = os.getenv("RELEASE_NAME", 'tao-api')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_namespace():
    """Returns the namespace of the environment"""
    if os.getenv("BACKEND") == "local-docker":
        return "default"
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        name_space = os.getenv('NAMESPACE', default="default")
        config.load_kube_config()
    else:
        with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r', encoding='utf-8') as f:
            current_name_space = f.read()
        name_space = os.getenv('NAMESPACE', default=current_name_space)
        config.load_incluster_config()
    return name_space


def get_service_in_cluster_ip(service_name, namespace="default"):
    """Get the cluster IP of a service"""
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    service = v1.read_namespaced_service(namespace=namespace, name=service_name)
    return service.spec.cluster_ip


def get_available_local_k8s_gpus():
    """Construct a dictionary where key is a UUID and value contains the gpu type and node it belongs to"""
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    nodes = v1.list_node().items
    available_local_k8s_gpus = {}
    for node in nodes:
        node_name = node.metadata.name
        labels = node.metadata.labels
        label_value = labels.get("accelerator")
        if label_value:
            for gpu_type in label_value.split(","):
                platform_id = str(uuid.uuid5(uuid.NAMESPACE_X500, f"local_k8s_{gpu_type}"))
                available_local_k8s_gpus[platform_id] = {"node": node_name,
                                                         "gpu_type": gpu_type}
    return available_local_k8s_gpus


def get_owner_reference():
    """Get the owner reference for K8s resources"""
    api_instance = client.AppsV1Api()
    name_space = get_namespace()
    workflow_deployment = api_instance.read_namespaced_deployment(
        name=f"{release_name}-workflow-pod",
        namespace=name_space)
    owner_reference = client.V1OwnerReference(
        api_version=workflow_deployment.api_version,
        kind=workflow_deployment.kind,
        controller=True,
        name=workflow_deployment.metadata.name,
        uid=workflow_deployment.metadata.uid
    )
    return owner_reference


def dependency_check(num_gpu=-1, accelerator=None):
    """Checks for GPU dependency"""
    from nvidia_tao_core.microservices.handlers.stateless_handlers import BACKEND

    if os.getenv("BACKEND", "") not in ("local-k8s", "local-docker"):
        return True
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    if BACKEND == "local-docker":
        from nvidia_tao_core.microservices.job_utils.gpu_manager import gpu_manager
        available_gpus = gpu_manager.get_available_gpus()
        return bool(available_gpus)
    label_selector = 'accelerator=' + str(accelerator)
    if not accelerator:
        label_selector = None
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    v1 = client.CoreV1Api()
    nodes = {}
    # how many GPUs allocatable per node
    ret = v1.list_node(label_selector=label_selector)
    if ret.items:
        for i in ret.items:
            if i.status and i.status.allocatable:
                for k, v in i.status.allocatable.items():
                    if k == 'nvidia.com/gpu':
                        nodes[i.metadata.name] = int(v)
                        break
    # how many GPUs requested for each node
    ret = v1.list_pod_for_all_namespaces()
    if ret.items:
        for i in ret.items:
            if i.spec.node_name is not None:
                if i.spec and i.spec.containers:
                    for c in i.spec.containers:
                        if c.resources and c.resources.requests:
                            for k, v in c.resources.requests.items():
                                if k == 'nvidia.com/gpu':
                                    current = nodes.get(i.spec.node_name, 0)
                                    nodes[i.spec.node_name] = max(0, current - int(v))
    # do I have enough GPUs on one of the nodes
    for k, v in nodes.items():
        if v >= num_gpu:
            return True
    return False


def get_cluster_ip(namespace='default'):
    """Get cluster IP of service"""
    try:
        # Load kubeconfig file (optional if running in-cluster)
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()
        api_instance = client.CoreV1Api()
        service = api_instance.read_namespaced_service(f"{release_name}-service", namespace)
        cluster_ip = service.spec.cluster_ip
        cluster_port = 8000
        for port in service.spec.ports:
            if port.name == "api":
                cluster_port = port.port
        return cluster_ip, cluster_port
    except Exception as e:
        logger.error(f"Error fetching ClusterIP: {e}")
        return None, None


def override_k8_status(job_name, k8_status):
    """Override kubernetes job status with toolkit status"""
    toolkit_status = get_toolkit_status(job_name)
    override_status = ""
    if k8_status == "Pending":  # We don't want to reverse done/error status to running
        if toolkit_status in ("STARTED", "RUNNING"):
            override_status = "Running"
        if not toolkit_status:
            override_status = "Pending"
    if toolkit_status == "SUCCESS":
        override_status = "Done"
    if toolkit_status == "FAILURE":
        override_status = "Error"
    return override_status


def check_service_ready(service_name, namespace):
    """Check if the specified service is ready."""
    try:
        _ = client.CoreV1Api().read_namespaced_service(name=service_name, namespace=namespace)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        raise e


def check_endpoints_ready(service_name, namespace):
    """Check if the specified service has ready endpoints."""
    try:
        endpoints = client.CoreV1Api().read_namespaced_endpoints(name=service_name, namespace=namespace)
        if not endpoints.subsets:
            return False
        for subset in endpoints.subsets:
            if subset.addresses:
                return True
        return False
    except ApiException as e:
        if e.status == 404:
            return False
        raise e


# Backward compatibility aliases for utility functions
_get_name_space = get_namespace
_get_owner_reference = get_owner_reference
