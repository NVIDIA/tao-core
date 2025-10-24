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

"""Kubernetes job manager modules"""
import os
import time
import uuid
import requests
import traceback
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging

from nvidia_tao_core.microservices.constants import MONAI_NETWORKS, NETWORK_CONTAINER_MAPPING
from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    BACKEND,
    get_handler_job_metadata,
    get_toolkit_status,
    internal_job_status_update,
    write_job_metadata,
    update_job_message,
    get_job_specs
)
from nvidia_tao_core.microservices.handlers.utilities import (
    get_statefulset_name,
    get_statefulset_service_name,
    send_microservice_request
)
from nvidia_tao_core.microservices.handlers.nvcf_handler import (
    create_function,
    deploy_function,
    get_function,
    delete_function_version,
    add_authorized_party,
    create_microservice_job_on_nvcf,
    get_nvcf_microservices_job_status
)
if os.getenv("BACKEND") == "local-docker":
    from nvidia_tao_core.microservices.job_utils.gpu_manager import gpu_manager
if os.getenv("BACKEND"):  # To see if the container is going to be used for Service pods or network jobs
    from nvidia_tao_core.microservices.handlers.mongo_handler import (
        mongo_secret,
        mongo_operator_enabled,
        mongo_namespace
    )
release_name = os.getenv("RELEASE_NAME", 'tao-api')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_name_space():
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
        for gpu_type in label_value.split(","):
            platform_id = str(uuid.uuid5(uuid.NAMESPACE_X500, f"local_k8s_{gpu_type}"))
            available_local_k8s_gpus[platform_id] = {"node": node_name,
                                                     "gpu_type": gpu_type}
    return available_local_k8s_gpus


def create(
    org_name,
    job_name,
    image,
    command,
    num_gpu=-1,
    num_nodes=1,
    accelerator=None,
    docker_env_vars=None,
    port=False,
    nv_job_metadata=None,
    automl_brain=False,
    automl_exp_job=False,
    cl_medical=False,
    local_cluster=False
):
    """Creates a kubernetes job"""
    name_space = _get_name_space()
    host_base_url = os.getenv("HOSTBASEURL", "no_url")
    if host_base_url == "no_url":
        raise ValueError(
            f"Base URL not set in values yaml. Please set it as "
            f"http(s)://<ip_address>:{release_name}-ingress-nginx-controller service's port number>"
        )
    if BACKEND == "NVCF" and nv_job_metadata:
        team_name = nv_job_metadata["teamName"]
        nvcf_backend_details = nv_job_metadata["nvcf_backend_details"]
        ngc_key = nv_job_metadata["TAO_ADMIN_KEY"]
        docker_image_name = nv_job_metadata["dockerImageName"]
        deployment_string = nv_job_metadata.get("deployment_string", "")
        current_available = nvcf_backend_details.get("current_available", 1)
        num_nodes = min(num_nodes, current_available)
        if not deployment_string:
            create_response = create_function(org_name, team_name, job_name, docker_image_name, ngc_key)
            if create_response.ok:
                logger.info(f"Function created successfully for job {job_name}")
                function_metadata = create_response.json()
                function_id = function_metadata["function"]["id"]
                version_id = function_metadata["function"]["versionId"]
                deploy_response = deploy_function(
                    org_name,
                    team_name,
                    function_metadata,
                    nvcf_backend_details,
                    ngc_key,
                    image=docker_image_name,
                    num_nodes=num_nodes
                )
                if deploy_response.ok:
                    deployment_string = f"{function_id}:{version_id}"
                    logger.info(f"Function deployment initiated successfully for job {job_name}")
                else:
                    internal_job_status_update(
                        job_name,
                        automl=automl_exp_job,
                        automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                        message="NVCF function could not be deployed"
                    )
                    logger.error(f"Function deployment request failed for job {job_name}")
                    logger.error(f"Deployment response {deploy_response.text}")
                    raise ValueError(f"Function deployment request failed for job {job_name}")
            else:
                internal_job_status_update(
                    job_name,
                    automl=automl_exp_job,
                    automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                    message="NVCF function couldn't be created, retry job again"
                )
                logger.error(f"Function creation request failed for job {job_name}")
                raise ValueError("NVCF function couldn't be created, retry job again")

        job_metadata = get_handler_job_metadata(job_name)
        job_metadata["backend_details"] = {}
        nv_job_metadata["deployment_string"] = deployment_string
        job_metadata["backend_details"]["nvcf_metadata"] = nv_job_metadata
        write_job_metadata(job_name, job_metadata)
        return

    command = 'umask 0 && ' + command
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))

    if BACKEND == "local-docker":
        docker_handler = DockerHandler(image)
        docker_env_vars = {
            "BACKEND": BACKEND,
            "HOST_PLATFORM": "local-docker",
            "MONGOSECRET": mongo_secret,
            "DOCKER_HOST": os.getenv("DOCKER_HOST", default="unix:///var/run/docker.sock"),
            "DOCKER_NETWORK": os.getenv("DOCKER_NETWORK", default="tao_default")
        }
        volumes = ['/var/run/docker.sock:/var/run/docker.sock'] if automl_brain else None
        docker_handler.start_container(
            job_name,
            command=["/bin/bash", "-c", command],
            num_gpus=num_gpu,
            volumes=volumes,
            docker_env_vars=docker_env_vars)
        return

    node_selector = None
    if accelerator:
        available_gpus = get_available_local_k8s_gpus()
        gpu_to_be_run_on = None
        if available_gpus:
            gpu_to_be_run_on = available_gpus.get(accelerator, "")
        node_selector = {'accelerator': gpu_to_be_run_on}

    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    name_space = _get_name_space()
    api_instance = client.BatchV1Api()

    volume_mounts = []
    if local_cluster:
        if os.getenv("INGRESSENABLED", "false") == "false":
            in_cluster_ip, cluster_port = get_cluster_ip()
        else:
            in_cluster_ip = get_service_in_cluster_ip(f"{release_name}-ingress-nginx-controller", namespace=name_space)
            cluster_port = 80
        # change the host_base_url to the in-cluster ip
        in_cluster_url = f"http://{in_cluster_ip}:{cluster_port}" if nv_job_metadata is None else None
        if "TAO_API_SERVER" in docker_env_vars:
            docker_env_vars["TAO_API_SERVER"] = docker_env_vars["TAO_API_SERVER"].replace(host_base_url, in_cluster_url)
        docker_env_vars["TAO_LOGGING_SERVER_URL"] = (
            docker_env_vars["TAO_LOGGING_SERVER_URL"].replace(host_base_url, in_cluster_url)
        )
    dshm_volume_mount = client.V1VolumeMount(
        name="dshm",
        mount_path="/dev/shm")
    volume_mounts.append(dshm_volume_mount)

    resources = client.V1ResourceRequirements(
        limits={
            'nvidia.com/gpu': str(num_gpu)
        })
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )
    backend_env = client.V1EnvVar(
        name="BACKEND",
        value=BACKEND)
    # CL job needs to set the environment variable to pass GPU checks (validate_num_gpu) for training jobs
    num_gpu_env = client.V1EnvVar(
        name="NUM_GPU_PER_NODE",
        value=str(num_gpu) if not cl_medical else os.getenv('NUM_GPU_PER_NODE', default='1'))
    mongo_secret_env = client.V1EnvVar(
        name="MONGOSECRET",
        value=mongo_secret  # pylint: disable=E0606
    )
    dynamic_docker_envs = []
    if os.getenv("BACKEND"):
        mongo_operator_enabled_env = client.V1EnvVar(
            name="MONGO_OPERATOR_ENABLED",
            value=str(mongo_operator_enabled).lower()
        )
        mongo_namespace_env = client.V1EnvVar(
            name="NAMESPACE",
            value=mongo_namespace
        )
        dynamic_docker_envs.append(mongo_operator_enabled_env)
        dynamic_docker_envs.append(mongo_namespace_env)
    if docker_env_vars:
        for docker_env_var_key, docker_env_var_value in docker_env_vars.items():
            kubernetes_env = client.V1EnvVar(
                name=docker_env_var_key,
                value=docker_env_var_value)
            dynamic_docker_envs.append(kubernetes_env)

    tis_ports = [
        client.V1ContainerPort(container_port=8000, name="http-triton"),
        client.V1ContainerPort(container_port=8001, name="grpc-triton"),
        client.V1ContainerPort(container_port=8002, name="metrics-triton")
    ]
    container = client.V1Container(
        name="container",
        image=image,
        env=[backend_env,
             num_gpu_env,
             mongo_secret_env] + dynamic_docker_envs,
        command=["/bin/bash", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=volume_mounts,
        ports=[] if port is False else tis_ports,
        security_context=security_context)
    dshm_volume = client.V1Volume(
        name="dshm",
        empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))
    restart_policy = "Always"
    if automl_brain:
        restart_policy = "Never"
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={"purpose": "tao-toolkit-job"}
        ),
        spec=client.V1PodSpec(
            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
            containers=[container],
            volumes=[dshm_volume],
            node_selector=node_selector,
            restart_policy=restart_policy))
    spec = client.V1JobSpec(
        ttl_seconds_after_finished=100,
        template=template,
        backoff_limit=0)
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=spec)

    try:
        api_instance.create_namespaced_job(
            body=job,
            namespace=name_space)
        return
    except Exception as e:
        logger.error(f"Exception thrown in executor create is {str(e)}")
        logger.error(traceback.format_exc())
        return


def create_service_unified(service_name, selector, ports, service_type="ClusterIP", labels=None,
                           return_info=False, add_owner_reference=True):
    """Unified function to create Kubernetes services with flexible configuration

    Args:
        service_name: Name of the service
        selector: Selector dictionary for the service
        ports: Either a single port (int), tuple (port, target_port), or list of tuples
               [(port, target_port, name), ...]
        service_type: "ClusterIP", "Headless", or other Kubernetes service types
        labels: Labels dictionary for the service
        return_info: Whether to return service information dict
        add_owner_reference: Whether to add owner reference to the service

    Returns:
        If return_info=True: dict with service_name, cluster_ip, and ports
        If return_info=False: None
    """
    try:
        name_space = _get_name_space()
        api_instance = client.CoreV1Api()

        # Set default labels
        if labels is None:
            labels = {}

        # Handle different port formats
        if isinstance(ports, int):
            # Single port number
            service_ports = [client.V1ServicePort(port=ports, target_port=ports)]
        elif isinstance(ports, tuple) and len(ports) == 2:
            # Single (port, target_port) tuple
            service_ports = [client.V1ServicePort(port=ports[0], target_port=ports[1])]
        elif isinstance(ports, list):
            # List of port configurations
            service_ports = []
            for port_config in ports:
                if isinstance(port_config, tuple):
                    if len(port_config) == 2:
                        # (port, target_port)
                        service_ports.append(client.V1ServicePort(
                            port=port_config[0],
                            target_port=port_config[1],
                            name=f"port-{port_config[0]}"
                        ))
                    elif len(port_config) == 3:
                        # (port, target_port, name)
                        service_ports.append(client.V1ServicePort(
                            port=port_config[0],
                            target_port=port_config[1],
                            name=port_config[2],
                            protocol="TCP"
                        ))
                else:
                    # Just port number
                    service_ports.append(client.V1ServicePort(
                        port=port_config,
                        target_port=port_config,
                        name=f"port-{port_config}"
                    ))
        else:
            raise ValueError(f"Unsupported ports format: {ports}")

        # Configure service spec
        cluster_ip = "None" if service_type == "Headless" else None
        spec = client.V1ServiceSpec(
            type=service_type if service_type != "Headless" else "ClusterIP",
            ports=service_ports,
            selector=selector,
            cluster_ip=cluster_ip
        )

        # Configure metadata
        metadata_kwargs = {"name": service_name, "labels": labels}
        if add_owner_reference:
            metadata_kwargs["owner_references"] = [_get_owner_reference()]

        # Create service object
        service = client.V1Service(
            api_version="v1",
            kind="Service",
            metadata=client.V1ObjectMeta(**metadata_kwargs),
            spec=spec
        )

        # Create the service
        api_response = api_instance.create_namespaced_service(
            namespace=name_space,
            body=service
        )
        logger.info(f"Service created: {service_name}")

        if return_info:
            return {
                "service_name": service_name,
                "cluster_ip": api_response.spec.cluster_ip,
                "ports": ports
            }
        return None

    except Exception as e:
        logger.error(f"Failed to create service {service_name}: {e}")
        if return_info:
            raise
        # For backward compatibility with create_service, don't raise for non-return cases
        logger.error(traceback.format_exc())
        return None


def create_service(service_name, selector, service_port, target_port, labels=None):
    """Legacy function - delegates to unified service creation"""
    if labels is None:
        labels = {}
    return create_service_unified(
        service_name=service_name,
        selector=selector,
        ports=(service_port, target_port),
        service_type="Headless",
        labels=labels,
        return_info=False,
        add_owner_reference=True
    )


def create_flask_service(job_id):
    """Create a service for a microservice pod"""
    service_name = f"flask-service-{job_id}"
    selector = {
        "app": "flask",
        "job-id": job_id
    }
    return create_service_unified(
        service_name=service_name,
        selector=selector,
        ports=(8000, 8000),
        service_type="Headless",
        labels={},
        return_info=False,
        add_owner_reference=True
    )


def create_statefulset_service(job_id, statefulset_type="multinode", ports=None, service_type="ClusterIP"):
    """Create a service for a statefulset with flexible configuration"""
    if statefulset_type == "inference_microservice":
        # Inference microservice service configuration
        service_name = f"ims-svc-{job_id}"
        statefulset_name = f"ims-{job_id}"
        selector = {"statefulset": statefulset_name}
        labels = {"app": "ims"}

        # Handle ports format for inference microservices
        if isinstance(ports, tuple) and len(ports) == 2:
            port_list = [
                (ports[0], ports[0], "http-port"),
                (ports[1], ports[1], "health-port")
            ]
        else:
            port_list = ports or [(8080, 8080, "http-port"), (8081, 8081, "health-port")]

        return create_service_unified(
            service_name=service_name,
            selector=selector,
            ports=port_list,
            service_type=service_type,
            labels=labels,
            return_info=True,
            add_owner_reference=False
        )

    # Multinode service configuration
    service_name = get_statefulset_service_name(job_id)
    selector = {
        "app": "multinode",
        "job-id": job_id
    }
    labels = {
        "app": "multinode",
        "job-id": job_id
    }
    # Default to single port if not specified
    port_info = ports[0] if ports else (8000, 8000)

    return create_service_unified(
        service_name=service_name,
        selector=selector,
        ports=port_info,
        service_type="Headless",
        labels=labels,
        return_info=False,
        add_owner_reference=True
    )


def _get_owner_reference():
    """Get the owner reference for K8s resources"""
    api_instance = client.AppsV1Api()
    name_space = _get_name_space()
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


def wait_for_statefulset_ready(statefulset_name, name_space):
    """Wait for the statefulset to be ready"""
    api_instance = client.AppsV1Api()
    stateful_set_ready = False
    while not stateful_set_ready:
        statefulset_response = api_instance.read_namespaced_stateful_set(
            statefulset_name,
            name_space
        )
        if statefulset_response:
            desired_replicas = statefulset_response.spec.replicas
            ready_replicas = statefulset_response.status.ready_replicas or 0
            if desired_replicas == ready_replicas:
                logger.info(f"Statefulset {statefulset_name} is ready with {ready_replicas} replicas")
                stateful_set_ready = True
            else:
                logger.info(
                    f"Statefulset {statefulset_name} pending with "
                    f"{ready_replicas}/{desired_replicas} ready"
                )
                time.sleep(10)
        else:
            logger.info(f"{statefulset_name} not found.")
            time.sleep(10)


def create_statefulset(job_id, num_gpu_per_node, num_nodes, image, api_port=8000, master_port=29500, accelerator=None,
                       statefulset_type="multinode", custom_command=None, custom_env_vars=None,
                       custom_ports=None, org_name=None, experiment_id=None, is_long_lived=False):
    """Create statefulset with flexible configuration for different types"""
    try:
        # Handle docker-compose backend for inference microservices
        if BACKEND == "local-docker" and statefulset_type == "inference_microservice":
            # Set default api_port for inference microservices if not explicitly provided
            if api_port == 8000:
                api_port = 8080  # Default port for inference microservices

            # Create docker container for inference microservice
            return create_docker_inference_microservice(
                job_id=job_id,
                image=image,
                custom_command=custom_command,
                api_port=api_port,
                num_gpu=num_gpu_per_node
            )

        # Set default api_port for inference microservices if not explicitly provided
        if statefulset_type == "inference_microservice" and api_port == 8000:
            api_port = 8080  # Default port for inference microservices

        # Create service before StatefulSet for better Kubernetes practices (enables immediate DNS resolution)
        if statefulset_type == "inference_microservice":
            # Use default inference microservice ports if no custom ports specified
            service_ports = custom_ports or [(8080, 8080), (8081, 8081)]
        else:
            # Use api_port for multinode services
            service_ports = custom_ports or [(api_port, api_port)]

        create_statefulset_service(job_id, statefulset_type=statefulset_type, ports=service_ports)

        name_space = _get_name_space()
        api_instance = client.AppsV1Api()

        # Set statefulset name and service name based on type
        if statefulset_type == "inference_microservice":
            statefulset_name = f"ims-{job_id}"
            service_name = f"ims-svc-{job_id}"
            app_label = "ims"
        else:
            statefulset_name = get_statefulset_name(job_id)
            service_name = get_statefulset_service_name(job_id)
            app_label = "multinode"

        # Configure labels based on type
        labels = {
            "app": app_label,
            "job-id": job_id
        }

        if statefulset_type == "inference_microservice":
            labels.update({
                "service-type": "long-lived" if is_long_lived else "temporary",
                "auto-cleanup": "false" if is_long_lived else "true",
                "statefulset": statefulset_name,
                "org": org_name or "default",
                "experiment": experiment_id or "",
                "job": job_id or ""
            })

        # Configure environment variables based on type
        env_vars = []
        if statefulset_type == "multinode":
            # Original multinode environment variables
            release_name_env_var = client.V1EnvVar(name="RELEASE_NAME", value=release_name)
            namespace_env_var = client.V1EnvVar(name="NAMESPACE", value=name_space)
            num_gpu_env_var = client.V1EnvVar(name="NUM_GPU_PER_NODE", value=str(num_gpu_per_node))
            world_size_env_var = client.V1EnvVar(name="WORLD_SIZE", value=str(num_nodes))
            node_rank_env_var = client.V1EnvVar(name="NODE_RANK", value_from=client.V1EnvVarSource(
                field_ref=client.V1ObjectFieldSelector(
                    field_path="metadata.labels['apps.kubernetes.io/pod-index']"
                )
            ))
            master_address_env_var = client.V1EnvVar(
                name="MASTER_ADDR",
                value=f"{statefulset_name}-0.{service_name}.{name_space}.svc.cluster.local"
            )
            master_port_env_var = client.V1EnvVar(name="MASTER_PORT", value=str(master_port))
            save_on_each_node_env_var = client.V1EnvVar("SAVE_ON_EACH_NODE", value="True")
            nccl_ib_disable_env_var = client.V1EnvVar(
                "NCCL_IB_DISABLE",
                value=os.getenv("NCCL_IB_DISABLE", default="0")
            )
            nccl_ib_ext_disable_env_var = client.V1EnvVar(
                "NCCL_IBEXT_DISABLE",
                value=os.getenv("NCCL_IBEXT_DISABLE", default="0")
            )
            service_prefix_env_var = client.V1EnvVar(name="JOB_SERVICE_PREFIX", value=statefulset_name)
            service_name_env_var = client.V1EnvVar(name="JOB_SERVICE_NAME", value=service_name)
            env_vars = [namespace_env_var, num_gpu_env_var, world_size_env_var, node_rank_env_var,
                        master_address_env_var, master_port_env_var, save_on_each_node_env_var,
                        nccl_ib_disable_env_var, nccl_ib_ext_disable_env_var, release_name_env_var,
                        service_prefix_env_var, service_name_env_var]
        elif statefulset_type == "inference_microservice":
            # Automatic environment variables for inference microservice
            env_vars = [client.V1EnvVar(name="JOB_ID", value=job_id or "")]

        # Add custom environment variables if provided
        if custom_env_vars:
            env_vars.extend(custom_env_vars)

        # Configure ports
        if statefulset_type == "inference_microservice":
            # Default inference microservice ports: HTTP API (8080) and health check (8081)
            if custom_ports:
                container_ports = [
                    client.V1ContainerPort(
                        container_port=port[0],
                        name=port[2] if len(port) > 2 else f"port-{port[0]}"
                    )
                    for port in custom_ports
                ]
            else:
                container_ports = [
                    client.V1ContainerPort(container_port=8080, name="http-ims"),
                    client.V1ContainerPort(container_port=8081, name="health-ims")
                ]
        elif custom_ports:
            container_ports = [
                client.V1ContainerPort(
                    container_port=port[0],
                    name=port[2] if len(port) > 2 else f"port-{port[0]}"
                )
                for port in custom_ports
            ]
        else:
            container_ports = [
                client.V1ContainerPort(container_port=api_port),
                client.V1ContainerPort(container_port=8080)]

        # Configure command
        if statefulset_type == "inference_microservice" and custom_command:
            # Auto-format inference microservice command with proper initialization
            container_command = ["/bin/bash", "-c"]
            inference_microservice_command = f"""
umask 0 &&
echo "Starting Inference Microservice..." &&
{custom_command}
"""
            container_args = [inference_microservice_command]
        elif custom_command:
            container_command = ["/bin/bash", "-c"]
            container_args = [custom_command]
        else:
            container_command = ["/bin/bash", "-c"]
            container_args = ["flask run --host 0.0.0.0 --port 8000"]

        # Configure container name
        container_name = f"{app_label}-container"

        dshm_volume_mount = client.V1VolumeMount(name="dshm", mount_path="/dev/shm")
        dshm_volume = client.V1Volume(
            name="dshm",
            empty_dir=client.V1EmptyDirVolumeSource(medium="Memory")
        )
        capabilities = client.V1Capabilities(
            add=['SYS_PTRACE']
        )
        security_context = client.V1SecurityContext(
            capabilities=capabilities
        )
        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
        node_selector = None
        if accelerator:
            available_gpus = get_available_local_k8s_gpus()
            gpu_to_be_run_on = None
            if available_gpus:
                gpu_to_be_run_on = available_gpus.get(accelerator, {}).get("gpu_type")
            node_selector = {'accelerator': gpu_to_be_run_on}

        # Configure probes (only for multinode, not for inference microservice)
        probes = {}
        if statefulset_type == "multinode":
            probes = {
                "readiness_probe": client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path="/api/v1/health/readiness",
                        port=8000
                    ),
                    initial_delay_seconds=10,
                    period_seconds=10,
                    timeout_seconds=5,
                    failure_threshold=3
                ),
                "liveness_probe": client.V1Probe(
                    http_get=client.V1HTTPGetAction(
                        path="/api/v1/health/liveness",
                        port=8000
                    ),
                    initial_delay_seconds=10,
                    period_seconds=10,
                    timeout_seconds=5,
                    failure_threshold=3
                )
            }

        # Create container
        container_spec = {
            "name": container_name,
            "image": image,
            "command": container_command,
            "args": container_args,
            "resources": client.V1ResourceRequirements(
                limits={
                    "nvidia.com/gpu": (
                        num_gpu_per_node if statefulset_type == "multinode"
                        else (num_gpu_per_node if num_gpu_per_node > 0 else 1)
                    )
                }
            ),
            "env": env_vars,
            "ports": container_ports,
            "volume_mounts": [dshm_volume_mount],
            "security_context": security_context
        }

        # Add probes if they exist
        if probes:
            container_spec.update(probes)

        container = client.V1Container(**container_spec)

        # Configure affinity (only for multinode)
        affinity = None
        if statefulset_type == "multinode":
            affinity = client.V1Affinity(
                pod_anti_affinity=client.V1PodAntiAffinity(
                    preferred_during_scheduling_ignored_during_execution=[
                        client.V1WeightedPodAffinityTerm(
                            weight=100,
                            pod_affinity_term=client.V1PodAffinityTerm(
                                label_selector=client.V1LabelSelector(
                                    match_expressions=[
                                        client.V1LabelSelectorRequirement(
                                            key="app",
                                            operator="In",
                                            values=["multinode"]
                                        )
                                    ]
                                ),
                                topology_key="kubernetes.io/hostname"
                            )
                        )
                    ]
                )
            )

        # Create metadata with owner references (only for multinode)
        metadata_spec = {"name": statefulset_name}
        if statefulset_type == "multinode":
            metadata_spec["owner_references"] = [_get_owner_reference()]

        # Add labels to metadata
        if statefulset_type == "inference_microservice":
            metadata_spec["labels"] = {
                "app": app_label,
                "service-type": "long-lived" if is_long_lived else "temporary",
                "auto-cleanup": "false" if is_long_lived else "true",
                "org": org_name or "default",
                "experiment": experiment_id or "",
                "job": job_id or ""
            }

        stateful_set = client.V1StatefulSet(
            api_version="apps/v1",
            kind="StatefulSet",
            metadata=client.V1ObjectMeta(**metadata_spec),
            spec=client.V1StatefulSetSpec(
                replicas=num_nodes,
                pod_management_policy="Parallel",
                selector=client.V1LabelSelector(
                    match_labels=(
                        {"statefulset": statefulset_name}
                        if statefulset_type == "inference_microservice"
                        else labels
                    )
                ),
                service_name=service_name,
                template=client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels=labels
                    ),
                    spec=client.V1PodSpec(
                        image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
                        containers=[container],
                        volumes=[dshm_volume],
                        node_selector=node_selector,
                        restart_policy="Always",
                        affinity=affinity
                    )
                )
            )
        )
        api_instance.create_namespaced_stateful_set(
            namespace=name_space,
            body=stateful_set
        )
        # Ensure the statefulset is ready
        wait_for_statefulset_ready(statefulset_name, name_space)
        return True
    except Exception as e:
        logger.error(f"Exception thrown in create_statefulset is {str(e)}")
        logger.error(traceback.format_exc())
        return False


def delete_service(job_id=None, service_name=None, service_type="default"):
    """Delete a microservice pod's service with flexible service name handling

    Args:
        job_id: Optional job ID (can be None if service_name is provided)
        service_name: Optional service name (can be None if job_id is provided and service_type is set)
        service_type: Type of service to determine naming pattern
                     - "default": uses job_id for service_name if service_name not provided
                     - "inference_microservice": extracts job_id from "ims-svc-{job_id}" pattern
                     - "flask": uses "flask-service-{job_id}" pattern
    """
    try:
        # Handle different service naming patterns
        if service_name is None:
            if job_id is None:
                raise ValueError("Either job_id or service_name must be provided")

            if service_type == "flask":
                service_name = f"flask-service-{job_id}"
            elif service_type == "statefulset":
                service_name = get_statefulset_service_name(job_id)
            else:
                # Default case
                service_name = job_id
        elif service_type == "inference_microservice" and job_id is None:
            # Extract job_id from inference microservice service name pattern
            if service_name.startswith("ims-svc-"):
                job_id = service_name.replace("ims-svc-", "")
            else:
                job_id = service_name  # fallback

        name_space = _get_name_space()
        core_v1 = client.CoreV1Api()
        service = core_v1.read_namespaced_service(name=service_name, namespace=name_space)
        if not service:
            logger.info(f"Service {service_name} not found in namespace {name_space}")
            return True  # Return True since the goal (service not existing) is achieved

        core_v1.delete_namespaced_service(name=service_name, namespace=name_space)
        logger.info(f"Successfully deleted service: {service_name}")
        return True
    except Exception as e:
        logger.error(f"Exception thrown in delete_service is {str(e)}")
        logger.error(traceback.format_exc())
        return False


def create_microservice_pod(job_name, image, num_gpu=-1, accelerator=None):
    """Create pod to invoke microservices"""
    create_flask_service(job_name)
    name_space = _get_name_space()
    api_instance = client.BatchV1Api()

    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')

    node_selector = None
    if accelerator:
        available_gpus = get_available_local_k8s_gpus()
        gpu_to_be_run_on = None
        if available_gpus:
            gpu_to_be_run_on = available_gpus.get(accelerator, {}).get("gpu_type")
        node_selector = {'accelerator': gpu_to_be_run_on}

    dshm_volume = client.V1Volume(
        name="dshm",
        empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))

    dshm_volume_mount = client.V1VolumeMount(
        name="dshm",
        mount_path="/dev/shm")

    resources = client.V1ResourceRequirements(
        limits={
            'nvidia.com/gpu': str(num_gpu)
        })
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )

    container = client.V1Container(
        name="container",
        image=image,
        command=["/bin/bash", "-c"],
        args=["flask run --host 0.0.0.0 --port 8000"],
        resources=resources,
        volume_mounts=[dshm_volume_mount],
        ports=[],
        readiness_probe=client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path="/api/v1/health/readiness",
                port=8000
            ),
            initial_delay_seconds=10,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3
        ),
        liveness_probe=client.V1Probe(
            http_get=client.V1HTTPGetAction(
                path="/api/v1/health/liveness",
                port=8000
            ),
            initial_delay_seconds=10,
            period_seconds=10,
            timeout_seconds=5,
            failure_threshold=3
        ),
        security_context=security_context)

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={
                "purpose": "tao-toolkit-job",
                "app": "flask",
                "job-id": job_name
            }
        ),
        spec=client.V1PodSpec(
            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
            containers=[container],
            volumes=[dshm_volume],
            node_selector=node_selector,
            restart_policy="Always"))

    spec = client.V1JobSpec(
        ttl_seconds_after_finished=100,
        template=template,
        backoff_limit=0)

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=spec)

    try:
        api_instance.create_namespaced_job(
            body=job,
            namespace=name_space)

        # Wait for the pod to be running
        core_v1 = client.CoreV1Api()
        pod_name = None
        while not pod_name:
            pods = core_v1.list_namespaced_pod(namespace='default', label_selector=f'job-name={job_name}')
            if pods.items:
                pod_name = pods.items[0].metadata.name
            time.sleep(10)

        logger.info(f"Pod {pod_name} is running. Waiting for it to be ready")

        # Ensure the pod is ready
        pod_ip = None
        pod_ready = False
        while not pod_ready or not pod_ip:
            pod = core_v1.read_namespaced_pod(name=pod_name, namespace='default')
            pod_ready = all(
                condition.status == 'True'
                for condition in pod.status.conditions
                if condition.type == 'Ready'
            )
            pod_ip = pod.status.pod_ip
            time.sleep(10)

        logger.info(f"Pod {pod_name} is ready with IP {pod_ip}.")
        time.sleep(10)
    except Exception as e:
        logger.error(f"Exception thrown in create_microservice_pod is {str(e)}")
        logger.error(traceback.format_exc())


def check_service_ready(service_name, namespace):
    """Check if the specified service is ready.

    Args:
        service_name (str): The name of the service to check.
        namespace (str): The namespace where the service is located.

    Returns:
        bool: True if the service is found, False otherwise.
    """
    try:
        _ = client.CoreV1Api().read_namespaced_service(name=service_name, namespace=namespace)
        return True
    except ApiException as e:
        if e.status == 404:
            return False
        raise e


def check_endpoints_ready(service_name, namespace):
    """Check if the specified service has ready endpoints.

    Args:
        service_name (str): The name of the service to check.
        namespace (str): The namespace where the service is located.

    Returns:
        bool: True if the service has ready endpoints, False otherwise.
    """
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


def wait_for_container(container_handler, job_id, port=8000):
    """Wait for the container to be ready."""
    start_time = time.time()
    while time.time() - start_time < 300:
        metadata_status = get_handler_job_metadata(job_id).get("status")
        if metadata_status in ("Canceled", "Canceling", "Paused", "Pausing"):
            return metadata_status
        if container_handler.check_container_health(port=port):
            logger.info(f"Container '{job_id}' is ready.")
            return "Running"
        logger.info(f"Waiting for container '{job_id}' to be ready...")
        time.sleep(10)
    logger.error("Timed out waiting for container to be ready.")
    return "Error"


def wait_for_service(job_id, service_name=None):
    """Wait until the specified service is ready or timeout is reached.

    Args:
        org_name (str): Org name under which job is submitted.
        handler_id (uuid): The handler id associated with the job.
        job_id (uuid): The job_id associated with the name of the service to wait for.
        handler_kind (str): If the job belongs to datasets or experiments.

    Returns:
        bool: True if the service is ready within the timeout period, False otherwise.
    """
    if not service_name:
        service_name = get_statefulset_service_name(job_id)
    namespace = _get_name_space()
    start_time = time.time()
    while time.time() - start_time < 300:
        metadata_status = get_handler_job_metadata(job_id).get("status")
        if metadata_status in ("Canceled", "Canceling", "Paused", "Pausing"):
            return metadata_status
        if check_service_ready(service_name, namespace) and check_endpoints_ready(service_name, namespace):
            logger.info(f"Service '{service_name}' is ready.")
            return "Running"
        logger.info(f"Waiting for service '{service_name}' to be ready...")
        time.sleep(10)
    logger.error(f"Timed out waiting for service '{service_name}' to be ready.")
    return "Error"


def create_microservice_and_send_request(
    api_endpoint,
    network,
    action,
    cloud_metadata={},
    specs={},
    microservice_pod_id="",
    nvcf_helm="",
    num_gpu=-1,
    microservice_container="",
    org_name="",
    handler_id="",
    handler_kind="",
    accelerator=None,
    docker_env_vars={},
    num_nodes=1,
):
    """Create a DNN container microservice pod and send request to the POD IP"""
    try:
        if not microservice_pod_id:
            microservice_pod_id = str(uuid.uuid4())
        if num_gpu == -1:
            num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
        if not microservice_container:
            microservice_container = os.getenv(f'IMAGE_{NETWORK_CONTAINER_MAPPING[network]}')
            if action == "gen_trt_engine":
                microservice_container = os.getenv('IMAGE_TAO_DEPLOY')

        if BACKEND == "local-docker":
            port = 8000
            # Use the reusable docker creation function
            if create_docker_inference_microservice(
                job_id=microservice_pod_id,
                image=microservice_container,
                custom_command=f"flask run --host 0.0.0.0 --port {port}",
                api_port=port,
                num_gpu=num_gpu
            ):
                docker_handler = DockerHandler.get_handler_for_container(microservice_pod_id)
                response = docker_handler.make_container_request(
                    api_endpoint,
                    network,
                    action,
                    cloud_metadata=cloud_metadata,
                    specs=specs,
                    job_id=microservice_pod_id,
                    docker_env_vars=docker_env_vars,
                    port=port
                )
                if response.status_code != 200 and response.text:
                    logger.error(f"Error when sending microservice request {response.text}")
                    internal_job_status_update(
                        microservice_pod_id,
                        message=f"Error when sending microservice request {response.text}"
                    )
                    docker_handler.stop_container()
                    gpu_manager.release_gpus(microservice_pod_id)
                    return None
                if api_endpoint != "post_action":
                    docker_handler.stop_container()
                    gpu_manager.release_gpus(microservice_pod_id)
                return response
            internal_job_status_update(
                microservice_pod_id,
                message=f"Error when creating microservice pod {microservice_pod_id}"
            )
            return None

        if BACKEND == "local-k8s":
            service_name = get_statefulset_service_name(microservice_pod_id)
            create_statefulset(
                microservice_pod_id,
                num_gpu,
                num_nodes,
                microservice_container,
                accelerator=accelerator
            )
            if wait_for_service(microservice_pod_id, service_name=service_name):
                response = send_microservice_request(
                    api_endpoint,
                    network,
                    action,
                    cloud_metadata=cloud_metadata,
                    specs=specs,
                    job_id=microservice_pod_id,
                    nvcf_helm=nvcf_helm,
                    docker_env_vars=docker_env_vars,
                    statefulset_replicas=num_nodes
                )
                if response.status_code != 200 and response.text:
                    logger.error(f"Error when sending microservice request {response.text}")
                    internal_job_status_update(
                        microservice_pod_id,
                        message=f"Error when sending microservice request {response.text}"
                    )
                    delete(microservice_pod_id, use_ngc=False)
                    return None
                if api_endpoint != "post_action":
                    delete(microservice_pod_id, use_ngc=False)
                return response
        return None
    except Exception as e:
        logger.error(f"Exception thrown in create_microservice_and_send_request is {str(e)}")
        logger.error("Exception in create ms pod and send request")
        logger.error(traceback.format_exc())
        internal_job_status_update(
            microservice_pod_id,
            message=f"Error when creating microservice pod {microservice_pod_id}"
        )
        delete(microservice_pod_id, use_ngc=False)
        return None


def create_triton_deployment(deployment_name, image, command, replicas, num_gpu=-1, ports=(8000, 8001, 8002)):
    """Creates a Triton deployment"""
    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    dshm_volume_mount = client.V1VolumeMount(
        name="dshm",
        mount_path="/dev/shm")
    resources = client.V1ResourceRequirements(
        limits={
            'nvidia.com/gpu': 1
            # can add other resources like cpu, memory
        })
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )
    tis_ports = [
        client.V1ContainerPort(container_port=ports[0], name="http-triton"),
        client.V1ContainerPort(container_port=ports[1], name="grpc-triton"),
        client.V1ContainerPort(container_port=ports[2], name="metrics-triton")
    ]
    container = client.V1Container(
        name="container",
        image=image,
        command=["/bin/bash", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=[dshm_volume_mount],
        ports=tis_ports,
        security_context=security_context)

    dshm_volume = client.V1Volume(
        name="dshm",
        empty_dir=client.V1EmptyDirVolumeSource(medium='Memory'))
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={
                "purpose": "tao-toolkit-job",
                "app": deployment_name,  # use deployment_name as the selector name
            }
        ),
        spec=client.V1PodSpec(
            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)],
            containers=[container],
            volumes=[dshm_volume],
        ))
    spec = client.V1DeploymentSpec(
        replicas=replicas,
        template=template,
        selector={"matchLabels": {"app": deployment_name}})
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=deployment_name),
        spec=spec)
    logger.info("Prepared deployment configs")
    try:
        api_instance.create_namespaced_deployment(
            body=deployment,
            namespace=name_space)
        logger.info("Start create deployment")
        return
    except Exception as e:
        logger.error(f"Create deployment got error: {e}")
        return


def create_tis_service(tis_service_name, deploy_label, ports=(8000, 8001, 8002)):
    """Create TIS service"""
    name_space = _get_name_space()
    tis_ports = [
        client.V1ServicePort(name="http", protocol="TCP", port=ports[0], target_port=ports[0]),
        client.V1ServicePort(name="grpc", protocol="TCP", port=ports[1], target_port=ports[1]),
        client.V1ServicePort(name="metrics", protocol="TCP", port=ports[2], target_port=ports[2]),
    ]
    spec = client.V1ServiceSpec(ports=tis_ports, selector={"app": deploy_label}, type="LoadBalancer")
    # add annotation, it will only works in Azure, but will not affect other cloud
    annotation = {
        "service.beta.kubernetes.io/azure-load-balancer-internal": "true"
    }
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=tis_service_name, labels={"app": tis_service_name}, annotations=annotation),
        spec=spec,
    )
    api_instance = client.CoreV1Api()
    logger.info("Prepared TIS Service configs")
    try:
        api_instance.create_namespaced_service(
            body=service,
            namespace=name_space)
        logger.info("Start create TIS Service")
        return
    except Exception as e:
        logger.error(f"Create TIS Service got error: {e}")
        return


def get_triton_deployment_pods(deployment_name):
    """Returns pods of a Triton deployment"""
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()
    label_selector = f"app={deployment_name}"
    try:
        pods = api_instance.list_namespaced_pod(
            namespace=name_space,
            label_selector=label_selector)
        pods_ip = []
        for pod in pods.items:
            pods_ip.append(pod.status.pod_ip)
        return pods_ip
    except Exception as e:
        logger.error(f"Got {type(e)} error: {e}")
        return []


def status_triton_deployment(deployment_name, replicas=1):
    """Returns status of Triton deployment

    Status definition:
    Running: The Triton deployment is ready and running
    ReplicaNotReady: at least one replica of the deployment is not ready.
    NotFound: cannot find the deployment.  This status is useful to check if the deployment is stopped.
    Error: meet exceptions except not found error when check the status.
    """
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.read_namespaced_deployment_status(
            name=deployment_name,
            namespace=name_space)
        available_replicas = api_response.status.available_replicas
        if not isinstance(available_replicas, int) or available_replicas < replicas:
            return {"status": "ReplicaNotReady"}
        return {"status": "Running"}
    except ApiException as e:
        if e.status == 404:
            logger.info("Trion Deployment not found.")
            # TODO: here defined a new status to find the situation that the deployment does not exists
            # This status is useful to check if the deployment is deleted or not created
            return {"status": "NotFound"}
        logger.error(f"Got other ApiException error: {e}")
        return {"status": "Error"}
    except Exception as e:
        logger.error(f"Got {type(e)} error: {e}")
        return {"status": "Error"}


def status_tis_service(tis_service_name, ports=(8000, 8001, 8002)):
    """Returns status of TIS Service

    Status definition:
    Running: The TIS Service is ready and running
    NotReady: the TIS Service is not ready.
    NotFound: cannot find the TIS Service. This status is useful to check if the service is stopped.
    Error: meet exceptions except not found error when check the status.
    """
    name_space = _get_name_space()
    try:
        tis_service_ip = get_service_in_cluster_ip(tis_service_name, namespace=name_space)
        # need to double confirm if the service is ready
        url = f"http://{tis_service_ip}:{ports[0]}/v2/health/ready"
        try:
            endpoint_response = requests.get(url, timeout=120)
            if endpoint_response.status_code == 200:
                return {"status": "Running", "tis_service_ip": tis_service_ip}
            # TODO: here defined a new status, in order to find the situation that
            # the TIS Service is started but not ready.
            return {"status": "NotReady"}
        except Exception as e:
            logger.error(f"Exception thrown in status_tis_service is {str(e)}")
            return {"status": "NotReady"}
    except ApiException as e:
        if e.status == 404:
            logger.info("TIS Service not found.")
            # TODO: here defined a new status, in order to find the situation that the TIS Service not exists
            # This status is useful to check if the TIS Service is deleted or not created
            return {"status": "NotFound"}
        logger.error(f"Got other ApiException error: {e}")
        return {"status": "Error"}
    except Exception as e:
        logger.error(f"Got {type(e)} error: {e}")
        return {"status": "Error"}


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


def status(
    org_name,
    handler_id,
    job_name,
    handler_kind,
    use_ngc=True,
    network="",
    action="",
    automl_exp_job=False,
    docker_env_vars={},
    authorized_party_nca_id="",
    automl_experiment_id="0"
):
    """Returns status of kubernetes job"""
    name_space = None
    if BACKEND == "local-k8s":
        name_space = _get_name_space()
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()

    if BACKEND == "NVCF" and use_ngc:
        try:
            job_metadata = get_handler_job_metadata(job_name)
            job_handler_id = job_metadata.get("handler_id", "")
            nv_job_metadata = job_metadata.get("backend_details", {}).get("nvcf_metadata", {})
            job_status = job_metadata.get("status", "Pending")
            team_name = nv_job_metadata.get("teamName", "")
            ngc_key = docker_env_vars.get("TAO_USER_KEY")
            deployment_string = nv_job_metadata.get("deployment_string", "")
            tao_api_status_callback_url = docker_env_vars.get("TAO_LOGGING_SERVER_URL")
            job_message_job_id = tao_api_status_callback_url.split("/")[-1]
            if job_status == "Pending":
                if deployment_string.find(":") != -1:
                    function_id, version_id = deployment_string.split(":")
                    nvcf_function_response = get_function(org_name, team_name, function_id, version_id, ngc_key)
                    if nvcf_function_response.status_code == 200:
                        nvcf_function_metadata = nvcf_function_response.json()
                        update_job_message(
                            job_handler_id,
                            job_message_job_id,
                            handler_kind,
                            "NVCF function is being deployed",
                            automl_expt_job_id=job_name,
                            update_automl_expt=True
                        )
                    else:
                        internal_job_status_update(
                            job_name,
                            automl=automl_exp_job,
                            automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                            message="NVCF function details cant be retrieved"
                        )
                        return "Error"
                    if nvcf_function_metadata.get("function", {}).get("status") == "ACTIVE":
                        logger.info("NVCF function is active, creating microservice job on NVCF")
                        deployment_string = (
                            f"{nvcf_function_metadata['function']['id']}:"
                            f"{nvcf_function_metadata['function']['versionId']}"
                        )
                        job_status, message = create_microservice_job_on_nvcf(
                            job_metadata, docker_env_vars=docker_env_vars
                        )
                        job_metadata["status"] = job_status
                        if job_metadata.get("job_details", {}).get(job_name, {}):
                            job_metadata["job_details"][job_name]["detailed_status"]["message"] = message
                        write_job_metadata(job_name, job_metadata)
                        if authorized_party_nca_id:
                            logger.info(
                                f"Adding authorized party {authorized_party_nca_id} for job {job_name}")
                            add_authorized_party(
                                org_name,
                                team_name,
                                function_id,
                                version_id,
                                authorized_party_nca_id,
                                ngc_key
                            )

                    if nvcf_function_metadata.get("function", {}).get("status") == "ERROR":
                        logger.error(f"Get function deployment status for job {job_name} returned error")
                        internal_job_status_update(
                            job_name,
                            automl=automl_exp_job,
                            automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                            message="NVCF function metadata has ERROR status"
                        )
                        return "Error"

            override_status = override_k8_status(job_name, job_status)
            job_status = get_nvcf_microservices_job_status(
                job_metadata,
                status=override_status,
                docker_env_vars=docker_env_vars
            )
            if override_status and override_status != job_status:
                job_status = override_status
                logger.warning(
                    f"job metadata status is {job_status}, Toolkit Status is {override_status}, so overwriting")
                logger.warning(f"Microservices job status via NVCF is {job_status}")
            return job_status
        except Exception as e:
            logger.error(f"Exception caught for {job_name} {e}")
            logger.error(traceback.format_exc())
            return "Error"

    # For local cluster jobs
    if network not in MONAI_NETWORKS:
        specs = get_job_specs(job_name, automl=automl_exp_job, automl_experiment_id=automl_experiment_id)
        if not specs:
            logger.error(f"Unable to retrieve specs for job {job_name}")
            return "Error"

        if BACKEND == "local-docker":
            docker_handler = DockerHandler.get_handler_for_container(job_name)
            if docker_handler:
                response = docker_handler.make_container_request(
                    api_endpoint="get_job_status",
                    network=network,
                    action=action,
                    job_id=job_name,
                    specs=specs,
                )
                if response and response.ok:
                    job_status = response.json()
                    status = job_status.get("status")
                    return status
                logger.error(f"Error when sending microservice request {response.text}")
            return "Error"

        service_status = wait_for_service(job_name)
        if service_status == "Running":
            response = send_microservice_request(
                api_endpoint="get_job_status",
                network=network,
                action=action,
                job_id=job_name,
                specs=specs,
            )
            if response and response.ok:
                job_status = response.json()
                status = job_status.get("status")
                return status
        elif service_status in ("Canceled", "Canceling", "Paused", "Pausing"):
            return service_status
        return "Error"

    api_instance = client.BatchV1Api()
    try:
        api_response = api_instance.read_namespaced_job_status(
            name=job_name,
            namespace=name_space)
        if api_response.status.succeeded is not None:
            return "Done"
        if api_response.status.failed is not None:
            return "Error"
        return "Running"
    except ApiException as e:
        logger.error(traceback.format_exc())
        if e.status == 404:
            logger.info("Job not found.")
            return "NotFound"
        return "Error"
    except Exception:
        logger.error(traceback.format_exc())
        return "Error"


def delete_triton_deployment(deployment_name):
    """Deletes a Triton deployment"""
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.delete_namespaced_deployment(
            name=deployment_name,
            namespace=name_space,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5))
        logger.info(f"Triton Deployment deleted. status='{str(api_response.status)}'")
        return
    except Exception as e:
        logger.error(f"Triton Deployment failed to delete, got error: {e}")
        return


def delete_tis_service(tis_service_name):
    """Deletes TIS service"""
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()
    try:
        api_response = api_instance.delete_namespaced_service(
            name=tis_service_name,
            namespace=name_space,
        )
        logger.info(f"TIS Service deleted. status='{str(api_response.status)}'")
        return
    except Exception as e:
        logger.error(f"TIS Service failed to delete, got error: {e}")
        return


def delete_nvcf_function(job_name):
    """Deletes an NVCF Function"""
    job_metadata = get_handler_job_metadata(job_name)
    org_name = job_metadata.get("org_name")
    nv_job_metadata = job_metadata.get("backend_details", {}).get("nvcf_metadata", {})
    team_name = nv_job_metadata["teamName"]
    ngc_key = nv_job_metadata["TAO_USER_KEY"]
    deployment_string = nv_job_metadata.get("deployment_string", "")
    if deployment_string.find(":") == -1:
        logger.warning(f"Deployment not active yet {job_name}")
        return
    function_id, version_id = deployment_string.split(":")
    delete_function_version(org_name, team_name, function_id, version_id, ngc_key)


def delete(job_name, use_ngc=True, resource_type="multinode"):
    """Deletes a Job or StatefulSet

    Args:
        job_name: Name of the job/resource to delete
        use_ngc: Whether to use NGC for NVCF functions
        resource_type: Type of resource to delete ("multinode" or "inference_microservice")

    Returns:
        bool: True if deletion successful or resource not found, False if error occurred
    """
    if BACKEND == "local-docker":
        docker_handler = DockerHandler.get_handler_for_container(job_name)
        if docker_handler:
            docker_handler.stop_container()
        else:
            logger.error(f"Docker container not found for job {job_name}")
        gpu_manager.release_gpus(job_name)
        return True

    name_space = _get_name_space()
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()
    if BACKEND == "NVCF" and use_ngc:
        delete_nvcf_function(job_name)
        return True
    api_instance = client.AppsV1Api()
    try:
        # Configure naming and service type based on resource type
        if resource_type == "inference_microservice":
            stateful_set_name = f"ims-{job_name}"
            service_type = "inference_microservice"
        else:
            stateful_set_name = get_statefulset_name(job_name)
            service_type = "statefulset"

        stateful_set = api_instance.read_namespaced_stateful_set(
            name=stateful_set_name,
            namespace=name_space
        )
        if not stateful_set:
            logger.info(f"Statefulset {stateful_set_name} not found in namespace {name_space}")
            return True  # Deletion goal achieved - resource doesn't exist
        api_response = api_instance.delete_namespaced_stateful_set(
            name=stateful_set_name,
            namespace=name_space,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5
            )
        )
        logger.info(f"Statefulset deleted. status='{str(api_response.status)}'")
        delete_service(job_id=job_name, service_type=service_type)
        return True
    except Exception as e:
        logger.error(f"Exception caught in delete_statefulset {str(e)}")
        logger.error("Statefulset failed to delete.")
        return False


def delete_job(job_name, use_ngc=True):
    """Deletes a kubernetes job"""
    name_space = _get_name_space()
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        config.load_kube_config()
    else:
        config.load_incluster_config()

    if BACKEND == "NVCF" and use_ngc:
        delete_nvcf_function(job_name)
        return

    api_instance = client.BatchV1Api()
    try:
        delete_service(job_id=job_name, service_type="flask")
        api_response = api_instance.delete_namespaced_job(
            name=job_name,
            namespace=name_space,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5))
        logger.info(f"Job deleted. status='{str(api_response.status)}'")
        return
    except Exception as e:
        logger.error(f"Exception caught in delete_job {str(e)}")
        logger.error("Job failed to delete.")
        return


def list_namespace_jobs():
    """List kubernetes job in a namespace"""
    name_space = _get_name_space()
    api_instance = client.BatchV1Api()
    api_response = None
    try:
        api_response = api_instance.list_namespaced_job(
            namespace=name_space,
            label_selector="purpose=tao-toolkit-job",
            watch=False,
            limit=1000
        )
    except Exception as e:
        logger.error(f"Exception thrown in list_namespace_jobs is {str(e)}")
        pass
    return api_response


def dependency_check(num_gpu=-1, accelerator=None):
    """Checks for GPU dependency"""
    if os.getenv("BACKEND", "") not in ("local-k8s", "local-docker"):
        return True
    if num_gpu == -1:
        num_gpu = int(os.getenv('NUM_GPU_PER_NODE', default='1'))
    if BACKEND == "local-docker":
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


def create_tensorboard_deployment(deployment_name, image, command, logs_image, logs_command, replicas):
    """Creates Tensorboard Deployment"""
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    logs_volume_mount = client.V1VolumeMount(
        name="tb-data",
        mount_path="/tfevents")
    capabilities = client.V1Capabilities(
        add=['SYS_PTRACE']
    )
    security_context = client.V1SecurityContext(
        capabilities=capabilities
    )

    tb_port = [
        client.V1ContainerPort(container_port=6006)
    ]
    resources = client.V1ResourceRequirements(
        limits={
            'memory': "600Mi",
            'cpu': "10m",
        },
        requests={
            'memory': '300Mi',
            'cpu': "5m"
        }
    )
    no_gpu = client.V1EnvVar(
        name="NVIDIA_VISIBLE_DEVICES",
        value="none")
    mongo_secret_env = client.V1EnvVar(
        name="MONGOSECRET",
        value=mongo_secret  # pylint: disable=E0606
    )
    mongo_operator_enabled_env = client.V1EnvVar(
        name="MONGO_OPERATOR_ENABLED",
        value=str(mongo_operator_enabled).lower()
    )
    mongo_namespace_env = client.V1EnvVar(
        name="NAMESPACE",
        value=mongo_namespace
    )
    backend_env = client.V1EnvVar(
        name="BACKEND",
        value=BACKEND,
    )
    image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
    tb_container = client.V1Container(
        name="tb-container",
        image=image,
        env=[no_gpu],
        command=["/bin/sh", "-c"],
        args=[command],
        resources=resources,
        volume_mounts=[logs_volume_mount],
        ports=tb_port,
        security_context=security_context)

    tb_logs_container = client.V1Container(
        name="tb-logs-container",
        image=logs_image,
        env=[no_gpu,
             mongo_secret_env,
             mongo_operator_enabled_env,
             mongo_namespace_env,
             backend_env],
        command=["/bin/sh", "-c"],
        resources=resources,
        args=[logs_command],
        volume_mounts=[logs_volume_mount],
        security_context=security_context,
    )

    logs_volume = client.V1Volume(
        name="tb-data",
        empty_dir=client.V1EmptyDirVolumeSource())

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(
            labels={
                "purpose": "tao-toolkit-tensorboard-job",
                "resource-type": "tensorboard",
                "app": deployment_name,  # use deployment_name as the selector name
            }
        ),
        spec=client.V1PodSpec(
            containers=[tb_container, tb_logs_container],
            volumes=[logs_volume],
            image_pull_secrets=[client.V1LocalObjectReference(name=image_pull_secret)]
        ))

    spec = client.V1DeploymentSpec(
        replicas=replicas,
        template=template,
        selector={"matchLabels": {"app": deployment_name}})

    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=deployment_name, labels={
            "resource-type": "tensorboard",
        }, owner_references=[_get_owner_reference()]),
        spec=spec)

    logger.info("Prepared deployment configs")
    try:
        api_instance.create_namespaced_deployment(
            body=deployment,
            namespace=name_space)
        logger.info("Start create deployment")
        return
    except Exception as e:
        logger.error(f"Create deployment got error: {e}")
        return


def create_tensorboard_service(tb_service_name, deploy_label):
    """Creates Tensorboard Service"""
    name_space = _get_name_space()
    tb_port = [
        client.V1ServicePort(name='tb-default-port', port=6006, target_port=6006, protocol="TCP")
    ]
    spec = client.V1ServiceSpec(ports=tb_port, selector={"app": deploy_label})
    # add annotation, it will only works in Azure, but will not affect other cloud
    annotation = {
        "service.beta.kubernetes.io/azure-load-balancer-internal": "true",
    }
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(
            name=tb_service_name,
            labels={
                "app": tb_service_name,
                "resource-type": "tensorboard",
            },
            owner_references=[_get_owner_reference()],
            annotations=annotation),
        spec=spec,
    )
    api_instance = client.CoreV1Api()

    logger.info("Prepared Tensorboard Service configs")
    try:
        api_instance.create_namespaced_service(
            body=service,
            namespace=name_space)
        logger.info("Start create Tensorboard Service")
        return
    except Exception as e:
        logger.error(f"Create Tensorboard Service got error: {e}")
        return


def create_tensorboard_ingress(tb_service_name, tb_ingress_name, tb_ingress_path):
    """Creates Tensorboard Ingress"""
    name_space = _get_name_space()
    networking_v1_api = client.NetworkingV1Api()
    ingress = client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(
            name=tb_ingress_name,
            namespace=name_space,
            labels={
                "resource-type": "tensorboard",
            },
            annotations={
                "kubernetes.io/ingress.class": "nginx",
                "nginx.ingress.kubernetes.io/client-max-body-size": "0m",
                "nginx.ingress.kubernetes.io/proxy-body-size": "0m",
                "nginx.ingress.kubernetes.io/body-size": "0m",
                "nginx.ingress.kubernetes.io/client-body-buffer-size": "50m",
                "nginx.ingress.kubernetes.io/proxy-buffer-size": "128k",
                "nginx.ingress.kubernetes.io/proxy-buffers-number": "4",
                "nginx.ingress.kubernetes.io/proxy-connect-timeout": "3600",
                "nginx.ingress.kubernetes.io/proxy-read-timeout": "3600",
                "nginx.ingress.kubernetes.io/proxy-send-timeout": "3600",
                "meta.helm.sh/release-name": release_name,
                "meta.helm.sh/release-namespace": name_space
            },
            owner_references=[_get_owner_reference()]),
        spec=client.V1IngressSpec(
            rules=[client.V1IngressRule(
                http=client.V1HTTPIngressRuleValue(
                    paths=[client.V1HTTPIngressPath(
                        path=tb_ingress_path,
                        path_type="Prefix",
                        backend=client.V1IngressBackend(
                            service=client.V1IngressServiceBackend(
                                port=client.V1ServiceBackendPort(
                                    name='tb-default-port'
                                ),
                                name=tb_service_name
                            )
                        )
                    )]
                )
            )]
        )
    )

    try:
        networking_v1_api.create_namespaced_ingress(
            body=ingress,
            namespace=name_space
        )
        logger.info("Created Tensorboard Ingress")
        return
    except Exception as e:
        logger.error(f"Create Tensorboard Ingress got error: {e}")
        return


def status_tensorboard_deployment(deployment_name, replicas=1):
    """Returns status of Tensorboard deployment

    Status definition:
    Running: The Tensorboard deployment is ready and running
    ReplicaNotReady: at least one replica of the deployment is not ready.
    NotFound: cannot find the deployment.  This status is useful to check if the deployment is stopped.
    Error: meet exceptions except not found error when check the status.
    """
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.read_namespaced_deployment_status(
            name=deployment_name,
            namespace=name_space)
        available_replicas = api_response.status.available_replicas
        if not isinstance(available_replicas, int) or available_replicas < replicas:
            return {"status": "ReplicaNotReady"}
        return {"status": "Running"}
    except ApiException as e:
        if e.status == 404:
            logger.info("Tensorboard Deployment not found.")
            # TODO: here defined a new status to find the situation that the deployment does not exists
            # This status is useful to check if the deployment is deleted or not created
            return {"status": "NotFound"}
        logger.error(f"Got other ApiException error: {e}")
        return {"status": "Error"}
    except Exception as e:
        logger.error(f"Got {type(e)} error: {e}")
        return {"status": "Error"}


def status_tb_service(tb_service_name, port=6006):
    """Returns status of TB Service

    Status definition:
    Running: The TB Service is ready and running
    NotReady: the TB Service is not ready.
    NotFound: cannot find the TB Service. This status is useful to check if the service is stopped.
    Error: meet exceptions except not found error when check the status.
    """
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()

    try:
        api_response = api_instance.read_namespaced_service(
            name=tb_service_name,
            namespace=name_space,
        )
        logger.info(f'TB Service API Response: {api_response}')
        tb_service_ip = api_response.spec.cluster_ip
        return {"status": "Running", "tb_service_ip": tb_service_ip}
    except ApiException as e:
        if e.status == 404:
            logger.info("TIS Service not found.")
            # TODO: here defined a new status, in order to find the situation that the TIS Service not exists
            # This status is useful to check if the TIS Service is deleted or not created
            return {"status": "NotFound"}
        logger.error(f"Got other ApiException error: {e}")
        return {"status": "Error"}
    except Exception as e:
        logger.error(f"Got {type(e)} error: {e}")
        return {"status": "Error"}


def delete_tensorboard_deployment(deployment_name):
    """Deletes Tensorboard Deployment"""
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.delete_namespaced_deployment(
            name=deployment_name,
            namespace=name_space,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5))
        logger.info(f"Tensorboard Deployment deleted. status='{str(api_response.status)}'")
        return
    except Exception as e:
        logger.error(f"Tensorboard Deployment failed to delete, got error: {e}")
        return


def delete_tensorboard_service(tb_service_name):
    """Deletes Tensorboard service"""
    name_space = _get_name_space()
    api_instance = client.CoreV1Api()
    try:
        api_response = api_instance.delete_namespaced_service(
            name=tb_service_name,
            namespace=name_space,
        )
        logger.info(f"Tensorboard Service deleted. status='{str(api_response.status)}'")
        return
    except Exception as e:
        logger.error(f"Tensorboard Service failed to delete, got error: {e}")
        return


def delete_tensorboard_ingress(tb_ingress_name):
    """Delete Tensorboard Ingress"""
    name_space = _get_name_space()
    networking_v1_api = client.NetworkingV1Api()
    try:
        api_response = networking_v1_api.delete_namespaced_ingress(
            name=tb_ingress_name,
            namespace=name_space
        )
        logger.info(f"Tensorboard Ingress deleted. status='{str(api_response.status)}'")
        return
    except Exception as e:
        logger.error(f"Tensorboard Ingress failed to delete, got error: {e}")
        return


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


def delete_inference_microservice_statefulset(statefulset_name):
    """Deletes a Inference Microservice StatefulSet"""
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()

    try:
        api_response = api_instance.delete_namespaced_stateful_set(
            name=statefulset_name,
            namespace=name_space,
            body=client.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5
            )
        )
        logger.info(f"Inference Microservice StatefulSet deleted. status='{str(api_response.status)}'")
        return True
    except Exception as e:
        logger.error(f"Inference Microservice StatefulSet failed to delete, got error: {e}")
        return False


def status_statefulset(statefulset_name, replicas=1, resource_type="StatefulSet"):
    """General function to get status of any StatefulSet

    Status definition:
    Running: The StatefulSet is ready and running
    ReplicaNotReady: at least one replica of the StatefulSet is not ready.
    NotFound: cannot find the StatefulSet. This status is useful to check if the StatefulSet is stopped.
    Error: meet exceptions except not found error when check the status.
    """
    name_space = _get_name_space()
    api_instance = client.AppsV1Api()
    try:
        api_response = api_instance.read_namespaced_stateful_set_status(
            name=statefulset_name,
            namespace=name_space)
        ready_replicas = api_response.status.ready_replicas or 0
        if ready_replicas < replicas:
            return {"status": "ReplicaNotReady", "replicas": {"ready": ready_replicas, "desired": replicas}}
        return {"status": "Running", "replicas": {"ready": ready_replicas, "desired": replicas}}
    except ApiException as e:
        if e.status == 404:
            logger.info(f"{resource_type} StatefulSet not found.")
            return {"status": "NotFound"}
        logger.error(f"Got other ApiException error: {e}")
        return {"status": "Error"}
    except Exception as e:
        logger.error(f"Got {type(e)} error: {e}")
        return {"status": "Error"}


def create_docker_inference_microservice(job_id, image, custom_command=None, api_port=8080, num_gpu=1):
    """Create a docker-compose inference microservice container

    Args:
        job_id: Unique identifier for the microservice
        image: Docker image to use
        custom_command: Custom command to run in the container (optional)
        api_port: Port for the microservice API
        num_gpu: Number of GPUs to allocate

    Returns:
        bool: True if container created successfully, False otherwise
    """
    try:
        docker_handler = DockerHandler(image)

        # Use custom command if provided, otherwise default to flask run
        if custom_command:
            # Split the custom command into a proper command array
            command_str = custom_command.strip()
            command = ["/bin/bash", "-c", command_str]
        else:
            command = ["/bin/bash", "-c", f"flask run --host 0.0.0.0 --port {api_port}"]

        # Start the container
        docker_handler.start_container(
            container_name=job_id,
            command=command,
            num_gpus=num_gpu
        )

        # Wait for container to be ready
        if wait_for_container(docker_handler, job_id, port=api_port):
            logger.info(f"Docker inference microservice {job_id} created successfully")
            return True

        logger.error(f"Failed to start docker inference microservice {job_id}")
        docker_handler.stop_container()
        gpu_manager.release_gpus(job_id)
        return False

    except Exception as e:
        logger.error(f"Error creating docker inference microservice {job_id}: {e}")
        return False
