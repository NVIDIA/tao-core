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

"""Microservice executor for microservice operations"""
import os
import time
import uuid
import traceback
import functools
from kubernetes import client

from nvidia_tao_core.microservices.constants import NETWORK_CONTAINER_MAPPING
# DockerHandler import moved to function level to avoid circular imports
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    BACKEND,
    get_handler_job_metadata,
    internal_job_status_update,
)
from nvidia_tao_core.microservices.utils.handler_utils import (
    get_statefulset_service_name,
    send_microservice_request
)

from .base_executor import BaseExecutor


def retry_docker_microservice(max_retries=3, delay_seconds=5):
    """Retry decorator for docker microservice creation.

    Retries when the wrapped function returns False or raises an Exception.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Assumes first arg is `self` (MicroserviceExecutor instance)
            self = args[0] if args else None
            logger = getattr(self, "logger", None)

            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Only retry on explicit False; True/None/etc. are treated as final
                    if result is not False:
                        return result

                    if logger:
                        logger.warning(
                            f"[DOCKER_MICROSERVICE_RETRY] {func.__name__} returned False "
                            f"(attempt {attempt}/{max_retries}), will retry after {delay_seconds}s"
                        )
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if logger:
                        logger.error(
                            f"[DOCKER_MICROSERVICE_RETRY] {func.__name__} raised {type(exc).__name__}: {exc} "
                            f"(attempt {attempt}/{max_retries})"
                        )
                        logger.error(traceback.format_exc())

                if attempt < max_retries:
                    time.sleep(delay_seconds)

            # Exhausted retries
            if last_exc is not None:
                # Re-raise the last exception after all retries
                raise last_exc

            # All attempts returned False
            return False

        return wrapper

    return decorator


class MicroserviceExecutor(BaseExecutor):
    """Handles microservice operations"""

    def create_microservice_pod(self, job_name, image, num_gpu=-1, accelerator=None):
        """Create pod to invoke microservices"""
        from .service_executor import ServiceExecutor
        service_executor = ServiceExecutor()
        service_executor.create_flask_service(job_name)
        name_space = self.get_namespace()
        api_instance = client.BatchV1Api()

        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')

        node_selector = None
        if accelerator:
            available_gpus = self.get_available_local_k8s_gpus()
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

            self.logger.info(f"Pod {pod_name} is running. Waiting for it to be ready")

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

            self.logger.info(f"Pod {pod_name} is ready with IP {pod_ip}.")
            time.sleep(10)
        except Exception as e:
            self.logger.error(f"Exception thrown in create_microservice_pod is {str(e)}")
            self.logger.error(traceback.format_exc())

    def wait_for_container(self, container_handler, job_id, port=8000):
        """Wait for the container to be ready."""
        start_time = time.time()
        while time.time() - start_time < 300:
            metadata_status = get_handler_job_metadata(job_id).get("status")
            if metadata_status in ("Canceled", "Canceling", "Paused", "Pausing"):
                return metadata_status
            if container_handler.check_container_health(port=port):
                self.logger.info(f"Container '{job_id}' is ready.")
                return "Running"
            self.logger.info(f"Waiting for container '{job_id}' to be ready...")
            time.sleep(10)
        self.logger.error("Timed out waiting for container to be ready.")
        return "Error"

    def create_microservice_and_send_request(
            self, api_endpoint, network, action, cloud_metadata={}, specs={},
            microservice_pod_id="", nvcf_helm="", num_gpu=-1,
            microservice_container="", org_name="", handler_id="",
            handler_kind="", accelerator=None, docker_env_vars={}, num_nodes=1):
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
                if self.create_docker_inference_microservice(
                    job_id=microservice_pod_id,
                    image=microservice_container,
                    custom_command=f"flask run --host 0.0.0.0 --port {port}",
                    api_port=port,
                    num_gpu=num_gpu
                ):
                    from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler
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
                        self.logger.error(f"Error when sending microservice request {response.text}")
                        internal_job_status_update(
                            microservice_pod_id,
                            message=f"Error when sending microservice request {response.text}"
                        )
                        docker_handler.stop_container()
                        # GPUs will be automatically reclaimed by lazy garbage collection on next assignment
                        return None
                    if api_endpoint != "post_action":
                        docker_handler.stop_container()
                        # GPUs will be automatically reclaimed by lazy garbage collection on next assignment
                    return response
                internal_job_status_update(
                    microservice_pod_id,
                    message=f"Error when creating microservice pod {microservice_pod_id}"
                )
                return None

            if BACKEND == "local-k8s":
                service_name = get_statefulset_service_name(microservice_pod_id)
                from .statefulset_executor import StatefulSetExecutor
                statefulset_executor = StatefulSetExecutor()
                statefulset_executor.create_statefulset(
                    microservice_pod_id,
                    num_gpu,
                    num_nodes,
                    microservice_container,
                    accelerator=accelerator
                )
                from .service_executor import ServiceExecutor
                service_executor = ServiceExecutor()
                if service_executor.wait_for_service(microservice_pod_id, service_name=service_name):
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
                        self.logger.error(f"Error when sending microservice request {response.text}")
                        internal_job_status_update(
                            microservice_pod_id,
                            message=f"Error when sending microservice request {response.text}"
                        )
                        statefulset_executor.delete_statefulset(microservice_pod_id, use_ngc=False)
                        return None
                    if api_endpoint != "post_action":
                        statefulset_executor.delete_statefulset(microservice_pod_id, use_ngc=False)
                    return response
            return None
        except Exception as e:
            self.logger.error(f"Exception thrown in create_microservice_and_send_request is {str(e)}")
            self.logger.error("Exception in create ms pod and send request")
            self.logger.error(traceback.format_exc())
            internal_job_status_update(
                microservice_pod_id,
                message=f"Error when creating microservice pod {microservice_pod_id}"
            )
            from .statefulset_executor import StatefulSetExecutor
            statefulset_executor = StatefulSetExecutor()
            statefulset_executor.delete_statefulset(microservice_pod_id, use_ngc=False)
            return None

    @retry_docker_microservice(max_retries=3, delay_seconds=5)
    def create_docker_inference_microservice(
        self,
        job_id,
        image,
        custom_command=None,
        api_port=8080,
        num_gpu=1,
    ):
        """Create a docker-compose inference microservice container"""
        try:
            from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler
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
            if self.wait_for_container(docker_handler, job_id, port=api_port) == "Running":
                self.logger.info(f"Docker inference microservice {job_id} created successfully")
                return True

            self.logger.error(f"Failed to start docker inference microservice {job_id}")
            docker_handler.stop_container()
            # GPUs will be automatically reclaimed by lazy garbage collection on next assignment
            return False

        except Exception as e:
            self.logger.error(f"Error creating docker inference microservice {job_id}: {e}")
            self.logger.error(traceback.format_exc())
            return False
