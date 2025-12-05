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

"""StatefulSet executor for StatefulSet operations"""
import os
import time
import traceback
import inspect
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# DockerHandler import moved to function level to avoid circular imports
from nvidia_tao_core.microservices.utils.stateless_handler_utils import BACKEND
from nvidia_tao_core.microservices.utils.handler_utils import (
    get_statefulset_name,
    get_statefulset_service_name
)

from .base_executor import BaseExecutor


class StatefulSetExecutor(BaseExecutor):
    """Handles StatefulSet operations"""

    def wait_for_statefulset_ready(self, statefulset_name, name_space):
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
                    self.logger.info(f"Statefulset {statefulset_name} is ready with {ready_replicas} replicas")
                    stateful_set_ready = True
                else:
                    self.logger.info(
                        f"Statefulset {statefulset_name} pending with "
                        f"{ready_replicas}/{desired_replicas} ready"
                    )
                    time.sleep(10)
            else:
                self.logger.info(f"{statefulset_name} not found.")
                time.sleep(10)

    def create_statefulset(
            self, job_id, num_gpu_per_node, num_nodes, image, api_port=8000,
            master_port=29500, accelerator=None, statefulset_type="multinode",
            custom_command=None, custom_env_vars=None, custom_ports=None,
            org_name=None, experiment_id=None, is_long_lived=False):
        """Create statefulset with flexible configuration for different types"""
        try:
            # Handle docker-compose backend for inference microservices
            if BACKEND == "local-docker" and statefulset_type == "inference_microservice":
                # Set default api_port for inference microservices if not explicitly provided
                if api_port == 8000:
                    api_port = 8080  # Default port for inference microservices

                # Create docker container for inference microservice
                from .microservice_executor import MicroserviceExecutor
                microservice_executor = MicroserviceExecutor()
                return microservice_executor.create_docker_inference_microservice(
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
            from .service_executor import ServiceExecutor
            service_executor = ServiceExecutor()
            if statefulset_type == "inference_microservice":
                # Use default inference microservice ports if no custom ports specified
                service_ports = custom_ports or [(8080, 8080), (8081, 8081)]
            else:
                # Use api_port for multinode services
                service_ports = custom_ports or [(api_port, api_port)]

            service_executor.create_statefulset_service(job_id, statefulset_type=statefulset_type, ports=service_ports)

            name_space = self.get_namespace()
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
                release_name_env_var = client.V1EnvVar(name="RELEASE_NAME", value=self.release_name)
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
                available_gpus = self.get_available_local_k8s_gpus()
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
                metadata_spec["owner_references"] = [self.get_owner_reference()]

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
            self.wait_for_statefulset_ready(statefulset_name, name_space)
            return True
        except Exception as e:
            self.logger.error(f"Exception thrown in create_statefulset is {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def delete_statefulset(self, job_name, use_ngc=True, resource_type="multinode"):
        """Deletes a Job or StatefulSet"""
        # Enhanced logging with call stack to understand WHY deletion was triggered
        caller_frame = inspect.currentframe().f_back
        caller_info = inspect.getframeinfo(caller_frame) if caller_frame else None
        caller_location = f"{caller_info.filename}:{caller_info.lineno}" if caller_info else "unknown"

        self.logger.debug(
            f"{'-' * 80}\n"
            f"DELETE_STATEFULSET CALLED\n"
            f"Job Name: {job_name}\n"
            f"Backend: {BACKEND}\n"
            f"Use NGC: {use_ngc}\n"
            f"Resource Type: {resource_type}\n"
            f"Called From: {caller_location}\n"
            f"Call Stack (top 5):\n"
        )

        # Log call stack to understand the termination trigger
        stack_lines = traceback.format_stack(limit=6)
        for line in stack_lines[-5:]:  # Last 5 frames
            self.logger.warning(f"  {line.strip()}")

        self.logger.debug(f"{'-' * 80}")

        if BACKEND == "local-docker":
            from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler
            self.logger.debug(f"Docker backend: Looking for container {job_name}")
            docker_handler = DockerHandler.get_handler_for_container(job_name)
            if docker_handler:
                self.logger.debug(f"Docker container found for {job_name}, stopping it now")
                docker_handler.stop_container()
                self.logger.info(f"Successfully stopped Docker container for {job_name}")
            else:
                self.logger.warning(f"Docker container not found for job {job_name}")
            return True

        # Handle NVCF backend
        if BACKEND == "NVCF" and use_ngc:
            from .job_executor import JobExecutor
            job_executor = JobExecutor()
            job_executor._delete_nvcf_function(job_name)
            return True

        # Handle local-k8s backend
        name_space = self.get_namespace()
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()

        api_instance = client.AppsV1Api()
        from .service_executor import ServiceExecutor
        service_executor = ServiceExecutor()
        try:
            # Configure naming and service type based on resource type
            if resource_type == "inference_microservice":
                stateful_set_name = f"ims-{job_name}"
                service_type = "inference_microservice"
            else:
                stateful_set_name = get_statefulset_name(job_name)
                service_type = "statefulset"

            # Delete service first, then statefulset
            service_executor.delete_service(job_id=job_name, service_type=service_type)
            stateful_set = api_instance.read_namespaced_stateful_set(
                name=stateful_set_name,
                namespace=name_space
            )
            if not stateful_set:
                self.logger.info(f"Statefulset {stateful_set_name} not found in namespace {name_space}")
                return True  # Deletion goal achieved - resource doesn't exist
            api_response = api_instance.delete_namespaced_stateful_set(
                name=stateful_set_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5
                )
            )
            self.logger.info(f"Statefulset deleted. status='{str(api_response.status)}'")
            return True
        except ApiException as e:
            if e.status == 404:
                # Resource not found - deletion goal already achieved (likely already deleted by another process)
                self.logger.info(
                    f"Statefulset {stateful_set_name} not found (404) - already deleted. Deletion successful."
                )
                return True
            self.logger.error(f"ApiException caught in delete_statefulset {str(e)}")
            self.logger.error("Statefulset failed to delete.")
            return False
        except Exception as e:
            self.logger.error(f"Exception caught in delete_statefulset {str(e)}")
            self.logger.error("Statefulset failed to delete.")
            return False

    def get_statefulset_status(self, statefulset_name, replicas=1, resource_type="StatefulSet"):
        """General function to get status of any StatefulSet"""
        name_space = self.get_namespace()
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
                self.logger.info(f"{resource_type} StatefulSet not found.")
                return {"status": "NotFound"}
            self.logger.error(f"Got other ApiException error: {e}")
            return {"status": "Error"}
        except Exception as e:
            self.logger.error(f"Got {type(e)} error: {e}")
            return {"status": "Error"}

    def wait_for_statefulset_termination(self, job_id, timeout_seconds=120):
        """Wait for a StatefulSet and its containers/pods to be fully terminated.

        Args:
            job_id: The job ID to wait for termination
            timeout_seconds: Maximum time to wait (default 120 seconds)

        Returns:
            bool: True if StatefulSet terminated, False if timeout
        """
        poll_interval = 5
        max_polls = timeout_seconds // poll_interval
        poll_count = 0
        sts_terminated = False

        self.logger.debug(
            f"Waiting for StatefulSet termination: job_id={job_id}, backend={BACKEND}, "
            f"timeout={timeout_seconds}s"
        )

        # Handle docker-compose backend
        if BACKEND == "local-docker":
            from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler

            while poll_count < max_polls:
                docker_handler = DockerHandler.get_handler_for_container(job_id)
                if docker_handler and docker_handler._container:
                    # Reload container status
                    try:
                        docker_handler._container.reload()
                        container_status = docker_handler._container.status
                        if container_status in ("exited", "dead", "removing", "removed"):
                            self.logger.debug(
                                f"Docker container terminated: job_id={job_id}, status={container_status}"
                            )
                            sts_terminated = True
                            break
                        self.logger.debug(
                            f"Docker container still running: job_id={job_id}, status={container_status}, "
                            f"poll={poll_count}/{max_polls}"
                        )
                    except Exception as e:
                        self.logger.debug(f"Docker container no longer exists: job_id={job_id}, error={str(e)}")
                        sts_terminated = True
                        break
                else:
                    # Container not found
                    self.logger.debug(f"Docker container not found: job_id={job_id}")
                    sts_terminated = True
                    break

                time.sleep(poll_interval)
                poll_count += 1

        # Handle NVCF backend
        elif BACKEND == "NVCF":
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_job_metadata
            # For NVCF, check job metadata status
            while poll_count < max_polls:
                job_metadata = get_handler_job_metadata(job_id)
                if not job_metadata:
                    self.logger.debug(f"NVCF job metadata not found: job_id={job_id}")
                    sts_terminated = True
                    break

                job_status = job_metadata.get("status", "")
                if job_status in ("Canceled", "Done", "Error", "Paused"):
                    self.logger.debug(f"NVCF job terminated: job_id={job_id}, status={job_status}")
                    sts_terminated = True
                    break

                self.logger.debug(
                    f"NVCF job still active: job_id={job_id}, status={job_status}, "
                    f"poll={poll_count}/{max_polls}"
                )
                time.sleep(poll_interval)
                poll_count += 1

        # Handle local-k8s backend
        elif BACKEND == "local-k8s":
            if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                config.load_kube_config()
            else:
                config.load_incluster_config()

            apps_v1 = client.AppsV1Api()
            core_v1 = client.CoreV1Api()
            namespace = os.getenv("NAMESPACE", "default")
            statefulset_name = f"tao-api-sts-{job_id}"

            while poll_count < max_polls:
                try:
                    # Check if K8s StatefulSet still exists
                    apps_v1.read_namespaced_stateful_set(statefulset_name, namespace)
                    self.logger.debug(
                        f"K8s StatefulSet still exists, waiting for deletion: job_id={job_id}, "
                        f"poll={poll_count}/{max_polls}"
                    )
                except client.exceptions.ApiException as e:
                    if e.status == 404:
                        self.logger.debug(f"K8s StatefulSet deleted: job_id={job_id}")
                        # Also check if pods are gone
                        try:
                            pods = core_v1.list_namespaced_pod(
                                namespace,
                                label_selector=f"job={job_id}"
                            )
                            if len(pods.items) == 0:
                                self.logger.debug(f"All K8s StatefulSet pods terminated: job_id={job_id}")
                                sts_terminated = True
                                break
                            self.logger.debug(
                                f"K8s StatefulSet pods still terminating: job_id={job_id}, "
                                f"count={len(pods.items)}"
                            )
                        except Exception as pod_err:
                            self.logger.warning(f"Error checking pods: {str(pod_err)}")
                            sts_terminated = True  # Assume terminated if we can't check
                            break
                    self.logger.error(f"Error checking K8s StatefulSet: {str(e)}")
                    break

                time.sleep(poll_interval)
                poll_count += 1
        else:
            self.logger.warning(f"Unknown BACKEND: {BACKEND}, assuming StatefulSet terminated")
            return True

        if not sts_terminated and poll_count >= max_polls:
            self.logger.warning(f"Timeout waiting for StatefulSet termination: job_id={job_id}, backend={BACKEND}")
            return False

        self.logger.info(f"StatefulSet termination confirmed: job_id={job_id}, backend={BACKEND}")
        return True
