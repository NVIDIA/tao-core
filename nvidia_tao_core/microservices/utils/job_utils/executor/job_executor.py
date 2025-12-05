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

"""Job executor for regular Kubernetes Job operations"""
import os
import time
import traceback
from kubernetes import client, config

# DockerHandler import moved to function level to avoid circular imports
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    BACKEND,
    get_handler_job_metadata,
    internal_job_status_update,
    write_job_metadata,
    update_job_message,
    get_job_specs
)
from nvidia_tao_core.microservices.utils.handler_utils import send_microservice_request
from nvidia_tao_core.microservices.utils.nvcf_utils import (
    create_function,
    deploy_function,
    get_function,
    delete_function_version,
    add_authorized_party,
    create_microservice_job_on_nvcf,
    get_nvcf_microservices_job_status
)

if os.getenv("BACKEND"):
    from nvidia_tao_core.microservices.utils.mongo_utils import (
        mongo_secret,
        mongo_operator_enabled,
        mongo_namespace
    )

from .base_executor import BaseExecutor
from nvidia_tao_core.microservices.utils.executor_utils import override_k8_status


class JobExecutor(BaseExecutor):
    """Handles regular Kubernetes Job operations"""

    def create_job(self, org_name, job_name, image, command, num_gpu=-1, num_nodes=1, accelerator=None,
                   docker_env_vars=None, port=False, nv_job_metadata=None, automl_brain=False,
                   automl_exp_job=False, local_cluster=False):
        """Creates a kubernetes job"""
        name_space = self.get_namespace()
        host_base_url = os.getenv("HOSTBASEURL", "no_url")
        if host_base_url == "no_url":
            raise ValueError(
                f"Base URL not set in values yaml. Please set it as "
                f"http(s)://<ip_address>:{self.release_name}-ingress-nginx-controller service's port number>"
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
                    self.logger.info(f"Function created successfully for job {job_name}")
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
                        self.logger.info(f"Function deployment initiated successfully for job {job_name}")
                    else:
                        internal_job_status_update(
                            job_name,
                            automl=automl_exp_job,
                            automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                            message="NVCF function could not be deployed"
                        )
                        self.logger.error(f"Function deployment request failed for job {job_name}")
                        self.logger.error(f"Deployment response {deploy_response.text}")
                        raise ValueError(f"Function deployment request failed for job {job_name}")
                else:
                    internal_job_status_update(
                        job_name,
                        automl=automl_exp_job,
                        automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                        message="NVCF function couldn't be created, retry job again"
                    )
                    self.logger.error(f"Function creation request failed for job {job_name}")
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
            from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler
            docker_handler = DockerHandler(image)
            docker_env_vars.update({
                "BACKEND": BACKEND,
                "HOST_PLATFORM": "local-docker",
                "MONGOSECRET": mongo_secret,
                "DOCKER_HOST": os.getenv("DOCKER_HOST", default="unix:///var/run/docker.sock"),
                "DOCKER_NETWORK": os.getenv("DOCKER_NETWORK", default="tao_default")
            })
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
            available_gpus = self.get_available_local_k8s_gpus()
            gpu_to_be_run_on = None
            if available_gpus:
                gpu_to_be_run_on = available_gpus.get(accelerator, "")
            node_selector = {'accelerator': gpu_to_be_run_on}

        image_pull_secret = os.getenv('IMAGEPULLSECRET', default='imagepullsecret')
        name_space = self.get_namespace()
        api_instance = client.BatchV1Api()

        volume_mounts = []
        if local_cluster:
            if os.getenv("INGRESSENABLED", "false") == "false":
                in_cluster_ip, cluster_port = self.get_cluster_ip()
            else:
                service_name = f"{self.release_name}-ingress-nginx-controller"
                in_cluster_ip = self.get_service_in_cluster_ip(
                    service_name, namespace=name_space
                )
                cluster_port = 80
            # change the host_base_url to the in-cluster ip
            in_cluster_url = f"http://{in_cluster_ip}:{cluster_port}" if nv_job_metadata is None else None
            if "TAO_API_SERVER" in docker_env_vars:
                docker_env_vars["TAO_API_SERVER"] = docker_env_vars[
                    "TAO_API_SERVER"
                ].replace(host_base_url, in_cluster_url)
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
            value=str(num_gpu))
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

        # Create metadata with Helm annotations for proper lifecycle management
        # This ensures the Job is deleted when helm delete is run
        job_metadata = {
            "name": job_name,
            "owner_references": [self.get_owner_reference()]
        }

        job = client.V1Job(
            api_version="batch/v1",
            kind="Job",
            metadata=client.V1ObjectMeta(**job_metadata),
            spec=spec)

        try:
            api_instance.create_namespaced_job(
                body=job,
                namespace=name_space)
            return
        except Exception as e:
            self.logger.error(f"Exception thrown in executor create is {str(e)}")
            self.logger.error(traceback.format_exc())
            return

    def delete_job(self, job_name, use_ngc=True):
        """Deletes a job (K8s Job, Docker container, or NVCF function)"""
        # Handle docker-compose backend
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
            return

        # Handle NVCF backend
        if BACKEND == "NVCF" and use_ngc:
            self._delete_nvcf_function(job_name)
            return

        # Handle local-k8s backend
        name_space = self.get_namespace()
        if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
            config.load_kube_config()
        else:
            config.load_incluster_config()

        api_instance = client.BatchV1Api()
        from .service_executor import ServiceExecutor
        service_executor = ServiceExecutor()
        try:
            service_executor.delete_service(job_id=job_name, service_type="flask")
            api_response = api_instance.delete_namespaced_job(
                name=job_name,
                namespace=name_space,
                body=client.V1DeleteOptions(
                    propagation_policy='Foreground',
                    grace_period_seconds=5))
            self.logger.info(f"Job deleted. status='{str(api_response.status)}'")
            return
        except Exception as e:
            self.logger.error(f"Exception caught in delete_job {str(e)}")
            self.logger.error("Job failed to delete.")
            return

    def list_namespace_jobs(self):
        """List kubernetes job in a namespace"""
        name_space = self.get_namespace()
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
            self.logger.error(f"Exception thrown in list_namespace_jobs is {str(e)}")
            pass
        return api_response

    def _delete_nvcf_function(self, job_name):
        """Deletes an NVCF Function"""
        job_metadata = get_handler_job_metadata(job_name)
        org_name = job_metadata.get("org_name")
        nv_job_metadata = job_metadata.get("backend_details", {}).get("nvcf_metadata", {})
        team_name = nv_job_metadata["teamName"]
        ngc_key = nv_job_metadata["TAO_USER_KEY"]
        deployment_string = nv_job_metadata.get("deployment_string", "")
        if deployment_string.find(":") == -1:
            self.logger.warning(f"Deployment not active yet {job_name}")
            return
        function_id, version_id = deployment_string.split(":")
        delete_function_version(org_name, team_name, function_id, version_id, ngc_key)

    def override_k8_status(self, job_name, k8_status):
        """Override kubernetes job status with toolkit status"""
        return override_k8_status(job_name, k8_status)

    def get_job_status(self, org_name, handler_id, job_name, handler_kind, use_ngc=True, network="",
                       action="", automl_exp_job=False, docker_env_vars={},
                       authorized_party_nca_id="", automl_experiment_id="0", skip_service_wait=False):
        """Returns status of kubernetes job

        Args:
            skip_service_wait: If True, skip waiting for service (useful for brain jobs or after deletion)
        """
        if BACKEND == "local-k8s":
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
                            self.logger.info("NVCF function is active, creating microservice job on NVCF")
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
                                self.logger.info(
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
                            self.logger.error(f"Get function deployment status for job {job_name} returned error")
                            internal_job_status_update(
                                job_name,
                                automl=automl_exp_job,
                                automl_experiment_number=nv_job_metadata.get("AUTOML_EXPERIMENT_NUMBER", "0"),
                                message="NVCF function metadata has ERROR status"
                            )
                            return "Error"

                override_status = self.override_k8_status(job_name, job_status)
                job_status = get_nvcf_microservices_job_status(
                    job_metadata,
                    status=override_status,
                    docker_env_vars=docker_env_vars
                )
                if override_status and override_status != job_status:
                    job_status = override_status
                    self.logger.warning(
                        f"job metadata status is {job_status}, Toolkit Status is {override_status}, so overwriting")
                    self.logger.warning(f"Microservices job status via NVCF is {job_status}")
                return job_status
            except Exception as e:
                self.logger.error(f"Exception caught for {job_name} {e}")
                self.logger.error(traceback.format_exc())
                return "Error"

        # For local cluster jobs
        specs = get_job_specs(job_name, automl=automl_exp_job, automl_experiment_id=automl_experiment_id)
        if not specs:
            self.logger.error(f"Unable to retrieve specs for job {job_name}")
            return "Error"

        if BACKEND == "local-docker":
            from nvidia_tao_core.microservices.handlers.docker_handler import DockerHandler
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
                    if status == "Error":
                        self.logger.error(f"Error when sending microservice request {response.text}")
                    return status
                self.logger.error(f"Error when sending microservice request {response.text}")
            return "Error"

        from .service_executor import ServiceExecutor
        service_executor = ServiceExecutor()

        # Skip service wait for brain jobs or when service was just deleted
        if skip_service_wait:
            self.logger.info(f"Skipping service wait for job {job_name} (skip_service_wait=True)")
            # Check job metadata directly for status
            job_metadata = get_handler_job_metadata(job_name)
            if job_metadata:
                status = job_metadata.get("status", "Error")
                self.logger.info(f"Job {job_name} status from metadata: {status}")
                return status
            self.logger.warning(f"Job {job_name} metadata not found, returning Error")
            return "Error"

        service_status = service_executor.wait_for_service(job_name)
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

    def wait_for_job_termination(self, job_id, timeout_seconds=120):
        """Wait for a Job and its containers/pods to be fully terminated.

        Args:
            job_id: The job ID to wait for termination
            timeout_seconds: Maximum time to wait (default 120 seconds)

        Returns:
            bool: True if job terminated, False if timeout
        """
        poll_interval = 5
        max_polls = timeout_seconds // poll_interval
        poll_count = 0
        job_terminated = False

        self.logger.debug(
            f"Waiting for job termination: job_id={job_id}, backend={BACKEND}, timeout={timeout_seconds}s"
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
                            job_terminated = True
                            break
                        self.logger.debug(
                            f"Docker container still running: job_id={job_id}, status={container_status}, "
                            f"poll={poll_count}/{max_polls}"
                        )
                    except Exception as e:
                        self.logger.debug(f"Docker container no longer exists: job_id={job_id}, error={str(e)}")
                        job_terminated = True
                        break
                else:
                    # Container not found
                    self.logger.debug(f"Docker container not found: job_id={job_id}")
                    job_terminated = True
                    break

                time.sleep(poll_interval)
                poll_count += 1

        # Handle NVCF backend
        elif BACKEND == "NVCF":
            # For NVCF, check job metadata status
            while poll_count < max_polls:
                job_metadata = get_handler_job_metadata(job_id)
                if not job_metadata:
                    self.logger.debug(f"NVCF job metadata not found: job_id={job_id}")
                    job_terminated = True
                    break

                job_status = job_metadata.get("status", "")
                if job_status in ("Canceled", "Done", "Error", "Paused"):
                    self.logger.debug(f"NVCF job terminated: job_id={job_id}, status={job_status}")
                    job_terminated = True
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

            batch_v1 = client.BatchV1Api()
            core_v1 = client.CoreV1Api()
            namespace = os.getenv("NAMESPACE", "default")

            while poll_count < max_polls:
                try:
                    # Check if K8s Job still exists
                    batch_v1.read_namespaced_job(job_id, namespace)
                    self.logger.debug(
                        f"K8s Job still exists, waiting for deletion: job_id={job_id}, "
                        f"poll={poll_count}/{max_polls}"
                    )
                except client.exceptions.ApiException as e:
                    if e.status == 404:
                        self.logger.debug(f"K8s Job deleted: job_id={job_id}")
                        # Also check if pods are gone
                        try:
                            pods = core_v1.list_namespaced_pod(
                                namespace,
                                label_selector=f"job-name={job_id}"
                            )
                            if len(pods.items) == 0:
                                self.logger.debug(f"All K8s job pods terminated: job_id={job_id}")
                                job_terminated = True
                                break
                            # Still waiting for pods to terminate
                            self.logger.debug(
                                f"K8s job pods still terminating: job_id={job_id}, count={len(pods.items)}"
                            )
                        except Exception as pod_err:
                            self.logger.warning(f"Error checking pods: {str(pod_err)}")
                            job_terminated = True  # Assume terminated if we can't check
                            break
                    self.logger.error(f"Error checking K8s job: {str(e)}")
                    break

                time.sleep(poll_interval)
                poll_count += 1
        else:
            self.logger.warning(f"Unknown BACKEND: {BACKEND}, assuming job terminated")
            return True

        if not job_terminated and poll_count >= max_polls:
            self.logger.warning(f"Timeout waiting for job termination: job_id={job_id}, backend={BACKEND}")
            return False

        self.logger.debug(f"Job termination confirmed: job_id={job_id}, backend={BACKEND}")
        return True
