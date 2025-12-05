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

"""Inference Microservice handler using StatefulSets for long-lived inference"""
import logging
import requests
from datetime import datetime, timezone
from typing import Dict, Any
import os

from .docker_images import DOCKER_IMAGE_MAPPER
from nvidia_tao_core.microservices.utils.handler_utils import (
    Code, add_workspace_to_cloud_metadata, get_model_results_path
)
from nvidia_tao_core.microservices.utils.job_utils.executor import (
    ServiceExecutor,
    StatefulSetExecutor
)
from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_metadata
from nvidia_tao_core.microservices.utils.core_utils import read_network_config


# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class InferenceMicroserviceHandler:
    """Handler class for long-lived Inference Microservice jobs using StatefulSets

    This handler manages:
    - Starting long-lived Inference Microservice using StatefulSets for persistence
    - Running both HTTP server (for file operations) and container_handler.py (for TAO jobs)
    - Managing service lifecycle (start, update, stop)
    - Following standard TAO container and job processing patterns
    """

    @staticmethod
    def start_inference_microservice(org_name: str, experiment_id: str, job_id: str,
                                     job_config: Dict[str, Any],
                                     replicas: int = 1, api_port: int = 8080) -> Code:
        """Starts a long-lived Inference Microservice using StatefulSet

        The network architecture is automatically determined from the experiment metadata.
        """
        from nvidia_tao_core.microservices.utils.stateless_handler_utils import write_job_metadata
        from nvidia_tao_core.microservices.utils import get_admin_key
        from nvidia_tao_core.microservices.utils.job_utils.executor.base_executor import get_cluster_ip

        logger.info("Starting Inference Microservice %s for experiment %s", job_id, experiment_id)

        # Get experiment metadata to determine network architecture
        experiment_metadata = get_handler_metadata(experiment_id, kind="experiments")
        network_arch = experiment_metadata.get("network_arch", "vila")  # Default to vila if not found
        logger.info("Network architecture from experiment metadata: %s", network_arch)

        folder_path_function = "parent_model"
        # Read network config to get docker image name
        try:
            network_config = read_network_config(network_arch.lower())

            if network_config:
                image_key = network_config.get('api_params', {}).get('image', network_arch.upper())
                image = DOCKER_IMAGE_MAPPER.get(image_key, "nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt")
                logger.info("Using Docker image: %s (from network_arch: %s)", image, network_arch)
                folder_path_function = network_config.get('spec_params', {}).get('inference', {}).get('model_path', "")
            else:
                # Fallback if network config is empty
                image = DOCKER_IMAGE_MAPPER.get(network_arch.upper(), "nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt")
                logger.info("Using fallback Docker image: %s", image)
        except Exception as e:
            logger.warning("Could not read network config for %s: %s. Using default image.", network_arch, str(e))
            image = DOCKER_IMAGE_MAPPER.get(network_arch.upper(), "nvcr.io/nvidia/tao/tao-toolkit:6.0.0-pyt")
            logger.info("Using fallback Docker image: %s", image)

        # Build command for Inference Microservice integrated into TAO container
        parent_id = job_config.get("parent_job_id", job_config.get("parent_id", ""))

        # Check if parent job is in Done state
        if parent_id:
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                get_handler_job_metadata
            )
            parent_job_metadata = get_handler_job_metadata(parent_id)
            if parent_job_metadata:
                parent_status = parent_job_metadata.get("status", "")
                if parent_status != "Done":
                    error_msg = (
                        f"Cannot start inference microservice: parent job {parent_id} "
                        f"is not in 'Done' state (current status: {parent_status})"
                    )
                    logger.error(error_msg)
                    return Code(400, {}, error_msg)
            else:
                error_msg = (
                    f"Cannot start inference microservice: parent job {parent_id} not found"
                )
                logger.error(error_msg)
                return Code(400, {}, error_msg)

        folder_path = "folder" in folder_path_function
        model_path = job_config.get("model_path", get_model_results_path(experiment_metadata, parent_id, folder_path))
        logger.info("Using model path: %s", model_path)
        if not model_path:
            return Code(400, {}, "Model path is required for Inference Microservice")

        # cli_args = convert_dict_to_cli_args(job_config)
        # cli_args = " ".join(cli_args)
        # logger.info("Using CLI args: %s", cli_args)

        docker_env_vars = experiment_metadata.get("docker_env_vars", {})
        docker_env_vars["TAO_EXECUTION_BACKEND"] = os.getenv("BACKEND", "local-k8s")
        docker_env_vars["TAO_API_JOB_ID"] = job_id

        # Set up environment variables for status callbacks (auto-deletion)
        docker_env_vars["CLOUD_BASED"] = "True"
        host_base_url = os.getenv("HOSTBASEURL", "no_url")
        if os.getenv("BACKEND", "local-k8s") == "local-k8s":
            cluster_ip, cluster_port = get_cluster_ip()
            if cluster_ip and cluster_port:
                host_base_url = f"http://{cluster_ip}:{cluster_port}"
        status_url = f"{host_base_url}/api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}"
        docker_env_vars["TAO_LOGGING_SERVER_URL"] = status_url
        docker_env_vars["TAO_ADMIN_KEY"] = get_admin_key()

        workspace_id = experiment_metadata.get("workspace", "")
        workspace_metadata = get_handler_metadata(workspace_id, kind="workspaces")
        cloud_metadata = {}
        add_workspace_to_cloud_metadata(workspace_metadata, cloud_metadata)

        cloud_type = workspace_metadata.get('cloud_type', '')
        cloud_details = workspace_metadata.get('cloud_specific_details', {})
        bucket_name = cloud_details.get('cloud_bucket_name', '')

        specs = {
            "model_path": model_path,
            "results_dir": f"{cloud_type}://{bucket_name}/results/{job_id}",
        }

        # Propagate additional parameters from job_config to specs
        # These include enable_lora, base_model_path, and any other user-provided configs
        for key, value in job_config.items():
            if key not in ["parent_id", "parent_job_id", "model_path"]:
                specs[key] = value
                logger.info(f"Propagating parameter to specs: {key} = {value}")

        job_metadata = {
            "job_id": job_id,
            "specs": specs,
            "cloud_metadata": cloud_metadata,
            "neural_network_name": network_arch,
        }

        # Clean TAO-compliant StatefulSet setup: Pure container_handler.py approach
        run_command = f"""
umask 0 &&


{network_arch}-inference-microservice --job "{str(job_metadata)}" --docker_env_vars "{str(docker_env_vars)}"
        """
        logger.info("Using run command: %s", run_command)

        try:
            # Create long-lived inference service StatefulSet
            # IMPORTANT: This overrides the default container entrypoint (e.g., "flask run")
            # with our custom command that starts the persistent model server + container_handler.py
            statefulset_executor = StatefulSetExecutor()
            success = statefulset_executor.create_statefulset(
                job_id=job_id,
                num_gpu_per_node=1,
                num_nodes=replicas,
                image=image,
                api_port=api_port,
                statefulset_type="inference_microservice",
                custom_command=run_command,
                org_name=org_name,
                experiment_id=experiment_id,
                is_long_lived=True
            )

            if not success:
                return Code(500, {}, "Failed to create Inference Microservice StatefulSet")

            # Wait for service to be ready before returning success
            service_id = f"ims-svc-{job_id}"
            logger.info("Waiting for Inference Microservice service %s to be ready", service_id)

            if os.getenv("BACKEND") == "local-k8s":
                service_executor = ServiceExecutor()
                service_status = service_executor.wait_for_service(job_id, service_name=service_id)
                if service_status != "Running":
                    logger.error("Inference Microservice service failed to become ready. Status: %s", service_status)
                    return Code(500, {}, f"Inference Microservice service failed to become ready: {service_status}")

            # For Kubernetes services, we typically use cluster IP for internal communication
            service_url = f"http://{service_id}:{api_port}"

            logger.info("Inference Microservice created at %s", service_url)

            # Save job metadata to database so status callbacks can find it

            job_metadata = {
                "id": job_id,
                "action": "inference_microservice",  # Special action for inference microservices
                "status": "Running",
                "created_on": datetime.now(tz=timezone.utc),
                "last_modified": datetime.now(tz=timezone.utc),
                "experiment_id": experiment_id,
                "org_name": org_name,
                "user_id": experiment_metadata.get("user_id"),
                "network": network_arch,
                "parent_id": job_config.get("parent_id", ""),
                "num_gpu": 1,
                "platform_id": None,
                "kind": "experiment",
                "specs": {},
                "workflow_status": "Running",  # Not enqueued since it's already running
            }

            write_job_metadata(job_id, job_metadata)
            logger.info("Saved inference microservice job metadata for %s", job_id)

            return Code(200, {
                "service_id": service_id,
                "service_url": service_url,
                "status": "Running",
                "endpoints": {
                    "inference": f"{service_url}/api/v1/inference",
                    "health": f"{service_url}/api/v1/health/liveness",
                    "readiness": f"{service_url}/api/v1/health/readiness",
                    "status": f"{service_url}/api/v1/status"
                },
                "job_id": job_id,
                "api_port": api_port
            }, "Inference Microservice started successfully")

        except Exception as e:
            logger.error("Error starting Inference Microservice: %s", str(e))
            return Code(500, {}, f"Failed to start Inference Microservice: {str(e)}. Try again")

    @staticmethod
    def stop_inference_microservice(job_id: str, auto_deletion: bool = False) -> Code:
        """Stop a running Inference Microservice

        Args:
            job_id: Job ID for the microservice to stop
            auto_deletion: True if called due to idle timeout, False if manual stop

        Returns:
            Code object with status and result information
        """
        action = "Auto-deleting" if auto_deletion else "Stopping"
        reason = "due to inactivity" if auto_deletion else "manually"

        logger.info("%s Inference Microservice %s %s", action, job_id, reason)

        try:
            # Delete the StatefulSet and associated service using the enhanced delete function
            statefulset_executor = StatefulSetExecutor()
            deletion_success = statefulset_executor.delete_statefulset(
                job_id, resource_type="inference_microservice"
            )

            if deletion_success:
                success_message = "auto-deleted due to inactivity" if auto_deletion else "stopped successfully"
                logger.info(
                    "Successfully %s Inference Microservice %s",
                    "auto-deleted" if auto_deletion else "stopped", job_id
                )

                # Update job status to Done in database
                from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
                    update_job_status, get_handler_job_metadata
                )
                job_metadata = get_handler_job_metadata(job_id)
                if job_metadata:
                    experiment_id = job_metadata.get("experiment_id")
                    if experiment_id:
                        update_job_status(
                            experiment_id,
                            job_id,
                            status="Done",
                            kind="experiments"
                        )
                        logger.info("Updated job status to Done for %s", job_id)

                result = {
                    "status": "success",
                    "message": f"Inference microservice for job {job_id} {success_message}",
                    "job_id": job_id,
                    "timestamp": datetime.now().isoformat(),
                    "auto_deletion": auto_deletion
                }
                return Code(200, result, f"Inference Microservice {success_message}")

            error_msg = f"Failed to {action.lower()} inference microservice for job {job_id}"
            logger.error(error_msg)
            return Code(500, {"error": error_msg}, f"Failed to {action.lower()} Inference Microservice")

        except Exception as e:
            logger.error("Error %s Inference Microservice %s: %s", action.lower(), job_id, str(e))
            return Code(500, {"error": str(e)}, f"Error {action.lower()} Inference Microservice")

    @staticmethod
    def get_inference_microservice_status(job_id: str) -> Code:
        """Gets the status of a Inference Microservice StatefulSet"""
        statefulset_name = f"ims-{job_id}"

        try:
            stat_dict = StatefulSetExecutor().get_statefulset_status(
                statefulset_name, replicas=1, resource_type="Inference Microservice"
            )
            status = stat_dict.get("status", "Unknown")

            return Code(200, {
                "job_id": job_id,
                "service_name": statefulset_name,
                "status": status,
                "replicas": stat_dict.get("replicas", {}),
                "pods": []
            }, f"Inference Microservice status: {status}")

        except Exception as e:
            logger.error("Error getting Inference Microservice status: %s", str(e))
            return Code(500, {}, f"Failed to get service status: {str(e)}")

    @staticmethod
    def check_inference_microservice_model_readiness(job_id: str, api_port: int = 8080) -> dict:
        """Check if Inference Microservice model is ready in StatefulSet containers

        Args:
            job_id: Job ID for the microservice
            api_port: Port number for the microservice

        Returns:
            Dictionary with readiness status and progress information
        """
        try:
            statefulset_name = f"ims-{job_id}"

            # Check if StatefulSet pods exist and are running
            try:
                stat_dict = StatefulSetExecutor().get_statefulset_status(
                    statefulset_name, replicas=1, resource_type="Inference Microservice"
                )
                statefulset_status = stat_dict.get("status", "Unknown")

                # If StatefulSet is running, get detailed status from the microservice
                if statefulset_status == "Running":
                    try:
                        # Get detailed status including progress
                        status_response = InferenceMicroserviceHandler.get_inference_microservice_status_direct(
                            job_id, api_port
                        )
                        return {
                            "job_id": job_id,
                            "status": "ready" if status_response.get("model_loaded") else "loading",
                            "loaded": status_response.get("model_loaded", False),
                            "loading": status_response.get("model_loading", False),
                            "initializing": status_response.get("server_initializing", False),
                            "statefulset_status": statefulset_status,
                            "progress": status_response.get("progress", {})
                        }
                    except Exception as status_err:
                        logger.warning(f"Could not get detailed status for {job_id}: {status_err}")
                        # Fallback to basic response
                        return {
                            "job_id": job_id,
                            "status": "ready",
                            "loaded": True,
                            "statefulset_status": statefulset_status
                        }
                return {
                    "job_id": job_id,
                    "status": "not_ready",
                    "loaded": False,
                    "statefulset_status": statefulset_status
                }
            except Exception:
                return {
                    "job_id": job_id,
                    "status": "not_found",
                    "loaded": False,
                    "statefulset_status": "NotFound"
                }

        except Exception as e:
            logger.error(f"Error checking Inference Microservice model readiness: {e}")
            return {"status": "error", "error": str(e), "loaded": False}

    @staticmethod
    def get_inference_microservice_url(job_id: str, endpoint: str = "inference", api_port: int = 8080) -> str:
        """Get the URL for inference microservice requests

        Args:
            job_id: Job ID for the microservice
            endpoint: Endpoint to call (inference, health, status)
            api_port: Port number for the microservice (default: 8080)

        Returns:
            Full URL for the request (always uses simple service name)
        """
        service_name = f"ims-svc-{job_id}"

        # Always use simple service name for both Kubernetes and docker-compose
        # This works reliably for intra-cluster communication and avoids DNS issues
        if os.environ.get('BACKEND', 'local-k8s') == 'local-docker':
            url = f"http://{job_id}:{api_port}/api/v1/{endpoint}"
        else:
            url = f"http://{service_name}:{api_port}/api/v1/{endpoint}"

        logger.info(f"Inference microservice URL: {url}")
        return url

    @staticmethod
    def process_inference_microservice_request_direct(
            job_id: str, request_data: Dict[str, Any], api_port: int = 8080
    ) -> Dict[str, Any]:
        """Process inference request directly to the StatefulSet microservice

        Args:
            job_id: Job ID for the microservice
            request_data: Request data to send to the microservice

        Returns:
            Response from the microservice
        """
        try:
            logger.info(f"Processing inference request for job {job_id}")

            # Get the inference URL
            inference_url = InferenceMicroserviceHandler.get_inference_microservice_url(
                job_id, "inference", api_port
            )

            # Make request to the microservice
            timeout = 300  # 5 minutes timeout for inference
            response = requests.post(
                inference_url,
                json=request_data,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Inference request completed successfully for job {job_id}")
                return result
            if response.status_code in [202, 503]:
                # Server is initializing or loading - return appropriate response
                result = response.json()
                logger.info(
                    f"Inference microservice not ready for job {job_id}: "
                    f"{result.get('message', 'Unknown status')}"
                )
                return result
            error_msg = f"Inference request failed with status {response.status_code}"
            logger.error(f"{error_msg} for job {job_id}")
            try:
                error_detail = response.json()
                error_msg += f": {error_detail.get('error', 'Unknown error')}"
            except (ValueError, KeyError):
                error_msg += f": {response.text}"

            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat()
            }

        except requests.exceptions.Timeout:
            error_msg = f"Inference request timed out after {timeout} seconds for job {job_id}"
            logger.error(error_msg)
            # Try to get current progress to include in error response
            try:
                status_response = (
                    InferenceMicroserviceHandler.get_inference_microservice_status_direct(
                        job_id, api_port
                    )
                )
                progress_info = status_response.get("progress", {})
            except Exception:
                progress_info = {}
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": progress_info
            }
        except requests.exceptions.ConnectionError:
            error_msg = f"Could not connect to inference microservice for job {job_id}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": "Connection failed - microservice may not be running",
                    "remaining_steps": [],
                    "details": {"error": "ConnectionError"}
                }
            }
        except Exception as e:
            error_msg = f"Unexpected error during inference request for job {job_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": f"Unexpected error: {str(e)}",
                    "remaining_steps": [],
                    "details": {"error": str(e)}
                }
            }

    @staticmethod
    def get_inference_microservice_status_direct(job_id: str, api_port: int = 8080) -> Dict[str, Any]:
        """Get status directly from the StatefulSet microservice

        Args:
            job_id: Job ID for the microservice

        Returns:
            Status response from the microservice including progress information
        """
        try:
            logger.info(f"Getting status for inference microservice job {job_id}")

            # Get the status URL
            status_url = InferenceMicroserviceHandler.get_inference_microservice_url(
                job_id, "status", api_port
            )

            # Make request to the microservice
            response = requests.get(status_url, timeout=30)

            if response.status_code == 200:
                result = response.json()
                logger.info(f"Status retrieved successfully for job {job_id}")
                # Progress information is already included in the result from the server
                return result
            error_msg = f"Status request failed with status {response.status_code}"
            logger.error(f"{error_msg} for job {job_id}")
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": error_msg,
                    "remaining_steps": [],
                    "details": {}
                }
            }

        except Exception as e:
            error_msg = f"Failed to get status for job {job_id}: {str(e)}"
            logger.error(error_msg)
            return {
                "status": "error",
                "error": error_msg,
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "stage": "error",
                    "message": f"Failed to connect to microservice: {str(e)}",
                    "remaining_steps": [],
                    "details": {"error": str(e)}
                }
            }

    @staticmethod
    def get_inference_microservice_status_detailed(job_id: str) -> dict:
        """Get Inference Microservice service status with model readiness information"""
        try:
            statefulset_name = f"ims-{job_id}"
            stat_dict = StatefulSetExecutor().get_statefulset_status(
                statefulset_name, replicas=1, resource_type="Inference Microservice"
            )

            # Check model readiness
            model_state = InferenceMicroserviceHandler.check_inference_microservice_model_readiness(job_id)

            return {
                "job_id": job_id,
                "service_name": statefulset_name,
                "status": stat_dict.get("status", "Unknown"),
                "replicas": stat_dict.get("replicas", {}),
                "model_loaded": model_state.get("loaded", False),
                "model_status": model_state.get("status", "unknown")
            }

        except Exception as e:
            logger.error(f"Error getting Inference Microservice service status: {e}")
            return {
                "job_id": job_id,
                "status": "error",
                "error": str(e)
            }
