# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Original source taken from https://github.com/NVIDIA/NeMo
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

"""Docker handler."""

import json
import logging
import os
import requests
import subprocess
import traceback
# pylint: disable=c-extension-no-member
import docker
from nvidia_tao_core.microservices.utils.core_utils import get_admin_key
if os.getenv("BACKEND") == "local-docker":
    from nvidia_tao_core.microservices.utils.job_utils.gpu_manager import gpu_manager

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)

DOCKER_NETWORK = os.getenv("DOCKER_NETWORK", "tao_default")
DOCKER_USERNAME = os.getenv("DOCKER_USERNAME", "$oauthtoken")
docker_client = docker.from_env() if os.getenv("DOCKER_HOST") else None


def is_tegra_platform():
    """Detect if running on NVIDIA Tegra/Jetson platform.

    This follows the same detection logic as tao_pt.py runner.
    Checks the uname output for 'tegra' string, which indicates
    Tegra/Jetson platforms like Thor.

    Returns:
        bool: True if Tegra/Jetson platform detected, False otherwise.
    """
    try:
        # Check if running on a tegra system like thor or jetson
        uname_output = subprocess.check_output(["uname", "-a"]).decode().strip()
        is_tegra = "tegra" in uname_output.lower()

        if is_tegra:
            logger.info(f"Tegra platform detected via uname: {uname_output}")
        else:
            logger.debug("Non-Tegra platform detected")

        return is_tegra
    except Exception as e:
        logger.warning(f"Error detecting platform type: {e}. Assuming non-Tegra platform.")
        return False


def should_use_nvidia_runtime():
    """Determine whether to use runtime="nvidia" or device_requests for GPU access.

    This follows the same logic as tao_pt.py runner:
    - Use runtime="nvidia" for Tegra/Jetson platforms (requires nvidia-docker2)
    - Use device_requests for standard x86 systems (requires nvidia-container-toolkit)

    Auto-detects the platform via uname and returns the appropriate runtime method.

    Returns:
        bool: True if runtime="nvidia" should be used, False for device_requests.
    """
    # Auto-detect platform (same logic as tao_pt.py)
    if is_tegra_platform():
        logger.info("Using runtime='nvidia' for Tegra/Jetson platform")
        return True
    logger.debug("Using device_requests for standard platform")
    return False


def docker_pull_progress(line):
    """Simple function to log docker pull progress."""
    logged_layers = set()
    if line['status'] == 'Downloading':
        if 'progressDetail' in line and 'current' in line['progressDetail'] and 'total' in line['progressDetail']:
            current = line['progressDetail']['current']
            total = line['progressDetail']['total']
            if total > 0:
                percentage = (current / total) * 100
                # Only log every 10% to reduce verbosity
                if percentage % 10 < 1 or percentage == 100:
                    logger.info(f"Downloading {line['id']}: {percentage:.0f}%")
        else:
            # Only log once per layer to avoid spam
            if line['id'] not in logged_layers:
                logger.info(f"Downloading {line['id']}")
                logged_layers.add(line['id'])
    elif line['status'] == 'Extracting':
        if 'progressDetail' in line and 'current' in line['progressDetail'] and 'total' in line['progressDetail']:
            current = line['progressDetail']['current']
            total = line['progressDetail']['total']
            if total > 0:
                percentage = (current / total) * 100
                # Only log every 25% to reduce verbosity
                if percentage % 25 < 1 or percentage == 100:
                    logger.info(f"Extracting {line['id']}: {percentage:.0f}%")
        else:
            # Only log once per layer to avoid spam
            if line['id'] not in logged_layers:
                logger.info(f"Extracting {line['id']}")
                logged_layers.add(line['id'])
    else:
        # Only log important statuses, skip verbose ones
        important_statuses = ['Pulling fs layer', 'Verifying Checksum', 'Download complete', 'Pull complete']
        if line['status'] in important_statuses:
            logger.info(f"Docker pull: {line['status']} for {line.get('id', 'unknown')}")
        logger.debug(f"Docker pull status: {line['status']} for {line.get('id', 'unknown')}")


class DockerHandler:
    """Docker Handler class"""

    def __init__(self, docker_image=None, container=None):
        """Initialize the docker handler object."""
        if not docker_client:
            raise ValueError("Docker client not initialized")
        self._docker_client = docker_client
        self._api_client = docker_client.api
        self._container = container
        self._docker_image = docker_image
        self._docker_registry, self._image_name, self._docker_tag = self.parse_image_string(docker_image)

    def login(self):
        """Login to the docker registry."""
        ngc_api_key = get_admin_key()
        if self._docker_registry == "nvcr.io" and ngc_api_key:
            try:
                response = self._api_client.login(
                    username=DOCKER_USERNAME,
                    password=ngc_api_key,
                    registry=self._docker_registry)
                logger.info(f"Logged in to NGC registry {self._docker_registry}")
                return response
            except Exception as e:
                logger.error(f"Error logging in to NGC registry {self._docker_registry}: {e}")
        return None

    @staticmethod
    def get_handler_for_container(container_name=None):
        """Initialize a docker handler from a container."""
        logger.info(f"Getting handler for container: {container_name}")
        if not docker_client:
            raise ValueError("Docker client not initialized")
        if not container_name:
            return None
        try:
            container = docker_client.containers.get(container_name)
            logger.info(f"Found container image {container.image.tags[0]} for {container_name}")
            return DockerHandler(container.image.tags[0], container=container)
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error getting handler for container {container_name}: {e}")
            return None

    def _check_image_exists(self):
        """Check if the image exists locally."""
        image_list = self._docker_client.images.list()
        assert isinstance(image_list, list), (
            "image_list should be a list."
        )
        for image in image_list:
            image_inspection_content = self._api_client.inspect_image(image.attrs["Id"])
            if image_inspection_content["RepoTags"]:
                if self._docker_image in image_inspection_content["RepoTags"]:
                    return True
        return False

    @staticmethod
    def parse_image_string(image_string):
        """Parse a Docker image string into registry, repository, and tag.

        Args:
            image_string (str): Docker image string (e.g., "nvcr.io/ea-tlt/tao_ea/vila_fine_tuning:latest")

        Returns:
            tuple: (registry, repository, tag)

        Examples:
            >>> parse_image_string("nvcr.io/ea-tlt/tao_ea/vila_fine_tuning:latest")
            ("nvcr.io", "ea-tlt/tao_ea/vila_fine_tuning", "latest")
            >>> parse_image_string("nvcr.io/ea-tlt/tao_ea/vila_fine_tuning")
            ("nvcr.io", "ea-tlt/tao_ea/vila_fine_tuning", "latest")
            >>> parse_image_string("ubuntu:20.04")
            ("docker.io", "ubuntu", "20.04")
        """
        if not image_string:
            raise ValueError("Image string cannot be empty")

        # Split by tag (after last colon)
        if ':' in image_string:
            # Handle cases like "localhost:5000/myapp:latest"
            # We need to find the last colon that's not part of a port number
            parts = image_string.split(':')
            if len(parts) == 2:
                # Simple case: image:tag
                repository_part, tag = parts
            else:
                # Complex case: registry:port/repository:tag
                # Find the last colon that's followed by a non-digit
                for i in range(len(parts) - 1, 0, -1):
                    if not parts[i].isdigit():
                        repository_part = ':'.join(parts[:i])
                        tag = ':'.join(parts[i:])
                        break
                else:
                    # All parts except the last are digits (port numbers)
                    repository_part = ':'.join(parts[:-1])
                    tag = parts[-1]
        else:
            repository_part = image_string
            tag = "latest"

        # Split repository by first slash to separate registry from image path
        if '/' in repository_part:
            # Check if the first part looks like a registry (contains dot or is 'localhost')
            first_part = repository_part.split('/')[0]
            if '.' in first_part or first_part == 'localhost':
                # First part is registry
                registry = first_part
                repository = '/'.join(repository_part.split('/')[1:])
            else:
                # No registry specified, use default
                registry = "docker.io"
                repository = repository_part
        else:
            # No slashes, this is a simple image name
            registry = "docker.io"
            repository = repository_part

        return registry, repository, tag

    def pull(self):
        """Pull the base docker."""
        logger.info(
            "Pulling the required container. This may take several minutes if you're doing this for the first time. "
            "Please wait here.\n...")
        self.login()
        try:
            repository = f"{self._docker_registry}/{self._image_name}"
            logger.info(f"Pulling from repository: {repository}")
            response = self._api_client.pull(
                repository=repository,
                tag=self._docker_tag,
                stream=True,
                decode=True
            )
            for line in response:
                docker_pull_progress(line)
        except docker.errors.APIError as e:
            logger.error(f"Docker pull failed. {e}")
            raise e
        logger.info("Container pull complete.")

    @staticmethod
    def get_device_requests(gpu_ids=[]):
        """Create device requests for the docker container."""
        device_requests = []
        if gpu_ids:
            device_requests = [docker.types.DeviceRequest(capabilities=[["gpu"]], device_ids=gpu_ids)]
        else:
            device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]
        return device_requests

    def start_container(
        self, container_name="", docker_env_vars={}, command=[], num_gpus=-1, volumes=None,
        pre_assigned_gpu_ids=None
    ):
        """Start a container with GPU access.

        Automatically detects the platform and uses the appropriate GPU access method:
        - Tegra/Jetson: Uses runtime="nvidia" with NVIDIA_VISIBLE_DEVICES
        - Standard x86: Uses device_requests

        Args:
            pre_assigned_gpu_ids: Optional list of GPU IDs already assigned by workflow.
                                 If provided, skips GPU assignment and uses these directly.

        Args:
            container_name: Name for the container
            docker_env_vars: Dictionary of environment variables
            command: Command to run in the container
            num_gpus: Number of GPUs to assign (-1 for all)
            volumes: Volume mounts for the container
        """
        try:
            # Check if the image exists locally. If not, pull it.
            if not self._check_image_exists():
                logger.info(
                    "The required docker doesn't exist locally/the manifest has changed. "
                    "Pulling a new docker.")
                self.pull()

            # Check for pre-assigned GPUs in three places (in order of priority):
            # 1. Passed directly as parameter
            # 2. Stored in MongoDB job metadata (from workflow pre-assignment)
            # 3. Assign new GPUs
            gpu_ids = None

            if pre_assigned_gpu_ids:
                logger.debug(
                    f"[GPU_ASSIGN] Using pre-assigned GPU IDs {pre_assigned_gpu_ids} "
                    f"for container {container_name} (passed as parameter)"
                )
                gpu_ids = pre_assigned_gpu_ids
            else:
                # Check MongoDB for pre-assigned GPUs
                from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_handler_job_metadata
                job_metadata = get_handler_job_metadata(container_name)
                if job_metadata and "pre_assigned_gpu_ids" in job_metadata:
                    gpu_ids = job_metadata["pre_assigned_gpu_ids"]
                    logger.debug(
                        f"[GPU_ASSIGN] Using pre-assigned GPU IDs {gpu_ids} "
                        f"for container {container_name} (from workflow pre-assignment in MongoDB)"
                    )

            if not gpu_ids:
                # Skip GPU assignment if num_gpus is 0 (job doesn't need GPUs)
                if num_gpus == 0:
                    logger.debug(
                        f"[GPU_ASSIGN] Job {container_name} does not require GPUs (num_gpus=0), "
                        f"skipping GPU assignment"
                    )
                    gpu_ids = []
                else:
                    logger.debug(f"[GPU_ASSIGN] Attempting to assign {num_gpus} GPU(s) for container {container_name}")
                    gpu_ids = gpu_manager.assign_gpus(container_name, num_gpus)
                    logger.debug(f"[GPU_ASSIGN] Assigned GPU IDs: {gpu_ids} for container {container_name}")

                # This prevents containers from starting with "all" GPUs when none are available
                # This should rarely happen now because:
                # - Workflow jobs: pre-assignment prevents dequeue without GPUs
                # - Direct spawns: availability check prevents calling start_container
                # But keep this as a safety net
                if num_gpus > 0 and num_gpus != -1 and not gpu_ids:
                    error_msg = (
                        f"[GPU_ASSIGN] FAILED: Cannot start container {container_name} - "
                        f"requested {num_gpus} GPU(s) but no GPUs available."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            logger.debug(f"Starting Container: {self._docker_image}")

            # Determine GPU access method based on platform
            use_runtime = should_use_nvidia_runtime()

            # Prepare container run parameters
            run_kwargs = {
                "image": self._docker_image,
                "command": command,
                "name": container_name,
                "network": DOCKER_NETWORK,
                "tmpfs": {"/dev/shm": ""},
                "detach": True,
                "remove": True,
                "environment": docker_env_vars,
                "volumes": volumes,
            }

            if use_runtime:
                # Method 1: Use runtime="nvidia" (for Tegra/Jetson and nvidia-docker2)
                # This matches the tao_pt.py runner behavior:
                # --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all -e NVIDIA_VISIBLE_DEVICES=<gpus>
                docker_env_vars = docker_env_vars.copy()
                docker_env_vars["NVIDIA_DRIVER_CAPABILITIES"] = "all"
                if gpu_ids:
                    docker_env_vars["NVIDIA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
                    logger.info(f"Using runtime='nvidia' with GPUs: {','.join(gpu_ids)}")
                else:
                    docker_env_vars["NVIDIA_VISIBLE_DEVICES"] = "all"
                    logger.info("Using runtime='nvidia' with all available GPUs")

                run_kwargs["runtime"] = "nvidia"
                docker_env_vars = docker_env_vars.copy()
                docker_env_vars["NVIDIA_DRIVER_CAPABILITIES"] = "all"
            else:
                # Method 2: Use device_requests (for standard Docker with nvidia-container-toolkit)
                device_requests = self.get_device_requests(gpu_ids)
                if gpu_ids:
                    logger.info(f"Using device_requests with GPUs: {','.join(gpu_ids)}")
                else:
                    logger.info("Using device_requests with all available GPUs")

                run_kwargs["device_requests"] = device_requests

            # Start the container
            self._container = self._docker_client.containers.run(**run_kwargs)
            logger.info(f"Container {container_name} started successfully")

        except Exception as e:
            logger.error(f"Exception thrown in start_container is {str(e)}")
            logger.error(traceback.format_exc())
            raise e

    def check_container_health(self, port=8000):
        """Check if the microservice container is running and healthy."""
        if self._container:
            logger.info(f"Checking container health: {self._container.name}")
            self._container.reload()
            if self._container.status.lower() == "running":
                container_name = self._container.name
                try:
                    base_url = f"http://{container_name}:{port}"
                    liveness_response = requests.get(f"{base_url}/api/v1/health/liveness", timeout=120)
                    if liveness_response.status_code == 200:
                        return True
                except Exception as e:
                    logger.error(f"Exception caught during checking container health: {e}")
                    return False
            else:
                logger.error(f"Container {self._container.name} is in {self._container.status} state")
                return False
        return False

    def check_container_readiness(self, port=8000):
        """Check if the microservice is ready to serve inference requests.

        Returns:
            True if model is loaded and ready for inference, False otherwise
        """
        if self._container:
            logger.info(f"Checking container readiness: {self._container.name}")
            self._container.reload()
            if self._container.status.lower() == "running":
                container_name = self._container.name
                try:
                    base_url = f"http://{container_name}:{port}"
                    readiness_response = requests.get(f"{base_url}/api/v1/health/readiness", timeout=30)
                    if readiness_response.status_code == 200:
                        logger.info(f"Container {container_name} is ready for inference")
                        return True
                    logger.info(f"Container {container_name} not ready yet: {readiness_response.status_code}")
                    return False
                except Exception as e:
                    logger.error(f"Exception caught during checking container readiness: {e}")
                    return False
            else:
                logger.error(f"Container {self._container.name} is in {self._container.status} state")
                return False
        return False

    def make_container_request(
        self,
        api_endpoint,
        network,
        action,
        cloud_metadata={},
        specs={},
        job_id="",
        docker_env_vars={},
        port=8000
    ):
        """Make a request to the microservice container."""
        if self._container:
            logger.info(f"Making microservice request to container: {self._container.name}")
            # Set default URLs if not provided
            if not docker_env_vars.get("TAO_API_SERVER"):
                docker_env_vars["TAO_API_SERVER"] = "https://nvidia.com"
            if not docker_env_vars.get("TAO_LOGGING_SERVER_URL"):
                docker_env_vars["TAO_LOGGING_SERVER_URL"] = "https://nvidia.com"

            # Normalize action name
            if action == "retrain":
                action = "train"
            if cloud_metadata:
                docker_env_vars["CLOUD_BASED"] = "True"

            # Prepare request metadata
            request_metadata = {
                "neural_network_name": network,
                "action_name": action,
                "specs": specs,
                "cloud_metadata": cloud_metadata,
                "docker_env_vars": docker_env_vars,
            }

            # Prepare request body
            if job_id:
                request_metadata["job_id"] = job_id
                request_metadata["docker_env_vars"]["JOB_ID"] = job_id

            base_url = f"http://{self._container.name}:{port}"
            endpoint = f"{base_url}/api/v1/internal/container_job"

            # Modify endpoint and request_metadata for get_job_status
            if api_endpoint == "get_job_status":
                endpoint = f"{base_url}/api/v1/internal/container_job:status"
                request_metadata = {"results_dir": specs.get("results_dir", "")}

            # Send request
            logger.debug("Sending request to %s with request_metadata %s", endpoint, request_metadata)
            try:
                if api_endpoint == "get_job_status":
                    response = requests.get(endpoint, params=request_metadata, timeout=120)
                else:
                    data = json.dumps(request_metadata)
                    response = requests.post(endpoint, data=data, timeout=120)
            except Exception as e:
                logger.error("Exception caught during sending a microservice request %s", e)
                raise e
            return response
        logger.error("No container to make request to")
        return None

    def stop_container(self):
        """Stop a container."""
        if self._container:
            logger.info(f"Stopping container: {self._container.name}")
            self._container.stop()
        else:
            logger.error("No container to stop")


def get_all_docker_running_containers():
    """Get all running Docker containers for TAO jobs

    Returns:
        list: List of container info dictionaries
    """
    backend = os.getenv("BACKEND", "local-k8s")
    if backend != "local-docker":
        return []

    try:
        if not docker_client:
            logger.error("Docker client not available")
            return []

        # List all running Docker containers with TAO labels
        containers = []
        all_containers = docker_client.containers.list(
            filters={'label': 'tao-toolkit'}
        )

        for container in all_containers:
            container_name = container.name
            container_id = container.id
            # Container name is typically the job_id
            containers.append({
                'job_id': container_name,
                'container_id': container_id,
                'status': 'Running'
            })

        return containers

    except Exception as e:
        logger.error(f"Error getting Docker containers: {e}")
        return []
