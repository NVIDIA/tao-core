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

"""Job queue handling."""


import json
import glob
import importlib
import os
import threading
import traceback
import yaml
import logging
import tarfile
from tqdm import tqdm

from nvidia_tao_core.api_utils import module_utils
from nvidia_tao_core.api_utils.entrypoint_mimicker import vlm_entrypoint
from nvidia_tao_core.cloud_handlers.utils import (
    download_files_from_spec,
    get_results_cloud_data,
    monitor_and_upload,
    cleanup_cuda_contexts,
    create_tarball,
    upload_tarball_to_cloud,
)
import nvidia_tao_core.loggers.logging as status_logging
from nvidia_tao_core.api_utils.module_utils import entrypoint_paths, entry_points
from nvidia_tao_core.microservices.utils import (
    safe_load_file,
    safe_dump_file,
    read_network_config,
    get_spec_backend_info
)
from nvidia_tao_core.microservices.specs_utils import json_to_kitti, json_to_yaml, json_to_toml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Spec backend to conversion functions mapping
SPEC_BACKEND_TO_FUNCTIONS = {
    "protobuf": json_to_kitti.kitti,
    "yaml": json_to_yaml.yml,
    "toml": json_to_toml.toml_format
}

IS_MASTER = int(os.environ.get("NODE_RANK", 0)) == 0


def prepare_data_before_job_run(job, docker_env_vars):
    """Prepare data before job run"""
    if docker_env_vars:
        os.environ.update(docker_env_vars)

    cloud_storage, specs = get_results_cloud_data(
        job.get("cloud_metadata"),
        job["specs"],
        f'/results/{job["job_id"]}'
    )

    ngc_key = docker_env_vars.get("TAO_API_KEY")
    if not ngc_key:
        ngc_key = docker_env_vars.get("TAO_USER_KEY")

    # Create results directory and download files
    os.makedirs(specs["results_dir"], exist_ok=True)
    reprocess_files = []

    # Handle additional downloads
    additional_downloads = specs.pop("additional_downloads", [])
    if additional_downloads:
        logger.info("Processing additional downloads: %s", additional_downloads)
        ContainerJobHandler._handle_additional_downloads(
            additional_downloads,
            job.get("cloud_metadata"),
            job["job_id"],
            job["neural_network_name"],
            ngc_key,
        )

    # Extract preserve_source_path_params if specified
    preserve_source_path_params = specs.pop("preserve_source_path_params", set())
    if isinstance(preserve_source_path_params, list):
        preserve_source_path_params = set(preserve_source_path_params)
    logger.info("Downloading files from normal spec (preserve_source_path_params=%s)", preserve_source_path_params)
    download_files_from_spec(
        cloud_data=job.get("cloud_metadata"),
        data=specs,
        job_id=job["job_id"],
        network_arch=job["neural_network_name"],
        ngc_key=ngc_key,
        reprocess_files=reprocess_files,
        preserve_source_path_params=preserve_source_path_params
    )

    # Save spec file with dynamic backend
    network_arch = job["neural_network_name"]
    spec_backend, file_extension = get_spec_backend_info(network_arch)
    spec_path = os.path.join(specs["results_dir"], f"spec.{file_extension}")

    if spec_backend == "yaml":
        # Use yaml format
        with open(spec_path, 'w+', encoding='utf-8') as spec_file:
            yaml.dump(specs, spec_file, default_flow_style=False)
    elif spec_backend in SPEC_BACKEND_TO_FUNCTIONS:
        # Use appropriate conversion function
        conversion_func = SPEC_BACKEND_TO_FUNCTIONS[spec_backend]
        converted_specs = conversion_func(specs)
        with open(spec_path, 'w+', encoding='utf-8') as spec_file:
            spec_file.write(converted_specs)
    else:
        # Fallback to yaml if unknown backend
        logger.warning(f"Unknown spec backend '{spec_backend}', falling back to yaml")
        with open(spec_path, 'w+', encoding='utf-8') as spec_file:
            yaml.dump(specs, spec_file, default_flow_style=False)

    if docker_env_vars.get("RECURSIVE_DATASET_FILE_DOWNLOAD", "False") == "True":
        logger.info("reprocess_files: %s", reprocess_files)
        if reprocess_files:
            for file_name in reprocess_files:
                file_type = file_name.split(".")[-1]
                reprocess_file_data = safe_load_file(file_name, file_type=file_type)
                if reprocess_file_data:
                    download_files_from_spec(
                        cloud_data=job.get("cloud_metadata"),
                        data=reprocess_file_data,
                        job_id=job["job_id"],
                        network_arch=job["neural_network_name"],
                        ngc_key=ngc_key,
                    )
                    if reprocess_file_data:
                        safe_dump_file(file_name, reprocess_file_data, file_type=file_type)
    return cloud_storage, specs, spec_path


class ContainerJobHandler:
    """Handler for processing jobs in a containerized environment."""

    @staticmethod
    def capture_directory_snapshot(directory):
        """Capture a snapshot of all files and directories in the given directory.

        Args:
            directory (str): Directory to snapshot

        Returns:
            set: Set of relative file paths from the directory
        """
        try:
            if not os.path.exists(directory):
                logger.warning("Directory does not exist for snapshot: %s", directory)
                return set()

            snapshot = set()
            for root, dirs, files in os.walk(directory):
                # Add all files
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, directory)
                    snapshot.add(rel_path)

                # Add all directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    rel_path = os.path.relpath(dir_path, directory)
                    snapshot.add(rel_path)

            logger.info("Captured snapshot of %d items in directory: %s", len(snapshot), directory)
            return snapshot

        except Exception as e:
            logger.error("Error capturing directory snapshot: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())
            return set()

    @staticmethod
    def create_and_upload_tarball(results_dir, cloud_storage, job_id, action_name,
                                  exclude_snapshot=None, exclude_patterns=None):
        """Create a tarball of the results directory and upload it.

        Args:
            results_dir (str): Directory to tarball
            cloud_storage: CloudStorage instance for uploading
            job_id (str): Job ID for naming the tarball
            action_name (str): Action name for naming the tarball
            exclude_snapshot (set, optional): Set of relative paths to exclude from tarball
            exclude_patterns (list, optional): List of regex patterns to exclude files from tarball
        """
        try:
            tarball_name = f"{action_name}_results.tar.gz"
            tarball_path = os.path.join(results_dir, tarball_name)

            # Create tarball using utility function
            if create_tarball(results_dir, tarball_path, exclude_snapshot, exclude_patterns):
                # Upload tarball using utility function
                if cloud_storage and os.path.exists(tarball_path):
                    upload_tarball_to_cloud(cloud_storage, tarball_path, remove_after_upload=True)
                else:
                    logger.warning("No cloud storage configured, tarball created but not uploaded: %s", tarball_path)
            else:
                logger.error("Failed to create tarball for job %s action %s", job_id, action_name)

        except Exception as e:
            logger.error("Error in create_and_upload_tarball: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())

    @staticmethod
    def entrypoint_wrapper(job):
        """Starts job asynchronously and handles cleanup after completion.

        Args:
            job (dict): Job configuration containing action, neural network and other metadata

        Returns:
            str: Job ID if launch successful, None otherwise
        """
        try:
            docker_env_vars = job.get('docker_env_vars', {})

            def async_setup_and_run():
                cloud_storage = None
                exit_event = None
                upload_thread = None
                status_logger = None
                results_dir_snapshot = None

                try:
                    # Setup cloud storage and specs
                    cloud_storage, specs, spec_path = prepare_data_before_job_run(job, docker_env_vars)

                    # Capture snapshot of results directory after downloads but before job execution
                    results_dir_snapshot = ContainerJobHandler.capture_directory_snapshot(specs["results_dir"])

                    # Get upload strategy and exclude patterns by reading network config directly
                    network = docker_env_vars.get("ORCHESTRATION_API_NETWORK",
                                                  job.get("neural_network_name", ""))
                    action = docker_env_vars.get("ORCHESTRATION_API_ACTION", job.get("action_name", ""))
                    upload_strategy, exclude_patterns = ContainerJobHandler.get_upload_strategy_from_config(
                        network, action)
                    logger.info("Using upload strategy for %s %s: %s", network, action, upload_strategy)
                    if exclude_patterns:
                        logger.info("Excluding patterns for %s %s: %s", network, action, exclude_patterns)

                    # Determine if we should start continuous monitoring
                    should_start_continuous = True
                    selective_tarball_config = None

                    if isinstance(upload_strategy, dict):
                        # Complex upload strategy
                        default_strategy = upload_strategy.get("default", "continuous")
                        selective_tarball_config = upload_strategy.get("selective_tarball")

                        if default_strategy != "continuous":
                            should_start_continuous = False

                        logger.info("Complex upload strategy - default: %s, selective_tarball: %s",
                                    default_strategy, bool(selective_tarball_config))
                    elif upload_strategy != "continuous":
                        # Simple non-continuous strategy
                        should_start_continuous = False

                    if cloud_storage and should_start_continuous:
                        exit_event = threading.Event()
                        upload_thread = threading.Thread(
                            target=monitor_and_upload,
                            args=(specs["results_dir"], cloud_storage, exit_event, 0,
                                  selective_tarball_config, exclude_patterns),
                            daemon=True
                        )
                        upload_thread.start()
                    else:
                        # For tarball_after_completion or complex strategies, we'll handle upload after job completion
                        exit_event = None
                        upload_thread = None

                    # Prepare entrypoint arguments
                    args = {
                        "subtask": job["action_name"],
                        "experiment_spec_file": spec_path,
                        "results_dir": specs["results_dir"]
                    }

                    module = entry_points[0].module_name.split('.')[0] if entry_points else None
                    entrypoint = importlib.import_module(entrypoint_paths[module]) if module else None

                    def run_entrypoint():
                        nonlocal status_logger
                        is_completed = False
                        status_file = None

                        def initialize_status_logger(status_file):
                            """Initialize or get existing status logger."""
                            nonlocal status_logger
                            if not status_logger:
                                status_logger = status_logging.StatusLogger(
                                    filename=status_file,
                                    is_master=IS_MASTER,
                                    verbosity=1,
                                    append=True
                                )
                                status_logging.set_status_logger(status_logger)
                            return status_logger

                        try:
                            # Launch entrypoint
                            if entrypoint:
                                try:
                                    _, actions = module_utils.get_neural_network_actions(job["neural_network_name"])
                                    entrypoint.launch(args, "", actions, network=job["neural_network_name"])
                                    is_completed = True
                                except SystemExit as e:
                                    is_completed = e.code == 0
                            else:
                                is_completed = vlm_entrypoint.vlm_launch(
                                    job["neural_network_name"],
                                    job["action_name"],
                                    specs,
                                    job["job_id"]
                                )

                        except Exception:
                            logger.error("Traceback")
                            logger.error(traceback.format_exc())
                            status_file = ContainerJobHandler.get_status_file(specs["results_dir"], job["action_name"])
                            status_logger = initialize_status_logger(status_file)
                            ContainerJobHandler._handle_failure(job, status_logger, status_file)
                        finally:
                            status_file = status_file or ContainerJobHandler.get_status_file(
                                specs["results_dir"],
                                job["action_name"]
                            )
                            status_logger = initialize_status_logger(status_file)
                            ContainerJobHandler._cleanup(
                                exit_event,
                                upload_thread,
                                job,
                                is_completed,
                                status_logger,
                                status_file,
                                cloud_storage,
                                upload_strategy,
                                specs["results_dir"],
                                selective_tarball_config,
                                results_dir_snapshot,
                                exclude_patterns
                            )

                    # Launch job asynchronously
                    entrypoint_thread = threading.Thread(target=run_entrypoint, daemon=True)
                    entrypoint_thread.start()

                except Exception:
                    logger.error("Traceback")
                    logger.error(traceback.format_exc())
                    if status_logger:
                        status_logging.get_status_logger().write(
                            message=(
                                f"{job['action_name']} action couldn't be launched "
                                f"for {job['neural_network_name']}"
                            ),
                            status_level=status_logging.Status.FAILURE
                        )
                    ContainerJobHandler._cleanup(
                        exit_event=exit_event,
                        upload_thread=upload_thread,
                        results_dir_snapshot=results_dir_snapshot
                    )

            # Launch the async setup and execution
            setup_thread = threading.Thread(target=async_setup_and_run, daemon=True)
            setup_thread.start()

            return job["job_id"]

        except Exception:
            logger.error("Traceback")
            logger.error(traceback.format_exc())
            return None

    @staticmethod
    def _handle_failure(job, status_logger, status_file):
        """Handle failure cases by creating status logger if needed and logging failure."""
        if not status_logger:
            try:
                status_logger = status_logging.StatusLogger(
                    filename=status_file,
                    is_master=IS_MASTER,
                    verbosity=1,
                    append=True
                )
                status_logging.set_status_logger(status_logger)
            except Exception:
                logger.error("Failed to create status logger")
        status_logging.get_status_logger().write(
            message=f"{job['action_name']} action failed for {job['neural_network_name']}",
            status_level=status_logging.Status.FAILURE
        )

    @staticmethod
    def _cleanup(
        exit_event=None,
        upload_thread=None,
        job=None,
        is_completed=None,
        status_logger=None,
        status_file=None,
        cloud_storage=None,
        upload_strategy="continuous",
        results_dir=None,
        selective_tarball_config=None,
        results_dir_snapshot=None,
        exclude_patterns=None
    ):
        """Clean up resources and log final status."""
        if exit_event:
            exit_event.set()
        if upload_thread:
            upload_thread.join()

        # Handle tarball upload strategy after job completion
        if (job and is_completed and cloud_storage and
           upload_strategy == "tarball_after_completion" and results_dir):
            logger.info("Job completed successfully, creating and uploading tarball...")
            ContainerJobHandler.create_and_upload_tarball(
                results_dir,
                cloud_storage,
                job["job_id"],
                job["action_name"],
                results_dir_snapshot,
                exclude_patterns
            )

        # Handle selective tarball creation if needed
        if selective_tarball_config and job and is_completed and cloud_storage:
            logger.info("Job completed successfully, creating and uploading selective tarball...")
            tarball_path = ContainerJobHandler.create_selective_tarball(
                results_dir,
                selective_tarball_config.get("patterns", []),
                selective_tarball_config.get("base_path", ""),
                job["job_id"],
                job["action_name"]
            )

            if tarball_path:
                upload_tarball_to_cloud(cloud_storage, tarball_path, remove_after_upload=True)

        if job and is_completed is not None:
            status = status_logging.Status.SUCCESS if is_completed else status_logging.Status.FAILURE
            result = "completed successfully" if is_completed else "failed"

            if not status_logger:
                try:
                    status_logger = status_logging.StatusLogger(
                        filename=status_file,
                        is_master=IS_MASTER,
                        verbosity=1,
                        append=True
                    )
                    status_logging.set_status_logger(status_logger)
                except Exception:
                    logger.error("Failed to create status logger")

            status_logging.get_status_logger().write(
                message=f"{job['action_name']} action {result} for {job['neural_network_name']}",
                status_level=status
            )
        # Clean up any stale CUDA contexts
        cleanup_cuda_contexts()

    @staticmethod
    def get_status_file(results_dir, action_name=""):
        """Get the path to the status file.

        Args:
            results_dir (str): Directory containing results and status files
            action_name (str, optional): Name of action for creating new status file path. Defaults to "".

        Returns:
            str: Path to existing status.json file if found, otherwise constructs new path
        """
        status_files = glob.glob(os.path.join(results_dir, "**", "status.json"), recursive=True)
        if not status_files:
            return os.path.join(results_dir, action_name, "status.json")
        return status_files[0]

    @staticmethod
    def _handle_additional_downloads(
        additional_downloads,
        cloud_metadata,
        job_id,
        network_arch,
        ngc_key,
    ):
        """Handle downloading additional files specified in additional_downloads.

        Args:
            additional_downloads (list): List of additional file/directory paths to download
            cloud_metadata (dict): Cloud storage metadata
            job_id (str): Current job ID
            network_arch (str): Network architecture name
            ngc_key (str): NGC API key
        """
        try:
            if not additional_downloads:
                return

            # Create a spec structure that includes all additional downloads
            # Use preserve_source_path=True to maintain original path structure
            additional_spec = {}

            for i, download_path in enumerate(additional_downloads):
                logger.info("Preparing additional download: %s", download_path)
                additional_spec[f"additional_download_{i}"] = download_path

            # Use the existing download utility with preserve_source_path=True
            download_files_from_spec(
                cloud_data=cloud_metadata,
                data=additional_spec,
                job_id=job_id,
                network_arch=network_arch,
                ngc_key=ngc_key,
                reprocess_files=[],
                preserve_source_path=True
            )
            logger.info("Additional downloads processed successfully")

        except Exception as e:
            logger.error("Error handling additional downloads: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())

    @staticmethod
    def get_current_job_status(results_dir):
        """Finds 'status.json' under specs['results_dir'] and returns the last entry's status."""
        if "://" in results_dir:
            bucket_name = results_dir.split("//")[1].split("/")[0]
            results_dir = results_dir[results_dir.find(bucket_name) + len(bucket_name):]

        if not results_dir:
            cleanup_cuda_contexts()
            raise ValueError("Empty 'results_dir' in specs.")
        if not os.path.isdir(results_dir):
            logger.error("results_dir directory %s does not exist", results_dir)
            return "Pending"

        file_path = ContainerJobHandler.get_status_file(results_dir)
        last_status = None

        if not os.path.exists(file_path):
            return "Pending"

        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    last_status = data.get("status")
                except json.JSONDecodeError:
                    continue

        return {
            "STARTED": "Running",
            "RUNNING": "Running",
            "SUCCESS": "Done",
            "FAILURE": "Error",
        }.get(last_status, "Pending")

    @staticmethod
    def find_files_by_patterns(base_dir, patterns):
        """Find files and directories matching the given patterns.

        Args:
            base_dir (str): Base directory to search in
            patterns (list): List of glob patterns to match

        Returns:
            list: List of file/directory paths that match the patterns
        """
        matched_paths = set()

        for pattern in patterns:
            # Convert the pattern to work with os.walk
            search_path = os.path.join(base_dir, pattern)

            # Use glob to find matching paths
            for match in glob.glob(search_path, recursive=True):
                if os.path.exists(match):
                    matched_paths.add(match)

        return list(matched_paths)

    @staticmethod
    def create_selective_tarball(results_dir, patterns, base_path, job_id, action_name):
        """Create a tarball containing only files matching the specified patterns.

        Args:
            results_dir (str): Results directory
            patterns (list): List of glob patterns to include in tarball
            base_path (str): Base path within results_dir to apply patterns
            job_id (str): Job ID for naming the tarball
            action_name (str): Action name for naming the tarball

        Returns:
            str: Path to created tarball or None if failed
        """
        try:
            search_dir = os.path.join(results_dir, base_path) if base_path else results_dir
            if not os.path.exists(search_dir):
                logger.warning("Search directory does not exist: %s", search_dir)
                return None

            # Find files matching patterns
            matched_files = ContainerJobHandler.find_files_by_patterns(search_dir, patterns)

            if not matched_files:
                logger.info("No files found matching patterns: %s", patterns)
                return None

            tarball_name = f"{action_name}_selective.tar.gz"
            tarball_path = os.path.join(results_dir, tarball_name)

            logger.info("Creating selective tarball with %d matching files/directories", len(matched_files))

            # Create tarball with only matching files
            with tarfile.open(tarball_path, 'w:gz') as tar:
                for file_path in tqdm(matched_files, desc="Adding files to tarball"):
                    # Calculate relative path for archive
                    if os.path.isfile(file_path):
                        rel_path = os.path.relpath(file_path, results_dir)
                        tar.add(file_path, arcname=rel_path)

            logger.info("Selective tarball created successfully: %s", tarball_path)
            return tarball_path

        except Exception as e:
            logger.error("Error creating selective tarball: %s", str(e))
            logger.error("Traceback: %s", traceback.format_exc())
            return None

    @staticmethod
    def get_upload_strategy_from_config(network, action):
        """Get upload strategy and exclude patterns by reading network config directly.

        Args:
            network (str): Network name
            action (str): Action name

        Returns:
            tuple: (upload_strategy, exclude_patterns) where upload_strategy is dict or str,
                   and exclude_patterns is list or None
        """
        try:
            network_config = read_network_config(network)
            if network_config and "cloud_upload" in network_config:
                cloud_upload_config = network_config["cloud_upload"]
                strategy = cloud_upload_config.get("upload_strategy", {}).get(action, "continuous")
                exclude_patterns = cloud_upload_config.get("exclude_patterns", {}).get(action)
                return strategy, exclude_patterns
            return "continuous", None  # Default to continuous if not specified
        except Exception as e:
            logger.error("Error reading upload strategy from network config: %s", str(e))
            return "continuous", None
