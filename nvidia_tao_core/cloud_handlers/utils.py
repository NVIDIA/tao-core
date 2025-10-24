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

"""Utility functions for Cloud Storage handler"""
import ast
import fnmatch
import glob
import logging
import os
import re
import requests
import subprocess
import sys
import tarfile
import time
import traceback
from huggingface_hub import snapshot_download

from nvidia_tao_core.microservices.handlers.cloud_storage import CloudStorage
from nvidia_tao_core.microservices.handlers.ngc_handler import download_ngc_model, split_ngc_path
from nvidia_tao_core.microservices.handlers.nvcf_handler import invoke_function
from nvidia_tao_core.microservices.handlers.stateless_handlers import get_internal_job_status_update_data
from nvidia_tao_core.distributed.decorators import master_node_only


logger = logging.getLogger(__name__)
NUM_RETRY = 3
REQUESTS_TIMEOUT = 180


def _untar_file(tar_path, dest, strip_components=0):
    """Function to untar a file.

    Args:
        tar_path (str): The path to the tar file to be untarred.
        dest (str): The destination directory where the contents will be extracted.
        strip_components (int, optional): The number of leading directory components to strip (default is 0).
    """
    os.makedirs(dest, exist_ok=True)
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar.getmembers():
            # Remove leading directory components using strip_components
            components = member.name.split(os.sep)
            if len(components) > strip_components:
                member.name = os.path.join(*components[strip_components:])
            if member.isdir():
                # Make subdirs ahead because tarfile extracts them with user permissions only
                os.makedirs(os.path.join(dest, member.name), exist_ok=True)
            tar.extract(member, path=dest, set_attrs=False)


def _extract_images(tar_path, dest):
    """Function to extract images, other directories on same level as images to root of dataset.

    Args:
        tar_path (str): The path to the tar file.
        dest (str): The destination directory where the contents will be extracted.
    """
    # Infer how many components to strip to get images,labels to top of dataset directory
    # Assumes: images, other necessary directories are in the same level
    with tarfile.open(tar_path) as tar:
        strip_components = 0
        names = [tinfo.name for tinfo in tar.getmembers()]
        for name in names:
            if "/images/" in name:
                strip_components = name.split("/").index("images")
                break
    # Build shell command for untarring
    logger.info("Untarring data started")
    _untar_file(tar_path, dest, strip_components)
    logger.info("Untarring data complete")

    # Remove .tar.gz file
    logger.info("Removing data tar file")
    os.remove(tar_path)
    logger.info("Deleted data tar file")


def search_for_ptm(root, network="", parameter_name=""):
    """Return path of the PTM file under the PTM root folder"""
    models = None
    models = (
        glob.glob(root + "/**/*.tlt", recursive=True) +
        glob.glob(root + "/**/*.hdf5", recursive=True) +
        glob.glob(root + "/**/*.pth", recursive=True) +
        glob.glob(root + "/**/*.pth.tar", recursive=True) +
        glob.glob(root + "/**/*.pt", recursive=True)
    )
    # TODO: remove after next nvaie release, Varun and Subha
    if network in ("classification_pyt", "visual_changenet", "nvdinov2"):
        models += glob.glob(root + "/**/*.ckpt", recursive=True)
    if network in ("classification_tf2", "efficientdet_tf2"):
        models = [os.path.join(root, os.listdir(root)[0])]
    if network == "stylegan_xl":
        if parameter_name == "inception_fid_path":
            models = glob.glob(root + "/**/*Inception*.pth", recursive=True)
        if parameter_name == "input_embeddings_path":
            models = glob.glob(root + "/**/*tf_efficientnet*.pth", recursive=True)
    if models:
        model_path = models[0]  # pick one arbitrarily
        logger.info("Found valid PTM at {}".format(model_path)) # noqa pylint: disable=C0209
        return model_path
    if os.path.exists(root):
        if network == "vila":
            return os.path.join(root, os.listdir(root)[0])
        return root
    logger.info("PTM can't be found")
    return None


def run_subprocess_command(command):
    """Run a subprocess command.

    Args:
        command (str): The command to run.
    """
    try:
        # Run the script.
        subprocess.check_call(
            ['/bin/bash', '-c', command],
            shell=False,
            stdout=sys.stdout,
            stderr=sys.stdout
        )
    except (KeyboardInterrupt, SystemExit) as e:
        logger.info("Command was interrupted due to {}".format(e)) # noqa pylint: disable=C0209
    except subprocess.CalledProcessError as e:
        if e.output is not None:
            logger.info("Called process error {}".format(e.output)) # noqa pylint: disable=C0209


def download_from_https_link(download_url, destination_folder):
    """Download a file from an HTTP link.

    Args:
        download_url (str): The URL of the file to download.
        destination_folder (str): The destination folder where the file will be saved.
    """
    cmnd = (
        f"until wget --timeout=1 --tries=1 --retry-connrefused "
        f"--no-verbose --directory-prefix={destination_folder}/ {download_url}; "
        f"do sleep 10; done"
    )
    run_subprocess_command(cmnd)
    file_name = download_url.split("/")[-1]
    if file_name.endswith(".tar") or file_name.endswith(".tar.gz"):
        _extract_images(f'{destination_folder}/{file_name}', f'{destination_folder}')


def download_huggingface_dataset(download_url, destination_folder, token):
    """Download a dataset from the Hugging Face dataset hub.

    Args:
        download_url (str): The URL of the dataset on the Hugging Face dataset hub.
        destination_folder (str): The destination folder where the dataset will be saved.
        token (str): The token for accessing private datasets (optional).
    """
    if token:
        match = re.match(r"https://huggingface.co/datasets/([^/]+)/", download_url)
        username = ""
        if match:
            username = match.group(1)
        cmnd = f"git clone https://{username}:{token}@{download_url.replace('https://', '')} {destination_folder}"
    else:
        cmnd = f"git clone {download_url} {destination_folder}"
    run_subprocess_command(cmnd)


def download_huggingface_model(download_url, destination_folder, token):
    """Download a model from the Hugging Face model hub.

    Args:
        download_url (str): The URL of the model on the Hugging Face model hub.
        destination_folder (str): The destination folder where the model will be saved.
        token (str): The token for accessing private models (optional).
    """
    if token:
        model_dir = snapshot_download(
            repo_id=download_url,
            local_dir=destination_folder,
            token=token
        )
    else:
        model_dir = snapshot_download(
            repo_id=download_url,
            local_dir=destination_folder)
    return model_dir


def extract_cloud_details(metadata):
    """Extract cloud-related details from metadata.

    Args:
        metadata (dict): Metadata dictionary.

    Returns:
        Tuple: Cloud details (cloud_type, bucket_name, access_key, secret_key, region, cloud_file_path).
    """
    cloud_type = metadata.get('cloud_type', '')

    # AWS, AZURE
    bucket_name = metadata.get('cloud_specific_details', {}).get('cloud_bucket_name', '')
    access_key = metadata.get('cloud_specific_details', {}).get('access_key', '')
    secret_key = metadata.get('cloud_specific_details', {}).get('secret_key', '')
    region = metadata.get('cloud_specific_details', {}).get('cloud_region', '')

    # Self_hosted, HuggingFace
    download_url = metadata.get('cloud_specific_details', {}).get('url', '')

    # HuggingFace
    token = metadata.get('cloud_specific_details', {}).get('token', '')

    return cloud_type, bucket_name, access_key, secret_key, region, download_url, token


def initialize_cloud_storage(cloud_type, bucket_name, region, access_key, secret_key, endpoint_url=None):
    """Initialize CloudStorage instance.

    Args:
        cloud_type (str): Type of cloud storage.
        bucket_name (str): Name of the bucket/container.
        region (str): Region for the cloud storage provider.
        access_key (str): Access key for authentication.
        secret_key (str): Secret key for authentication.
        endpoint_url (str): Endpoint URL for the cloud storage provider.

    Returns:
        CloudStorage: Initialized CloudStorage instance.
    """
    # Prepare client_kwargs - only include endpoint_url if it's provided
    client_kwargs = {}
    if endpoint_url and endpoint_url != "":
        client_kwargs["endpoint_url"] = endpoint_url

    return CloudStorage(
        cloud_type=cloud_type,
        bucket_name=bucket_name,
        region=region,
        key=access_key,
        secret=secret_key,
        client_kwargs=client_kwargs
    )


def search_for_dataset(root):
    """Return path of the dataset file"""
    datasets = (
        glob.glob(root + "/*.tar.gz", recursive=False) +
        glob.glob(root + "/*.tgz", recursive=False) +
        glob.glob(root + "/*.tar", recursive=False)
    )

    if datasets:
        dataset_path = datasets[0]  # pick one arbitrarily
        return dataset_path
    return None


def download_files(cloud_storage, cloud_file_path, local_path):
    """Download files from cloud storage.

    Args:
        cloud_storage (CloudStorage): Initialized CloudStorage instance.
        cloud_file_path (str): Path of the file in cloud storage.
        local_path (str): Local path for downloading the file.
    """
    if not local_path.endswith("/"):
        local_path = f"{local_path}/"
    cloud_storage.download_file(cloud_file_path, local_path)
    if cloud_file_path.endswith(".tar") or cloud_file_path.endswith(".tar.gz"):
        _extract_images(os.path.join(local_path, os.path.basename(cloud_file_path)), local_path)
        dir_name = cloud_file_path.split("/")[-1].split(".")[0]
        destination_path = f"{local_path}{dir_name}"
        if not os.path.exists(destination_path):
            cleanup_cuda_contexts()
            raise ValueError("Folder name not same as the file name")
    else:
        file_name = cloud_file_path.split("/")[-1]
        destination_path = f"{local_path}{file_name}"
        if not os.path.isfile(destination_path):
            cleanup_cuda_contexts()
            raise ValueError("Unable to download the file")
    return destination_path


def get_file_modification_time(local_path):
    """Gets file modification time and ignores any issue in getting so.

    Args:
        local_path (str): The local path to monitor.
        cloud_storage: An instance of the CloudStorage class for uploading files.
        file_last_modified: Dictionary to find modified files
    """
    try:
        return os.path.getmtime(local_path)
    except Exception:
        return 0


@master_node_only
def upload_files(local_path, cloud_storage, file_last_modified, selective_tarball_config=None, exclude_patterns=None):
    """Uploads any detected changes to the specified cloud storage.

    Args:
        local_path (str): The local path to monitor.
        cloud_storage: An instance of the CloudStorage class for uploading files.
        file_last_modified: Dictionary to find modified files
        selective_tarball_config (dict): Configuration for selective tarball patterns to skip.
        exclude_patterns (list, optional): List of regex patterns to exclude files from upload.
    """
    for root, _, files in os.walk(local_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            current_last_modified = get_file_modification_time(file_path)
            if current_last_modified:

                # Check if the file is new or modified
                if (
                    file_path not in file_last_modified or
                    current_last_modified > file_last_modified[file_path]
                ) and ("checkpoint-" not in file_path and "tmp" not in file_path):

                    # Check exclude_patterns
                    should_exclude = False
                    if exclude_patterns:
                        rel_path = os.path.relpath(file_path, local_path)
                        for pattern in exclude_patterns:
                            try:
                                if re.search(pattern, rel_path) or re.search(pattern, filename):
                                    should_exclude = True
                                    logger.debug("Excluding file from upload due to pattern '%s': %s",
                                                 pattern, rel_path)
                                    break
                            except re.error:
                                logger.warning("Invalid regex pattern '%s', skipping", pattern)

                    if should_exclude:
                        # Update modification time but don't upload
                        file_last_modified[file_path] = current_last_modified
                        continue

                    # Skip files that will be included in selective tarball
                    if should_skip_file_for_tarball(file_path, local_path, selective_tarball_config):
                        # Update modification time but don't upload
                        file_last_modified[file_path] = current_last_modified
                        continue

                    logger.info("File event created/modified {}".format(file_path))  # noqa pylint: disable=C0209
                    try:
                        time.sleep(10)
                        cloud_storage.upload_file(file_path, file_path)
                    except Exception as e:  # pylint: disable=broad-except
                        logger.error(
                            "Failed to upload file: {} - Error: {}".format(  # noqa pylint: disable=C0209
                                file_path, str(e)
                            )
                        )
                    # Remove file after successful upload only if size > 50MB
                    try:
                        if cloud_storage.is_file(file_path):
                            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                            if file_size_mb > 50:
                                os.remove(file_path)
                                logger.info(
                                    "Large file (%.2f MB) successfully uploaded and removed: %s",
                                    file_size_mb, file_path
                                )
                            else:
                                logger.info(
                                    "File (%.2f MB) successfully uploaded but retained (under 50MB): %s",
                                    file_size_mb, file_path
                                )
                    except Exception as e:  # pylint: disable=broad-except
                        logger.error(
                            "Failed to remove file after upload: {} - Error: {}".format(  # noqa pylint: disable=C0209
                                file_path, str(e)
                            )
                        )

                    # Update the last modification time for the file
                    file_last_modified[file_path] = current_last_modified
            else:
                logger.error("File could not be uploaded: %s", file_path)


def get_log_file_name():
    """Return log file name"""
    job_id = os.getenv("JOB_ID")
    logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG', '/results')
    log_file = f'{logs_dir}/{job_id}/microservices_log.txt'
    return log_file


def send_logs_to_server(seek_position, retry=0):
    """Sends TTY logs back to Hosted API"""
    if os.getenv("CLOUD_BASED") == "True":
        if retry >= NUM_RETRY:
            cleanup_cuda_contexts()
            raise ValueError("Log Callback was unsuccessfull")

        log_file = get_log_file_name()

        if os.path.isfile(log_file):
            with open(log_file, 'r', encoding='utf-8') as log_file:
                log_file.seek(seek_position)
                log_contents = log_file.read()
                seek_position = log_file.tell()
                ngc_key = os.getenv("TAO_USER_KEY")
                headers = {"Authorization": f"Bearer {ngc_key}"}
                if log_contents and headers:
                    nvcf_helm_deployment = os.getenv("NVCF_HELM")
                    log_callback_url = os.getenv("TAO_LOGGING_SERVER_URL") + ":log_update"
                    if log_callback_url:
                        headers['Content-Type'] = 'application/json'
                        data = {
                            'experiment_number': os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                            'log_contents': log_contents
                        }
                        if not nvcf_helm_deployment:
                            try:
                                response = requests.post(
                                    log_callback_url,
                                    json=data,
                                    headers=headers,
                                    timeout=REQUESTS_TIMEOUT
                                )
                                if response.ok:
                                    return seek_position
                                logger.info(
                                    "Failed to send logs. Status code: {}".format(response.status_code)  # noqa pylint: disable=C0209
                                )
                                seek_position -= len(log_contents)
                                retry += 1

                            except requests.RequestException as e:
                                logger.info("Exception during log sending: {}".format(e))  # noqa pylint: disable=C0209
                                seek_position -= len(log_contents)
                                retry += 1

                            time.sleep(5)
                            return send_logs_to_server(seek_position, retry)
    return seek_position


def status_callback(data_string, retry=0):
    """Sends status update data back to the server.

    Args:
        data_string (str): The status data to be sent.
        retry (int, optional): The current retry attempt (default is 0).
    """
    if os.getenv("CLOUD_BASED") == "True":
        if retry >= NUM_RETRY:
            cleanup_cuda_contexts()
            raise ValueError("Status Callback was unsuccessful after multiple retries")

        ngc_key = os.getenv("TAO_ADMIN_KEY")
        headers = {"Authorization": f"Bearer {ngc_key}"}
        if data_string and headers:
            status_url = os.getenv("TAO_LOGGING_SERVER_URL", "") + ":status_update"
            if status_url:
                data = {
                    "experiment_number": os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                    "status": data_string,
                }
                nvcf_helm_deployment = os.getenv("NVCF_HELM")
                if nvcf_helm_deployment:
                    url_parts = os.getenv("TAO_LOGGING_SERVER_URL", "").split('/')
                    # Extract kind, handler_id, and job_id based on their positions
                    kind = url_parts[7]
                    handler_id = url_parts[8]
                    job_id = url_parts[10]
                    docker_env_vars = {
                        "TAO_USER_KEY": ngc_key,
                    }
                    invoke_function(
                        deployment_string=nvcf_helm_deployment,
                        microservice_action="status_update",
                        docker_env_vars=docker_env_vars,
                        kind=kind,
                        handler_id=handler_id,
                        job_id=job_id,
                        request_body=data,
                    )
                else:
                    try:
                        response = requests.post(status_url, json=data, headers=headers, timeout=REQUESTS_TIMEOUT)
                        if response.ok:
                            logger.info(f"Status update with data {data} sent successfully")
                            return
                        logger.error(
                            "Failed to send status update. Status code: {}".format(  # noqa pylint: disable=C0209
                                response.status_code
                            )
                        )
                        retry += 1

                    except requests.RequestException as e:
                        logger.error(
                            "Exception during status update sending: {}".format(e)  # noqa pylint: disable=C0209
                        )
                        retry += 1

                    time.sleep(5)
                    status_callback(data_string, retry)


def should_skip_file_for_tarball(file_path, local_path, selective_tarball_config):
    """Check if a file should be skipped because it will be included in selective tarball.

    Args:
        file_path (str): Full path to the file
        local_path (str): Base local path
        selective_tarball_config (dict): Selective tarball configuration

    Returns:
        bool: True if file should be skipped, False otherwise
    """
    if not selective_tarball_config:
        return False

    patterns = selective_tarball_config.get("patterns", [])
    base_path = selective_tarball_config.get("base_path", "")

    if not patterns:
        return False

    # Calculate the search directory
    search_dir = os.path.join(local_path, base_path) if base_path else local_path

    # Check if file is within the search directory
    if not file_path.startswith(search_dir):
        return False

    # Get relative path from search directory
    rel_path = os.path.relpath(file_path, search_dir)

    # Check if file matches any pattern
    for pattern in patterns:
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(file_path, os.path.join(search_dir, pattern)):
            logger.debug("Skipping continuous upload for tarball file: %s (matches pattern: %s)", rel_path, pattern)
            return True

    return False


@master_node_only
def monitor_and_upload(local_path, cloud_storage, exit_event, seek_position=0,
                       selective_tarball_config=None, exclude_patterns=None):
    """Monitors the specified local path and its subdirectories for new or modified files.

    Args:
        local_path (str): The local path to monitor.
        cloud_storage: An instance of the CloudStorage class for uploading files.
        exit_event (threading.Event): An event to signal the thread to exit.
        seek_position (int): Initial seek position for log reading.
        selective_tarball_config (dict): Configuration for selective tarball patterns to skip.
        exclude_patterns (list, optional): List of regex patterns to exclude files from upload.

    Returns:
        None
    """
    logger.info("monitor_and_upload :: Entering")
    if selective_tarball_config:
        patterns = selective_tarball_config.get("patterns", [])
        base_path = selective_tarball_config.get("base_path", "")
        logger.info("Selective tarball enabled - skipping patterns: %s in base_path: %s", patterns, base_path)
    if exclude_patterns:
        logger.info("Exclude patterns enabled for continuous upload: %s", exclude_patterns)
    file_last_modified = {}

    # Initialize file_last_modified with files that are already part of results dir
    for root, _, files in os.walk(local_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_last_modified[file_path] = os.path.getmtime(file_path)

    try:
        while True:
            upload_files(local_path, cloud_storage, file_last_modified, selective_tarball_config, exclude_patterns)
            seek_position = send_logs_to_server(seek_position)
            if exit_event.is_set():
                upload_files(local_path, cloud_storage, file_last_modified, selective_tarball_config, exclude_patterns)
                seek_position = send_logs_to_server(seek_position)
                break
            time.sleep(30)  # Adjust the sleep interval as needed

    except (KeyboardInterrupt, SystemExit, Exception):
        logger.error("traceback: %s", traceback.format_exc())
        exit_event.set()


def get_file_path_from_cloud_string(value):
    """Get the cloud storage class object from the value"""
    csp_provider = value.split(":")[0]
    bucket_name = value.split("//")[1].split("/")[0]
    cloud_file_path = value[value.find(bucket_name) + len(bucket_name):]
    return csp_provider, bucket_name, cloud_file_path


def get_cloud_storage_class_object(cloud_data, cloud_string):
    """Initalize Apache LibCloud class"""
    csp_provider, bucket_name, cloud_file_path = get_file_path_from_cloud_string(cloud_string)
    cloud_storage = initialize_cloud_storage(
        cloud_type=csp_provider,
        bucket_name=bucket_name,
        region=cloud_data[csp_provider][bucket_name].get("region"),
        access_key=cloud_data[csp_provider][bucket_name].get("access_key"),
        secret_key=cloud_data[csp_provider][bucket_name].get("secret_key"),
        endpoint_url=cloud_data[csp_provider][bucket_name].get("endpoint_url")
    )
    while cloud_file_path.find("//") != -1:
        cloud_file_path = cloud_file_path.replace("//", "/")
    return cloud_storage, cloud_file_path


def download_from_user_storage(
    cloud_storage=None, job_id="", cloud_data={}, value="", dictionary={}, key="",
    preserve_source_path=False, reset_value=False
):
    """Download a file/folder from user storage"""
    try:
        if not cloud_storage:
            cloud_storage, cloud_file_path = get_cloud_storage_class_object(cloud_data, value)
        else:
            cloud_file_path = value

        local_path_of_dataset_file = f"/results/{job_id}/{cloud_file_path}"
        if preserve_source_path:
            local_path_of_dataset_file = cloud_file_path
        if reset_value:
            # Update the dictionary value with the local path
            if dictionary and key:
                dictionary[key] = local_path_of_dataset_file.replace(".tar.gz", "")
        destination_path = local_path_of_dataset_file
        if cloud_file_path.startswith("/"):
            cloud_file_path = cloud_file_path[1:]

        # Create destination directory
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        if cloud_storage.is_file(cloud_file_path):
            cloud_storage.download_file(cloud_file_path, destination_path)
            if cloud_file_path.endswith(".tar") or cloud_file_path.endswith(".tar.gz"):
                _extract_images(destination_path, os.path.dirname(destination_path))
        else:
            cloud_storage.download_folder(cloud_file_path, destination_path)
            for root, _, files in os.walk(destination_path):
                for file in files:
                    abs_filepath = os.path.join(root, file)
                    if abs_filepath.endswith(".tar") or abs_filepath.endswith(".tar.gz"):
                        _extract_images(abs_filepath, os.path.dirname(abs_filepath))

        logger.info("Downloaded: {}".format(cloud_file_path))  # noqa pylint: disable=C0209
        return local_path_of_dataset_file.replace(".tar.gz", "")
    except Exception as e:
        logger.error("Error downloading cloud file: %s", str(e))
        logger.error(traceback.format_exc())
        callback_data = get_internal_job_status_update_data(
            automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
            message=f"Error downloading cloud file {value}"
        )
        status_callback(callback_data)
        raise e


def download_files_from_cloud(
    cloud_data,
    dictionary,
    key,
    value,
    job_id,
    network_arch,
    ngc_key,
    reset_value=False,
    preserve_source_path=False
):
    """Based on the cloud dype, download the file"""
    if "'link': 'https://" in value:
        try:
            https_dictionary = ast.literal_eval(value)
            link = https_dictionary.get("link", "")
            destination_path = https_dictionary.get("destination_path", f"/ptm/{network_arch}/download")
            destination_folder = os.path.dirname(destination_path)
            download_from_https_link(link, destination_folder)
            dictionary[key] = destination_path
            return destination_path
        except Exception as e:
            logger.error("Error downloading public hosted file: %s", str(e))
            logger.error(traceback.format_exc())
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=f"Error downloading public hosted file {value}"
            )
            status_callback(callback_data)
            raise e

    if value.startswith("ngc://"):
        try:
            if not ngc_key:
                cleanup_cuda_contexts()
                raise ValueError("NGC Personal key has not been provided")
            ngc_model = value.split("ngc://")[-1]
            org, team, model_name, model_version = split_ngc_path(ngc_model)
            if not download_ngc_model(
                ngc_model,
                f"/ptm/{org}/{team}/{model_name}/{model_version}/model",
                ngc_key,
            ):
                cleanup_cuda_contexts()
                raise ValueError("Unable to download the PTM")
            ptm_path = search_for_ptm(f"/ptm/{org}/{team}/{model_name}/{model_version}/model", network_arch, key)
            dictionary[key] = ptm_path
            return ptm_path
        except Exception as e:
            logger.error("Error downloading NGC model: %s", str(e))
            logger.error(traceback.format_exc())
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=f"Error downloading NGC model {value}"
            )
            status_callback(callback_data)
            raise e

    if value.startswith("hf_model://"):
        try:
            huggingface_model = value.split("hf_model://")[-1]
            download_huggingface_model(huggingface_model, "/ptm/huggingface_models", os.getenv("HF_TOKEN", ""))
            dictionary[key] = "/ptm/huggingface_models"
            return "/ptm/huggingface_models"
        except Exception as e:
            logger.error("Error downloading Hugging Face model: %s", str(e))
            logger.error(traceback.format_exc())
            callback_data = get_internal_job_status_update_data(
                automl_experiment_number=os.getenv("AUTOML_EXPERIMENT_NUMBER", "0"),
                message=f"Error downloading Hugging Face model {value}"
            )
            status_callback(callback_data)
            raise e

    if "://" in value:
        return download_from_user_storage(
            cloud_data=cloud_data,
            value=value,
            job_id=job_id,
            dictionary=dictionary,
            key=key,
            preserve_source_path=preserve_source_path,
            reset_value=reset_value
        )
    return None


def download_files_from_spec(
    cloud_data,
    data,
    job_id,
    network_arch=None,
    ngc_key=None,
    reprocess_files=None,
    preserve_source_path=False,
    preserve_source_path_params=None,
    current_path=""
):
    """Recursively download files from a nested dictionary."""
    if preserve_source_path_params is None:
        preserve_source_path_params = set()
    if isinstance(data, dict):
        for key, value in data.items():
            # Build the current parameter path
            new_path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, dict):
                download_files_from_spec(
                    cloud_data,
                    value,
                    job_id,
                    network_arch=network_arch,
                    ngc_key=ngc_key,
                    reprocess_files=reprocess_files,
                    preserve_source_path=preserve_source_path,
                    preserve_source_path_params=preserve_source_path_params,
                    current_path=new_path
                )
            elif isinstance(value, list):
                override_list = []
                for list_element in value:
                    if isinstance(list_element, str):
                        # Check if this parameter should preserve source path
                        param_preserve_source_path = preserve_source_path or new_path in preserve_source_path_params
                        override_value = download_files_from_cloud(
                            cloud_data,
                            data,
                            key,
                            list_element,
                            job_id,
                            network_arch,
                            ngc_key,
                            preserve_source_path=param_preserve_source_path
                        )
                        if not override_value:
                            override_value = list_element
                        if (reprocess_files is not None and override_value and
                                (list_element.endswith(".yaml") or list_element.endswith(".json"))):
                            reprocess_files.append(override_value)
                        override_list.append(override_value)
                    elif isinstance(list_element, dict):
                        override_dict = {}
                        for list_dict_key, list_dict_value in list_element.items():
                            if isinstance(list_dict_value, str):
                                # Check if this parameter should preserve source path
                                param_preserve_source_path = (
                                    preserve_source_path or new_path in preserve_source_path_params
                                )
                                override_value = download_files_from_cloud(
                                    cloud_data,
                                    data,
                                    key,
                                    list_dict_value,
                                    job_id,
                                    network_arch,
                                    ngc_key,
                                    preserve_source_path=param_preserve_source_path
                                )
                                if (reprocess_files is not None and override_value and
                                        (list_dict_value.endswith(".yaml") or list_dict_value.endswith(".json"))):
                                    reprocess_files.append(override_value)
                                if not override_value:
                                    override_value = list_dict_value
                            else:
                                override_value = list_dict_value
                            override_dict[list_dict_key] = override_value
                        override_list.append(override_dict)
                    else:
                        override_list.append(list_element)
                data[key] = override_list
            else:
                if isinstance(value, str):
                    # Check if this parameter should preserve source path
                    param_preserve_source_path = preserve_source_path or new_path in preserve_source_path_params
                    override_value = download_files_from_cloud(
                        cloud_data,
                        data,
                        key,
                        value,
                        job_id,
                        network_arch,
                        ngc_key,
                        reset_value=True,
                        preserve_source_path=param_preserve_source_path
                    )
                    if (reprocess_files is not None and override_value and
                            (value.endswith(".yaml") or value.endswith(".json"))):
                        reprocess_files.append(override_value)


def get_results_cloud_data(cloud_data, spec_data, dest_dir=None):
    """Obtain the CloudStorage instance for uploading the results.

    Args:
    cloud_data: Cloud storage metadata.
    spec_data: Model spec.
    dest_dir: Directory where the results directorys is present.

    Returns:
    cloud_storage: CloudStorage instance.
    spec_data: Updated model spec.
    """
    results_dir = spec_data["results_dir"]
    if "://" in results_dir:
        cloud_storage, cloud_file_path = get_cloud_storage_class_object(cloud_data, results_dir)
        spec_data["results_dir"] = cloud_file_path
        return cloud_storage, spec_data
    if not dest_dir:
        cleanup_cuda_contexts()
        raise ValueError("Destination directory is not provided")
    spec_data["results_dir"] = f'{dest_dir}/{spec_data["results_dir"]}'
    return None, spec_data


def create_tarball(source_dir, tarball_path, exclude_paths=None, exclude_patterns=None):
    """Create a tarball from the source directory.

    Args:
        source_dir (str): Directory to tarball
        tarball_path (str): Path where the tarball will be created
        exclude_paths (set, optional): Set of relative paths to exclude from tarball
        exclude_patterns (list, optional): List of regex patterns to exclude files from tarball

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        exclusion_info = []
        if exclude_paths:
            exclusion_info.append(f"{len(exclude_paths)} pre-existing items")
        if exclude_patterns:
            exclusion_info.append(f"files matching patterns: {exclude_patterns}")
        if exclusion_info:
            logger.info("Creating tarball excluding %s: %s from source: %s",
                        ", ".join(exclusion_info), tarball_path, source_dir)
        else:
            logger.info("Creating tarball: %s from source: %s", tarball_path, source_dir)

        with tarfile.open(tarball_path, 'w:gz') as tar:
            if exclude_paths or exclude_patterns:
                # Walk through directory and selectively add files
                files_added = 0
                files_excluded = 0

                for root, dirs, files in os.walk(source_dir):
                    # Process files
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, source_dir)
                        # Check if file should be excluded
                        should_exclude = False
                        # Check exclude_paths
                        if exclude_paths and rel_path in exclude_paths:
                            should_exclude = True
                        # Check exclude_patterns
                        if exclude_patterns and not should_exclude:
                            for pattern in exclude_patterns:
                                try:
                                    if re.search(pattern, rel_path) or re.search(pattern, file):
                                        should_exclude = True
                                        break
                                except re.error:
                                    logger.warning("Invalid regex pattern '%s', skipping", pattern)

                        if not should_exclude:
                            tar.add(file_path, arcname=rel_path)
                            files_added += 1
                        else:
                            files_excluded += 1

                    # Process directories (only add empty directories that weren't in exclusion set)
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        rel_path = os.path.relpath(dir_path, source_dir)

                        # Check if directory is empty and wasn't in exclusion set
                        # Empty directory not in exclusion set
                        if (not exclude_paths or rel_path not in exclude_paths) and not os.listdir(dir_path):
                            tar.add(dir_path, arcname=rel_path)

                logger.info("Tarball created successfully: %s (%d files added, %d files excluded)",
                            tarball_path, files_added, files_excluded)
            else:
                # Original behavior - add entire directory
                tar.add(source_dir, arcname=os.path.basename(source_dir))
                logger.info("Tarball created successfully: %s", tarball_path)

        return True
    except Exception as e:
        logger.error("Error creating tarball: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        return False


@master_node_only
def upload_tarball_to_cloud(cloud_storage, tarball_path, remove_after_upload=True):
    """Upload a tarball to cloud storage.

    Args:
        cloud_storage: CloudStorage instance
        tarball_path (str): Path to the tarball to upload
        remove_after_upload (bool): Whether to remove the local tarball after upload

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Uploading tarball to cloud: %s", tarball_path)

        cloud_storage.upload_file(tarball_path, tarball_path)
        logger.info("Tarball uploaded successfully: %s", tarball_path)

        if remove_after_upload and os.path.exists(tarball_path):
            os.remove(tarball_path)
            logger.info("Local tarball removed: %s", tarball_path)

        return True
    except Exception as e:
        logger.error("Error uploading tarball: %s", str(e))
        logger.error("Traceback: %s", traceback.format_exc())
        return False


def cleanup_cuda_contexts():
    """Clean up any stale CUDA contexts.

    Call this function in cleanup routines or when throwing exceptions.
    """
    try:
        # Check if PyCUDA is imported in this environment
        pycuda_imported = "pycuda" in sys.modules
        if pycuda_imported:
            import pycuda.driver as cuda

            # Clean up any active contexts
            if cuda.Context.get_current() is not None:
                try:
                    # Pop the current context
                    logger.info("Found active CUDA context. Cleaning up...")
                    cuda.Context.pop()
                    logger.info("Active CUDA context cleaned up!")
                except Exception as e:
                    logger.warning(f"Error cleaning up CUDA context: {str(e)}")
    except Exception as e:
        logger.warning(f"Unexpected error during CUDA cleanup: {str(e)}")
