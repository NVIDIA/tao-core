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

"""API Stateless handlers modules"""
import json
import os
import sys
import shutil
import requests
import logging
from ngcbase import errors

from nvidia_tao_core.microservices.handlers.encrypt import NVVaultEncryption
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    get_handler_metadata, get_jobs_root, get_user, get_workspace_string_identifier
)
from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance
from nvidia_tao_core.microservices.utils import (
    sha256_checksum, read_network_config,
    retry_method, get_admin_key
)

NUM_OF_RETRY = 3
TIMEOUT = 120

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorResponse:
    """Custom error response object"""

    def __init__(self, status_code):
        """Init function for ErrorResponse class"""
        self.status_code = status_code
        self.ok = False


@retry_method(response=True)
def send_ngc_api_request(endpoint, requests_method, request_body, json=False, ngc_key="", accept_encoding="identity"):
    """Send NGC API requests with token refresh, retries, and timeout handling"""
    headers = {"Authorization": f"Bearer {ngc_key}"}
    if accept_encoding:
        headers['Accept-Encoding'] = accept_encoding
    if requests_method == "POST":
        if json:
            headers['accept'] = 'application/json'
            headers['Content-Type'] = 'application/json'
        response = requests.post(url=endpoint, data=request_body, headers=headers, timeout=TIMEOUT)
    elif requests_method == "GET":
        response = requests.get(url=endpoint, headers=headers, timeout=TIMEOUT)
    elif requests_method == "DELETE":
        response = requests.delete(url=endpoint, headers=headers, timeout=TIMEOUT)
    else:
        raise ValueError(f"Unsupported request method: {requests_method}")
    return response


def split_ngc_path(ngc_path):
    """Split ngc path into org, team and model name, model version"""
    path_split = ngc_path.replace("/no-team", "").split("/")
    if len(path_split) == 3:
        org, team, model_name = path_split
    elif len(path_split) == 2:
        org, model_name = path_split
        team = ""
    else:
        raise ValueError(f"Invalid ngc_path: {ngc_path}")
    if ":" in model_name:
        model_name, model_version = model_name.split(":")
    else:
        model_version = ""
    return org, team, model_name, model_version


def get_user_key(user_id, org_name, admin_key_override=False) -> str:
    """Return user API key"""
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        return ""

    if admin_key_override and os.getenv("USE_ADMIN_KEY", "").lower() == "true":
        return get_admin_key()

    ngc_user_details = get_user(user_id)
    encrypted_ngc_key = ngc_user_details.get("key", {}).get(org_name, "")

    # Decrypt the ngc key
    if encrypted_ngc_key:
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path:
            encryption = NVVaultEncryption(config_path)
            if encryption.check_config()[0]:
                return encryption.decrypt(encrypted_ngc_key)
            logger.error("Failed to decrypt NGC key for user=%s org=%s", user_id, org_name)
            return ""

    return encrypted_ngc_key


def get_user_info(ngc_key: str, accept_encoding: str = "identity") -> requests.Response:
    """Get NGC user info from NGC"""
    endpoint = "https://api.ngc.nvidia.com/v2/users/me"

    try:
        response = send_ngc_api_request(
            endpoint=endpoint,
            requests_method="GET",
            request_body={},
            json=True,
            ngc_key=ngc_key,
            accept_encoding=accept_encoding
        )
    except Exception as e:
        print("Exception caught during getting NGC user info", e, file=sys.stderr)
        raise e
    return response


def get_model(org_name, team_name, model_name, ngc_key):
    """Get NGC Model information"""
    endpoint = "https://api.ngc.nvidia.com/v2"

    endpoint += f"/org/{org_name}"
    if team_name:
        endpoint += f"/team/{team_name}"
    endpoint += f"/models/{model_name}"

    response = send_ngc_api_request(
        endpoint=endpoint,
        requests_method="GET",
        request_body={},
        json=True,
        ngc_key=ngc_key
    )

    status_code = response.status_code
    logger.info("get_model %s status code is %s", model_name, status_code)
    if status_code == 200:
        return response.json()
    return None


def create_model(org_name, team_name, handler_metadata, source_file, ngc_key, display_name, description):
    """Create model in ngc private registry if not exist"""
    model = get_model(org_name, team_name, handler_metadata.get("network_arch"), ngc_key)
    if model:
        return 200, "Model already exists"

    """Create model in ngc private registry"""
    endpoint = "https://api.ngc.nvidia.com/v2"

    endpoint += f"/org/{org_name}"
    if team_name:
        endpoint += f"/team/{team_name}"
    endpoint += "/models"

    network = handler_metadata.get("network_arch")
    network_config = read_network_config(network)
    framework = network_config.get("api_params", {}).get("image", "tao-pytorch")

    model_format = os.path.splitext(source_file)[1]

    data = {"application": f"TAO {network}",
            "framework": framework,
            "modelFormat": model_format,
            "name": network,
            "precision": "FP32",
            "shortDescription": description,
            "displayName": display_name,
            }

    response = send_ngc_api_request(
        endpoint=endpoint,
        requests_method="POST",
        request_body=json.dumps(data),
        json=True,
        ngc_key=ngc_key
    )

    status_code = response.status_code
    message = ""
    if status_code != 200:
        message = response.json().get("requestStatus").get("statusDescription")
    return status_code, message


def upload_model(org_name, team_name, handler_metadata, source_files, ngc_key, job_id, job_action):
    """Upload model to ngc private registry"""
    logger.info("Publishing %s", source_files)
    network = handler_metadata.get("network_arch")

    checkpoint_choose_method = handler_metadata.get("checkpoint_choose_method", "best_model")
    epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
    epoch_number = epoch_number_dictionary.get(f"{checkpoint_choose_method}_{job_id}", 0)
    num_epochs_trained = handler_metadata["checkpoint_epoch_number"].get(f"latest_model_{job_id}", 0)
    workspace_id = handler_metadata.get("workspace")

    workspace_identifier = get_workspace_string_identifier(workspace_id, workspace_cache={})

    workspace_metadata = get_handler_metadata(workspace_id, "workspaces")
    cs_instance, _ = create_cs_instance(workspace_metadata)
    jobs_root = get_jobs_root(handler_metadata.get("user_id"), org_name=org_name)
    local_dir = os.path.join(jobs_root, "publish_model_artifacts")
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
    for source_file in source_files:
        cloud_path = source_file[len(workspace_identifier):]
        artifact_name = os.path.basename(cloud_path[:-1] if cloud_path[-1] == '/' else cloud_path)
        local_path = os.path.join(jobs_root, "publish_model_artifacts", artifact_name)
        cs_instance.download_file(cloud_path, local_path)

    target_version = f"{org_name}/{team_name}/{network}:{job_action}_{job_id}_{epoch_number}"

    try:
        os.environ["NGC_CLI_HOME"] = "/tmp/.ngc"
        from ngcsdk import Client  # pylint: disable=C0415
        clt = Client()
        clt.configure(api_key=ngc_key, org_name=org_name, team_name=team_name)
        clt.registry.model.upload_version(target=target_version, source=local_dir, num_epochs=num_epochs_trained)
        clt.clear_config()
    except Exception as e:
        shutil.rmtree(local_dir)
        logger.error("Exception in model_upload: %s, %s", str(e), type(e))
        return 404, str(e)
    shutil.rmtree(local_dir)
    return 200, "Published model into requested org"


def download_ngc_model(ngc_path, ptm_root, key):
    """Download models from NGC model registry or use local models in air-gapped mode.

    Args:
        ngc_path (str): The NGC path to the desired model in the format 'org/team/model:version'.
        ptm_root (str): The directory where the downloaded model will be saved.

    Returns:
        bool: True if the download is successful, False otherwise.
    """
    if ngc_path == "":
        logger.info("Invalid ngc path.")
        return False

    # Check if running in air-gapped mode
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        logger.info("Air-gapped mode detected, using local model registry")
        return _download_local_model(ngc_path, ptm_root)

    # Original NGC download logic
    ngc_configs = ngc_path.split('/')
    org = ngc_configs[0]
    team = ""
    if len(ngc_configs) == 3:
        team = ngc_configs[1]

    # Get access token using k8s admin secret
    if not key:
        logger.info("Personal key is None")
        return False

    # Download model with ngc sdk
    from ngcsdk import Client  # pylint: disable=C0415
    clt = Client()

    try:
        clt.configure(api_key=key, org_name=org, team_name=team)
    except Exception as e:
        if not ("Invalid org" in str(e) or "Invalid team" in str(e) or "doesn't match organization" in str(e)):
            logger.error(
                "Can't configure the passed NGC KEY for Org {}, team {}".format(org, team)  # noqa pylint: disable=C0209
            )
            logger.error("Error: %s", e)
            return False
        msg = (
            "Can't validate the passed NGC KEY for Org {}, team {}, "
            "going to try download without configuring credentials"
        ).format(org, team)
        logger.info(msg)  # noqa pylint: disable=C0209

    try:
        if not os.path.exists(ptm_root):
            os.makedirs(ptm_root, exist_ok=True)
            clt.registry.model.download_version(ngc_path, destination=ptm_root)
            logger.info("Saving base_experiment file to {}".format(ptm_root)) # noqa pylint: disable=C0209
        else:
            logger.info("Base_experiment already present in {}".format(ptm_root)) # noqa pylint: disable=C0209
    except errors.ResourceNotFoundException as e:
        logger.error("Model {} not found. Error: {}".format(ngc_path, e))  # noqa pylint: disable=C0209
        return False
    except errors.NgcException as e:
        logger.error("Failed to download {}. Error: {}".format(ngc_path, e))  # noqa pylint: disable=C0209
        return False
    except Exception as e:
        logger.error("Failed to download {}. Error: {}".format(ngc_path, e))  # noqa pylint: disable=C0209
        return False

    return True


def _download_local_model(ngc_path, ptm_root):
    """Download model from local registry in air-gapped mode.

    Args:
        ngc_path (str): The NGC path to the desired model in the format 'org/team/model:version'.
        ptm_root (str): The directory where the downloaded model will be saved.

    Returns:
        bool: True if the copy is successful, False otherwise.
    """
    try:
        # Parse NGC path
        from nvidia_tao_core.microservices.pretrained_models import split_ngc_path
        org, team, model_version = split_ngc_path(ngc_path)
        model_name, version = model_version.split(':')

        # Get local model registry path
        local_registry = os.getenv("LOCAL_MODEL_REGISTRY", "/shared-storage/models")
        local_model_path = os.path.join(local_registry, org, team, model_name, version)

        if not os.path.exists(local_model_path):
            logger.error(f"Model {ngc_path} not found in local registry at {local_model_path}")
            return False

        # Create destination directory
        os.makedirs(ptm_root, exist_ok=True)

        # Copy model from local registry to PTM root
        for item in os.listdir(local_model_path):
            src = os.path.join(local_model_path, item)
            dst = os.path.join(ptm_root, item)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        logger.info(f"Copied local model {ngc_path} to {ptm_root}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy local model {ngc_path}. Error: {e}")
        return False


def delete_model(org_name, team_name, handler_metadata, ngc_key, job_id, job_action):
    """Delete model from ngc registry"""
    network = handler_metadata.get("network_arch")

    checkpoint_choose_method = handler_metadata.get("checkpoint_choose_method", "best_model")
    epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
    epoch_number = epoch_number_dictionary.get(f"{checkpoint_choose_method}_{job_id}", 0)

    endpoint = "https://api.ngc.nvidia.com/v2"
    endpoint += f"/org/{org_name}"
    if team_name:
        endpoint += f"/team/{team_name}"
    endpoint += f"/models/{network}/versions/{job_action}_{job_id}_{epoch_number}"
    logger.info("Deleting: %s/%s/%s:%s_%s_%s", org_name, team_name, network, job_action, job_id, epoch_number)

    response = send_ngc_api_request(endpoint=endpoint, requests_method="DELETE", request_body={}, ngc_key=ngc_key)

    logger.info("Delete model response: %s", response)
    logger.info("Delete model response.text: %s", response.text)
    return response


def validate_ptm_download(base_experiment_folder, sha256_digest):
    """Validate if downloaded files are not corrupt"""
    if sha256_digest:
        for dirpath, _, filenames in os.walk(base_experiment_folder):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.isfile(file_path):
                    if sha256_digest.get(filename):
                        downloaded_file_checksum = sha256_checksum(file_path)
                        if sha256_digest[filename] != downloaded_file_checksum:
                            logger.error(
                                f"{filename} sha256 checksum not matched. "
                                f"Expected checksum is {sha256_digest.get(filename)}"
                                f"wheras downloaded file checksum is {downloaded_file_checksum}"
                            )
                        return sha256_digest[filename] == downloaded_file_checksum
    return True


def get_org_products(user_id, org_name):
    """Return the products the ORG has subscribe to"""
    try:
        ngc_key = get_user_key(user_id, org_name)
    except Exception as e:
        logger.error("Error getting NGC key for user %s and org %s: %s", user_id, org_name, e)
        return []
    headers = {}
    headers['Authorization'] = 'Bearer ' + ngc_key
    headers['Accept-Encoding'] = "True"
    url = f'https://api.ngc.nvidia.com/v2/orgs/{org_name}'
    response = requests.get(url, headers=headers, timeout=120)
    products = []
    if response.ok:
        org_metadata = response.json().get("organizations", {})
        product_enablements = org_metadata.get("productEnablements", [])
        for product_enablement in product_enablements:
            if product_enablement.get("productName", "") in ("TAO", "MONAI", "MAXINE"):
                products.append(product_enablement.get("productName"))
    return products


def get_ngc_token_from_api_key(ngc_api_key, org=None, team=None):
    """Get NGC token from API key"""
    url = "https://authn.nvidia.com/token"
    params = {"service": "ngc", "scope": "group/ngc"}
    if org:
        params["scope"] = f"group/ngc:{org}"
    if team:
        params["scope"] += f"&group/ngc:{org}/{team}"
    headers = {"Accept": "application/json"}
    auth = ("$oauthtoken", ngc_api_key)
    response = requests.get(url, headers=headers, auth=auth, params=params, timeout=TIMEOUT)
    if response.status_code == 200:
        return response.json()["token"]
    logger.error(f"Failed to get NGC token from API key: {response.text}, {response.status_code}")
    return None
