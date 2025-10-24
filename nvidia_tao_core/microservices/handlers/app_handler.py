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

"""API handler modules"""
import re
import copy
from datetime import datetime, timezone
import glob
import os
import shutil
import tarfile
import threading
import time
import traceback
import uuid
import logging

from nvidia_tao_core.microservices.constants import (
    AUTOML_DISABLED_NETWORKS,
    TENSORBOARD_DISABLED_NETWORKS,
    TENSORBOARD_EXPERIMENT_LIMIT,
    VALID_MODEL_DOWNLOAD_TYPE,
    TAO_NETWORKS,
    MEDICAL_CUSTOM_ARCHITECT,
    MAXINE_NETWORKS,
    MISSING_EPOCH_FORMAT_NETWORKS
)
from nvidia_tao_core.microservices.airgapped_utils import AirgappedExperimentLoader
from nvidia_tao_core.microservices.enum_constants import DatasetType, ExperimentNetworkArch
from nvidia_tao_core.microservices.handlers import ngc_handler, stateless_handlers
from nvidia_tao_core.microservices.handlers.nvcf_handler import get_available_nvcf_instances
from nvidia_tao_core.microservices.handlers.automl_handler import AutoMLHandler
from nvidia_tao_core.microservices.handlers.cloud_storage import create_cs_instance
from nvidia_tao_core.microservices.handlers.dataset_handler import validate_dataset
from nvidia_tao_core.microservices.handlers.encrypt import NVVaultEncryption
# from nvidia_tao_core.microservices.handlers import nvcf_handler
from nvidia_tao_core.microservices.handlers.monai.helpers import (
    CapGpuUsage,
    download_from_url,
    validate_monai_bundle,
    CUSTOMIZED_BUNDLE_URL_FILE,
    CUSTOMIZED_BUNDLE_URL_KEY
)
from nvidia_tao_core.microservices.handlers.monai_dataset_handler import MONAI_DATASET_ACTIONS, MonaiDatasetHandler
from nvidia_tao_core.microservices.handlers.monai_model_handler import MonaiModelHandler
# TODO: force max length of code line to 120 chars
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    check_read_access,
    check_write_access,
    get_base_experiment_metadata,
    get_job_specs,
    infer_action_from_job,
    is_valid_uuid4,
    printc,
    resolve_existence,
    resolve_metadata,
    resolve_root,
    get_handler_log_root,
    get_handler_job_metadata,
    get_jobs_root,
    save_job_specs,
    sanitize_handler_metadata,
    write_handler_metadata,
    is_request_automl,
    get_automl_controller_info,
    get_automl_current_rec,
    get_handler_status,
    validate_automl_settings
)
from nvidia_tao_core.microservices.handlers.tis_handler import TISHandler
from nvidia_tao_core.microservices.handlers.tensorboard_handler import TensorboardHandler
from nvidia_tao_core.microservices.handlers.utilities import (
    Code,
    download_log_from_cloud,
    download_dataset,
    get_files_from_cloud,
    prep_tis_model_repository,
    resolve_checkpoint_root_and_search,
    validate_and_update_experiment_metadata,
    validate_num_gpu,
    get_num_gpus_from_spec
)
if os.getenv("BACKEND"):  # To see if the container is going to be used for Service pods or network jobs
    from nvidia_tao_core.microservices.handlers.mongo_handler import (
        MongoHandler,
    )
from nvidia_tao_core.microservices.job_utils import executor as jobDriver
from nvidia_tao_core.microservices.job_utils.workflow_driver import create_job_context, on_delete_job, on_new_job
from nvidia_tao_core.microservices.job_utils.automl_job_utils import on_delete_automl_job
from nvidia_tao_core.microservices.specs_utils import csv_to_json_schema
from nvidia_tao_core.microservices.utils import (
    read_network_config,
    merge_nested_dicts,
    override_dicts,
    check_and_convert,
    safe_dump_file,
    log_monitor,
    get_microservices_network_and_action,
    DataMonitorLogTypeEnum,
)

from nvidia_tao_core.scripts.generate_schema import generate_schema, validate_and_clean_merged_spec

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Identify if workflow is on NGC
BACKEND = os.getenv("BACKEND", "local-k8s")
# Identify if nginx-ingress is enabled (should be disabled for NVCF deployments)
ingress_enabled = os.getenv("INGRESSENABLED", "false") == "true"


# Helpers
def resolve_job_existence(job_id):
    """Return whether job exists or not"""
    mongo_jobs = MongoHandler("tao", "jobs")
    job = mongo_jobs.find_one({'id': job_id})
    if job:
        return True
    return False


def delete_jobs_for_handler(handler_id, kind):
    """Deletes job metadatas associated with handler_id"""
    mongo_jobs = MongoHandler("tao", "jobs")
    mongo_jobs.delete_many({f"{kind}_id": handler_id})


def resolve_metadata_with_jobs(user_id, org_name, kind, handler_id):
    """Reads job_id.json in jobs_metadata folder and return it's contents"""
    if not user_id:
        logger.error("Can't resolve job metadata without user information")
        return {}
    handler_id = "*" if handler_id in ("*", "all") else handler_id
    metadata = {} if handler_id == "*" else resolve_metadata(kind, handler_id)
    if metadata or handler_id == "*":
        metadata["jobs"] = []
        jobs = stateless_handlers.get_jobs_for_handler(handler_id, kind)
        for job_meta in jobs:
            job_meta.pop('num_gpu', None)
            metadata["jobs"].append(job_meta)
        return metadata
    return {}


def get_org_experiments(org_name):
    """Returns a list of experiment IDs that are available for the given org_name"""
    mongo_experiments = MongoHandler("tao", "experiments")
    org_experiments = mongo_experiments.find({'org_name': org_name})
    experiments = []
    for experiment in org_experiments:
        experiment_id = experiment.get('id')
        experiments.append(experiment_id)
    return experiments


def get_org_datasets(org_name):
    """Returns a list of dataset IDs that are available for the given org_name"""
    mongo_datasets = MongoHandler("tao", "datasets")
    org_datasets = mongo_datasets.find({'org_name': org_name})
    datasets = []
    for dataset in org_datasets:
        dataset_id = dataset.get('id')
        datasets.append(dataset_id)
    return datasets


def get_org_workspaces(org_name):
    """Returns a list of workspace IDs that are available in given org_name"""
    mongo_workspaces = MongoHandler("tao", "workspaces")
    org_workspaces = mongo_workspaces.find({'org_name': org_name})
    workspaces = []
    for workspace in org_workspaces:
        workspace_id = workspace.get('id')
        workspaces.append(workspace_id)
    return workspaces


def get_user_experiments(user_id, mongo_users=None):
    """Returns a list of experiments that are available for the user"""
    user = stateless_handlers.get_user(user_id, mongo_users)
    experiments = user.get("experiments", [])
    return experiments


def get_user_datasets(user_id, mongo_users=None):
    """Returns a list of datasets that are available for the user"""
    user = stateless_handlers.get_user(user_id, mongo_users)
    datasets = user.get("datasets", [])
    return datasets


def get_user_workspaces(user_id, mongo_users=None):
    """Returns a list of datasets that are available for the user in given org_name"""
    user = stateless_handlers.get_user(user_id, mongo_users)
    workspaces = user.get("workspaces", [])
    return workspaces


def get_job(job_id):
    """Returns job from DB"""
    mongo_jobs = MongoHandler("tao", "jobs")
    job_query = {'id': job_id}
    job = mongo_jobs.find_one(job_query)
    return job


def get_experiment(experiment_id):
    """Returns experiment from DB"""
    mongo_experiments = MongoHandler("tao", "experiments")
    experiment_query = {'id': experiment_id}
    experiment = mongo_experiments.find_one(experiment_query)
    return experiment


def get_dataset(dataset_id):
    """Returns dataset from DB"""
    mongo_datasets = MongoHandler("tao", "datasets")
    dataset_query = {'id': dataset_id}
    dataset = mongo_datasets.find_one(dataset_query)
    return dataset


def get_workspace(workspace_id):
    """Returns workspace from DB"""
    mongo_workspaces = MongoHandler("tao", "workspaces")
    workspace_query = {'id': workspace_id}
    workspace = mongo_workspaces.find_one(workspace_query)
    return workspace


def create_blob_dataset(org_name, kind, handler_id):
    """Creates a blob dataset"""
    # Make a placeholder for S3 blob dataset
    msg = "Doesn't support the blob dataset for now."
    return Code(400, {}, msg)


def get_dataset_actions(ds_type, ds_format):
    """Reads the dataset's network config and returns the valid actions of the given dataset type and format"""
    actions_default = read_network_config(ds_type)["api_params"]["actions"]

    # Define all anamolous formats where actions are not same as ones listed in the network config
    TYPE_FORMAT_ACTIONS_MAP = {("object_detection", "raw"): [],
                               ("object_detection", "coco_raw"): [],
                               ("segmentation", "raw"): [],
                               }

    actions_override = TYPE_FORMAT_ACTIONS_MAP.get((ds_type, ds_format), actions_default)
    return actions_override


def nested_update(source, additions, allow_overwrite=True):
    """Merge one dictionary(additions) into another(source)"""
    if not isinstance(additions, dict):
        return source
    for key, value in additions.items():
        if isinstance(value, dict):
            # Initialize key in source if not present
            if key not in source:
                source[key] = {}
            source[key] = nested_update(source[key], value, allow_overwrite=allow_overwrite)
        else:
            source[key] = value if allow_overwrite else source.get(key, value)
    return source


def get_job_logs(log_file_path):
    """Yield lines from job log file"""
    with open(log_file_path, 'r', encoding="utf-8") as log_file:
        while True:
            log_line = log_file.readline()
            if not log_line:
                break

            yield log_line


def is_maxine_request(handler_id, handler_kind, handler_metadata={}):
    """Check if the request is related to Maxine.

    Args:
        handler_id (str): The ID of the handler.
        handler_kind (str): The kind of handler.
        handler_metadata (dict): The metadata of the handler.

    Returns:
        bool: True if the request is related to Maxine, False otherwise.
    """
    if not handler_metadata:
        handler_metadata = stateless_handlers.get_handler_metadata(handler_id, handler_kind)
    if handler_kind in ("workspaces", "workspace"):
        return True
    if handler_kind in ("datasets", "dataset"):
        return handler_metadata.get("type", "") == "maxine_dataset"
    if handler_kind in ("experiments", "experiment"):
        return handler_metadata.get("network_arch", "") == "maxine_eye_contact"
    return False


def handler_level_access_control(user_id, org_name, handler_id="", handler_kind="",
                                 handler_metadata={}, base_experiment=False):
    """Control access to handlers based on user permissions and product entitlements.

    Args:
        user_id (str): The ID of the user.
        org_name (str): The name of the organization.
        handler_id (str, optional): The ID of the handler. Defaults to "".
        handler_kind (str, optional): The kind of handler. Defaults to "".
        handler_metadata (dict, optional): The metadata of the handler. Defaults to {}.
        base_experiment (bool, optional): Whether this is a base experiment. Defaults to False.

    Returns:
        bool: True if the user has access, False otherwise.
    """
    if base_experiment or is_maxine_request(handler_id, handler_kind, handler_metadata):
        logger.info("Checking if user has MAXINE entitlement")
        if "MAXINE" not in ngc_handler.get_org_products(user_id, org_name):
            logger.info("User does not have MAXINE entitlement")
            return False
        mongo = MongoHandler("tao", "users")
        user_metadata = mongo.find_one({'id': user_id})
        member_of = user_metadata.get('member_of', [])
        if f"{org_name}/:MAXINE_USER" not in member_of:
            logger.info("User does not have MAXINE entitlement in NGC metadata")
            return False
    return True


class AppHandler:
    """Handles dataset, workspace, experiment, job creation, updating, deletion and retrieval."""

    # Workspace API
    @staticmethod
    def list_workspaces(user_id, org_name):
        """Retrieve a list of workspaces accessible by the given user.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.

        Returns:
            list[dict]: A list of workspace metadata dictionaries.
        """
        # Collect all metadatas
        metadatas = []
        for workspace_id in list(set(get_org_workspaces(org_name))):
            handler_metadata = stateless_handlers.get_handler_metadata(workspace_id, 'workspaces')
            shared_workspace = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_workspace or handler_metadata.get("user_id") == user_id:
                    metadatas.append(handler_metadata)
            else:
                # Something is wrong. The user metadata has a workspace that doesn't exist in the system.
                contexts = {"user_id": user_id, "org_name": org_name, "handler_id": workspace_id}
                printc("Workspace not found. Skipping.", contexts)
        return metadatas

    @staticmethod
    def retrieve_workspace(user_id, org_name, workspace_id):
        """Retrieve metadata for a specific workspace if the user has access.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.

        Returns:
            Code:
                - 200 with metadata if successful.
                - 404 if the workspace is not found or access is denied.
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(404, {}, "Workspace not found")

        if not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not found")

        return Code(200, handler_metadata, "Workspace retrieved")

    @staticmethod
    def retrieve_cloud_datasets(user_id, org_name, workspace_id, dataset_type, dataset_format, dataset_intention):
        """Retrieve paths of cloud datasets accessible within a workspace.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.
            dataset_type (str): Type of the dataset.
            dataset_format (str): Format of the dataset.
            dataset_intention (str): Purpose of dataset usage.

        Returns:
            Code:
                - 200 with dataset paths if successful.
                - 404 if the workspace is not found or access is denied.
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(404, {}, "Workspace not found")

        if not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not found")

        cloud_instance, _ = create_cs_instance(handler_metadata)
        cloud_files, _ = cloud_instance.list_files_in_folder("data")
        suggestions = set([])
        for cloud_file_path in cloud_files:
            cloud_folder = os.path.dirname(cloud_file_path)
            if dataset_type in ("segmentation", "pose_classification"):
                cloud_folder = os.path.dirname(cloud_folder)
            if dataset_type == "ml_recog":
                index_of_folder = cloud_folder.find("metric_learning_recognition")
                if index_of_folder != "-1":
                    cloud_folder = cloud_folder[0:index_of_folder]
            dataset_handler_metadata = {
                "type": dataset_type,
                "format": dataset_format,
                "use_for": dataset_intention
            }
            is_cloud_dataset_present, _ = validate_dataset(
                org_name,
                dataset_handler_metadata,
                temp_dir=f"/{cloud_folder}",
                workspace_metadata=handler_metadata
            )
            if is_cloud_dataset_present:
                suggestions.add(f"/{cloud_folder}")
        suggestions = list(suggestions)
        return_response_data = {"dataset_paths": suggestions}
        if suggestions:
            return Code(200, return_response_data, "Dataset folder path suggestions retrieved")
        return Code(200, return_response_data, "Dataset folder path suggestion couldn't be retrieved")

    @staticmethod
    def create_workspace(user_id, org_name, request_dict):
        """Create a new workspace with specified parameters.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            request_dict (dict): Workspace request following WorkspaceRspSchema.
                - "type" (required)
                - "format" (required)

        Returns:
            Code:
                - 209 with metadata if creation is successful.
                - 400 if there are errors in the request.
        """
        workspace_id = str(uuid.uuid4())
        # Create metadata dict and create some initial folders
        metadata = {"id": workspace_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Workspace"),
                    "shared": request_dict.get("shared", False),
                    "version": request_dict.get("version", "1.0.0"),
                    "cloud_type": request_dict.get("cloud_type", ""),
                    "cloud_specific_details": request_dict.get("cloud_specific_details", {}),
                    }

        encrypted_metadata = copy.deepcopy(metadata)

        # Encrypt Cloud details
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if encrypted_metadata["cloud_specific_details"]:
            cloud_specific_details = encrypted_metadata["cloud_specific_details"]
            if config_path and cloud_specific_details:
                encryption = NVVaultEncryption(config_path)
                for key, value in cloud_specific_details.items():
                    if encryption.check_config()[0]:
                        encrypted_metadata["cloud_specific_details"][key] = encryption.encrypt(value)
                    elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                        return Code(400, {}, "Vault service does not work, can't save cloud workspace")

        try:
            if encrypted_metadata["cloud_type"] in ("aws", "azure"):
                create_cs_instance(encrypted_metadata)
        except Exception as e:
            logger.error("Exception thrown in create workspace is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, {}, "Provided cloud credentials are invalid")

        write_handler_metadata(workspace_id, encrypted_metadata, "workspace")
        mongo_users = MongoHandler("tao", "users")
        workspaces = get_user_workspaces(user_id, mongo_users)
        workspaces.append(workspace_id)
        mongo_users.upsert({'id': user_id}, {'id': user_id, 'workspaces': workspaces})

        ret_Code = Code(200, metadata, "Workspace created")
        return ret_Code

    @staticmethod
    def update_workspace(user_id, org_name, workspace_id, request_dict):
        """Update an existing workspace with new metadata.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.
            request_dict (dict): Update request following WorkspaceRspSchema.
                - "type" (required)
                - "format" (required)

        Returns:
            Code:
                - 200 with updated metadata if successful.
                - 404 if the workspace is not found or access is denied.
                - 400 if the update request is invalid.
        """
        metadata = resolve_metadata("workspace", workspace_id)
        if not metadata:
            return Code(404, {}, "Workspace not found")

        if not check_write_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, {}, "Workspace not available")

        update_keys = request_dict.keys()
        for key in ["name", "version", "cloud_type", "shared"]:
            if key in update_keys:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)

        encrypted_metadata = copy.deepcopy(metadata)
        if "cloud_specific_details" in request_dict.keys():
            # Encrypt Cloud details
            for key, value in request_dict["cloud_specific_details"].items():
                if key == "cloud_type":
                    encrypted_metadata["cloud_type"] = value
                if key == "cloud_specific_details":
                    config_path = os.getenv("VAULT_SECRET_PATH", None)
                    if config_path:
                        if not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                            return Code(400, {}, "Vault service does not work, can't save cloud workspace")
                        encryption = NVVaultEncryption(config_path)
                        for cloud_key, cloud_value in request_dict["cloud_specific_details"].items():
                            encrypted_metadata["cloud_specific_details"][cloud_key] = cloud_value
                            if encryption.check_config()[0]:
                                encrypted_metadata["cloud_specific_details"][cloud_key] = (
                                    encryption.encrypt(cloud_value)
                                )

        if encrypted_metadata["cloud_type"] in ("aws", "azure"):
            try:
                if "cloud_type" in request_dict.keys() or "cloud_specific_details" in request_dict.keys():
                    if encrypted_metadata["cloud_type"] in ("aws", "azure"):
                        create_cs_instance(encrypted_metadata)
            except Exception as e:
                logger.error("Exception thrown in update_workspace is %s", str(e))
                logger.error(traceback.format_exc())
                return Code(400, {}, "Provided cloud credentials are invalid")

        write_handler_metadata(workspace_id, encrypted_metadata, "workspace")
        ret_Code = Code(200, metadata, "Workspace updated")
        return ret_Code

    @staticmethod
    def delete_workspace(org_name, workspace_id):
        """Delete a workspace if it belongs to the user.

        Args:
            org_name (str): Name of the organization.
            workspace_id (str): UUID of the workspace.

        Returns:
            Code:
                - 200 if deletion is successful.
                - 404 if the workspace cannot be accessed or deleted.
        """
        handler_metadata = resolve_metadata("workspace", workspace_id)
        if not handler_metadata:
            return Code(200, {}, "Workspace deleted")
        user_id = handler_metadata.get("user_id")

        if workspace_id not in get_user_workspaces(user_id):
            return Code(404, {}, "Workspace cannot be deleted as it's doesn't belong to user")

        # If workspace is being used by user's experiments or datasets.
        experiments = get_user_experiments(user_id)

        for experiment_id in experiments:
            experiment_metadata = get_experiment(experiment_id)
            experiment_workspace = experiment_metadata.get("workspace", "")
            if experiment_workspace and workspace_id in experiment_workspace:
                return Code(
                    400,
                    {},
                    f"Experiment {experiment_metadata['id']} "
                    f"({experiment_metadata['id']}) in use; Delete experiment first"
                )

            train_datasets = experiment_metadata.get("train_datasets", [])
            if not isinstance(train_datasets, list):
                train_datasets = [train_datasets]
            for dataset_id in train_datasets:
                dataset_metadata = get_dataset(dataset_id)
                dataset_workspace = dataset_metadata.get("workspace", "")
                if workspace_id == dataset_workspace:
                    return Code(
                        400,
                        {},
                        f"Dataset {dataset_metadata['id']} "
                        f"({dataset_metadata['id']}) in use; Delete dataset first"
                    )

            for key in ["eval_dataset", "inference_dataset", "calibration_dataset"]:
                additional_dataset_id = experiment_metadata.get(key)
                if additional_dataset_id:
                    dataset_metadata = get_dataset(additional_dataset_id)
                    dataset_workspace = dataset_metadata.get("workspace", "")
                    if workspace_id == dataset_workspace:
                        return Code(
                            400,
                            {},
                            f"Dataset {dataset_metadata['id']} "
                            f"({dataset_metadata['id']}) in use; Delete dataset first"
                        )

        mongo_users = MongoHandler("tao", "users")
        user = stateless_handlers.get_user(user_id, mongo_users)
        workspaces = user.get("workspaces", [])
        if workspace_id in workspaces:
            workspaces.remove(workspace_id)
            mongo_users.upsert({'id': user_id}, {'id': user_id, 'workspaces': workspaces})
        mongo_workspaces = MongoHandler("tao", "workspaces")
        mongo_workspaces.delete_one({'id': workspace_id})
        return Code(200, {"message": "Workspace deleted"}, "")

    # Dataset API
    @staticmethod
    def list_datasets(user_id, org_name):
        """Retrieve a list of datasets accessible by the given user.

        Args:
            user_id (str): UUID representing the user.
            org_name (str): Name of the organization.

        Returns:
            list[dict]: A list of dataset metadata dictionaries.
        """
        # Collect all metadatas
        metadatas = []
        for dataset_id in list(set(get_org_datasets(org_name) + stateless_handlers.get_public_datasets())):
            handler_metadata = stateless_handlers.get_handler_metadata(dataset_id, 'datasets')
            shared_dataset = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_dataset or handler_metadata.get("user_id") == user_id:
                    handler_metadata = sanitize_handler_metadata(handler_metadata)
                    metadatas.append(handler_metadata)
            else:
                # Something is wrong. The user metadata has a dataset that doesn't exist in the system.
                contexts = {"user_id": user_id, "org_name": org_name, "handler_id": dataset_id}
                printc("Dataset not found. Skipping.", contexts)
        return metadatas

    @staticmethod
    def get_dataset_formats(dataset_type):
        """Retrieve available dataset formats for a given dataset type.

        Args:
            dataset_type (str): The type of dataset.

        Returns:
            list[str]: A list of supported dataset formats.
        """
        try:
            dataset_formats = []
            accepted_dataset_intents = []
            api_params = read_network_config(dataset_type).get("api_params", {})
            if api_params:
                if api_params.get("formats", []):
                    dataset_formats += api_params.get("formats", [])
                if api_params.get("accepted_ds_intents", []):
                    accepted_dataset_intents += api_params.get("accepted_ds_intents", [])
            return Code(
                200,
                {
                    "dataset_formats": dataset_formats,
                    "accepted_dataset_intents": accepted_dataset_intents
                },
                ""
            )
        except Exception:
            logger.error("Exception caught during getting dataset formats: %s", traceback.format_exc())
            return Code(404, [], "Exception caught during getting dataset formats")

    # Create dataset
    @staticmethod
    def create_dataset(user_id, org_name, request_dict, dataset_id=None, from_ui=False):
        """Creates a new dataset with the given parameters.

        Args:
            user_id (str): The unique identifier of the user.
            org_name (str): The name of the organization.
            request_dict (dict): Dictionary containing dataset creation parameters.
                - "type" (str): Required dataset type.
                - "format" (str): Required dataset format.
            dataset_id (str, optional): A predefined dataset ID. Defaults to None.
            from_ui (bool, optional): Flag indicating if the request is from UI. Defaults to False.

        Returns:
            Code: Response object containing status and metadata of the created dataset.
        """
        workspace_id = request_dict.get("workspace", None)
        if workspace_id and not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, None, f"Workspace {workspace_id} not found")

        # Gather type,format fields from request
        ds_type = request_dict.get("type", None)
        if ds_type == "ocrnet":
            intention = request_dict.get("use_for", [])
            if not (intention in (["training"], ["evaluation"])):
                return Code(
                    400,
                    {},
                    "Use_for in dataset metadata is not set ['training'] or ['evaluation']. "
                    "Please set use_for appropriately"
                )

        ds_format = request_dict.get("format", None)
        # Perform basic checks - valid type and format?
        if ds_type not in DatasetType.__members__.values():
            msg = "Invalid dataset type"
            return Code(400, {}, msg)

        # For monai dataset, don't check the dataset type.
        if ds_format not in read_network_config(ds_type)["api_params"]["formats"] and ds_format != "monai":
            msg = "Incompatible dataset format and type"
            return Code(400, {}, msg)

        intention = request_dict.get("use_for", [])
        if ds_format in ("raw", "coco_raw") and intention:
            if intention != ["testing"] and ds_type != "maxine_dataset":
                msg = "raw or coco_raw's format should be associated with ['testing'] intent"
                return Code(400, {}, msg)

        # Create a dataset ID and its root
        pull = False
        if not dataset_id:
            pull = True
            dataset_id = str(uuid.uuid4())

        if request_dict.get("public", False):
            stateless_handlers.add_public_dataset(dataset_id)

        dataset_actions = get_dataset_actions(ds_type, ds_format) if ds_format != "monai" else MONAI_DATASET_ACTIONS

        # Create metadata dict and create some initial folders
        metadata = {"id": dataset_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "authorized_party_nca_id": request_dict.get("authorized_party_nca_id", ""),
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Dataset"),
                    "shared": request_dict.get("shared", False),
                    "description": request_dict.get("description", "My TAO Dataset"),
                    "version": request_dict.get("version", "1.0.0"),
                    "docker_env_vars": request_dict.get("docker_env_vars", {}),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "type": ds_type,
                    "format": ds_format,
                    "actions": dataset_actions,
                    "client_url": request_dict.get("client_url", None),
                    "client_id": request_dict.get("client_id", None),
                    "client_secret": request_dict.get("client_secret", None),  # TODO:: Store Secrets in Vault
                    "filters": request_dict.get("filters", None),
                    "cloud_file_path": request_dict.get("cloud_file_path"),
                    "url": request_dict.get("url"),
                    "workspace": request_dict.get("workspace"),
                    "use_for": intention,
                    "base_experiment": request_dict.get("base_experiment", []),
                    }

        if not handler_level_access_control(user_id, org_name, dataset_id, "datasets", handler_metadata=metadata):
            return Code(403, {}, "Not allowed to work with this org")

        # Set status based on skip_validation flag
        skip_validation = request_dict.get("skip_validation", False)
        if skip_validation:
            metadata["status"] = "pull_complete"
        else:
            metadata["status"] = request_dict.get("status", "starting") if ds_format != "monai" else "pull_complete"

        if metadata.get("url", ""):
            if not metadata.get("url").startswith("https"):
                return Code(400, {}, "Invalid pull URL passed")

        # Encrypt the MLOPs keys
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path and metadata["docker_env_vars"]:
            encryption = NVVaultEncryption(config_path)
            for key, value in metadata["docker_env_vars"].items():
                if encryption.check_config()[0]:
                    metadata["docker_env_vars"][key] = encryption.encrypt(value)
                elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                    return Code(400, {}, "Vault service does not work, can't enable MLOPs services")

        # Encrypt the client secret if the dataset is a monai dataset and the vault agent has been set.
        if config_path and metadata["client_secret"] and ds_format == "monai":
            encryption = NVVaultEncryption(config_path)
            if encryption.check_config()[0]:
                metadata["client_secret"] = encryption.encrypt(metadata["client_secret"])
            elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                return Code(400, {}, "Cannot create dataset because vault service does not work.")

        # For MONAI dataset only
        if ds_format == "monai":
            client_url = request_dict.get("client_url", None)
            log_content = (
                f"user_id:{user_id}, "
                f"org_name:{org_name}, "
                f"from_ui:{from_ui}, "
                f"dataset_url:{client_url}, "
                f"action:creation"
            )
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_dataset, log_content=log_content)
            if client_url is None:
                msg = "Must provide a url to create a MONAI dataset."
                return Code(400, {}, msg)

            status, m = MonaiDatasetHandler.status_check(metadata)
            if not status or m:
                return Code(400, {}, m)

        write_handler_metadata(dataset_id, metadata, "dataset")
        mongo_users = MongoHandler("tao", "users")
        user_query = {'id': user_id}
        datasets = get_user_datasets(user_id, mongo_users)
        datasets.append(dataset_id)
        mongo_users.upsert(user_query, {'id': user_id, 'datasets': datasets})

        # Pull dataset in background if known URL and not skipping validation
        if pull and not skip_validation:
            job_run_thread = threading.Thread(target=AppHandler.pull_dataset, args=(user_id, org_name, dataset_id,))
            job_run_thread.start()

        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Dataset created")
        return ret_Code

    @staticmethod
    def create_dataset_dict_from_experiment_metadata(dataset_id, action, handler_metadata):
        """Generates a dataset request dictionary from existing experiment metadata.

        Args:
            dataset_id (str): The unique identifier of the new dataset.
            action (str): The action performed that triggered dataset creation.
            handler_metadata (dict): Metadata from the source dataset or experiment.

        Returns:
            dict: A request dictionary containing dataset creation parameters.
        """
        infer_ds = handler_metadata.get("inference_dataset", None)
        if infer_ds:
            dataset_metadata = stateless_handlers.get_handler_metadata(infer_ds, "datasets")
        else:
            dataset_metadata = copy.deepcopy(handler_metadata)
        request_dict = {}
        output_dataset_type = dataset_metadata.get("type")
        output_dataset_format = dataset_metadata.get("format")
        use_for = dataset_metadata.get("use_for")
        request_dict["type"] = output_dataset_type
        request_dict["status"] = dataset_metadata.get("status", "pull_complete")
        request_dict["format"] = output_dataset_format
        request_dict["use_for"] = use_for
        request_dict["workspace"] = dataset_metadata.get("workspace")
        request_dict["cloud_file_path"] = os.path.join("/results/", dataset_id)
        request_dict["name"] = f"{dataset_metadata.get('name')} (created from Data services {action} action)"
        request_dict["shared"] = dataset_metadata.get("shared", False)
        request_dict["use_for"] = dataset_metadata.get("use_for", [])
        request_dict["docker_env_vars"] = dataset_metadata.get("docker_env_vars", {})
        return request_dict

    # Update existing dataset for user based on request dict
    @staticmethod
    def update_dataset(org_name, dataset_id, request_dict):
        """Updates an existing dataset with new metadata.

        Args:
            org_name (str): The name of the organization.
            dataset_id (str): The unique identifier of the dataset.
            request_dict (dict): Dictionary containing update parameters.
                - "type" (str): Required dataset type (unchangeable).
                - "format" (str): Required dataset format (unchangeable).

        Returns:
            Code: Response object indicating success (200) or failure (404 or 400).
        """
        metadata = resolve_metadata("dataset", dataset_id)
        if not metadata:
            return Code(404, {}, "Dataset not found")

        user_id = metadata.get("user_id")
        if not handler_level_access_control(user_id, org_name, dataset_id, "datasets", handler_metadata=metadata):
            return Code(403, {}, "Not allowed to work with this org")
        if not check_write_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not available")
        if request_dict.get("public", None):
            if request_dict["public"]:
                stateless_handlers.add_public_dataset(dataset_id)
            else:
                stateless_handlers.remove_public_dataset(dataset_id)
        pull = False
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["type", "format"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change dataset {key}"
                    return Code(400, {}, msg)

            if key in [
                "name", "description", "version", "logo", "shared",
                "base_experiment", "authorized_party_nca_id"
            ]:
                requested_value = request_dict[key]
                if requested_value:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)

            if key == "cloud_file_path":
                if metadata["status"] not in ("pull_complete", "invalid_pull"):
                    return Code(
                        400,
                        {},
                        f"Cloud file_path can be updated only when status is pull_complete or "
                        f"invalid_pull, the current status is {metadata['status']}. Try again after sometime"
                    )
                pull = True
                metadata["status"] = "starting"
                metadata["cloud_file_path"] = request_dict[key]

            if key == "docker_env_vars":
                # Encrypt the MLOPs keys
                requested_value = request_dict[key]
                config_path = os.getenv("VAULT_SECRET_PATH", None)
                if config_path:
                    encryption = NVVaultEncryption(config_path)
                    for mlops_key, value in requested_value.items():
                        if encryption.check_config()[0]:
                            metadata["docker_env_vars"][mlops_key] = encryption.encrypt(value)
                        elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                            return Code(400, {}, "Vault service does not work, can't enable MLOPs services")
                else:
                    metadata["docker_env_vars"] = requested_value

        # Pull dataset in background if known URL
        if pull:
            job_run_thread = threading.Thread(target=AppHandler.pull_dataset, args=(user_id, org_name, dataset_id,))
            job_run_thread.start()

        write_handler_metadata(dataset_id, metadata, "dataset")
        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Dataset updated")
        return ret_Code

    # Retrieve existing dataset for user based on request dict
    @staticmethod
    def retrieve_dataset(org_name, dataset_id):
        """Retrieves metadata for an existing dataset.

        Args:
            org_name (str): The name of the organization.
            dataset_id (str): The unique identifier of the dataset.

        Returns:
            Code: Response object containing dataset metadata if found (200) or an error (404).
        """
        handler_metadata = resolve_metadata("dataset", dataset_id)
        if not handler_metadata:
            return Code(404, {}, "Dataset not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not found")

        return_metadata = sanitize_handler_metadata(handler_metadata)
        if return_metadata.get("status") == "invalid_pull":
            # Include detailed validation error information if available
            validation_details = return_metadata.get("validation_details", {})
            if validation_details:
                error_msg = validation_details.get("error_details", "Dataset validation failed")
                return Code(404, return_metadata, error_msg, use_data_as_response=True)
            return Code(404, return_metadata, "Dataset pulled from cloud doesn't match folder structure required")
        return Code(200, return_metadata, "Dataset retrieved")

    # Delete a user's dataset
    @staticmethod
    def delete_dataset(org_name, dataset_id):
        """Deletes a dataset if it is not in use or restricted.

        Args:
            org_name (str): Name of the organization requesting the deletion.
            dataset_id (str): UUID of the dataset to be deleted.

        Returns:
            Code: Response object containing:
                - 200 with metadata of the deleted dataset if successful.
                - 404 if the user lacks access to the dataset.
                - 400 if the dataset is in use by a running job or an active experiment.
        """
        handler_metadata = resolve_metadata("dataset", dataset_id)
        if not handler_metadata:
            return Code(200, {}, f"Dataset {dataset_id} deleted")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, f"Dataset {dataset_id} not available")

        # If dataset is being used by user's experiments.
        experiments = get_user_experiments(user_id)
        for experiment_id in experiments:
            metadata = stateless_handlers.get_handler_metadata(experiment_id, "experiment")
            datasets_in_use = set(metadata.get("train_datasets", []))
            for key in ["eval_dataset", "inference_dataset", "calibration_dataset"]:
                additional_dataset_id = metadata.get(key)
                if additional_dataset_id:
                    datasets_in_use.add(additional_dataset_id)
            if dataset_id in datasets_in_use:
                return Code(400, {}, f"Dataset {dataset_id} in use by {experiment_id}")

        # Check if any job running
        for job in handler_metadata.get("jobs", {}):
            if handler_metadata["jobs"][job]["status"] == "Running":
                return Code(400, {}, f"Dataset {dataset_id} in use by job {job}")

        # Check if dataset is public, then someone could be running it
        if handler_metadata.get("public", False):
            return Code(400, {}, f"Dataset {dataset_id} is Public. Cannot delete")

        # Check if dataset is read only, if yes, cannot delete
        if handler_metadata.get("read_only", False):
            return Code(400, {}, f"Dataset {dataset_id} is read only. Cannot delete")

        mongo_users = MongoHandler("tao", "users")
        datasets = get_user_datasets(user_id, mongo_users)
        if dataset_id in datasets:
            datasets.remove(dataset_id)
        user_query = {'id': user_id}
        mongo_users.upsert(user_query, {'id': user_id, 'datasets': datasets})

        mongo_datasets = MongoHandler("tao", "datasets")
        dataset_query = {'id': dataset_id}
        mongo_datasets.delete_one(dataset_query)
        delete_jobs_for_handler(dataset_id, "dataset")
        # TODO: Delete logs for dataset
        return_metadata = sanitize_handler_metadata(handler_metadata)
        return Code(200, return_metadata, "Dataset deleted")

    @staticmethod
    def validate_dataset(user_id, org_name, dataset_id, temp_dir=None, file_path=None):
        """Validates a dataset and updates its status accordingly.

        Args:
            user_id (str): UUID of the user requesting validation.
            org_name (str): Name of the organization.
            dataset_id (str): UUID of the dataset to be validated.
            temp_dir (str, optional): Path to the temporary directory for dataset processing.
            file_path (str, optional): Path to the dataset file or folder.

        Returns:
            Code: Response object containing:
                - 200 with an empty dictionary if validation starts successfully.
                - 404 if the dataset is not found or access is denied.
                - 400 if validation fails due to structural issues.
        """
        metadata = resolve_metadata("dataset", dataset_id)
        if not metadata:
            return Code(404, {}, "Dataset not found")

        if not check_write_access(user_id, org_name, dataset_id, kind="datasets"):
            return Code(404, {}, "Dataset not available")

        if metadata.get("format") == "monai":
            return Code(404, metadata, "Uploading external data is not supported for MONAI Dataset")

        try:
            metadata["status"] = "in_progress"
            write_handler_metadata(dataset_id, metadata, "dataset")

            def validate_dataset_thread():
                try:
                    # For cloud-based validation, resolve workspace metadata
                    workspace_metadata = None
                    if metadata.get("workspace"):
                        workspace_metadata = resolve_metadata("workspace", metadata.get("workspace"))

                    valid_dataset_structure, validation_result = validate_dataset(
                        org_name,
                        metadata,
                        temp_dir=temp_dir,
                        workspace_metadata=workspace_metadata
                    )
                    # Only remove temp_dir if it was actually created (not empty for cloud validation)
                    if temp_dir and os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

                    if valid_dataset_structure:
                        metadata["status"] = "pull_complete"
                    else:
                        metadata["status"] = "invalid_pull"
                        # Store detailed validation information in metadata for user feedback
                        metadata["validation_details"] = {
                            "error_details": validation_result.get("error_details", "Unknown validation error"),
                            "expected_structure": validation_result.get("expected_structure", {}),
                            "actual_structure": validation_result.get("actual_structure", []),
                            "missing_files": validation_result.get("missing_files", []),
                            "network_type": validation_result.get("network_type", ""),
                            "dataset_format": validation_result.get("dataset_format", ""),
                            "dataset_intent": validation_result.get("dataset_intent", [])
                        }
                        logger.error(
                            "Dataset structure validation failed for dataset %s. "
                            "Expected structure: %s. Actual files: %s. Missing files: %s. Error: %s",
                            dataset_id,
                            validation_result.get("expected_structure", {}),
                            validation_result.get("actual_structure", []),
                            validation_result.get("missing_files", []),
                            validation_result.get("error_details", ""))

                    write_handler_metadata(dataset_id, metadata, "dataset")
                except Exception as e:
                    logger.error("Exception thrown in validate_dataset_thread is %s", str(e))
                    logger.error(traceback.format_exc())
                    metadata["status"] = "invalid_pull"
                    metadata["validation_details"] = {
                        "error_details": f"Validation process failed: {str(e)}",
                        "expected_structure": {},
                        "actual_structure": [],
                        "missing_files": [],
                        "network_type": metadata.get("type", ""),
                        "dataset_format": metadata.get("format", ""),
                        "dataset_intent": metadata.get("use_for", [])
                    }
                    write_handler_metadata(dataset_id, metadata, "dataset")

            thread = threading.Thread(target=validate_dataset_thread)
            thread.start()
            return Code(200, {}, "Server recieved file and upload process started")
        except Exception as e:
            logger.error("Exception thrown in validate_dataset is %s", str(e))
            logger.error(traceback.format_exc())
            metadata["status"] = "invalid_pull"
            write_handler_metadata(dataset_id, metadata, "dataset")
            return Code(404, [], "Exception caught during upload")

    @staticmethod
    def pull_dataset(user_id, org_name, dataset_id):
        """Initiates the process of validating a dataset, optimizing for cloud-based datasets.

        Args:
            user_id (str): UUID of the user requesting the dataset pull.
            org_name (str): Name of the organization.
            dataset_id (str): UUID of the dataset to be pulled.

        Notes:
            - For cloud-based datasets: validates structure directly without downloading.
            - For public URLs/HuggingFace: downloads first then validates.
            - Updates dataset status upon failure.
        """
        try:
            metadata = resolve_metadata("dataset", dataset_id)
            if not metadata:
                logger.error("Dataset metadata not found for %s", dataset_id)
                return

            # Check if this is a cloud-based dataset that can use cloud peek validation
            cloud_file_path = metadata.get("cloud_file_path")
            workspace_id = metadata.get("workspace")
            dataset_url = metadata.get("url")

            # Determine if we can use cloud peek validation (avoid download)
            can_use_cloud_peek = (
                cloud_file_path and
                workspace_id and
                not dataset_url  # No external URL means it's cloud storage based
            )

            if can_use_cloud_peek:
                # Validate directly from cloud without downloading
                AppHandler.validate_dataset(user_id, org_name, dataset_id, temp_dir="", file_path="")
            else:
                logger.info("Using download validation for dataset %s (url: %s)", dataset_id, dataset_url)
                temp_dir, file_path = download_dataset(dataset_id)
                AppHandler.validate_dataset(user_id, org_name, dataset_id, temp_dir=temp_dir, file_path=file_path)
        except Exception as e:
            logger.error("Exception thrown in pull_dataset is %s", str(e))
            logger.error(traceback.format_exc())
            metadata = resolve_metadata("dataset", dataset_id)
            metadata["status"] = "invalid_pull"
            write_handler_metadata(dataset_id, metadata, "dataset")

    # Spec API

    @staticmethod
    def get_spec_schema(user_id, org_name, handler_id, action, kind):
        """Retrieves the specification schema for a dataset or experiment.

        Args:
            user_id (str): UUID of the user requesting the schema.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the dataset or experiment.
            action (str): The specific action for which the schema is required.
            kind (str): Type of entity, either "experiment" or "dataset".

        Returns:
            Code: Response object containing:
                - 200 with the schema in JSON format if retrieval is successful.
                - 404 if the dataset/experiment is not found or access is denied.
                - 404 if the requested action is invalid.
        """
        metadata = resolve_metadata(kind, handler_id)
        if not metadata:
            return Code(404, {}, "Spec schema not found")

        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, "Spec schema not available")

        # Action not available
        if action not in metadata.get("actions", []):
            if not (kind == "dataset" and action == "validate_images"):
                return Code(404, {}, "Action not found")

        base_experiment_spec = {}
        if metadata.get("base_experiment", []):
            for base_experiment_id in metadata["base_experiment"]:
                base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
                base_exp_meta = base_experiment_metadata.get("base_experiment_metadata", {})
                if base_experiment_metadata and base_exp_meta.get("spec_file_present"):
                    base_experiment_spec = base_exp_meta.get("specs", {})
                    if not base_experiment_spec:
                        return Code(404, {}, "Base specs not present.")

        # Read csv from spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}

        if kind == "dataset" and metadata.get("format") == "monai":
            json_schema = MonaiDatasetHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

        if kind == "experiment" and metadata.get("type").lower() == "medical":
            json_schema = MonaiModelHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

        network = metadata.get("network_arch", None)
        if not network:
            # Used for dataset jobs
            network = metadata.get("type", None)

        microservices_network, microservices_action = get_microservices_network_and_action(network, action)

        try:
            json_schema = generate_schema(microservices_network, microservices_action)
        except Exception as e:
            logger.error("Exception thrown in get_spec_schema is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                fmt = metadata.get("format", "_")
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{fmt}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    Code(404, {}, "Default specs do not exist for action")
            json_schema = csv_to_json_schema.convert(CSV_PATH)

        if "default" in json_schema and base_experiment_spec:
            # Merge the base experiment spec with the default schema
            merged_default = merge_nested_dicts(json_schema["default"], base_experiment_spec)
            # Validate and clean the merged spec to remove any invalid keys from corrupt base_experiment_spec
            logger.info("Validating merged base_experiment_spec")
            json_schema["default"] = validate_and_clean_merged_spec(json_schema, merged_default)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_spec_schema_for_job(user_id, org_name, handler_id, job_id, kind):
        """Retrieves the specification schema for a specific job within an experiment or dataset.

        Args:
            user_id (str): UUID of the user requesting the schema.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the dataset or experiment.
            job_id (str): UUID of the job.
            kind (str): Type of entity, either "experiment" or "dataset".

        Returns:
            Code: Response object containing:
                - 200 with the schema in JSON format if retrieval is successful.
                - 404 if the dataset/experiment or job is not found or access is denied.
        """
        metadata = resolve_metadata(kind, handler_id)
        if not metadata:
            return Code(404, {}, "Spec schema not found")

        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, "Spec schema not available")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "Job trying to get schema for not found")
        action = job_metadata.get("action", "")

        job_specs = get_job_specs(job_id)

        json_schema = {}

        if kind == "dataset" and metadata.get("format") == "monai":
            json_schema = MonaiDatasetHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

        if kind == "experiment" and metadata.get("type").lower() == "medical":
            json_schema = MonaiModelHandler.get_schema(action)
            return Code(200, json_schema, "Schema retrieved")

        network = metadata.get("network_arch", None)
        if not network:
            # Used for dataset jobs
            network = metadata.get("type", None)

        microservices_network, microservices_action = get_microservices_network_and_action(network, action)

        try:
            json_schema = generate_schema(microservices_network, microservices_action)
        except Exception as e:
            logger.error("Exception thrown in get_spec_schema_for_job is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                fmt = metadata.get("format", "_")
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{fmt}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    Code(404, {}, "Default specs do not exist for action")
            json_schema = csv_to_json_schema.convert(CSV_PATH)

        json_schema["default"] = job_specs
        if "popular" in json_schema and job_specs:
            json_schema["popular"] = override_dicts(json_schema["popular"], job_specs)
        if is_request_automl(handler_id, action, kind):
            json_schema["automl_default_parameters"] = (
                metadata.get("automl_settings", {}).get("automl_hyperparameters", "[]")
            )
        # elif BACKEND == "NVCF":
        #     json_schema = {}
        #     deployment_string = os.getenv(f'FUNCTION_{NETWORK_CONTAINER_MAPPING[microservices_network]}')
        #     if action == "gen_trt_engine":
        #         deployment_string = os.getenv('FUNCTION_TAO_DEPLOY')
        #     nvcf_response = nvcf_handler.invoke_function(
        #         deployment_string=deployment_string,
        #         network=microservices_network,
        #         action=microservices_action,
        #         microservice_action="get_schema"
        #     )
        #     if nvcf_response.status_code != 200:
        #         if nvcf_response.status_code == 202:
        #             return Code(404, {}, "Schema from NVCF couldn't be obtained in 60 seconds, Retry again")
        #         return Code(nvcf_response.status_code, {}, str(nvcf_response.json()))
        #     json_schema = nvcf_response.json().get("response")

        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_base_experiment_spec_schema(experiment_id, action):
        """Retrieves the base experiment specification schema.

        This method fetches the JSON schema for a base experiment spec based on the
        experiment ID and action. It checks the availability of the schema either
        through microservices or locally stored spec files.

        Args:
            experiment_id (str): UUID corresponding to the base experiment.
            action (str): A valid action for a dataset.

        Returns:
            Code: A response code object containing the status and the JSON schema
                  (if found) or error details:
                  - 200: JSON schema in a schema format.
                  - 404: If experiment/action not found or user cannot access.
        """
        base_experiment_spec = {}
        base_experiment_metadata = get_base_experiment_metadata(experiment_id)
        base_experiment_network = base_experiment_metadata.get("network_arch", "")
        if action not in base_experiment_metadata.get("actions", []):
            return Code(404, {}, "Action not found")

        base_exp_meta = base_experiment_metadata.get("base_experiment_metadata", {})
        if base_experiment_metadata and base_exp_meta.get("spec_file_present"):
            base_experiment_spec = base_exp_meta.get("specs", {})
            if not base_experiment_spec:
                return Code(404, {}, "Base specs not present.")

        # Read csv from spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        if base_experiment_network in TAO_NETWORKS:
            try:
                json_schema = generate_schema(base_experiment_network, action)
            except Exception as e:
                logger.error("Exception thrown in get_base_experiment_spec_schema is %s", str(e))
                logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", base_experiment_network,
                                    f"{base_experiment_network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                Code(404, {}, "Default specs do not exist for action")
            json_schema = csv_to_json_schema.convert(CSV_PATH)
        if "default" in json_schema and base_experiment_spec:
            # Merge the base experiment spec with the default schema
            merged_default = merge_nested_dicts(json_schema["default"], base_experiment_spec)
            # Validate and clean the merged spec to remove any invalid keys from corrupt base_experiment_spec
            logger.info("Validating merged base_experiment_spec for base_experiment_id: %s, action: %s",
                        experiment_id, action)
            json_schema["default"] = validate_and_clean_merged_spec(json_schema, merged_default)
            if (base_experiment_network == "visual_changenet_segment" and
                    "train" in json_schema["default"]):
                json_schema["default"]["train"].pop("tensorboard", None)
        if "popular" in json_schema and base_experiment_spec:
            # Merge the base experiment spec with the popular schema
            merged_popular = override_dicts(json_schema["popular"], base_experiment_spec)
            # Validate and clean the merged spec to remove any invalid keys from corrupt base_experiment_spec
            logger.info("Validating merged base_experiment_spec (popular) for base_experiment_id: %s, action: %s",
                        experiment_id, action)
            json_schema["popular"] = validate_and_clean_merged_spec(json_schema, merged_popular)
            if (base_experiment_network == "visual_changenet_segment" and
                    "train" in json_schema["popular"]):
                json_schema["popular"]["train"].pop("tensorboard", None)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_spec_schema_without_handler_id(org_name, network, dataset_format, action, train_datasets):
        """Retrieves the specification schema without handler ID.

        This method generates a JSON schema for a given network and action, either
        through direct retrieval or by reading from CSV files stored locally.

        Args:
            org_name (str): The organization name.
            network (str): A valid network architecture name supported.
            dataset_format (str): A valid format of the architecture, if necessary.
            action (str): A valid action for a dataset.
            train_datasets (list): A list of UUIDs corresponding to training datasets.

        Returns:
            Code: A response code object containing the status and the JSON schema
                  (if found) or error details:
                  - 200: JSON schema in a schema format.
                  - 404: If experiment/dataset not found or user cannot access.
        """
        # Action not available
        if not network:
            return Code(404, {}, "Pass network name to the request")
        if not action:
            return Code(404, {}, "Pass action name to the request")

        # Read csv from spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        if network in TAO_NETWORKS:
            try:
                json_schema = generate_schema(network, action)
            except Exception as e:
                logger.error("Exception thrown in get_spec_schema_without_handler_id is %s", str(e))
                logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{dataset_format}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    Code(404, {}, "Default specs do not exist for action")

            json_schema = csv_to_json_schema.convert(CSV_PATH)
        return Code(200, json_schema, "Schema retrieved")

    @staticmethod
    def get_gpu_types(user_id, org_name):
        """Retrieves available GPU types for the given user and organization.

        This method checks the backend for available GPU resources based on the
        provided user and organization information.

        Args:
            user_id (str): The user's UUID.
            org_name (str): The organization's name.

        Returns:
            Code: A response code object containing the status and available GPU
                  types or error details:
                  - 200: A list of available GPUs.
                  - 404: If GPUs cannot be retrieved for the specified backend.
        """
        if BACKEND == "NVCF":
            available_nvcf_instances = get_available_nvcf_instances(user_id, org_name)
            if available_nvcf_instances:
                return Code(200, available_nvcf_instances, "Retrieved available GPU info")
            return Code(404, [], f"NVCF GPU's are not available for {org_name}")
        if BACKEND == "local-k8s":
            available_gpu_types = jobDriver.get_available_local_k8s_gpus()
            if available_gpu_types:
                return Code(200, available_gpu_types, "Retrieved available GPU info")
            return Code(
                404, [],
                "Requested GPU's are not available in the current deployment. "
                "Check if all nodes has accelerator labels"
            )
        return Code(404, [], f"GPU types can't be retrieved for deployed Backend {BACKEND}")

    # Job API

    @staticmethod
    def job_run(
        org_name,
        handler_id,
        parent_job_id,
        action,
        kind,
        specs=None,
        name=None,
        description=None,
        num_gpu=-1,
        platform_id=None,
        from_ui=False
    ):
        """Runs a job based on the specified parameters.

        This method initiates a job based on the given organization, experiment,
        dataset details, and other job specifications. It handles different job
        scenarios including AutoML and regular jobs, and validates all necessary
        conditions before scheduling the job.

        Args:
            org_name (str): The organization name.
            handler_id (str): UUID corresponding to experiment or dataset.
            parent_job_id (str): UUID of the parent job.
            action (str): The action to be performed.
            kind (str): The type of resource ("experiment" or "dataset").
            specs (dict, optional): Specifications for the job.
            name (str, optional): The job's name.
            description (str, optional): The job's description.
            num_gpu (int, optional): The number of GPUs to allocate.
            platform_id (str, optional): The platform ID for job execution.
            from_ui (bool, optional): Indicates whether the job call is from the UI.

        Returns:
            Code: A response code object containing the status and job ID or error details:
                  - 200: Job successfully queued.
                  - 400: If job execution was unsuccessful.
                  - 404: If dataset/experiment/action not found or access is denied.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{handler_id} {kind} doesn't exist")

        user_id = handler_metadata.get("user_id")
        network_arch = handler_metadata.get("network_arch", "")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{org_name} has no write access to {handler_id}")

        if not action:
            return Code(404, [], "action not sent")

        if action not in handler_metadata.get("actions", []):
            if not (kind == "dataset" and action == "validate_images"):
                return Code(
                    404, {},
                    f"Action {action} requested not in {','.join(handler_metadata.get('actions', []))}"
                )

        if not user_id:
            return Code(
                404, [],
                "User ID couldn't be found in the experiment metadata. "
                "Try creating the experiment again"
            )

        if parent_job_id:
            parent_job_metadata = stateless_handlers.get_handler_job_metadata(parent_job_id)
            if not parent_job_metadata:
                return Code(404, [], f"Parent job {parent_job_id} not found")

            if kind != "dataset":
                parent_handler_id = parent_job_metadata.get("dataset_id")
                parent_kind = "dataset"
                if not parent_handler_id:
                    parent_kind = "experiment"
                    parent_handler_id = parent_job_metadata.get("experiment_id")
                if not parent_handler_id:
                    return Code(404, [], f"Unable to identify {parent_kind} id for parent job {parent_job_id}")

                if parent_kind == "experiment":
                    if parent_handler_id != handler_id:
                        return Code(
                            404, [],
                            f"Parent job {parent_job_id} trying to assign doesn't belong to current experiment "
                            f"{handler_id}, it belongs to experiment {parent_handler_id}"
                        )

        if BACKEND == "NVCF":
            available_nvcf_instances = get_available_nvcf_instances(user_id, org_name)
            if not available_nvcf_instances:
                platform_id = "052fc221-ffaa-5c15-8d22-b663e7339349"
            else:
                if not platform_id:
                    def get_powers_of_2(start: int):
                        power = 1
                        powers = []
                        while power <= 8:
                            if power >= start:
                                powers.append(power)
                            power *= 2
                        return powers

                    num_gpu = 1
                    if specs:
                        num_gpu = get_num_gpus_from_spec(specs, action, network=network_arch, default=1)
                    gpu_based_subset = {}
                    valid_gpu_counts = get_powers_of_2(num_gpu)
                    for nvcf_instance_id, nvcf_instance_info in available_nvcf_instances.items():
                        cluster = available_nvcf_instances[nvcf_instance_id].get('cluster', '')
                        instance_type = available_nvcf_instances[nvcf_instance_id].get('instance_type', '')
                        if cluster == 'GFN':
                            instance_type = instance_type.replace("2x", "1x").replace("4x", "2x")
                        for valid_gpu_count in valid_gpu_counts:
                            if f"{valid_gpu_count}x" in instance_type:
                                gpu_based_subset[nvcf_instance_id] = nvcf_instance_info

                    if gpu_based_subset:
                        sorted_platform_ids = sorted(
                            gpu_based_subset,
                            key=lambda x: gpu_based_subset[x]['current_available'],
                            reverse=True
                        )
                    else:
                        sorted_platform_ids = sorted(
                            available_nvcf_instances,
                            key=lambda x: available_nvcf_instances[x]['current_available'],
                            reverse=True
                        )
                    platform_id = sorted_platform_ids[0]
                if platform_id not in available_nvcf_instances:
                    return Code(
                        404, [],
                        f"Requested NVCF resource {platform_id} not available. "
                        f"Valid platform_id options are {str(available_nvcf_instances.keys())}"
                    )
                if available_nvcf_instances[platform_id]["current_available"] == 0:
                    return Code(
                        404, [],
                        f"Requested NVCF resource {platform_id} maxed out. Choose other platform_id options, "
                        f"valid options are: {str(available_nvcf_instances.keys())}"
                    )

        if kind == "experiment" and handler_metadata.get("type").lower() == "medical":
            if action not in handler_metadata.get("actions", []):
                return Code(404, {}, "Action not found")

            if not isinstance(specs, dict):
                return Code(404, [], f"{specs} must be a dictionary. Received {type(specs)}")

            default_spec = read_network_config(handler_metadata["network_arch"])["spec_params"].get(action, {})
            nested_update(specs, default_spec, allow_overwrite=False)
            if "num_gpus" in specs:
                return Code(400, [], "num_gpus is not a valid key in the specs. Use num_gpu instead.")
            num_gpu, err_msg = validate_num_gpu(specs.get("num_gpu", None), action)
            if num_gpu <= 0 and err_msg:
                return Code(400, [], err_msg)
            log_content = (
                f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, "
                f"job_type:experiment, network_arch:{network_arch}, action:{action}, "
                f"num_gpu:{num_gpu}"
            )
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_job, log_content=log_content)
            if action == "inference":
                return MonaiModelHandler.run_inference(org_name, handler_id, handler_metadata, specs)

            # regular async jobs
            if action == "annotation":
                all_metadata = stateless_handlers.get_jobs_for_handler(handler_id, kind)
                if [m for m in all_metadata if m["action"] == "annotation" and m["status"] in ("Running", "Pending")]:
                    return Code(400, [], "There is one running/pending annotation job. Please stop it first.")
                if handler_metadata.get("eval_dataset", None) is None:
                    return Code(404, {}, "Annotation job requires eval dataset in the model metadata.")

        if kind == "dataset" and handler_metadata.get("format") == "monai":
            log_content = (
                f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, "
                f"job_type:dataset, action:{action}"
            )
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_job, log_content=log_content)
            return MonaiDatasetHandler.run_job(org_name, handler_id, handler_metadata, action, specs)

        try:
            job_id = str(uuid.uuid4())
            if specs:
                spec_schema_response = AppHandler.get_spec_schema(user_id, org_name, handler_id, action, kind)
                if spec_schema_response.code == 200:
                    spec_schema = spec_schema_response.data
                    default_spec = spec_schema["default"]
                    check_and_convert(specs, default_spec)
            msg = ""
            if is_request_automl(handler_id, action, kind):
                logger.info("Creating AutoML job %s", job_id)
                AutoMLHandler.start(
                    user_id,
                    org_name,
                    handler_id,
                    job_id,
                    handler_metadata,
                    name=name,
                    platform_id=platform_id
                )
                msg = "AutoML "
            else:
                logger.info("Creating job %s", job_id)
                job_context = create_job_context(
                    parent_job_id,
                    action,
                    job_id,
                    handler_id,
                    user_id,
                    org_name,
                    kind,
                    handler_metadata=handler_metadata,
                    specs=specs,
                    name=name,
                    description=description,
                    num_gpu=num_gpu,
                    platform_id=platform_id
                )
                on_new_job(job_context)
            if specs:
                save_job_specs(job_id, specs)
            return Code(200, job_id, f"{msg}Job scheduled")
        except Exception as e:
            logger.error("Exception thrown in job_run is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(500, [], "Exception in job_run fn")

    @staticmethod
    def job_retry(org_name, handler_id, kind, job_id, from_ui=False):
        """Retries a job based on its status and metadata.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            job_id (str): UUID of the job to retry.
            from_ui (bool): Whether the retry is triggered from the UI (default is False).

        Returns:
            Code: A response indicating the result of the retry action:
                - 200 with the new job UUID if successfully queued.
                - 400 if the job status prevents retrying.
                - 404 if the job or related resources are not found.
        """
        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], f"Job {job_id} doesn't exist")
        status = job_metadata.get('status', 'Unknown')
        if status in ("Done", "Running", "Pending", "Canceling", "Resuming", "Pausing", "Paused"):
            return Code(400, [], f"Unable to retry job with {status} status")

        parent_job_id = job_metadata.get('parent_id')
        action = job_metadata.get('action')
        specs = job_metadata.get('specs')
        name = job_metadata.get('name', "Job") + " Retry"
        description = job_metadata.get('description')
        platform_id = job_metadata.get('platform_id')
        job_response = AppHandler.job_run(org_name, handler_id, parent_job_id, action, kind, specs, name, description,
                                          platform_id=platform_id, from_ui=from_ui)
        return job_response

    @staticmethod
    def job_get_epoch_numbers(user_id, org_name, handler_id, job_id, kind):
        """Retrieves the epoch numbers associated with a given job.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 with a list of epoch numbers if found.
                - 404 if no epoch numbers are found or an error occurs.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        handler_metadata = stateless_handlers.get_handler_metadata(handler_id, kind + "s")
        if not handler_metadata or "user_id" not in handler_metadata:
            return Code(404, [], "job trying to update not found")

        job = get_job(job_id)
        if not job:
            return Code(404, [], "job trying to update not found")
        try:
            job_files, _, _, _ = get_files_from_cloud(handler_metadata, job_id)
            epoch_numbers = []
            for job_file in job_files:
                # Extract numbers before the extension using regex
                match = re.search(r'(\d+)(?=\.(pth|hdf5|tlt)$)', job_file)
                if match:
                    epoch_numbers.append(match.group(1))
            return Code(200, {"data": epoch_numbers}, "Job status updated")
        except Exception:
            logger.error(traceback.format_exc())
            return Code(404, [], "Exception caught during getting epoch numbers")

    @staticmethod
    def job_status_update(org_name, handler_id, job_id, kind, callback_data):
        """Updates the status of a given job.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            callback_data (dict): Data containing the new status information.

        Returns:
            Code: A response indicating the result:
                - 200 if the job status was successfully updated.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        action = job.get("action", "")
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        automl = False
        if is_request_automl(handler_id, action, kind) and action == "train":
            automl = True
        stateless_handlers.save_dnn_status(job_id, automl, callback_data, handler_id, kind)
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_log_update(org_name, handler_id, job_id, kind, callback_data):
        """Appends log contents to the job's log file.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job.
            kind (str): Type of resource, either 'experiment' or 'dataset'.
            callback_data (dict): Data containing the log contents to append.

        Returns:
            Code: A response indicating the result:
                - 200 if the log was successfully updated.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id", None)
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        action = job.get("action", "")
        if not job or not user_id:
            return Code(404, [], "job trying to update not found")

        handler_log_root = stateless_handlers.get_handler_log_root(user_id, org_name, handler_id)
        log_file = os.path.join(handler_log_root, job_id + ".txt")
        if is_request_automl(handler_id, action, kind):
            job_root = os.path.join(stateless_handlers.get_jobs_root(user_id, org_name), job_id)
            experiment_number = callback_data.get("experiment_number", "0")
            handler_log_root = f"{job_root}/experiment_{experiment_number}"
            log_file = f"{job_root}/experiment_{experiment_number}/log.txt"
        if not os.path.exists(handler_log_root):
            os.makedirs(handler_log_root, exist_ok=True)
        with open(log_file, "a", encoding='utf-8') as file_ptr:
            file_ptr.write(callback_data["log_contents"])
            file_ptr.write("\n")
        return Code(200, [], "Job status updated")

    @staticmethod
    def job_list(user_id, org_name, handler_id, kind):
        """Retrieves a list of jobs associated with a given handler.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 with a list of jobs if found.
                - 404 if the handler or jobs are not found.
        """
        if handler_id not in ("*", "all") and not resolve_existence(kind, handler_id):
            return Code(404, [], f"{kind} not found")

        if handler_id not in ("*", "all") and not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        return_metadata = resolve_metadata_with_jobs(user_id, org_name, kind, handler_id).get("jobs", [])

        return Code(200, return_metadata, "Jobs retrieved")

    @staticmethod
    def job_cancel(org_name, handler_id, job_id, kind):
        """Cancels a job based on its current status.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job to cancel.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if the job was successfully canceled or if the job cannot be canceled due to its current status.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if job_id not in handler_metadata.get("jobs", {}).keys():
            return Code(404, [], f"Job to cancel not found in the {kind}.")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to cancel not found")
        action = job_metadata.get("action", "")

        if is_request_automl(handler_id, action, kind):
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            automl_response = AutoMLHandler.stop(user_id, org_name, handler_id, job_id)
            # Remove any pending jobs from Workflow queue
            try:
                on_delete_automl_job(job_id)
            except Exception as e:
                logger.error("Exception thrown in automl_job_cancel is %s", str(e))
                return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            return automl_response

        # If job is error / done, then cancel is NoOp
        job_status = job_metadata.get("status", "Error")

        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            return Code(
                200,
                {
                    f"Job {job_id} with current status {job_status} can't be attemped to cancel. "
                    "Current status should be one of Running, Pending, Resuming"
                }
            )
        specs = job_metadata.get("specs", None)
        use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")

        if job_status == "Pending":
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
            on_delete_job(job_id)
            jobDriver.delete(job_id, use_ngc=use_ngc)
            stateless_handlers.update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
            return Code(200, {"message": f"Pending job {job_id} cancelled"})

        if job_status == "Running":
            try:
                # Delete K8s job
                stateless_handlers.update_job_status(handler_id, job_id, status="Canceling", kind=kind + "s")
                jobDriver.delete(job_id, use_ngc=use_ngc)
                k8s_status = jobDriver.status(
                    org_name,
                    handler_id,
                    job_id,
                    kind + "s",
                    use_ngc=use_ngc,
                    automl_exp_job=False
                )
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        break
                    k8s_status = jobDriver.status(
                        org_name,
                        handler_id,
                        job_id,
                        kind + "s",
                        use_ngc=use_ngc,
                        automl_exp_job=False
                    )
                    time.sleep(5)
                stateless_handlers.update_job_status(handler_id, job_id, status="Canceled", kind=kind + "s")
                return Code(200, {"message": f"Running job {job_id} cancelled"})
            except Exception as e:
                logger.error("Exception thrown in job_cancel is %s", str(e))
                logger.error("Cancel traceback: %s", traceback.format_exc())
                return Code(404, [], "job not found in platform")
        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def job_pause(org_name, handler_id, job_id, kind):
        """Pauses a job based on its current status.

        Args:
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            job_id (str): UUID of the job to pause.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if the job was successfully paused or if the job cannot be paused due to its current status.
                - 404 if the job or related resources are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to pause not found")
        action = job_metadata.get("action", "")

        if is_request_automl(handler_id, action, kind):
            stateless_handlers.update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            automl_response = AutoMLHandler.stop(user_id, org_name, handler_id, job_id)
            # Remove any pending jobs from Workflow queue
            try:
                on_delete_automl_job(job_id)
            except Exception as e:
                logger.error("Exception thrown in automl job_pause is %s", str(e))
                return Code(200, {"message": f"job {job_id} cancelled, and no pending recommendations"})
            stateless_handlers.update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            return automl_response

        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "retrain"):
            return Code(
                404, [],
                f"Only train, distill, quantize or retrain jobs can be paused. The current action is {job_action}"
            )
        job_status = job_metadata.get("status", "Error")

        # If job is error / done, or one of cancel or pause states then pause is NoOp
        if job_status in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"]:
            return Code(
                200,
                {
                    f"Job {job_id} with current status {job_status} can't be attemped to pause. "
                    "Current status should be one of Running, Pending, Resuming"
                }
            )
        specs = job_metadata.get("specs", None)
        use_ngc = not (specs and "cluster" in specs and specs["cluster"] == "local")

        if job_status == "Pending":
            stateless_handlers.update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
            on_delete_job(job_id)
            jobDriver.delete(job_id, use_ngc=use_ngc)
            stateless_handlers.update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
            return Code(200, {"message": f"Pending job {job_id} paused"})

        if job_status == "Running":
            try:
                # Delete K8s job
                stateless_handlers.update_job_status(handler_id, job_id, status="Pausing", kind=kind + "s")
                jobDriver.delete(job_id, use_ngc=use_ngc)
                k8s_status = jobDriver.status(
                    org_name,
                    handler_id,
                    job_id,
                    kind + "s",
                    use_ngc=use_ngc,
                    automl_exp_job=False
                )
                while k8s_status in ("Done", "Error", "Running", "Pending"):
                    if k8s_status in ("Done", "Error"):
                        break
                    k8s_status = jobDriver.status(
                        org_name,
                        handler_id,
                        job_id,
                        kind + "s",
                        use_ngc=use_ngc,
                        automl_exp_job=False
                    )
                    time.sleep(5)
                stateless_handlers.update_job_status(handler_id, job_id, status="Paused", kind=kind + "s")
                return Code(200, {"message": f"Running job {job_id} paused"})
            except Exception as e:
                logger.error("Exception thrown in job_pause is %s", str(e))
                logger.error("Pause traceback: %s", traceback.format_exc())
                return Code(404, [], "job not found in platform")

        else:
            return Code(404, [], "job status not found")

    @staticmethod
    def all_job_cancel(user_id, org_name, handler_id, kind):
        """Cancels all jobs associated with a given handler.

        Args:
            user_id (str): UUID of the user.
            org_name (str): Name of the organization.
            handler_id (str): UUID of the experiment or dataset.
            kind (str): Type of resource, either 'experiment' or 'dataset'.

        Returns:
            Code: A response indicating the result:
                - 200 if all jobs within the experiment can be canceled.
                - 404 if the handler or jobs are not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        def cancel_jobs_within_handler(cancel_handler_id, cancel_kind):
            cancel_success = True
            cancel_message = ""
            jobs = stateless_handlers.get_jobs_for_handler(cancel_handler_id, cancel_kind)
            for job_metadata in jobs:
                job_id = job_metadata.get("id")
                job_status = job_metadata.get("status", "Error")
                if job_status not in ["Error", "Done", "Canceled", "Canceling", "Pausing", "Paused"] and job_id:
                    cancel_response = AppHandler.job_cancel(org_name, cancel_handler_id, job_id, cancel_kind)
                    if cancel_response.code != 200:
                        if (type(cancel_response.data) is dict and
                                cancel_response.data.get("error_desc", "") != "incomplete job not found"):
                            cancel_success = False
                            cancel_message += f"Cancelation for job {job_id} failed due to {str(cancel_response.data)} "
            return cancel_success, cancel_message

        if handler_metadata.get("all_jobs_cancel_status") == "Canceling":
            return Code(200, {"message": "Canceling all jobs is already triggered"})

        try:
            handler_metadata["all_jobs_cancel_status"] = "Canceling"
            write_handler_metadata(handler_id, handler_metadata, kind)

            appended_message = ""
            for train_dataset in handler_metadata.get("train_datasets", []):
                jobs_cancel_sucess, message = cancel_jobs_within_handler(train_dataset, "dataset")
                appended_message += message

            eval_dataset = handler_metadata.get("eval_dataset", None)
            if eval_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(eval_dataset, "dataset")
                appended_message += message

            inference_dataset = handler_metadata.get("inference_dataset", None)
            if inference_dataset:
                jobs_cancel_sucess, message = cancel_jobs_within_handler(inference_dataset, "dataset")
                appended_message += message

            jobs_cancel_sucess, message = cancel_jobs_within_handler(handler_id, kind)
            appended_message += message

            handler_metadata = resolve_metadata(kind, handler_id)
            if jobs_cancel_sucess:
                handler_metadata["all_jobs_cancel_status"] = "Canceled"
                write_handler_metadata(handler_id, handler_metadata, kind)
                return Code(200, {"message": "All jobs within experiment canceled"})
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], appended_message)
        except Exception as e:
            logger.error("Exception thrown in all_job_cancel is %s", str(e))
            logger.error(traceback.format_exc())
            handler_metadata["all_jobs_cancel_status"] = "Error"
            write_handler_metadata(handler_id, handler_metadata, kind)
            return Code(404, [], "Runtime exception caught during deleting a job")

    @staticmethod
    def job_retrieve(org_name, handler_id, job_id, kind, return_specs=False):
        """Retrieve the specified job based on its ID and kind (experiment or dataset).

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job to be retrieved.
        kind (str): The type of job, either "experiment" or "dataset".
        return_specs (bool): Flag indicating whether to return specs.

        Returns:
        Code: A response code (200 if the job is found, 404 if not found).
              - 200: Returns a dictionary following the JobResultSchema.
              - 404: Returns an empty dictionary if the job or handler is not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, {}, f"{kind} not found")

        job_meta = get_job(job_id)
        if not job_meta:
            return Code(404, {}, "Job trying to retrieve not found")
        job_meta.pop('num_gpu', None)
        if not return_specs:
            if "specs" in job_meta:
                _ = job_meta.pop("specs")
        return Code(200, job_meta, "Job retrieved")

    @staticmethod
    def publish_model(org_name, team_name, experiment_id, job_id, display_name, description):
        """Publish a model with the specified details after validating the job status.

        Parameters:
        org_name (str): The name of the organization.
        team_name (str): The name of the team.
        experiment_id (str): UUID corresponding to the experiment.
        job_id (str): UUID corresponding to the job.
        display_name (str): Display name for the model.
        description (str): Description for the model.

        Returns:
        Code: A response code (200 if the model is successfully published, 404 or 403 for errors).
              - 200: Model successfully created and uploaded.
              - 404: If experiment, job, or relevant files are not found.
              - 403: If the user does not have permission to publish the model.
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {}, "Experiment not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "Experiment cant be read")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {}, "Job trying to retrieve not found")

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(404, {}, "Job is not in success or Done state")
        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(
                404,
                {},
                "Publish model is available only for train, distill, quantize, prune, retrain, export, "
                "gen_trt_engine actions"
            )

        try:
            network_arch = handler_metadata.get('network_arch')
            source_files = []
            if job_action == 'gen_trt_engine' and network_arch in MAXINE_NETWORKS:
                encoder_regex = r'.*encoder.*\.(engine|engine\.trtpkg)$'
                encoder_file = resolve_checkpoint_root_and_search(handler_metadata, job_id, regex=encoder_regex)
                if encoder_file:
                    source_files.append(encoder_file)
                decoder_regex = r'.*decoder.*\.(engine|engine\.trtpkg)$'
                decoder_file = resolve_checkpoint_root_and_search(handler_metadata, job_id, regex=decoder_regex)
                if decoder_file:
                    source_files.append(decoder_file)
            else:
                source_file = resolve_checkpoint_root_and_search(handler_metadata, job_id)
                source_files.append(source_file)
            if not source_files:
                return Code(404, [], "Unable to find a model for the given job")

            # Create NGC model
            ngc_key = ngc_handler.get_user_key(user_id, org_name)
            if not ngc_key:
                return Code(403, {}, "User does not have access to publish model")

            code, message = ngc_handler.create_model(
                org_name, team_name, handler_metadata, source_files[0], ngc_key, display_name, description
            )
            if code not in [200, 201]:
                logger.error("Error while creating NGC model")
                return Code(code, {}, message)

            # Upload model version
            response_code, response_message = ngc_handler.upload_model(
                org_name, team_name, handler_metadata, source_files, ngc_key, job_id, job_action
            )
            if "already exists" in response_message:
                response_message = (
                    "Version trying to upload already exists, use remove_published_model endpoint to reupload the model"
                )
            return Code(response_code, {}, response_message)
        except Exception as e:
            logger.error("Exception thrown in publish_model is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, {}, "Unable to publish model")

    @staticmethod
    def remove_published_model(org_name, team_name, experiment_id, job_id):
        """Remove a previously published model.

        Parameters:
        org_name (str): The name of the organization.
        team_name (str): The name of the team.
        experiment_id (str): UUID corresponding to the experiment.
        job_id (str): UUID corresponding to the job.

        Returns:
        Code: A response code (200 if the model is successfully removed, 404 for errors).
              - 200: Successfully deleted the model.
              - 404: If experiment, job, or the published model is not found.
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(404, {}, "Experiment not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "Experiment cant be read")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, {}, "Job trying to retrieve not found")

        job_status = job_metadata.get("status", "Error")
        if job_status not in ("Success", "Done"):
            return Code(404, {}, "Job is not in success or Done state")
        job_action = job_metadata.get("action", "")
        if job_action not in ("train", "distill", "quantize", "prune", "retrain", "export", "gen_trt_engine"):
            return Code(
                404,
                {},
                "Delete published model is available only for train, distill, ",
                "quantize, prune, retrain, export, gen_trt_engine actions"
            )

        try:
            ngc_key = ngc_handler.get_user_key(user_id, org_name)
            if not ngc_key:
                return Code(403, {}, "User does not have access to remove published model")

            response = ngc_handler.delete_model(
                org_name, team_name, handler_metadata, ngc_key, job_id, job_action
            )
            if response.ok:
                return Code(response.status_code, {}, "Sucessfully deleted model")
            return Code(response.status_code, {}, "Unable to delete published model")
        except Exception as e:
            logger.error("Exception thrown in remove_published_model is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, {}, "Unable to delete published model")

    # Delete job
    @staticmethod
    def job_delete(org_name, handler_id, job_id, kind):
        """Delete the specified job.

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job to be deleted.
        kind (str): The type of job, either "experiment" or "dataset".

        Returns:
        Code: A response code:
                 - 200 if the job is successfully deleted
                 - 404 if not found
                 - 400 if deletion is not allowed
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "job trying to delete not found")

        try:
            # If job is running, cannot delete
            if job_metadata.get("status", "Error") in ["Running", "Pending"]:
                return Code(400, [], "job cannot be deleted")
            # Delete job metadata
            mongo_jobs = MongoHandler("tao", "jobs")
            mongo_jobs.delete_one({'id': job_id})
            # Delete job from handler metadata
            if "jobs" in handler_metadata and job_id in handler_metadata["jobs"]:
                del handler_metadata["jobs"][job_id]
                handler_metadata["status"] = get_handler_status(handler_metadata)
                handler_metadata["last_modified"] = datetime.now(tz=timezone.utc)
                write_handler_metadata(handler_id, handler_metadata, kind)
            # Delete job logs
            job_log_path = os.path.join(
                stateless_handlers.get_handler_log_root(user_id, org_name, handler_id),
                job_id + ".txt"
            )
            if os.path.exists(job_log_path):
                os.remove(job_log_path)
            return Code(200, [job_id], "job deleted")
        except Exception as e:
            logger.error("Exception thrown in job_delete is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, [], "job cannot be deleted")

    # Download experiment job
    @staticmethod
    def job_download(
        org_name,
        handler_id,
        job_id,
        kind,
        file_lists=None,
        best_model=None,
        latest_model=None,
        tar_files=True,
        export_type="tao"
    ):
        """Download files associated with the specified job.

        Parameters:
        org_name (str): The name of the organization.
        handler_id (str): UUID corresponding to the experiment or dataset.
        job_id (str): UUID corresponding to the job.
        kind (str): The type of job, either "experiment" or "dataset".
        file_lists (list, optional): List of files to download (defaults to None).
        best_model (bool, optional): Whether to download the best model (defaults to None).
        latest_model (bool, optional): Whether to download the latest model (defaults to None).
        tar_files (bool, optional): Whether to compress the downloaded files into a tarball (defaults to True).
        export_type (str, optional): The type of export (defaults to "tao").

        Returns:
        Code: A response code (200 if the files are successfully downloaded, 404 if not found, 400 for errors).
              - 200: Successfully downloaded the files.
              - 404: If the job or handler is not found.
              - 400: If there are errors during file download.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, None, f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, None, f"{kind} not found")

        handler_job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not handler_job_metadata:
            return Code(404, None, "job trying to download not found")

        try:
            if export_type not in VALID_MODEL_DOWNLOAD_TYPE:
                return Code(404, None, f"Export format {export_type} not found.")
            root = stateless_handlers.get_jobs_root(user_id, org_name)

            # Following is for `if export_type == "tao":`
            # Copy job logs from root/logs/<job_id>.txt to root/<job_id>/logs_from_toolkit.txt
            out_tar = os.path.join(root, job_id + ".tar.gz")
            files = [os.path.join(root, job_id)]
            if file_lists or best_model or latest_model:
                files = []
                for file in file_lists:
                    if os.path.exists(os.path.join(root, file)):
                        files.append(os.path.join(root, file))
                action = handler_job_metadata.get("action", "")
                epoch_number_dictionary = handler_metadata.get("checkpoint_epoch_number", {})
                best_checkpoint_epoch_number = epoch_number_dictionary.get(f"best_model_{job_id}", 0)
                latest_checkpoint_epoch_number = epoch_number_dictionary.get(f"latest_model_{job_id}", 0)
                if (not best_model) and latest_model:
                    best_checkpoint_epoch_number = latest_checkpoint_epoch_number
                network = handler_metadata.get("network_arch", "")
                if network in MISSING_EPOCH_FORMAT_NETWORKS:
                    format_epoch_number = str(best_checkpoint_epoch_number)
                else:
                    format_epoch_number = f"{best_checkpoint_epoch_number:03}"
                if best_model or latest_model:
                    job_root = os.path.join(root, job_id)
                    if (handler_metadata.get("automl_settings", {}).get("automl_enabled") is True and
                       action in ("train", "distill", "quantize")):
                        job_root = os.path.join(job_root, "best_model")
                    find_trained_tlt = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.tlt") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.tlt") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.tlt")
                    )
                    find_trained_pth = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.pth") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.pth") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.pth")
                    )
                    find_trained_hdf5 = (
                        glob.glob(f"{job_root}/*{format_epoch_number}.hdf5") +
                        glob.glob(f"{job_root}/train/*{format_epoch_number}.hdf5") +
                        glob.glob(f"{job_root}/weights/*{format_epoch_number}.hdf5")
                    )
                    if find_trained_tlt:
                        files.append(find_trained_tlt[0])
                    if find_trained_pth:
                        files.append(find_trained_pth[0])
                    if find_trained_hdf5:
                        files.append(find_trained_hdf5[0])
                if not files:
                    return Code(404, None, "Atleast one of the requested files not present")

            log_root = get_handler_log_root(user_id, org_name, handler_id)
            log_file = glob.glob(f"{log_root}/**/*{job_id}.txt", recursive=True)
            files = list(set(files))

            if tar_files or (not tar_files and len(files) > 1) or files == [os.path.join(root, job_id)]:
                if files == [os.path.join(root, job_id)]:
                    files += log_file
                    files = list(set(files))

                def get_files_recursively(directory):
                    return [
                        file for file in glob.glob(os.path.join(directory, '**'), recursive=True)
                        if os.path.isfile(file) and not file.endswith(".lock")
                    ]
                all_files = []
                for file in files:
                    if os.path.isdir(file):
                        all_files.extend(get_files_recursively(file))
                    elif os.path.isfile(file):
                        all_files.append(file)

                # Appending UUID to not overwrite the tar file created at end of job complete
                out_tar = out_tar.replace(
                    ".tar.gz",
                    str(uuid.uuid4()) + ".tar.gz"
                )
                with tarfile.open(out_tar, "w:gz") as tar:
                    for file_path in all_files:
                        tar.add(file_path, arcname=file_path.replace(root, "", 1).replace(log_root, "", 1))
                return Code(200, out_tar, "selective files of job downloaded")

            if files and os.path.exists(os.path.join(root, files[0])):
                return Code(200, os.path.join(root, files[0]), "single file of job downloaded")
            return Code(404, None, "job output not found")

        except Exception as e:
            logger.error("Exception thrown in job_download is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(404, None, "job output not found")

    @staticmethod
    def job_list_files(org_name, handler_id, job_id, kind):
        """Lists the files associated with a specific job.

        Args:
            org_name (str): The name of the organization.
            handler_id (str): The UUID corresponding to the experiment or dataset.
            job_id (str): The UUID of the job whose files need to be listed.
            kind (str): The type of handler, either 'experiment' or 'dataset'.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 with a list of file paths if files are found.
                - 404 with an error message if no files are found or access is denied.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, [], f"{kind} not found")

        user_id = handler_metadata.get("user_id")
        if not check_read_access(user_id, org_name, handler_id, kind=kind + "s"):
            return Code(404, [], f"{kind} not found")

        job = get_job(job_id)
        if not job:
            return Code(404, None, "job trying to view not found")

        files, _, _, _ = get_files_from_cloud(handler_metadata, job_id)
        if files:
            return Code(200, files, "Job files retrieved")
        return Code(200, files, "No downloadable files for this job is found")

    # Get realtime job logs
    @staticmethod
    def get_job_logs(org_name, handler_id, job_id, kind, automl_experiment_index=None):
        """Retrieves real-time logs for a specific job.

        Args:
            org_name (str): The name of the organization.
            handler_id (str): The UUID corresponding to the experiment or dataset.
            job_id (str): The UUID of the job for which logs are being retrieved.
            kind (str): The type of handler, either 'experiment' or 'dataset'.
            automl_experiment_index (int, optional): The index of the AutoML experiment, if applicable.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 with the log content or a detailed message if logs are not yet available.
                - 404 if the log file is not found.
        """
        handler_metadata = resolve_metadata(kind, handler_id)
        if not handler_metadata:
            return Code(404, {}, f"{kind} not found.")

        # Get log file path
        # Normal action log is saved at /orgs/<org_name>/users/<user_id>/logs/<job_id>.txt
        # AutoML train  log is saved at:
        # /orgs/<org_name>/users/<user_id>/jobs/<job_id>/experiment_<recommendation_index>/log.txt
        user_id = handler_metadata.get("user_id")
        log_file_path = os.path.join(get_handler_log_root(user_id, org_name, handler_id), str(job_id) + ".txt")
        job_metadata = get_handler_job_metadata(job_id)
        automl_index = None
        if (handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and
                job_metadata.get("action", "") == "train"):
            root = os.path.join(get_jobs_root(user_id, org_name), job_id)
            automl_index = get_automl_current_rec(job_id)
            if automl_experiment_index is not None:
                automl_index = int(automl_experiment_index)
            log_file_path = os.path.join(root, f"experiment_{automl_index}", "log.txt")

        if (job_metadata.get("status", "") not in ("Done", "Error", "Canceled", "Paused") or
                not os.path.exists(log_file_path)):
            workspace_id = handler_metadata.get("workspace", "")
            if not workspace_id:
                return Code(404, {}, "Handler doesn't have workspace assigned, can't download logs.")
            download_log_from_cloud(handler_metadata, job_id, log_file_path, automl_index)

        # File not present - Use detailed message or job status
        if not os.path.exists(log_file_path):
            detailed_result_msg = (
                job_metadata.get("job_details", {})
                .get(job_id, {})
                .get("detailed_status", {})
                .get("message", "")
            )
            if detailed_result_msg:
                return Code(200, detailed_result_msg)

            if (handler_metadata.get("automl_settings", {}).get("automl_enabled", False) and
                    job_metadata.get("action", "") == "train"):
                if handler_metadata.get("status") in ["Canceled", "Canceling"]:
                    return Code(200, "AutoML training has been canceled.")
                if handler_metadata.get("status") in ["Paused", "Pausing"]:
                    return Code(200, "AutoML training has been paused.")
                if handler_metadata.get("status") == "Resuming":
                    return Code(200, "AutoML training is resuming.")
                if handler_metadata.get("status") == "Running":
                    return Code(200, "Generating new recommendation for AutoML experiment.")
            return Code(404, {}, "Logs for the job are not available yet.")
        return Code(200, get_job_logs(log_file_path))

    # Experiment API
    @staticmethod
    def list_experiments(user_id, org_name, user_only=False):
        """Lists experiments accessible by the user.

        Args:
            user_id (str): The UUID of the user.
            org_name (str): The name of the organization.
            user_only (bool): Flag to indicate whether to list only experiments owned by the user.

        Returns:
            list(dict): A list of dictionaries containing metadata of experiments accessible by the user.
        """
        # Collect all metadatas
        metadatas = []
        for experiment_id in list(set(get_org_experiments(org_name))):
            handler_metadata = stateless_handlers.get_handler_metadata(experiment_id, "experiments")
            shared_experiment = handler_metadata.get("shared", False)
            if handler_metadata:
                if shared_experiment or handler_metadata.get("user_id") == user_id:
                    handler_metadata = sanitize_handler_metadata(handler_metadata)
                    handler_metadata["status"] = get_handler_status(handler_metadata)
                    metadatas.append(handler_metadata)
        if not user_only:
            maxine_request = handler_level_access_control(user_id, org_name, base_experiment=True)
            public_experiments_metadata = stateless_handlers.get_public_experiments(maxine=maxine_request)
            metadatas += public_experiments_metadata
        return metadatas

    @staticmethod
    def list_base_experiments(user_id, org_name):
        """Lists public base experiments.

        Returns:
            list(dict): A list of dictionaries containing metadata of publicly accessible base experiments.
        """
        # Collect all metadatas
        metadatas = []
        maxine_request = handler_level_access_control(user_id, org_name, base_experiment=True)
        public_experiments_metadata = stateless_handlers.get_public_experiments(maxine=maxine_request)
        metadatas += public_experiments_metadata
        return metadatas

    @staticmethod
    def load_airgapped_experiments(user_id, org_name, workspace_id):
        """Load airgapped experiments from cloud storage using workspace credentials.

        Args:
            user_id (str): The UUID of the user.
            org_name (str): The name of the organization.
            workspace_id (str): The UUID of the workspace containing cloud credentials.
            models_base_dir (str, optional): Base directory for searching model files.

        Returns:
            Code: A response object indicating the result of the operation.
                - 200 if experiments are loaded successfully.
                - 400 if there's an error with the request or configuration.
                - 403 if access denied to workspace.
                - 404 if workspace not found.
        """
        # Get workspace metadata
        workspace_metadata = stateless_handlers.get_handler_metadata(workspace_id, "workspaces")
        if not workspace_metadata:
            return Code(404, {"error_desc": f"Workspace {workspace_id} not found", "error_code": 1},
                        f"Workspace {workspace_id} not found")

        # Check workspace access
        workspace_user_id = workspace_metadata.get('user_id')
        if workspace_user_id != user_id:
            return Code(403, {"error_desc": "Access denied to workspace", "error_code": 1},
                        "Access denied to workspace")

        # Map workspace cloud credentials to cloud_config format
        cloud_type = workspace_metadata.get("cloud_type", "seaweedfs")
        cloud_specific_details = workspace_metadata.get("cloud_specific_details", {})

        cloud_config = {
            "cloud_type": cloud_type,
            "bucket_name": cloud_specific_details.get("cloud_bucket_name", "tao-storage"),
            "region": cloud_specific_details.get("cloud_region"),
            "access_key": cloud_specific_details.get("access_key"),
            "secret_key": cloud_specific_details.get("secret_key"),
            "endpoint_url": cloud_specific_details.get("endpoint_url")
        }

        # Initialize and run airgapped loader (dry_run=False to save to MongoDB)
        try:
            loader = AirgappedExperimentLoader(
                cloud_config=cloud_config
            )

            # Load and import experiments to MongoDB
            success = loader.load_and_import()

            if success:
                return_metadata = {
                    "success": True,
                    "message": "Successfully loaded airgapped experiments to MongoDB",
                    "experiments_loaded": 1,  # We don't have exact counts from the loader
                    "experiments_failed": 0
                }
                return Code(200, return_metadata, "Successfully loaded airgapped experiments to MongoDB")
            return_metadata = {
                "success": False,
                "message": "Failed to load airgapped experiments",
                "experiments_loaded": 0,
                "experiments_failed": 1
            }
            return Code(400, return_metadata, "Failed to load airgapped experiments")

        except Exception as e:
            return_metadata = {
                "success": False,
                "message": f"Error loading airgapped experiments: {str(e)}",
                "experiments_loaded": 0,
                "experiments_failed": 1,
                "error_desc": str(e),
                "error_code": 1
            }
            return Code(400, return_metadata, f"Error loading airgapped experiments: {str(e)}")
        finally:
            # Clean up
            try:
                if 'loader' in locals():
                    loader.cleanup()
            except Exception:
                pass  # Ignore cleanup errors

    @staticmethod
    def create_experiment(user_id, org_name, request_dict, experiment_id=None, from_ui=False):
        """Creates a new experiment with the specified metadata.

        Args:
            user_id (str): The UUID of the user.
            org_name (str): The name of the organization.
            request_dict (dict): A dictionary containing the experiment details, adhering to the `ExperimentReqSchema`.
            experiment_id (str, optional): The ID of the experiment to be created (auto-generated if not provided).
            from_ui (bool): Flag indicating if the experiment creation request originated from the UI.

        Returns:
            Code: A response object indicating the result of the experiment creation.
                - 200 if the experiment is created successfully with metadata.
                - 400 if invalid data is provided in the request (e.g., missing or incorrect fields).
        """
        # Create a dataset ID and its root
        experiment_id = experiment_id or str(uuid.uuid4())

        workspace_id = request_dict.get("workspace", None)
        if not check_read_access(user_id, org_name, workspace_id, kind="workspaces"):
            return Code(404, None, f"Workspace {workspace_id} not found")

        # Gather type,format fields from request
        mdl_nw = request_dict.get("network_arch", None)
        # Perform basic checks - valid type and format?
        if mdl_nw not in ExperimentNetworkArch.__members__:
            msg = "Invalid network arch"
            return Code(400, {}, msg)

        if request_dict.get("public", False):
            stateless_handlers.add_public_experiment(experiment_id)

        mdl_type = request_dict.get("type", "vision")
        if str(mdl_nw).startswith("monai_"):
            mdl_type = "medical"

        # Create metadata dict and create some initial folders
        # Initially make datasets, base_experiment None
        metadata = {"id": experiment_id,
                    "user_id": user_id,
                    "org_name": org_name,
                    "authorized_party_nca_id": request_dict.get("authorized_party_nca_id", ""),
                    "created_on": datetime.now(tz=timezone.utc),
                    "last_modified": datetime.now(tz=timezone.utc),
                    "name": request_dict.get("name", "My Experiment"),
                    "shared": request_dict.get("shared", False),
                    "description": request_dict.get("description", "My Experiments"),
                    "version": request_dict.get("version", "1.0.0"),
                    "logo": request_dict.get("logo", "https://www.nvidia.com"),
                    "ngc_path": request_dict.get("ngc_path", ""),
                    "encryption_key": request_dict.get("encryption_key", "tlt_encode"),
                    "read_only": request_dict.get("read_only", False),
                    "public": request_dict.get("public", False),
                    "network_arch": mdl_nw,
                    "type": mdl_type,
                    "dataset_type": read_network_config(mdl_nw)["api_params"]["dataset_type"],
                    "dataset_formats": read_network_config(mdl_nw)["api_params"].get(
                        "formats",
                        read_network_config(
                            read_network_config(mdl_nw)["api_params"]["dataset_type"]
                        ).get("api_params", {}).get("formats", None)
                    ),
                    "accepted_dataset_intents": read_network_config(mdl_nw)["api_params"].get(
                        "accepted_ds_intents",
                        []
                    ),
                    "actions": read_network_config(mdl_nw)["api_params"]["actions"],
                    "docker_env_vars": request_dict.get("docker_env_vars", {}),
                    "train_datasets": [],
                    "eval_dataset": None,
                    "inference_dataset": None,
                    "additional_id_info": None,
                    "checkpoint_choose_method": request_dict.get("checkpoint_choose_method", "best_model"),
                    "checkpoint_epoch_number": request_dict.get("checkpoint_epoch_number", {}),
                    "calibration_dataset": None,
                    "base_experiment": [],
                    "automl_settings": request_dict.get("automl_settings", {}),
                    "metric": request_dict.get("metric", "kpi"),
                    "realtime_infer": False,
                    "realtime_infer_support": False,
                    "realtime_infer_endpoint": None,
                    "realtime_infer_model_name": None,
                    "realtime_infer_request_timeout": request_dict.get("realtime_infer_request_timeout", 60),
                    "model_params": request_dict.get("model_params", {}),
                    "tensorboard_enabled": request_dict.get("tensorboard_enabled", False),
                    "workspace": request_dict.get("workspace", None),
                    "experiment_actions": request_dict.get('experiment_actions', []),
                    "tags": list({t.lower(): t for t in request_dict.get("tags", [])}.values()),
                    }

        if not handler_level_access_control(user_id, org_name, experiment_id, "experiments", handler_metadata=metadata):
            return Code(403, {}, "Not allowed to work with this org")

        if metadata.get("automl_settings", {}).get("automl_enabled") and mdl_nw in AUTOML_DISABLED_NETWORKS:
            return Code(400, {}, "automl_enabled cannot be True for unsupported network")
        if metadata.get("automl_settings", {}).get("automl_enabled") and BACKEND == "NVCF":
            return Code(400, {}, "Automl not supported on NVCF backend, use baremetal deployments of TAO-API")

        if BACKEND == "NVCF" and metadata.get("tensorboard_enabled", False):
            return Code(400, {}, "Tensorboard not supported on NVCF backend, use baremetal deployments of TAO-API")
        if mdl_nw in TAO_NETWORKS and (not metadata.get("workspace")):
            return Code(400, {}, "Workspace must be provided for experiment creation")
        if not ingress_enabled and metadata.get("tensorboard_enabled", False):
            return Code(400, {}, "Tensorboard not available without ingress-nginx")

        automl_validation_error = validate_automl_settings(metadata.get("automl_settings", {}))
        if automl_validation_error:
            return Code(400, {}, automl_validation_error)

        # Encrypt the MLOPs keys
        config_path = os.getenv("VAULT_SECRET_PATH", None)
        if config_path and metadata["docker_env_vars"]:
            encryption = NVVaultEncryption(config_path)
            for key, value in metadata["docker_env_vars"].items():
                if encryption.check_config()[0]:
                    metadata["docker_env_vars"][key] = encryption.encrypt(value)
                elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                    return Code(400, {}, "Vault service does not work, can't enable MLOPs services")

        # Update datasets and base_experiments if given.
        # "realtime_infer" will be checked later, since in some cases (in MEDICAL_CUSTOM_ARCHITECT),
        # need to prepare base_experiment first
        metadata, error_code = validate_and_update_experiment_metadata(
            user_id,
            org_name,
            request_dict,
            metadata,
            [
                "train_datasets",
                "eval_dataset",
                "inference_dataset",
                "calibration_dataset",
                "base_experiment"
            ]
        )
        if error_code:
            return error_code

        def clean_on_error(experiment_id=experiment_id):
            mongo_experiments = MongoHandler("tao", "experiments")
            mongo_experiments.delete_one({'id': experiment_id})

        if mdl_type == "medical":
            is_custom_bundle = metadata["network_arch"] in MEDICAL_CUSTOM_ARCHITECT
            is_auto3seg_inference = metadata["network_arch"] == "monai_automl_generated"
            no_ptm = (metadata["base_experiment"] is None) or (len(metadata["base_experiment"]) == 0)
            if no_ptm and is_custom_bundle:
                # If base_experiment is not provided, then we will need to create a model
                # to host the files downloaded from NGC.
                bundle_url = request_dict.get("bundle_url", None)
                if bundle_url is None:
                    return Code(
                        400,
                        {},
                        "Either `bundle_url` or `ngc_path` needs to be defined for MONAI Custom Model."
                    )
                base_experiment_id = str(uuid.uuid4())
                ptm_metadata = metadata.copy()
                ptm_metadata["id"] = base_experiment_id
                ptm_metadata["name"] = "base_experiment_" + metadata["name"]
                ptm_metadata["description"] = " PTM auto-generated. " + metadata["description"]
                ptm_metadata["train_datasets"] = []
                ptm_metadata["eval_dataset"] = None
                ptm_metadata["inference_dataset"] = None
                # since "realtime_infer" is not updated by update_metadata,
                # specify it from request_dict here first for download.
                ptm_metadata["realtime_infer"] = request_dict.get("realtime_infer", False)
                ptm_metadata["realtime_infer_support"] = ptm_metadata["realtime_infer"]
                write_handler_metadata(base_experiment_id, ptm_metadata, "experiment")
                # Download it from the provided url
                download_from_url(bundle_url, base_experiment_id)

                # The base_experiment is downloaded, now we need to make sure it is correct.
                bundle_checks = []
                if ptm_metadata["realtime_infer"]:
                    bundle_checks.append("infer")
                ptm_file = validate_monai_bundle(base_experiment_id, checks=bundle_checks)
                if (ptm_file is None) or (not os.path.isdir(ptm_file)):
                    clean_on_error(experiment_id=base_experiment_id)
                    return Code(
                        400,
                        {},
                        "Failed to download base experiment, or the provided bundle does not follow "
                        "MONAI bundle format."
                    )

                ptm_metadata["base_experiment_pull_complete"] = "pull_complete"
                write_handler_metadata(base_experiment_id, ptm_metadata, "experiment")
                bundle_url_path = os.path.join(
                    resolve_root(org_name, "experiment", base_experiment_id),
                    CUSTOMIZED_BUNDLE_URL_FILE
                )
                safe_dump_file(bundle_url_path, {CUSTOMIZED_BUNDLE_URL_KEY: bundle_url})
                metadata["base_experiment"] = [base_experiment_id]
            elif no_ptm and is_auto3seg_inference:
                bundle_url = request_dict.get("bundle_url", None)
                if bundle_url is None:
                    return Code(
                        400,
                        {},
                        "Either `bundle_url` or `ngc_path` needs to be defined for MONAI Custom Model."
                    )
                bundle_url_path = os.path.join(
                    resolve_root(org_name, "experiment", experiment_id),
                    CUSTOMIZED_BUNDLE_URL_FILE
                )
                os.makedirs(resolve_root(org_name, "experiment", experiment_id), exist_ok=True)
                safe_dump_file(bundle_url_path, {CUSTOMIZED_BUNDLE_URL_KEY: bundle_url})

            network_arch = metadata.get("network_arch", None)
            log_content = (
                f"user_id:{user_id}, org_name:{org_name}, from_ui:{from_ui}, "
                f"network_arch:{network_arch}, action:creation, no_ptm:{no_ptm}"
            )
            log_monitor(log_type=DataMonitorLogTypeEnum.medical_experiment, log_content=log_content)

        # check "realtime_infer"
        metadata, error_code = validate_and_update_experiment_metadata(
            user_id,
            org_name,
            request_dict,
            metadata,
            ["realtime_infer"]
        )
        if error_code:
            return error_code

        if mdl_type == "medical" and metadata["realtime_infer"]:
            base_experiment_id = metadata["base_experiment"][0]
            model_params = metadata["model_params"]
            job_id = None
            if metadata["network_arch"] not in MEDICAL_CUSTOM_ARCHITECT:
                # Need to download the base_experiment to set up the TIS for realtime infer
                # Customizd model already has the base_experiment downloaded in the previous step thus skip here
                additional_id_info = request_dict.get("additional_id_info", None)
                job_id = additional_id_info if additional_id_info and is_valid_uuid4(additional_id_info) else None
                if not job_id:
                    return Code(
                        400,
                        {},
                        f"Non-NGC base_experiment {base_experiment_id} needs job_id in the request for path location"
                    )
            success, model_name, msg, bundle_metadata = prep_tis_model_repository(
                model_params,
                base_experiment_id,
                org_name,
                user_id,
                experiment_id,
                job_id=job_id
            )
            if not success:
                clean_on_error(experiment_id=experiment_id)
                return Code(400, {}, msg)

            # If Labels not supplied then pick from Model Bundle
            if model_params.get("labels", None) is None:
                _, pred = next(iter(bundle_metadata["network_data_format"]["outputs"].items()))
                labels = {k: v.lower() for k, v in pred.get("channel_def", {}).items() if v.lower() != "background"}
                model_params["labels"] = labels
            else:
                labels = model_params.get("labels")
                if len(labels) == 0:
                    return Code(400, {}, "Labels cannot be empty")

            # get replicas data to determine if create multiple replicas
            replicas = model_params.get("replicas", 1)
            success, msg = CapGpuUsage.schedule(org_name, replicas)
            if not success:
                return Code(400, {}, msg)
            response = TISHandler.start(org_name, experiment_id, metadata, model_name, replicas)
            if response.code != 200:
                TISHandler.stop(experiment_id, metadata)
                CapGpuUsage.release_used(org_name, replicas)
                clean_on_error(experiment_id=experiment_id)
                return response
            metadata["realtime_infer_endpoint"] = response.data["pod_ip"]
            metadata["realtime_infer_model_name"] = model_name
            metadata["realtime_infer_support"] = True

        # Create Tensorboard deployment if enabled
        if metadata.get("tensorboard_enabled", False):
            if mdl_nw in TENSORBOARD_DISABLED_NETWORKS:
                clean_on_error(experiment_id)
                return Code(400, {}, f"Network {mdl_nw} not supported for Tensorboard")
            workspace_id = request_dict.get("workspace", None)
            if TensorboardHandler.check_user_metadata(user_id) and workspace_id:
                response = TensorboardHandler.start(org_name, experiment_id, user_id, workspace_id)
                if response.code != 200:
                    TensorboardHandler.stop(experiment_id, user_id)
                    clean_on_error(experiment_id)
                    return response
            else:
                clean_on_error(experiment_id)
                return Code(
                    400,
                    {},
                    f"Maximum of {TENSORBOARD_EXPERIMENT_LIMIT} Tensorboard Experiments allowed per user."
                )

        # Actual "creation" happens here...
        write_handler_metadata(experiment_id, metadata, "experiment")

        mongo_users = MongoHandler("tao", "users")
        experiments = get_user_experiments(user_id, mongo_users)
        experiments.append(experiment_id)
        if mdl_type == 'medical':
            for base_exp in metadata.get("base_experiment", []):
                experiments.append(base_exp)
        mongo_users.upsert({'id': user_id}, {'id': user_id, 'experiments': experiments})

        experiment_actions = request_dict.get('experiment_actions', [])
        retry_experiment_id = request_dict.get('retry_experiment_id', None)
        error_response = None
        if retry_experiment_id:
            error_response = AppHandler.retry_experiment(org_name, user_id, retry_experiment_id, experiment_id, from_ui)
        elif experiment_actions:
            error_response = AppHandler.retry_experiment_actions(
                user_id,
                org_name,
                experiment_id,
                experiment_actions,
                from_ui
            )
        if error_response:
            clean_on_error(experiment_id)
            return error_response

        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Experiment created")

        # TODO: may need to call "monai_triton_client" with dummy request to accelerate
        return ret_Code

    # Retry existing experiment
    @staticmethod
    def retry_experiment(org_name, user_id, retry_experiment_id, new_experiment_id, from_ui):
        """Retries all jobs within an experiment by reusing the specs from already run jobs.

        Args:
            org_name (str): Organization name.
            user_id (str): User ID initiating the retry.
            retry_experiment_id (str): ID of the experiment whose jobs are to be retried.
            new_experiment_id (str): ID of the new experiment to create jobs for.
            from_ui (bool): Indicates whether the retry was triggered from the UI.

        Returns:
            Response: A response indicating the outcome of the operation (201 for success, error response otherwise).
        """
        handler_metadata = stateless_handlers.get_handler_metadata_with_jobs(retry_experiment_id, "experiments")
        handler_jobs = handler_metadata.get("jobs", [])
        job_map = {}
        for job in handler_jobs:
            job_id = job.get('id')
            job_action = job.get('action')
            if job_id and job_action:
                logger.info("Loading existing specs from job %s", job_id)
                specs = get_job_specs(job_id)
                name = job.get('name')
                description = job.get('description')
                retry_parent_job_id = job.get('parent_id', None)
                parent_job_id = None
                if retry_parent_job_id:
                    retry_parent_job = get_job(retry_parent_job_id)
                    parent_action = retry_parent_job.get('action')
                    parent_job_id = job_map.get(parent_action, None)
                response = AppHandler.job_run(
                    org_name=org_name,
                    handler_id=new_experiment_id,
                    parent_job_id=parent_job_id,
                    action=job_action,
                    kind='experiment',
                    specs=specs,
                    name=name,
                    description=description,
                    from_ui=from_ui
                )
                if response.code == 200:
                    job_id = response.data
                    job_map[job_action] = job_id
                    logger.info(
                        f"Created {job_action} job with id {job_id} for experiment {new_experiment_id}"
                    )
                else:
                    return response
        return None

    # Retry experiment actions
    @staticmethod
    def retry_experiment_actions(user_id, org_name, experiment_id, experiment_actions, from_ui):
        """Retries specific jobs within an experiment based on the provided actions.

        Args:
            user_id (str): User ID initiating the retry.
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to retry jobs within.
            experiment_actions (list): List of actions with details to be retried.
            from_ui (bool): Indicates whether the retry was triggered from the UI.

        Returns:
            Response: A response indicating the outcome of the operation (201 for success, error response otherwise).
        """
        raw_actions = []
        action_lookup = {}
        for action_dict in experiment_actions:
            action = action_dict.get('action')
            specs = action_dict.get('specs', {})
            name = action_dict.get('name')
            description = action_dict.get('description')
            num_gpu = action_dict.get('num_gpu', -1)
            platform_id = action_dict.get('platform_id', None)
            action_data = {
                'specs': specs,
                'name': name,
                'description': description,
                'num_gpu': num_gpu,
                'platform_id': platform_id
            }
            if action:
                raw_actions.append(action)
                action_lookup[action] = action_data

        if raw_actions and action_lookup:
            job_mapping = stateless_handlers.validate_chained_actions(raw_actions)
            if not job_mapping:
                return Code(400, {}, "Invalid workflow chaining")

            job_action_to_id = {}
            for mapping in job_mapping:
                child_action = mapping.get('child')
                parent_action = mapping.get('parent', None)
                if child_action in action_lookup:
                    lookup_data = action_lookup[child_action]
                    specs = {}
                    if not specs and not lookup_data.get('specs', {}):
                        specs_response = AppHandler.get_spec_schema(
                            user_id,
                            org_name,
                            experiment_id,
                            child_action,
                            'experiment'
                        )
                        if specs_response.code == 200:
                            spec_schema = specs_response.data
                            specs = spec_schema["default"]
                            logger.info("Retrieved specs from DNN: %s", specs)
                        else:
                            return specs_response
                    else:
                        specs = action_lookup[child_action].get('specs', {})
                    name = lookup_data.get('name')
                    description = lookup_data.get('description')
                    num_gpu = lookup_data.get('num_gpu', -1)
                    platform_id = lookup_data.get('platform_id', None)
                    parent_job_id = job_action_to_id.get(parent_action, None)
                    response = AppHandler.job_run(
                        org_name=org_name,
                        handler_id=experiment_id,
                        parent_job_id=parent_job_id,
                        action=child_action,
                        kind='experiment',
                        specs=specs,
                        name=name,
                        description=description,
                        num_gpu=num_gpu,
                        platform_id=platform_id,
                        from_ui=from_ui
                    )
                    if response.code == 200:
                        job_id = response.data
                        logger.info(
                            f"Created {child_action} job with id {job_id} for experiment {experiment_id}"
                        )
                        job_action_to_id[child_action] = job_id
                    else:
                        return response
        return None

    # Update existing experiment for user based on request dict
    @staticmethod
    def update_experiment(org_name, experiment_id, request_dict):
        """Updates the metadata of an existing experiment based on the provided request data.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to update.
            request_dict (dict): Dictionary containing the fields to update, following the ExperimentReqSchema.

        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        metadata = resolve_metadata("experiment", experiment_id)
        if not metadata:
            return Code(400, {}, "Experiment does not exist")

        user_id = metadata.get("user_id")
        if not handler_level_access_control(user_id, org_name, experiment_id, "experiments", handler_metadata=metadata):
            return Code(403, {}, "Not allowed to work with this org")
        if not check_write_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(400, {}, "User doesn't have write access to experiment")

        # if public is set to True => add it to public_experiments, if it is set to False => take it down
        # if public is not there, do nothing
        if request_dict.get("public", None):
            if request_dict["public"]:
                stateless_handlers.add_public_experiment(experiment_id)
            else:
                stateless_handlers.remove_public_experiment(experiment_id)

        user_id = metadata.get("user_id")
        workspace_id = metadata.get('workspace')
        for key in request_dict.keys():

            # Cannot process the update, so return 400
            if key in ["network_arch", "experiment_params", "base_experiment_metadata"]:
                if request_dict[key] != metadata.get(key):
                    msg = f"Cannot change experiment {key}"
                    return Code(400, {}, msg)

            if (key == "realtime_infer") and (request_dict[key] != metadata.get(key)):
                if request_dict[key] is False:
                    response = TISHandler.stop(experiment_id, metadata)
                    replicas = metadata.get("model_params", {}).get("replicas", 1)
                    CapGpuUsage.release_used(org_name, replicas)
                    if response.code != 200:
                        return response
                    metadata[key] = False
                else:
                    return Code(400, {}, f"Can only change {key} from True to False.")

            if key in ["name", "description", "version", "logo",
                       "ngc_path", "encryption_key", "read_only",
                       "metric", "public", "shared", "tags", "authorized_party_nca_id"]:
                requested_value = request_dict[key]
                if requested_value is not None:
                    metadata[key] = requested_value
                    metadata["last_modified"] = datetime.now(tz=timezone.utc)
                    if key == "tags":
                        metadata[key] = list({t.lower(): t for t in requested_value}.values())

            if key == "docker_env_vars":
                # Encrypt the MLOPs keys
                requested_value = request_dict[key]
                config_path = os.getenv("VAULT_SECRET_PATH", None)
                if config_path:
                    encryption = NVVaultEncryption(config_path)
                    for mlops_key, value in requested_value.items():
                        if encryption.check_config()[0]:
                            metadata["docker_env_vars"][mlops_key] = encryption.encrypt(value)
                        elif not os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
                            return Code(400, {}, "Vault service does not work, can't enable MLOPs services")
                else:
                    metadata["docker_env_vars"] = requested_value

            metadata, error_code = validate_and_update_experiment_metadata(
                user_id,
                org_name,
                request_dict,
                metadata,
                [
                    "train_datasets",
                    "eval_dataset",
                    "inference_dataset",
                    "calibration_dataset",
                    "base_experiment",
                    "checkpoint_choose_method",
                    "checkpoint_epoch_number"
                ]
            )
            if error_code:
                return error_code

            automl_enabled = metadata.get("automl_settings", {}).get("automl_enabled", False)
            tensorboard_enabled = metadata.get("tensorboard_enabled", False)
            if key == "automl_settings":
                value = request_dict[key]
                automl_enabled = value.get('automl_enabled', False)
                # If False, can set. If True, need to check if AutoML is supported
                if value:
                    mdl_nw = metadata.get("network_arch", "")
                    if automl_enabled and BACKEND == "NVCF":
                        return Code(
                            400,
                            {},
                            "Automl not supported on NVCF backend, use baremetal deployments of TAO-API"
                        )
                    if tensorboard_enabled and BACKEND == "NVCF":
                        return Code(
                            400,
                            {},
                            "Tensorboard not supported on NVCF backend, use baremetal deployments of TAO-API"
                        )
                    if mdl_nw not in AUTOML_DISABLED_NETWORKS:
                        metadata[key] = request_dict.get(key, {})
                    else:
                        return Code(400, {}, "automl_enabled cannot be True for unsupported network")
                    automl_validation_error = validate_automl_settings(value)
                    if automl_validation_error:
                        return Code(400, {}, automl_validation_error)
                else:
                    metadata[key] = value

            if key == "tensorboard_enabled":
                value = request_dict[key]
                mdl_nw = metadata.get("network_arch", "")
                if not tensorboard_enabled and value:  # Enable Tensorboard
                    if mdl_nw in TENSORBOARD_DISABLED_NETWORKS:
                        return Code(400, {}, f"Network {mdl_nw} not supported for Tensorboard")
                    if automl_enabled:
                        return Code(400, {}, "AutoML not supported yet for Tensorboard")
                    if TensorboardHandler.check_user_metadata(user_id) and workspace_id:
                        response = TensorboardHandler.start(org_name, experiment_id, user_id, workspace_id)
                        if response.code != 200:
                            TensorboardHandler.stop(experiment_id, user_id)
                            return response
                    else:
                        return Code(
                            400,
                            {},
                            f"Maximum of {TENSORBOARD_EXPERIMENT_LIMIT} Tensorboard Experiments allowed per user. "
                        )
                elif tensorboard_enabled and not value:  # Disable Tensorboard
                    response = TensorboardHandler.stop(experiment_id, user_id)
                    if response.code != 200:
                        return response
                metadata[key] = value

        write_handler_metadata(experiment_id, metadata, "experiment")
        # Read this metadata from saved file...
        return_metadata = sanitize_handler_metadata(metadata)
        ret_Code = Code(200, return_metadata, "Experiment updated")
        return ret_Code

    @staticmethod
    def retrieve_experiment(org_name, experiment_id):
        """Retrieves experiment metadata.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to retrieve.

        Returns:
            Response: A response indicating the outcome of the operation (200 with metadata for success,
                     404 if not found).
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if experiment_id not in ("*", "all") and not handler_metadata:
            return Code(404, {}, "Experiment not found")

        user_id = handler_metadata.get("user_id")
        if (experiment_id not in ("*", "all") and
                not check_read_access(user_id, org_name, experiment_id, kind="experiments")):
            return Code(404, {}, "Experiment not found")

        handler_metadata["status"] = get_handler_status(handler_metadata)
        return_metadata = sanitize_handler_metadata(handler_metadata)
        return Code(200, return_metadata, "Experiment retrieved")

    @staticmethod
    def delete_experiment(org_name, experiment_id):
        """Deletes an experiment if it is not in use by any job or other experiments.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to delete.

        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        handler_metadata = resolve_metadata("experiment", experiment_id)
        if not handler_metadata:
            return Code(200, {}, "Experiment deleted")
        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, {}, "User doesn't have write access to experiment")

        # If experiment is being used by user's experiments.
        experiments = get_user_experiments(user_id)

        if experiment_id not in experiments:
            return Code(404, {}, f"Experiment {experiment_id} cannot be deleted")

        for handler_id in experiments:
            metadata = get_experiment(handler_id)
            if experiment_id == metadata.get("base_experiment", None):
                return Code(400, {}, f"Experiment {experiment_id} in use as a base_experiment")

        for job in handler_metadata.get("jobs", {}):
            if handler_metadata["jobs"][job]["status"] in ("Pending", "Running"):
                return Code(400, {}, f"Experiment {experiment_id} in use by job {job}")

        # Check if experiment is public, then someone could be running it
        if handler_metadata.get("public", False):
            return Code(400, {}, f"Experiment {experiment_id} is Public. Cannot delete")

        # Check if experiment is read only, if yes, cannot delete
        if handler_metadata.get("read_only", False):
            return Code(400, {}, f"Experiment {experiment_id} is read only. Cannot delete")

        # Check if the experiment is being used by a realtime infer job
        if handler_metadata.get("realtime_infer", False):
            response = TISHandler.stop(experiment_id, handler_metadata)
            replicas = metadata.get("model_params", {}).get("replicas", 1)
            CapGpuUsage.release_used(org_name, replicas)
            if response is not None and response.code != 200:
                return response

        if handler_metadata.get("tensorboard_enabled", False):
            response = TensorboardHandler.stop(experiment_id, user_id)
            if response.code != 200:
                return response

        if experiment_id in experiments:
            experiments.remove(experiment_id)
            mongo_users = MongoHandler("tao", "users")
            mongo_users.upsert({'id': user_id}, {'id': user_id, 'experiments': experiments})

        delete_jobs_for_handler(experiment_id, "experiment")
        mongo_experiments = MongoHandler("tao", "experiments")
        mongo_experiments.delete_one({'id': experiment_id})
        return_metadata = sanitize_handler_metadata(handler_metadata)
        return Code(200, return_metadata, "Experiment deleted")

    @staticmethod
    def resume_experiment_job(
        org_name,
        experiment_id,
        job_id,
        kind,
        parent_job_id=None,
        specs=None,
        name=None,
        description=None,
        num_gpu=-1,
        platform_id=None
    ):
        """Resumes a paused experiment job, adding it back to the queue for processing.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment to which the job belongs.
            job_id (str): ID of the job to resume.
            kind (str): Type of the experiment (e.g., "experiment").
            parent_job_id (str, optional): ID of the parent job if applicable.
            specs (dict, optional): Specifications for the resumed job.
            name (str, optional): Name of the job.
            description (str, optional): Description of the job.
            num_gpu (int, optional): Number of GPUs to allocate.
            platform_id (str, optional): Platform ID for the job.

        Returns:
            Response: A response indicating the outcome of the operation (200 for success, error responses for failure).
        """
        handler_metadata = resolve_metadata(kind, experiment_id)
        if not handler_metadata:
            return Code(404, [], "Experiment not found")

        user_id = handler_metadata.get("user_id")
        if not check_write_access(user_id, org_name, experiment_id, kind="experiments"):
            return Code(404, [], "Experiment not found")

        job_metadata = stateless_handlers.get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, None, "job trying to resume not found")
        action = job_metadata.get("action", "")
        action = infer_action_from_job(experiment_id, job_id)
        status = job_metadata.get("status", "")
        if status != "Paused":
            return Code(400, [], f"Job status should be paused, not {status}")
        if action not in ("train", "distill", "quantize", "retrain"):
            return Code(400, [], f"Action should be train, distill, quantize, retrain, not {action}")
        network = handler_metadata.get("network_arch", None)
        if network in MAXINE_NETWORKS:
            return Code(400, [], "Maxine networks do not support resume.")
        if not user_id:
            return Code(
                404,
                [],
                "User ID couldn't be found in the experiment metadata. Try creating the experiment again"
            )

        msg = ""
        try:
            stateless_handlers.update_job_status(experiment_id, job_id, status="Resuming", kind=kind + "s")
            if not name:
                name = job_metadata.get("name", "")
            if not platform_id:
                logger.info("Loading existing platform_id from paused job")
                platform_id = job_metadata.get("platform_id", "")
            if is_request_automl(experiment_id, action, kind):
                msg = "AutoML "
                AutoMLHandler.resume(
                    user_id,
                    org_name,
                    experiment_id,
                    job_id,
                    handler_metadata,
                    name=name,
                    platform_id=platform_id
                )
            else:
                # Create a job and run it
                if not specs:
                    logger.info("Loading existing specs from paused job")
                    specs = get_job_specs(job_id)
                if not parent_job_id:
                    logger.info("Loading existing parent_job_id from paused job")
                    parent_job_id = handler_metadata.get('parent_job_id', None)
                if not description:
                    description = job_metadata.get("description", "")
                if num_gpu == -1:
                    num_gpu = job_metadata.get("num_gpu", -1)
                job_context = create_job_context(
                    parent_job_id,
                    "train",
                    job_id,
                    experiment_id,
                    user_id,
                    org_name,
                    kind,
                    handler_metadata=handler_metadata,
                    specs=specs,
                    name=name,
                    description=description,
                    num_gpu=num_gpu,
                    platform_id=platform_id
                )
                on_new_job(job_context)
            return Code(200, {"message": f"{msg}Action for job {job_id} resumed"})
        except Exception as e:
            logger.error("Exception thrown in resume_experiment_job is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, [], "Action cannot be resumed")

    @staticmethod
    def automl_details(org_name, experiment_id, job_id):
        """Retrieves AutoML details for a specific experiment and job.

        Args:
            org_name (str): Organization name.
            experiment_id (str): ID of the experiment.
            job_id (str): ID of the job.

        Returns:
            - Code(200, details, None) if successful.
            - Code(404, {}, "AutoML details not found") if the experiment does not have AutoML details
              or the experiment is not found.
        """
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            user_id = handler_metadata.get("user_id")
            root = stateless_handlers.get_jobs_root(user_id, org_name)
            jobdir = os.path.join(root, job_id)
            automl_controller_data = get_automl_controller_info(job_id)

            automl_interpretable_result = {}

            # Get current experiment id
            current_rec = get_automl_current_rec(job_id)
            if not current_rec:
                current_rec = 0
            automl_interpretable_result["current_experiment_id"] = current_rec

            # Get per experiment result and status
            automl_interpretable_result["experiments"] = {}
            for experiment_details in automl_controller_data:
                automl_interpretable_result["metric"] = experiment_details.get("metric")
                exp_id = experiment_details.get("id")
                automl_interpretable_result["experiments"][exp_id] = {}
                automl_interpretable_result["experiments"][exp_id]["result"] = experiment_details.get("result")
                automl_interpretable_result["experiments"][exp_id]["status"] = experiment_details.get("status")

            # Get the best experiment id
            if os.path.exists(os.path.join(jobdir, "best_model")):
                rec_files = glob.glob(os.path.join(jobdir, "best_model", "recommendation*.yaml"))
                if rec_files:
                    experiment_name = os.path.splitext(os.path.basename(rec_files[0]))[0]
                    experiment_id = experiment_name.split("_")[1]
                    automl_interpretable_result["best_experiment_id"] = int(experiment_id)
            return Code(200, automl_interpretable_result, "AutoML results compiled")
        except Exception as e:
            logger.error("Exception thrown in automl_details fetch is %s", str(e))
            logger.error(traceback.format_exc())
            return Code(400, [], "Error in constructing AutoML results")
