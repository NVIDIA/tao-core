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

"""Spec handler module for managing specification schemas"""
import os
import logging

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    check_read_access,
    get_base_experiment_metadata,
    get_handler_job_metadata,
    get_job_specs,
    is_request_automl
)
from nvidia_tao_core.microservices.utils.handler_utils import Code
from nvidia_tao_core.microservices.utils.executor_utils import get_available_local_k8s_gpus
from nvidia_tao_core.microservices.utils.nvcf_utils import get_available_nvcf_instances
from nvidia_tao_core.microservices.utils.specs_utils import csv_to_json_schema
from nvidia_tao_core.microservices.utils.core_utils import (
    merge_nested_dicts,
    override_dicts,
    get_microservices_network_and_action
)
from nvidia_tao_core.scripts.generate_schema import generate_schema, validate_and_clean_merged_spec

from ..utils.basic_utils import resolve_metadata

# Configure logging
logger = logging.getLogger(__name__)

# Identify if workflow is on NGC
BACKEND = os.getenv("BACKEND", "local-k8s")


class SpecHandler:
    """Handles specification schema operations."""

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
        if metadata.get("base_experiment_ids", []):
            for base_experiment_id in metadata["base_experiment_ids"]:
                base_experiment_metadata = get_base_experiment_metadata(base_experiment_id)
                base_exp_meta = base_experiment_metadata.get("base_experiment_metadata", {})
                if base_experiment_metadata and base_exp_meta.get("spec_file_present"):
                    base_experiment_spec = base_exp_meta.get("specs", {})
                    if not base_experiment_spec:
                        return Code(404, {}, "Base specs not present.")

        # Read csv from utils/spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}

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
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                fmt = metadata.get("format", "_")
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "utils",
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{fmt}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    return Code(404, {}, "Default specs do not exist for action")
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

        job_metadata = get_handler_job_metadata(job_id)
        if not job_metadata:
            return Code(404, [], "Job trying to get schema for not found")
        action = job_metadata.get("action", "")

        job_specs = get_job_specs(job_id)

        json_schema = {}

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
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                fmt = metadata.get("format", "_")
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "utils",
                    "specs_utils",
                    "specs",
                    network,
                    f"{network} - {action}__{fmt}.csv"
                )
                if not os.path.exists(CSV_PATH):
                    return Code(404, {}, "Default specs do not exist for action")
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

        # Read csv from utils/spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        try:
            json_schema = generate_schema(base_experiment_network, action)
        except Exception as e:
            logger.error("Exception thrown in get_base_experiment_spec_schema is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", base_experiment_network,
                                    f"{base_experiment_network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                return Code(404, {}, "Default specs do not exist for action")
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

        # Read csv from utils/spec_utils/specs/<network_name>/action.csv
        # Convert to json schema
        json_schema = {}
        microservices_network, microservices_action = get_microservices_network_and_action(network, action)
        try:
            json_schema = generate_schema(microservices_network, microservices_action)
        except Exception as e:
            logger.error("Exception thrown in get_spec_schema_without_handler_id is %s", str(e))
            logger.error("Unable to fetch schema from tao_core")

        if not json_schema:
            DIR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

            # Try regular format for CSV_PATH => "<network> - <action>.csv"
            CSV_PATH = os.path.join(DIR_PATH, "utils", "specs_utils", "specs", network, f"{network} - {action}.csv")
            if not os.path.exists(CSV_PATH):
                # Try secondary format for CSV_PATH => "<network> - <action>__<dataset-format>.csv"
                CSV_PATH = os.path.join(
                    DIR_PATH,
                    "utils",
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
            available_gpu_types = get_available_local_k8s_gpus()
            if available_gpu_types:
                return Code(200, available_gpu_types, "Retrieved available GPU info")
            return Code(
                404, [],
                "Requested GPU's are not available in the current deployment. "
                "Check if all nodes has accelerator labels"
            )
        return Code(404, [], f"GPU types can't be retrieved for deployed Backend {BACKEND}")
