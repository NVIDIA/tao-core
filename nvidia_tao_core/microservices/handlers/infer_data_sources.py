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

"""Functions to infer data sources"""
import os
import re
import logging

from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    get_handler_metadata,
    get_workspace_string_identifier,
    get_handler_job_metadata
)
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


def contains_results_uuid(data_path):
    """Check if data path is from another TAO API job"""
    # Define the regex pattern to match "/results/{uuid}"
    pattern = r"/results/[0-9a-fA-F-]{36}"
    # Search the pattern in the input string
    match = re.search(pattern, data_path)
    # Return True if the pattern is found, otherwise False
    return bool(match)


def get_datasets_from_metadata(metadata, source_key):
    """Gets a list of datasets from metadata based on source key.

    Args:
        metadata (dict): Handler metadata containing dataset information
        source_key (str): Key to lookup in metadata (e.g. 'train_datasets')

    Returns:
        list: List of dataset IDs, or empty list if not found
    """
    dataset = metadata.get(source_key)
    return dataset if isinstance(dataset, list) else [dataset] if dataset else []


def get_nested_config_value(config, path):
    """Gets a value from nested config using dot notation path."""
    parts = path.replace("]", "").replace("[", ".").split(".")
    current = config

    for part in parts:
        if part.isdigit():
            part = int(part)
        if isinstance(current, dict):
            if part not in current:
                return None
            current = current[part]
        elif isinstance(current, list):
            if part >= len(current):
                return None
            current = current[part]
        else:
            return None

    return current


def remove_nested_config_value(config, path):
    """Removes a value from nested config using dot notation path."""
    parts = path.replace("]", "").replace("[", ".").split(".")
    current = config

    for part in parts[:-1]:
        if part.isdigit():
            part = int(part)
        if part not in current:
            return
        current = current[part]

    if parts[-1] in current:
        del current[parts[-1]]


def set_nested_config_value(config, path, value):
    """Sets a value in nested config using dot notation path.

    Args:
        config (dict): Config dictionary to modify
        path (str): Dot notation path (e.g. "dataset.train_data_sources")
        value: Value to set at the path

    Example:
        >>> config = {}
        >>> set_nested_config_value(config, "a.b.c", 123)
        >>> config
        {'a': {'b': {'c': 123}}}
    """
    parts = path.replace("]", "").replace("[", ".").split(".")
    current = config

    for part in parts[:-1]:
        if part.isdigit():
            part = int(part)
        if part not in current:
            # Create list if next part is numeric, dict otherwise
            next_part = parts[parts.index(part) + 1]
            current[part] = [] if next_part.isdigit() else {}
        current = current[part]

    last_part = parts[-1]
    if last_part.isdigit():
        last_part = int(last_part)

    # If both are dictionaries, merge them instead of replacing
    if isinstance(current.get(last_part), dict) and isinstance(value, dict):
        current[last_part].update(value)
    else:
        current[last_part] = value


def get_job_id_of_action(dataset_id, kind, action):
    """Gets job ID for a specific action on a dataset."""
    # Implementation of getting job ID from dataset and action
    dataset_metadata = get_handler_metadata(dataset_id, kind)
    job_id = None
    for job in dataset_metadata.get("jobs", []):
        job_metadata = get_handler_job_metadata(job)
        if job_metadata.get("action") == action and job_metadata.get("status") == "Done":
            job_id = job_metadata.get("id")
    return job_id


def get_dataset_convert_downloaded_locally(network_config):
    """Get if dataset convert is downloaded locally"""
    dataset_convert_downloaded_locally = False
    if network_config and "cloud_upload" in network_config:
        cloud_upload = network_config["cloud_upload"]
        upload_strategy = cloud_upload.get("upload_strategy", {})
        dataset_convert_strategy = upload_strategy.get("dataset_convert")
        # If dataset_convert has tarball strategy, it will be downloaded locally
        if (dataset_convert_strategy == "tarball_after_completion" or
                (isinstance(dataset_convert_strategy, dict) and "selective_tarball" in dataset_convert_strategy)):
            dataset_convert_downloaded_locally = True
    return dataset_convert_downloaded_locally


def apply_transforms(
    value,
    transforms,
    source_root=None,
    source_ds=None,
    dataset_convert_action=None,
    workspace_identifier=None,
    dataset_convert_downloaded_locally=None
):
    """Apply a list of transforms to a value.

    Args:
        value: The value to transform
        transforms: List of transforms to apply
        source_root: Root path of the source dataset
        source_ds: Source dataset ID
        dataset_convert_action: Action for dataset conversion
        workspace_identifier: Workspace identifier for dataset convert job paths
    """
    if isinstance(transforms, str):
        transforms = [transforms]

    for transform in transforms:
        if transform == "handle_tar_path":
            value = os.path.join(source_root, "images") if contains_results_uuid(source_root) else value
        elif transform == "wrap_in_list":
            value = [value]
        elif transform == "use_dataset_convert_job":
            dataset_convert_job_id = get_job_id_of_action(
                source_ds, kind="datasets", action=dataset_convert_action
            ) or ""
            if "{dataset_convert_job_id}" in value and not dataset_convert_job_id:
                logger.warning(
                    "Unable to resolve dataset-convert job for dataset %s; skipping transform.",
                    source_ds,
                )
                return value

            # Check if the value already has the results path format
            value = value.replace("{dataset_convert_job_id}", dataset_convert_job_id)
            if dataset_convert_downloaded_locally:
                return value

            if value.startswith("/results/"):
                # It's already in the correct format, just prepend workspace identifier
                value = f"{workspace_identifier}{value}"
            else:
                # Legacy format - apply the old logic
                corrected_value = value.replace(source_root, "")
                value = f"{workspace_identifier}{corrected_value.lstrip('/')}"

    return value


def get_source_root(source_ds_metadata, workspace_identifier):
    """Get the source root path for a dataset."""
    return f"{workspace_identifier}{source_ds_metadata.get('cloud_file_path')}"


def get_dataset_metadata_and_paths(source_ds, workspace_cache, kind="datasets"):
    """Helper function to get common dataset metadata and paths."""
    source_ds_metadata = get_handler_metadata(source_ds, kind=kind)
    workspace_identifier = get_workspace_string_identifier(
        source_ds_metadata.get('workspace'),
        workspace_cache
    )
    source_root = get_source_root(source_ds_metadata, workspace_identifier)
    return source_ds_metadata, workspace_identifier, source_root


def get_source_datasets_from_config(config_source, handler_metadata):
    """Helper function to get source datasets from config."""
    if config_source == "id":
        return [handler_metadata.get("id")]
    return get_datasets_from_metadata(handler_metadata, config_source)


def process_convert_job_spec_path(spec_config, source_ds, dataset_convert_action):
    """Helper function to process path from convert job spec."""
    spec_path = spec_config.get("spec_path")
    mapping = spec_config.get("mapping", {})

    # Get the dataset convert job for this dataset
    dataset_convert_job_id = get_job_id_of_action(
        source_ds, kind="datasets", action=dataset_convert_action
    )

    if not dataset_convert_job_id:
        return None, None

    # Get the job metadata to access specs
    convert_job_metadata = get_handler_job_metadata(dataset_convert_job_id)
    if not convert_job_metadata:
        return None, None

    # Get the spec value from the convert job
    job_specs = convert_job_metadata.get("specs", {})
    spec_value = get_nested_config_value(job_specs, spec_path)

    # Use the mapping to determine the path
    path_template = (mapping.get(spec_value, mapping.get("*", ""))
                     if spec_value else mapping.get("*", ""))

    return path_template, dataset_convert_job_id


def process_mapping_path_and_transforms(mapping, source_root, source_ds, dataset_convert_action,
                                        workspace_identifier, dataset_convert_downloaded_locally):
    """Helper function to process mapping path and apply transforms."""
    # Skip optional paths that don't exist
    if mapping.get("optional") and not check_file_exists(source_root, mapping["path"]):
        return None

    # Get the path value
    value = os.path.join(source_root, mapping["path"]) if mapping.get("path") else source_root

    # Apply any transforms
    if "transform" in mapping:
        value = apply_transforms(
            value, mapping["transform"],
            source_root, source_ds, dataset_convert_action,
            workspace_identifier, dataset_convert_downloaded_locally)

    return value


def apply_workspace_identifier_to_results_path(path, workspace_identifier):
    """Helper function to prepend workspace identifier to results paths."""
    if path and path.startswith("/results/"):
        return workspace_identifier + path
    return path


def replace_placeholder_and_apply_workspace_id(value, placeholder, replacement, workspace_identifier):
    """Helper function to replace placeholders and apply workspace identifier."""
    if isinstance(value, str) and placeholder in value:
        value = value.replace(placeholder, replacement)
        # If this is a path to results, prepend workspace identifier
        if "results/" in value:
            value = workspace_identifier + value
    return value


def process_mapping_entry(mapping, source_root, source_ds, dataset_convert_action,
                          workspace_identifier, dataset_convert_downloaded_locally=False):
    """Process a single mapping entry.

    Handles three types of mappings:
    1. Simple mappings with direct path and transform (like ann_file)
    2. Nested mappings with sub-mappings (like data_prefix with pts and img)
    3. String mappings that reference dataset metadata fields (like dataset_format)
    """
    # Handle string mappings that reference dataset metadata fields
    if isinstance(mapping, str):
        # Get dataset metadata to resolve the field reference
        source_ds_metadata = get_handler_metadata(source_ds, kind="datasets")
        if mapping == "dataset_format":
            return source_ds_metadata.get("format")
        if mapping == "dataset_type":
            return source_ds_metadata.get("type")
        if mapping == "dataset_intent":
            use_for = source_ds_metadata.get("use_for", [])
            return use_for[0] if use_for else None
        # For any other string, try to get it directly from metadata
        return source_ds_metadata.get(mapping)
    # Check if this is a nested mapping (like data_prefix with pts and img)
    if any(isinstance(v, dict) and "path" in v for k, v in mapping.items()):
        # This is a nested mapping (e.g., data_prefix with pts and img)
        result = {}
        for key, sub_mapping in mapping.items():
            if isinstance(sub_mapping, dict) and "path" in sub_mapping:
                value = process_mapping_path_and_transforms(
                    sub_mapping, source_root, source_ds, dataset_convert_action,
                    workspace_identifier, dataset_convert_downloaded_locally)
                if value is not None:
                    result[key] = value
        return result if result else None

    # This is a simple mapping (e.g., ann_file with direct path and transform)
    if "path" in mapping:
        return process_mapping_path_and_transforms(
            mapping, source_root, source_ds, dataset_convert_action,
            workspace_identifier, dataset_convert_downloaded_locally)

    return None


def get_metadata_value(metadata, path_type):
    """Helper function to get metadata values safely"""
    if path_type == "intent":
        use_for = metadata.get("use_for", [])
        return use_for[0] if use_for else None
    return metadata.get(path_type) if path_type in ["type", "format"] else None


def process_additional_downloads(
    network_config, job_context, handler_metadata, workspace_cache, dataset_convert_action, endpoint_action
):
    """Process additional downloads configuration from network config"""
    additional_downloads = []

    # Auto-generate additional downloads based on upload strategy for dataset_convert
    if job_context.action in ["train", "evaluate", "inference", "retrain", "prune", "export"]:
        cloud_upload = network_config.get("cloud_upload", {})
        upload_strategy = cloud_upload.get("upload_strategy", {})
        dataset_convert_strategy = upload_strategy.get("dataset_convert")

        if dataset_convert_strategy:
            # Get datasets that might have dataset_convert results
            source_datasets = (get_datasets_from_metadata(handler_metadata, "train_datasets") or
                               get_datasets_from_metadata(handler_metadata, "eval_dataset") or
                               get_datasets_from_metadata(handler_metadata, "inference_dataset"))
            if source_datasets:
                dataset_convert_job_id = get_job_id_of_action(
                    source_datasets[0], kind="datasets", action=dataset_convert_action
                )

                if dataset_convert_job_id:
                    # Get workspace identifier
                    source_ds_metadata = get_handler_metadata(source_datasets[0], kind="datasets")
                    workspace_identifier = get_workspace_string_identifier(
                        source_ds_metadata.get('workspace'),
                        workspace_cache
                    )

                    # Generate download path based on strategy
                    if dataset_convert_strategy == "tarball_after_completion":
                        # For simple tarball strategy (like pointpillars)
                        download_path = (f"{workspace_identifier}/results/{dataset_convert_job_id}/"
                                         f"{endpoint_action}_results.tar.gz")
                        additional_downloads.append(download_path)
                    elif isinstance(dataset_convert_strategy, dict) and "selective_tarball" in dataset_convert_strategy:
                        # For selective tarball strategy (like sparse4d)
                        download_path = (f"{workspace_identifier}/results/{dataset_convert_job_id}/"
                                         f"{endpoint_action}_selective.tar.gz")
                        additional_downloads.append(download_path)

    # Get additional downloads for the current action
    downloads_config = network_config.get("additional_download", {}).get(job_context.action, [])

    if not downloads_config:
        return additional_downloads

    for download_config in downloads_config:
        # Get source datasets
        source_datasets = get_source_datasets_from_config(download_config["source"], handler_metadata)
        if not source_datasets:
            continue

        # Process each source dataset
        for source_ds in source_datasets:
            (source_ds_metadata,
             workspace_identifier,
             _) = get_dataset_metadata_and_paths(source_ds, workspace_cache)

            # Handle path from convert job spec
            if "path_from_convert_job_spec" in download_config:
                path_template, dataset_convert_job_id = process_convert_job_spec_path(
                    download_config["path_from_convert_job_spec"], source_ds, dataset_convert_action
                )

                if path_template:
                    # Replace {dataset_convert_job_id} with actual job ID
                    download_path = path_template.replace("{dataset_convert_job_id}", dataset_convert_job_id)

                    # Replace {dataset_path} with dataset-specific path
                    if "{dataset_path}" in download_path:
                        # Use dataset ID or a default path component
                        dataset_path = source_ds_metadata.get("cloud_file_path", source_ds)
                        download_path = download_path.replace("{dataset_path}", dataset_path)

                    # Prepend workspace identifier if this is a results path
                    download_path = apply_workspace_identifier_to_results_path(download_path, workspace_identifier)
                    additional_downloads.append(download_path)

            # Handle direct path (fallback)
            elif "path" in download_config:
                path = download_config["path"]
                if path:
                    # Replace {dataset_convert_job_id} if present
                    dataset_convert_job_id = get_job_id_of_action(
                        source_ds, kind="datasets", action=dataset_convert_action
                    )
                    if dataset_convert_job_id and "{dataset_convert_job_id}" in path:
                        path = path.replace("{dataset_convert_job_id}", dataset_convert_job_id)

                    # Replace {dataset_path} with dataset-specific path
                    if "{dataset_path}" in path:
                        # Use dataset ID or a default path component
                        dataset_path = source_ds_metadata.get("cloud_file_path", source_ds)
                        path = path.replace("{dataset_path}", dataset_path)
                        path = workspace_identifier + path

                    # Prepend workspace identifier if this is a results path
                    path = apply_workspace_identifier_to_results_path(path, workspace_identifier)
                    additional_downloads.append(path)

    return list(set(additional_downloads))


def apply_data_source_config(config, job_context, handler_metadata):
    """Generic data source configuration using config file"""
    workspace_cache = {}
    dataset_convert_action = "dataset_convert"

    job_network, job_action = (("image", "validate") if job_context.action == "validate_images"
                               else (job_context.network, job_context.action))
    network_config = read_network_config(job_network)
    dataset_convert_downloaded_locally = get_dataset_convert_downloaded_locally(network_config)

    # Keep track of paths that have already been set by special handlers
    already_configured_paths = set()

    # Handle dynamic config adjustments first
    if "dynamic_config" in network_config:
        dynamic_config = network_config["dynamic_config"]
        model_type_key = dynamic_config.get("model_type_key")
        model_type = get_nested_config_value(config, model_type_key) if model_type_key else None

        # Check parent action rules
        if "parent_action_rules" in dynamic_config and job_context.parent_id:
            parent_job_metadata = get_handler_job_metadata(job_context.parent_id)
            parent_action = parent_job_metadata.get("action") if parent_job_metadata else None

            if parent_action and parent_action in dynamic_config["parent_action_rules"]:
                rules = dynamic_config["parent_action_rules"][parent_action]

                # Handle direct set_value rules
                if "set_value" in rules:
                    # Check if there's an action_restriction
                    action_restriction = rules.get("action_restriction", None)

                    # Check conditional if present
                    conditional_pass = True
                    if "conditional" in rules:
                        cond = rules["conditional"]
                        if "metadata_key" in cond:
                            meta_value = handler_metadata.get(cond["metadata_key"], None)
                            conditional_pass = not (("equals" in cond and meta_value != cond["equals"]) or
                                                    ("not_equals" in cond and meta_value == cond["not_equals"]))
                        elif "config_path" in cond:
                            config_value = get_nested_config_value(config, cond["config_path"])
                            conditional_pass = not (("equals" in cond and config_value != cond["equals"]) or
                                                    ("not_equals" in cond and config_value == cond["not_equals"]))

                    if conditional_pass:
                        for config_path, value in rules["set_value"].items():
                            # Check if this config_path is restricted to specific actions
                            if action_restriction:
                                is_list_restriction = (isinstance(action_restriction, list) and
                                                       job_action not in action_restriction)
                                is_dict_restriction = (isinstance(action_restriction, dict) and
                                                       config_path in action_restriction and
                                                       job_action not in action_restriction[config_path])
                                if is_list_restriction or is_dict_restriction:
                                    continue

                            # Replace {parent_id} with actual parent ID if present
                            workspace_identifier = get_workspace_string_identifier(
                                handler_metadata.get('workspace'),
                                workspace_cache
                            )
                            value = replace_placeholder_and_apply_workspace_id(
                                value, "{parent_id}", job_context.parent_id, workspace_identifier
                            )

                            set_nested_config_value(config, config_path, value)
                            already_configured_paths.add(config_path)

                # Handle check_parent_specs rules
                if "check_parent_specs" in rules:
                    # Get action_restriction if present
                    action_restriction = rules.get("action_restriction", None)

                    for spec_path, expected_value in rules["check_parent_specs"].items():
                        if spec_path == "if_match":
                            continue

                        # Get the actual value from parent specs
                        parent_specs = parent_job_metadata.get("specs", {})
                        actual_value = get_nested_config_value(parent_specs, spec_path)

                        # If the value matches, apply the rules in if_match
                        if actual_value == expected_value and "if_match" in rules["check_parent_specs"]:
                            match_rules = rules["check_parent_specs"]["if_match"]

                            if "set_value" in match_rules:
                                for config_path, value in match_rules["set_value"].items():
                                    # Check if this config_path is restricted to specific actions
                                    if action_restriction:
                                        if isinstance(action_restriction, list):
                                            # Simple list of allowed actions
                                            if job_action not in action_restriction:
                                                continue
                                        elif isinstance(action_restriction, dict):
                                            # Dict mapping config paths to allowed actions
                                            if config_path in action_restriction:
                                                if job_action not in action_restriction[config_path]:
                                                    continue
                                            else:
                                                # If config_path not in action_restriction dict, apply to all actions
                                                pass

                                    # Replace {parent_id} with actual parent ID if present
                                    workspace_identifier = get_workspace_string_identifier(
                                        handler_metadata.get('workspace'),
                                        workspace_cache
                                    )
                                    value = replace_placeholder_and_apply_workspace_id(
                                        value, "{parent_id}", job_context.parent_id, workspace_identifier
                                    )

                                    set_nested_config_value(config, config_path, value)
                                    already_configured_paths.add(config_path)

        # Check action rules
        if "action_rules" in dynamic_config:
            if job_action in dynamic_config["action_rules"]:
                rules = dynamic_config["action_rules"][job_action]
                if "set_value" in rules:
                    for config_path, value in rules["set_value"].items():
                        set_nested_config_value(config, config_path, value)
                if "remove" in rules:
                    for path in rules["remove"]:
                        remove_nested_config_value(config, path)

        # Handle model type specific rules
        if model_type and "rules" in dynamic_config:
            rules = dynamic_config["rules"].get(model_type, {})

            # Handle removals
            for path in rules.get("remove", []):
                remove_nested_config_value(config, path)

            # Handle action-specific removals
            for path in rules.get("remove_if_action", {}).get(job_action, []):
                remove_nested_config_value(config, path)

            # Handle joint model path splitting
            if (rules.get("transform") == "split_pretrained_paths" and
                    "rgb_pretrained_model_path" in config.get("model", {})):
                ptm_paths = config["model"]["rgb_pretrained_model_path"].split(",")
                rgb_path = next((p for p in ptm_paths if "_rgb_" in p), ptm_paths[0])
                of_path = next((p for p in ptm_paths if "_of_" in p), ptm_paths[1])
                config["model"]["rgb_pretrained_model_path"] = rgb_path
                config["model"]["of_pretrained_model_path"] = of_path

        # Handle defaults
        for path, value in dynamic_config.get("defaults", {}).items():
            if not get_nested_config_value(config, path):
                set_nested_config_value(config, path, value)

    # Apply data source mappings
    data_sources = network_config.get("data_sources", {}).get(job_action, {})

    for config_path, source_config in data_sources.items():
        # Get source datasets
        if config_path in already_configured_paths:
            continue

        # Handle special source: parent_job_specs
        if source_config["source"] == "parent_job_specs":
            if job_context.parent_id and "check_key_exists" in source_config:
                parent_job_metadata = get_handler_job_metadata(job_context.parent_id)
                if parent_job_metadata:
                    parent_specs = parent_job_metadata.get("specs", {})
                    key_path = source_config["check_key_exists"]

                    # Check if the key exists in parent specs
                    key_exists = get_nested_config_value(parent_specs, key_path) is not None
                    set_nested_config_value(config, config_path, key_exists)
                    already_configured_paths.add(config_path)
                    logger.info(f"Set {config_path} = {key_exists} based on parent job specs key '{key_path}'")
            continue

        source_datasets = get_source_datasets_from_config(source_config["source"], handler_metadata)
        if not source_datasets:
            continue
        (source_ds_metadata,
         workspace_identifier,
         source_root) = get_dataset_metadata_and_paths(source_datasets[0], workspace_cache)

        # Handle value from metadata
        if "value_from_metadata" in source_config:
            meta_config = source_config["value_from_metadata"]
            meta_value = source_ds_metadata.get(meta_config["key"], None)
            value = meta_config["mapping"].get(meta_value, meta_config["mapping"].get("*", meta_config["default"]))
            set_nested_config_value(config, config_path, value)
            already_configured_paths.add(config_path)  # Mark as configured
            continue

        # Handle path from convert job spec
        if "path_from_convert_job_spec" in source_config:
            path, dataset_convert_job_id = process_convert_job_spec_path(
                source_config["path_from_convert_job_spec"], source_datasets[0], dataset_convert_action
            )

            if path:
                # Check if we're using the dataset convert job transform
                if "use_dataset_convert_job" in source_config.get("transform", []):
                    # Build the results path template for the transform
                    value = f"/results/{dataset_convert_job_id}/{path}"
                else:
                    value = os.path.join(source_root, path)
                value = apply_transforms(
                    value, source_config.get("transform", []),
                    source_root, source_datasets[0], dataset_convert_action,
                    workspace_identifier, dataset_convert_downloaded_locally)
                set_nested_config_value(config, config_path, value)
                already_configured_paths.add(config_path)  # Mark as configured
                continue

        # Handle path from type/intent/source/model_type cases
        for path_type in ["type", "format", "intent", "source", "model_type"]:
            key = f"path_from_{path_type}"
            if key in source_config:
                meta_value = {
                    "model_type": lambda: get_nested_config_value(config, model_type_key) if model_type_key else None,
                    "source": lambda x=source_config: x["source"],
                    "type": lambda x=source_ds_metadata: get_metadata_value(x, "type"),
                    "format": lambda x=source_ds_metadata: get_metadata_value(x, "format"),
                    "intent": lambda x=source_ds_metadata: get_metadata_value(x, "intent")
                }[path_type]()

                if path_type == "model_type" and meta_value == "openpose":
                    meta_value = "kinetics"

                if meta_value in source_config[key]:
                    path_config = source_config[key][meta_value]
                    # Handle array of paths (fallback mechanism)
                    if isinstance(path_config, list):
                        path = next((p for p in path_config
                                    if check_file_exists_in_cloud(source_ds_metadata, source_root, p)),
                                    path_config[0])
                    else:
                        path = path_config
                elif path_type != "intent":  # intent doesn't use fallback
                    path = source_config[key].get("*")
                else:
                    continue

                value = os.path.join(source_root, path)
                if path_type == "source":
                    value = apply_transforms(
                        value, source_config.get("transform", []),
                        source_root, source_datasets[0], dataset_convert_action,
                        workspace_identifier, dataset_convert_downloaded_locally)
                set_nested_config_value(config, config_path, value)
                already_configured_paths.add(config_path)  # Mark as configured
                continue

        # Skip rest of processing if already configured by a special handler
        if config_path in already_configured_paths:
            continue

        # Check conditional
        if "conditional" in source_config:
            cond = source_config["conditional"]
            meta_value = source_ds_metadata.get(cond["metadata_key"], None)
            if (("equals" in cond and meta_value != cond["equals"]) or
                    ("not_equals" in cond and meta_value == cond["not_equals"])):
                continue

        if source_config.get("multiple_sources", False):
            # Handle multiple sources
            result_list = []
            for source_ds in source_datasets:
                (source_ds_metadata,
                 workspace_identifier,
                 source_root) = get_dataset_metadata_and_paths(source_ds, workspace_cache)

                if "mapping" in source_config:
                    entry = {}
                    for key, mapping in source_config["mapping"].items():
                        value = process_mapping_entry(
                            mapping, source_root, source_ds,
                            dataset_convert_action, workspace_identifier, network_config)
                        if value is not None:
                            entry[key] = value
                    if entry:
                        result_list.append(entry)
                else:
                    path = source_config.get("path", "")
                    value = source_root if path == "" else os.path.join(source_root, path)
                    result_list.append(value)

            if result_list:
                set_nested_config_value(config, config_path, result_list)

        else:
            # Handle single source
            source_ds = source_datasets[0]
            if source_config.get("value") is not None:
                # Replace {job_id} with actual job ID if present
                value = source_config["value"]
                if isinstance(value, str) and "{job_id}" in value:
                    value = value.replace("{job_id}", job_context.id)
                set_nested_config_value(config, config_path, value)
                continue

            (source_ds_metadata,
             workspace_identifier,
             source_root) = get_dataset_metadata_and_paths(source_ds, workspace_cache)

            if "mapping" in source_config:
                result = {}
                for key, mapping in source_config["mapping"].items():
                    value = process_mapping_entry(
                        mapping, source_root, source_ds,
                        dataset_convert_action, workspace_identifier, network_config)
                    if value is not None:
                        result[key] = value

                if result:
                    set_nested_config_value(config, config_path, result)
            else:
                path = source_config.get("path", "")
                value = (source_root if path == "" else
                         (path if dataset_convert_downloaded_locally else os.path.join(source_root, path)))

                value = apply_transforms(
                    value, source_config.get("transform", []),
                    source_root, source_ds, dataset_convert_action,
                    workspace_identifier, dataset_convert_downloaded_locally)
                set_nested_config_value(config, config_path, value)

    # Add preserve_source_path_params from network config if specified
    preserve_source_path_params = network_config.get("preserve_source_path_params")
    if preserve_source_path_params:
        config["preserve_source_path_params"] = preserve_source_path_params
        logger.info("Added preserve_source_path_params to config: %s", preserve_source_path_params)

    # Process additional downloads
    endpoint_action = (
        network_config.get("actions_mapping", {})
        .get(dataset_convert_action, {})
        .get("action", dataset_convert_action)
    )
    additional_downloads = process_additional_downloads(
        network_config, job_context, handler_metadata, workspace_cache, dataset_convert_action, endpoint_action
    )

    # Handle custom data loader script for cosmos-rl and other networks
    if "custom_script" in config:
        custom_script_path = config.get("custom_script")
        if custom_script_path:
            logger.info("Custom data loader script found in config: %s", custom_script_path)
            additional_downloads.append(custom_script_path)

    if additional_downloads:
        config["additional_downloads"] = additional_downloads

    return config


def check_file_exists(root_path, file_path):
    """Check if a file exists in the given root path."""
    full_path = os.path.join(root_path, file_path)
    return os.path.exists(full_path)


def check_file_exists_in_cloud(source_ds_metadata, source_root, file_path):
    """Check if a file exists, supporting both cloud and local storage."""
    file_exists = False

    # Try cloud storage check first
    if source_ds_metadata.get('workspace'):
        try:
            from nvidia_tao_core.microservices.utils.cloud_utils import create_cs_instance
            workspace_metadata = get_handler_metadata(source_ds_metadata.get('workspace'), kind="workspace")
            if workspace_metadata:
                cloud_instance, _ = create_cs_instance(workspace_metadata)
                if cloud_instance:
                    cloud_file_path = source_ds_metadata.get('cloud_file_path', '')
                    cloud_path = f"{cloud_file_path.strip('/')}/{file_path}"
                    file_exists = cloud_instance.is_file(cloud_path)
                    logger.info(f"Cloud check for {cloud_path}: {file_exists}")
                    return file_exists
        except Exception as e:
            logger.warning(f"Cloud storage check failed: {e}")

    # Fallback to local check
    file_exists = check_file_exists(source_root, file_path)
    logger.info(f"Local check for {file_path}: {file_exists}")
    return file_exists
