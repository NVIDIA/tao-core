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

"""Generating JSON schemas"""

from nvidia_tao_core.api_utils import dataclass2json_converter
from nvidia_tao_core.microservices import enum_constants


def generate_schema(neural_network_name, action=""):
    """Generates JSON schema for network"""
    imported_module = dataclass2json_converter.import_module_from_path(
        f"nvidia_tao_core.config.{neural_network_name}.default_config"
    )
    if neural_network_name == "bevfusion" and action == "dataset_convert":
        expConfig = imported_module.BEVFusionDataConvertExpConfig()
    elif neural_network_name == "stylegan_xl" and action == "dataset_convert":
        imported_module = dataclass2json_converter.import_module_from_path(
            f"nvidia_tao_core.config.{neural_network_name}.dataset"
        )
        expConfig = imported_module.DataConvertExpConfig()
    else:
        expConfig = imported_module.ExperimentConfig()
    json_with_meta_config = dataclass2json_converter.dataclass_to_json(expConfig)
    schema = dataclass2json_converter.create_json_schema(json_with_meta_config)
    # Only keep relevant top-level keys
    valid_actions = enum_constants._get_valid_config_json_param_for_network(neural_network_name, "actions")
    schema = filter_schema(schema, valid_actions, action)
    return schema


def filter_schema(schema, valid_actions, current_action):
    """Filter the schema to only include the allowed keys"""
    # Always keep 'train' and the current action, plus all non-action keys
    allowed_keys = set(['train', 'distill', 'quantize', current_action])
    # Add all non-action keys (not in valid_actions)
    allowed_keys.update([k for k in schema['properties'] if k not in valid_actions])

    # Filter top-level properties and default
    schema['properties'] = {k: v for k, v in schema['properties'].items() if k in allowed_keys}
    schema['default'] = {k: v for k, v in schema['default'].items() if k in allowed_keys}
    return schema
