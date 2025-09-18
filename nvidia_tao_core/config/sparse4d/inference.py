# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Configuration hyperparameter schema to run inference on model."""

from dataclasses import dataclass
from omegaconf import MISSING

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    BOOL_FIELD,
    DATACLASS_FIELD
)
from nvidia_tao_core.config.sparse4d.dataset import Sparse4DTrackingConfig
from nvidia_tao_core.config.common.common_config import InferenceConfig


@dataclass
class Sparse4DInferenceConfig(InferenceConfig):
    """Inference configuration for Sparse4D."""

    checkpoint: str = STR_FIELD(
        value=MISSING,
        default_value="",
        description="Path to checkpoint file",
        display_name="Path to checkpoint file"
    )
    jsonfile_prefix: str = STR_FIELD(
        value="sparse4d_pred",
        default_value="sparse4d_pred",
        description="JSON file prefix",
        display_name="JSON file prefix"
    )
    output_nvschema: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        description="Output NVSchema",
        display_name="Output NVSchema"
    )
    tracking: Sparse4DTrackingConfig = DATACLASS_FIELD(
        Sparse4DTrackingConfig(),
        description="Tracking config",
        display_name="Tracking config"
    )
