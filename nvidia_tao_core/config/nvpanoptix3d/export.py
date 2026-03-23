# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Configuration hyperparameter schema for export."""

from dataclasses import dataclass
from nvidia_tao_core.config.common.common_config import ExportConfig

from nvidia_tao_core.config.utils.types import (
    INT_FIELD,
    STR_FIELD
)


@dataclass
class NVPanoptix3DExportExpConfig(ExportConfig):
    """NVPanoptix3D export ONNX experiment config."""

    onnx_file_2d: str = STR_FIELD(
        value="",
        default_value="",
        display_name="onnx file 2d",
        description="Path to the onnx model 2d file."
    )
    onnx_file_3d: str = STR_FIELD(
        value="",
        default_value="",
        display_name="onnx file 3d",
        description="Path to the onnx model 3d file."
    )
    max_voxels: int = INT_FIELD(
        value=700000,
        default_value=700000,
        valid_min=1,
        description="The maximum number of voxels in the input Tensor for the engine.",
        display_name="max voxels"
    )
