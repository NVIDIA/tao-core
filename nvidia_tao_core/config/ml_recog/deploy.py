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

"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD,
)
from nvidia_tao_core.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig,
    CalibrationConfig
)


@dataclass
class MLTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        display_name="data_type",
        description="[Optional] The precision to be used for the TensorRT engine.",
        valid_options="FP32, FP16, INT8",
    )
    calibration: CalibrationConfig = DATACLASS_FIELD(
        CalibrationConfig(),
        description="The calibration configuration for the model.",
        display_name="calibration",
    )


@dataclass
class MLGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: MLTrtConfig = DATACLASS_FIELD(
        MLTrtConfig(),
        description="The TensorRT configuration for the model.",
        display_name="tensorrt",
    )
