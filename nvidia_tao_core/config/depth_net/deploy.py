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

"""Configuration hyperparameter schema to deploy the model."""

from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    DATACLASS_FIELD,
    STR_FIELD
)
from nvidia_tao_core.config.common.common_config import (
    GenTrtEngineConfig,
    TrtConfig
)


@dataclass
class DepthNetTrtConfig(TrtConfig):
    """Trt config."""

    data_type: str = STR_FIELD(
        value="FP32",
        default_value="FP32",
        description="The precision to be set for building the TensorRT engine.",
        display_name="data type",
        valid_options=",".join(["FP32", "FP16"])
    )


@dataclass
class DepthNetGenTrtEngineExpConfig(GenTrtEngineConfig):
    """Gen TRT Engine experiment config."""

    tensorrt: DepthNetTrtConfig = DATACLASS_FIELD(
        DepthNetTrtConfig(),
        description="Hyper parameters to configure the TensorRT Engine builder.",
        display_name="TensorRT hyper params."
    )
