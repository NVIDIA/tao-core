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

"""Default config file"""

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    DATACLASS_FIELD,
    BOOL_FIELD,
)


@dataclass
class InferenceConfig:
    """Inference config."""

    media: str = STR_FIELD(
        default_value="",
        value="",
        display_name="File path",
        description="File to be used for inference"
    )
    prompt: str = STR_FIELD(
        default_value="Describe this video.",
        value="Describe this video.",
        display_name="Prompt",
        description="Prompt for inference"
    )
    fps: Optional[int] = INT_FIELD(
        default_value=4,
        value=4,
        display_name="FPS",
        description="FPS for inference"
    )
    total_pixels: Optional[int] = INT_FIELD(
        default_value=6422528,
        value=6422528,
        display_name="Total pixels",
        description="Total pixels for inference"
    )
    max_new_tokens: Optional[int] = INT_FIELD(
        default_value=4096,
        value=4096,
        display_name="Max new tokens",
        description="Max new tokens for inference"
    )
    num_gpus: Optional[int] = INT_FIELD(
        default_value=1,
        value=1,
        valid_min=1,
        valid_max=8,
        display_name="Number of GPUs",
        description="Number of GPUs to use for inference (enables multi-GPU tensor parallelism)"
    )
    enable_lora: bool = BOOL_FIELD(
        default_value=False,
        value=False,
        display_name="Enable LoRA merging",
        description="Enable LoRA model merging (merge LoRA weights with base model before evaluation)"
    )
    base_model_path: Optional[str] = STR_FIELD(
        default_value="",
        value="",
        display_name="Base model path",
        description="Path to base model for LoRA merging (used when enable_lora is True)"
    )


@dataclass
class ExperimentConfig:
    """Experiment config."""

    inference: InferenceConfig = DATACLASS_FIELD(InferenceConfig(), description="Inference config.")
