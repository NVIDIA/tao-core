# Copyright (c) 2025  , NVIDIA CORPORATION.  All rights reserved.
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
"""Configuration hyperparameter schema for the evaluation."""

from typing import List
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    INT_FIELD,
    LIST_FIELD,
    STR_FIELD
)


@dataclass
class OneFormerInferenceConfig:
    """Evaluation configuration for OneFormer."""

    num_gpus: int = INT_FIELD(
        value=1,
        valid_min=1,
        display_name="Number of GPUs",
        description="The number of GPUs to run the evaluation job.",
        popular="yes",
    )
    gpu_ids: List[int] = LIST_FIELD(
        arrList=[0],
        display_name="GPU IDs",
        description="List of GPU IDs to run the evaluation on",
        popular="yes",
    )
    checkpoint: str = STR_FIELD(
        value="",
        description="Path to the checkpoint used for evaluation.",
        display_name="Checkpoint path",
    )
    results_dir: str = STR_FIELD(
        value="",
        description="Path to the results directory.",
        display_name="Results directory",
    )
    mode: str = STR_FIELD(
        value="semantic",
        description="Mode to run inference.",
        display_name="Mode",
        valid_options="semantic,instance,panoptic"
    )
    images_dir: str = STR_FIELD(
        value="",
        description="Path to the images directory.",
        display_name="Images directory",
    )
    image_size: List[int] = LIST_FIELD(
        arrList=[1024, 1024],
        description="Size of the image.",
        display_name="Image size",
    )
