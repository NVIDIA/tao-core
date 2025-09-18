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

"""Configuration hyperparameter schema for the dataset."""

from typing import Optional, List
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    LIST_FIELD,
)


@dataclass
class DataConvertExpConfig:
    """Configuration parameters for Data Converter"""

    source: str = STR_FIELD(
        value="/path/to/your/torchvision_datasets_ImageFolder",
        default_value="/path/to/your/torchvision_datasets_ImageFolder",
        display_name="Sorce dataset",
        description="Sorce dataset which follows torchvision.datasets.ImageFolder format",
    )
    results_dir: str = STR_FIELD(
        value="/path/to/your/results_dir",
        default_value="/path/to/your/results_dir",
        display_name="Result directory",
        description="Result directory",
    )
    dest_file_name: str = STR_FIELD(
        value="torchvision_datasets_ImageFolder.zip",
        default_value="torchvision_datasets_ImageFolder.zip",
        display_name="Destination zipped file name",
        description="Destination zipped file name generated from source dataset",
    )
    resolution: List[int] = LIST_FIELD(
        arrList=[128, 128],
        display_name="Resized resolution",
        description="The moving average parameter for adaptive learning rate."
    )
    transform: Optional[str] = STR_FIELD(
        value=None,
        default_value=None,
        display_name="Transformation before resizing",
        description="Transformation such as 'center-crop' before resizing can avoid distortion",
        valid_options="center-crop"
    )
