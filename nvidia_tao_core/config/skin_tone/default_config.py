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

"""Default config file."""

from typing import List, Optional
from dataclasses import dataclass
from omegaconf import MISSING
from nvidia_tao_core.config.utils.types import (
    STR_FIELD,
    INT_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
)


@dataclass
class HueConfig:
    """Hue configuration template."""

    angle: int = INT_FIELD(value=0)


@dataclass
class BrightnessConfig:
    """Contrast configuration template."""

    offset: int = INT_FIELD(value=0)


@dataclass
class ColorAugmentationConfig:
    """Color augmentation configuration template."""

    hue: HueConfig = DATACLASS_FIELD(HueConfig())
    lum: BrightnessConfig = DATACLASS_FIELD(BrightnessConfig())


@dataclass
class DataConfig:
    """Dataset configuration template."""

    input_dir: str = STR_FIELD(value=MISSING, default_value="<specify image directory>")
    image_size: int = INT_FIELD(value=448)


@dataclass
class ExperimentConfig:
    """Experiment configuration template."""

    gpu_id: List[int] = LIST_FIELD(arrList=[0])
    dataset: DataConfig = DATACLASS_FIELD(DataConfig())
    color_aug: ColorAugmentationConfig = DATACLASS_FIELD(ColorAugmentationConfig())
    results_dir: Optional[str] = STR_FIELD(
        "/results", default_value="/results"
    )
