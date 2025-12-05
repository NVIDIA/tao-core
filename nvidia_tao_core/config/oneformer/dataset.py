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
"""Configuration hyperparameter schema for the dataset."""

from typing import List
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    INT_FIELD,
    STR_FIELD,
    LIST_FIELD,
    DATACLASS_FIELD,
    DICT_FIELD,
    FLOAT_FIELD
)


@dataclass
class Dataset:
    """Dataset config."""

    batch_size: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Batch size",
        math_cond=">0",
        valid_min=1,
        valid_max="inf",
        display_name="batch size"
    )
    num_workers: int = INT_FIELD(
        value=1,
        default_value=1,
        description="Number of workers",
        valid_min=0,
        valid_max="inf",
        display_name="Number of workers"
    )
    images: str = STR_FIELD(
        value="",
        default_value="",
        display_name="image root",
        description="A path to image root"
    )
    annotations: str = STR_FIELD(
        value="",
        default_value="",
        display_name="annotation root",
        description="A path to annotation root"
    )
    panoptic: str = STR_FIELD(
        value="",
        default_value="",
        display_name="panoptic root",
        description="A path to panoptic root"
    )


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    train_min_size: List[int] = LIST_FIELD(
        arrList=[800],
        description="A list of sizes to perform random resize.",
        display_name="Train min size"
    )
    train_max_size: int = INT_FIELD(
        value=1333,
        valid_min=32,
        valid_max="inf",
        description="The maximum random crop size for training data",
        automl_enabled="TRUE",
        display_name="Train max size"
    )
    train_crop_size: List[int] = LIST_FIELD(
        arrList=[1024, 1024],
        description="The random crop size for training data in [H, W]",
        display_name="Train crop size"
    )
    test_min_size: int = INT_FIELD(
        value=800,
        valid_min=32,
        valid_max="inf",
        description="The minimum resize size for test data",
        automl_enabled="TRUE",
        display_name="Test min size"
    )
    test_max_size: int = INT_FIELD(
        value=1333,
        valid_min=32,
        valid_max="inf",
        description="The maximum resize size for test",
        automl_enabled="TRUE",
        display_name="Test max size"
    )


@dataclass
class OneFormerDatasetConfig:
    """Data config."""

    train: Dataset = DATACLASS_FIELD(
        Dataset(),
        description="Configurable parameters to construct the train dataset.",
    )
    val: Dataset = DATACLASS_FIELD(
        Dataset(),
        description="Configurable parameters to construct the validation dataset.",
    )
    test: Dataset = DATACLASS_FIELD(
        Dataset(),
        description="Configurable parameters to construct the test dataset.",
    )
    workers: int = INT_FIELD(
        value=8,
        default_value=8,
        valid_min=1,
        valid_max="inf",
        description="The number of parallel workers processing data",
        display_name="workers"
    )
    pin_memory: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="pin_memory",
        description="Flag to enable the dataloader to allocate pagelocked memory"
    )
    pixel_mean: List[float] = LIST_FIELD(
        arrList=[123.675, 116.28, 103.53],
        description="The input mean for RGB frames",
        display_name="input mean per pixel"
    )
    pixel_std: List[float] = LIST_FIELD(
        arrList=[58.395, 57.12, 57.375],
        description="The input standard deviation per pixel for RGB frames",
        display_name="input std per pixel"
    )
    augmentation: AugmentationConfig = DATACLASS_FIELD(
        AugmentationConfig(),
        description="Configuration parameters for data augmentation",
    )
    contiguous_id: bool = BOOL_FIELD(
        value=True,
        default_value=True,
        display_name="contiguous id",
        description="Flag to enable contiguous ids for labels."
    )
    label_map: str = STR_FIELD(
        value="",
        display_name="label map",
        description="A path to label map file"
    )
    task_prob_train: dict = DICT_FIELD(
        hashMap={
            "semantic": 0.33,
            "instance": 0.66,
            "panoptic": 0.01
        },
        description="Task probabilities",
        display_name="task probabilities"
    )
    task_prob_val: dict = DICT_FIELD(
        hashMap={
            "semantic": 0.33,
            "instance": 0.66,
            "panoptic": 0.01
        },
        description="Task probabilities",
        display_name="task probabilities"
    )
    task_seq_len: int = INT_FIELD(
        value=77,
        description="Task sequence length",
        display_name="task sequence length"
    )
    max_seq_len: int = INT_FIELD(
        value=77,
        description="Maximum sequence length",
        display_name="maximum sequence length"
    )
    image_size: int = INT_FIELD(
        value=1024,
        default_value=1024,
        description="Image size",
        display_name="image size"
    )
    min_scale: float = FLOAT_FIELD(
        value=0.1,
        description="Minimum scale",
        display_name="minimum scale"
    )
    max_scale: float = FLOAT_FIELD(
        value=2.0,
        description="Maximum scale",
        display_name="maximum scale"
    )
    cutmix_prob: float = FLOAT_FIELD(
        value=0.0,
        description="Cutmix probability",
        display_name="cutmix probability"
    )
