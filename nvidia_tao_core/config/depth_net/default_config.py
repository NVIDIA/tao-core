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

"""Default config file."""

from typing import List, Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    BOOL_FIELD,
    STR_FIELD,
    LIST_FIELD
)

from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    InferenceConfig
)
from nvidia_tao_core.config.depth_net.dataset import DepthNetDatasetConfig
from nvidia_tao_core.config.depth_net.model import DepthNetModelConfig
from nvidia_tao_core.config.depth_net.train import DepthNetTrainExpConfig


@dataclass
class WandBConfig:
    """Configuration element wandb client."""

    enable: bool = BOOL_FIELD(value=True)
    project: str = STR_FIELD(value="TAO Toolkit")
    entity: Optional[str] = STR_FIELD(value="")
    tags: List[str] = LIST_FIELD(arrList=["tao-toolkit"])
    reinit: bool = BOOL_FIELD(value=False)
    sync_tensorboard: bool = BOOL_FIELD(value=False)
    save_code: bool = BOOL_FIELD(value=False)
    name: str = BOOL_FIELD(value="TAO Toolkit Training")
    run_id: str = STR_FIELD(value="")


@dataclass
class DepthNetInferenceExpConfig(InferenceConfig):
    """Inference experiment config."""

    conf_threshold: float = FLOAT_FIELD(
        value=0.5,
        default_value=0.5,
        description="""The value of the confidence threshold to be used when
                    filtering out the final list of boxes.""",
        display_name="confidence threshold"
    )
    input_width: Optional[int] = INT_FIELD(
        value=None,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=1,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=1,
    )


@dataclass
class DepthNetEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    input_width: Optional[int] = INT_FIELD(
        value=736,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=1,
    )
    input_height: Optional[int] = INT_FIELD(
        value=320,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=1,
    )


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    dataset: DepthNetDatasetConfig = DATACLASS_FIELD(
        DepthNetDatasetConfig(),
        description="Configurable parameters to construct the dataset for a DepthNet experiment.",
    )
    model: DepthNetModelConfig = DATACLASS_FIELD(
        DepthNetModelConfig(),
        description="Configurable parameters to construct the model for a DepthNet experiment.",
    )
    inference: DepthNetInferenceExpConfig = DATACLASS_FIELD(
        DepthNetInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for a DepthNet experiment.",
    )
    evaluate: DepthNetEvalExpConfig = DATACLASS_FIELD(
        DepthNetEvalExpConfig(),
        description="Configurable parameters to construct the evaluator for a DepthNet experiment.",
    )
    train: DepthNetTrainExpConfig = DATACLASS_FIELD(
        DepthNetTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a RT-DETR experiment.",
    )
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(),
        description="Configurable parameters to construct the wandb client for a DepthNet experiment.",
    )
