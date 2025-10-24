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

from typing import Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    DATACLASS_FIELD,
    FLOAT_FIELD,
    INT_FIELD,
    BOOL_FIELD,
)

from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig,
    EvaluateConfig,
    InferenceConfig,
    ExportConfig
)
from nvidia_tao_core.config.depth_net.dataset import DepthNetDatasetConfig
from nvidia_tao_core.config.depth_net.model import DepthNetModelConfig
from nvidia_tao_core.config.depth_net.train import DepthNetTrainExpConfig
from nvidia_tao_core.config.depth_net.deploy import DepthNetGenTrtEngineExpConfig
from nvidia_tao_core.config.common.mlops import WandBConfig


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
    save_raw_pfm: Optional[bool] = BOOL_FIELD(
        value=False,
        default_value=False,
        description="Whether to save the raw pfm output during inference.",
        display_name="Save PFM Output"
    )


@dataclass
class DepthNetEvalExpConfig(EvaluateConfig):
    """Evaluation experiment config."""

    input_width: Optional[int] = INT_FIELD(
        value=None,
        default_value=736,
        description="Width of the input image tensor.",
        display_name="input width",
        valid_min=1,
    )
    input_height: Optional[int] = INT_FIELD(
        value=None,
        default_value=320,
        description="Height of the input image tensor.",
        display_name="input height",
        valid_min=1,
    )


@dataclass
class DepthNetExportExpConfig(ExportConfig):
    """Inference experiment config."""

    valid_iters: Optional[int] = INT_FIELD(
        value=22,
        default_value=22,
        description="Number of GRU iterations to export the model.",
        display_name="Valid Iterations",
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
        description="Configurable parameters to construct the trainer for a DepthNet experiment.",
    )
    wandb: WandBConfig = DATACLASS_FIELD(
        WandBConfig(),
        description="Configurable parameters to construct the wandb client for a DepthNet experiment.",
    )
    export: DepthNetExportExpConfig = DATACLASS_FIELD(
        DepthNetExportExpConfig(),
        description="Configurable parameters to construct the onnx export for a DepthNet experiment."
    )
    gen_trt_engine: DepthNetGenTrtEngineExpConfig = DATACLASS_FIELD(
        DepthNetGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a DepthNet experiment.",
    )

    def __post_init__(self):
        """Set default model name for DepthNet."""
        if self.model_name is None:
            self.model_name = "depth_net"
