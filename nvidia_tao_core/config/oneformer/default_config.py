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

from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    DATACLASS_FIELD
)
from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig
)

from nvidia_tao_core.config.oneformer.dataset import OneFormerDatasetConfig
from nvidia_tao_core.config.oneformer.model import OneFormerModelConfig
from nvidia_tao_core.config.oneformer.train import OneFormerTrainExpConfig
from nvidia_tao_core.config.oneformer.export import OneFormerExportExpConfig
from nvidia_tao_core.config.oneformer.evaluate import OneFormerEvaluateConfig
from nvidia_tao_core.config.oneformer.inference import OneFormerInferenceConfig
from nvidia_tao_core.config.oneformer.deploy import OneFormerGenTrtEngineExpConfig


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: OneFormerModelConfig = DATACLASS_FIELD(
        OneFormerModelConfig(),
    )
    dataset: OneFormerDatasetConfig = DATACLASS_FIELD(
        OneFormerDatasetConfig(),
    )
    train: OneFormerTrainExpConfig = DATACLASS_FIELD(
        OneFormerTrainExpConfig(),
        description="Configurable parameters to construct the trainer for a OneFormer experiment.",
    )
    evaluate: OneFormerEvaluateConfig = DATACLASS_FIELD(
        OneFormerEvaluateConfig(),
        description="Configurable parameters to construct the evaluator for a OneFormer experiment.",
    )
    inference: OneFormerInferenceConfig = DATACLASS_FIELD(
        OneFormerInferenceConfig(),
        description="Configurable parameters to construct the inference for a OneFormer experiment.",
    )
    export: OneFormerExportExpConfig = DATACLASS_FIELD(
        OneFormerExportExpConfig(),
        description="Configurable parameters to construct the exporter for a OneFormer checkpoint.",
    )
    gen_trt_engine: OneFormerGenTrtEngineExpConfig = DATACLASS_FIELD(
        OneFormerGenTrtEngineExpConfig(),
        description="Configurable parameters to construct the deployer for a OneFormer checkpoint.",
    )
