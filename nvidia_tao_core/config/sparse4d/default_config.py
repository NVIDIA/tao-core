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

"""Default config file for Sparse4D."""

from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    DATACLASS_FIELD
)
from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig,
    ExportConfig,
)
from nvidia_tao_core.config.sparse4d.model import Sparse4DModelConfig
from nvidia_tao_core.config.sparse4d.dataset import Omniverse3DDetTrackDatasetConfig
from nvidia_tao_core.config.sparse4d.train import Sparse4DTrainConfig
from nvidia_tao_core.config.sparse4d.inference import Sparse4DInferenceConfig
from nvidia_tao_core.config.sparse4d.evaluate import Sparse4DEvaluateConfig
from nvidia_tao_core.config.sparse4d.visualize import Sparse4DVisualizeConfig


@dataclass
class Sparse4DExportConfig(ExportConfig):
    """Export configuration for Sparse4D."""

    pass


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment configuration for Sparse4D."""

    train: Sparse4DTrainConfig = DATACLASS_FIELD(
        Sparse4DTrainConfig(),
        description="Train config",
        display_name="Train config"
    )
    model: Sparse4DModelConfig = DATACLASS_FIELD(
        Sparse4DModelConfig(),
        description="Model config",
        display_name="Model config"
    )
    dataset: Omniverse3DDetTrackDatasetConfig = DATACLASS_FIELD(
        Omniverse3DDetTrackDatasetConfig(),
        description="Dataset config",
        display_name="Dataset config"
    )
    inference: Sparse4DInferenceConfig = DATACLASS_FIELD(
        Sparse4DInferenceConfig(),
        description="Inference config",
        display_name="Inference config"
    )
    evaluate: Sparse4DEvaluateConfig = DATACLASS_FIELD(
        Sparse4DEvaluateConfig(),
        description="Evaluate config",
        display_name="Evaluate config"
    )
    export: Sparse4DExportConfig = DATACLASS_FIELD(
        Sparse4DExportConfig(),
        description="Export config",
        display_name="Export config"
    )
    visualize: Sparse4DVisualizeConfig = DATACLASS_FIELD(
        Sparse4DVisualizeConfig(),
        description="Visualize config",
        display_name="Visualize config"
    )

    def __post_init__(self):
        """Set default model name for Sparse4D."""
        if self.model_name is None:
            self.model_name = "sparse4d"
