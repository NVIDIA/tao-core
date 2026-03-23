# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
    DATACLASS_FIELD,
)
from nvidia_tao_core.config.common.common_config import (
    CommonExperimentConfig
)

from nvidia_tao_core.config.nvpanoptix3d.dataset import NVPanoptix3DDatasetConfig
from nvidia_tao_core.config.nvpanoptix3d.model import NVPanoptix3DModelConfig
from nvidia_tao_core.config.nvpanoptix3d.train import NVPanoptix3DTrainExpConfig
from nvidia_tao_core.config.nvpanoptix3d.deploy import NVPanoptix3DGenTRTEngineExpConfig
from nvidia_tao_core.config.nvpanoptix3d.inference import NVPanoptix3DInferenceExpConfig
from nvidia_tao_core.config.nvpanoptix3d.evaluate import NVPanoptix3DEvaluateExpConfig
from nvidia_tao_core.config.nvpanoptix3d.export import NVPanoptix3DExportExpConfig


@dataclass
class ExperimentConfig(CommonExperimentConfig):
    """Experiment config."""

    model: NVPanoptix3DModelConfig = DATACLASS_FIELD(
        NVPanoptix3DModelConfig(),
        description="Configurable parameters to construct the model for the NVPanoptix3D experiment.",
    )
    dataset: NVPanoptix3DDatasetConfig = DATACLASS_FIELD(
        NVPanoptix3DDatasetConfig(),
        description="Configurable parameters to construct the dataset for the NVPanoptix3D experiment.",
    )
    train: NVPanoptix3DTrainExpConfig = DATACLASS_FIELD(
        NVPanoptix3DTrainExpConfig(),
        description="Configurable parameters to construct the trainer for the NVPanoptix3D experiment.",
    )
    inference: NVPanoptix3DInferenceExpConfig = DATACLASS_FIELD(
        NVPanoptix3DInferenceExpConfig(),
        description="Configurable parameters to construct the inferencer for the NVPanoptix3D experiment.",
    )
    evaluate: NVPanoptix3DEvaluateExpConfig = DATACLASS_FIELD(
        NVPanoptix3DEvaluateExpConfig(),
        description="Configurable parameters to construct the evaluator for the NVPanoptix3D experiment.",
    )
    export: NVPanoptix3DExportExpConfig = DATACLASS_FIELD(
        NVPanoptix3DExportExpConfig(),
        description="Configurable parameters to construct the exporter for the NVPanoptix3D experiment.",
    )
    gen_trt_engine: NVPanoptix3DGenTRTEngineExpConfig = DATACLASS_FIELD(
        NVPanoptix3DGenTRTEngineExpConfig(),
        description="Configurable parameters to construct the TensorRT engine builder for a NVPanoptix3D experiment.",
    )
