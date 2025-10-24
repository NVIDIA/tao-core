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

"""Default config file"""

from typing import List, Optional
from dataclasses import dataclass

from nvidia_tao_core.config.utils.types import (
    BOOL_FIELD,
    LIST_FIELD,
    STR_FIELD,
)


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
    name: str = STR_FIELD(value="TAO Toolkit Training")
    run_id: str = STR_FIELD(value="")


@dataclass
class ClearMLConfig:
    """Configration element for clearml client."""

    project: str = STR_FIELD(value="TAO Toolkit")
    task: str = STR_FIELD("train")
    deferred_init: bool = BOOL_FIELD(value=False)
    reuse_last_task_id: bool = BOOL_FIELD(value=False)
    continue_last_task: bool = BOOL_FIELD(value=False)
    tags: List[str] = LIST_FIELD(arrList=["tao-toolkit"])
