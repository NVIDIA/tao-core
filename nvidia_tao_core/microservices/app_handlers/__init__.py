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

"""App handlers module - provides direct access to specialized handler classes"""

# Import all handlers for direct usage
from nvidia_tao_core.microservices.app_handlers.workspace_handler import WorkspaceHandler
from nvidia_tao_core.microservices.app_handlers.dataset_handler import DatasetHandler
from nvidia_tao_core.microservices.app_handlers.experiment_handler import ExperimentHandler
from nvidia_tao_core.microservices.app_handlers.job_handler import JobHandler
from nvidia_tao_core.microservices.app_handlers.spec_handler import SpecHandler
from nvidia_tao_core.microservices.app_handlers.mongo_handler import MongoBackupHandler
from nvidia_tao_core.microservices.app_handlers.model_handler import ModelHandler

# Export all handlers for direct access
__all__ = [
    'WorkspaceHandler',
    'DatasetHandler',
    'ExperimentHandler',
    'JobHandler',
    'SpecHandler',
    'MongoBackupHandler',
    'ModelHandler'
]
