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

"""API handlers module - provides direct access to specialized handler classes"""

# Import all handlers for direct access
from .dataset_handler import DatasetHandler
from .workspace_handler import WorkspaceHandler
from .experiment_handler import ExperimentHandler
from .job_handler import JobHandler
from .spec_handler import SpecHandler
from .mongo_handler import MongoBackupHandler
from .model_handler import ModelHandler

# Export all handlers
__all__ = [
    'DatasetHandler',
    'WorkspaceHandler',
    'ExperimentHandler',
    'JobHandler',
    'SpecHandler',
    'MongoBackupHandler',
    'ModelHandler'
]
