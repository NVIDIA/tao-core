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

"""API v2 blueprints for organizing endpoints by version."""

from .auth import auth_bp_v2
from .workspaces import workspaces_bp_v2
from .datasets import datasets_bp_v2
from .jobs import jobs_bp_v2
from .health import health_bp_v2
from .admin import admin_bp_v2
from .inference_microservices import inference_microservices_bp_v2
from .automl import automl_params_bp_v2

# Import all schemas from v2 schemas module
from .schemas import (
    AWSCloudPull,
    AutoMLParameterDetail,
    AutoMLParameterDetailsRsp,
    AutoMLResults,
    AutoML,
    AutoMLUpdateParameterRangesReq,
    AzureCloudPull,
    BaseExperimentMetadata,
    BulkOpsRsp,
    BulkOps,
    Category,
    CategoryWise,
    CloudSpecificDetails,
    DatasetActions,
    DatasetJob,
    DatasetJobReq,
    DatasetJobRsp,
    DatasetListRsp,
    DatasetPathLst,
    DatasetReq,
    DatasetRsp,
    DetailedStatus,
    ErrorRsp,
    ExperimentActions,
    ExperimentJobReq,
    ExperimentJobRsp,
    GpuDetails,
    Graph,
    HuggingFaceCloudPull,
    InferenceMicroserviceReq,
    InferenceMicroserviceRsp,
    InferenceReq,
    JobListRsp,
    JobReq,
    JobResult,
    JobResume,
    JobRsp,
    JobSubset,
    KPI,
    LoadAirgappedExperimentsReq,
    LoadAirgappedExperimentsRsp,
    LoginReq,
    LoginRsp,
    LstStr,
    MessageOnly,
    MissingFile,
    NVCFReq,
    PaginationInfo,
    ParameterRangeSchema,
    PublishModel,
    Stats,
    ValidationDetails,
    WorkspaceBackupReq,
    WorkspaceListRsp,
    WorkspaceReq,
    WorkspaceRsp
)

# V2 Blueprint Schemas - List of tuples with (name, schema_class)
V2_SCHEMAS = [
    ("AWSCloudPull", AWSCloudPull),
    ("AutoMLParameterDetail", AutoMLParameterDetail),
    ("AutoMLParameterDetailsRsp", AutoMLParameterDetailsRsp),
    ("AutoMLResults", AutoMLResults),
    ("AutoML", AutoML),
    ("AutoMLUpdateParameterRangesReq", AutoMLUpdateParameterRangesReq),
    ("AzureCloudPull", AzureCloudPull),
    ("BaseExperimentMetadata", BaseExperimentMetadata),
    ("BulkOpsRsp", BulkOpsRsp),
    ("BulkOps", BulkOps),
    ("Category", Category),
    ("CategoryWise", CategoryWise),
    ("CloudSpecificDetails", CloudSpecificDetails),
    ("DatasetActions", DatasetActions),
    ("DatasetJob", DatasetJob),
    ("DatasetJobReq", DatasetJobReq),
    ("DatasetJobRsp", DatasetJobRsp),
    ("DatasetListRsp", DatasetListRsp),
    ("DatasetPathLst", DatasetPathLst),
    ("DatasetReq", DatasetReq),
    ("DatasetRsp", DatasetRsp),
    ("DetailedStatus", DetailedStatus),
    ("ErrorRsp", ErrorRsp),
    ("ExperimentActions", ExperimentActions),
    ("ExperimentJobReq", ExperimentJobReq),
    ("ExperimentJobRsp", ExperimentJobRsp),
    ("GpuDetails", GpuDetails),
    ("Graph", Graph),
    ("HuggingFaceCloudPull", HuggingFaceCloudPull),
    ("InferenceMicroserviceReq", InferenceMicroserviceReq),
    ("InferenceMicroserviceRsp", InferenceMicroserviceRsp),
    ("InferenceReq", InferenceReq),
    ("JobListRsp", JobListRsp),
    ("JobReq", JobReq),
    ("JobResult", JobResult),
    ("JobResume", JobResume),
    ("JobRsp", JobRsp),
    ("JobSubset", JobSubset),
    ("KPI", KPI),
    ("LoadAirgappedExperimentsReq", LoadAirgappedExperimentsReq),
    ("LoadAirgappedExperimentsRsp", LoadAirgappedExperimentsRsp),
    ("LoginReq", LoginReq),
    ("LoginRsp", LoginRsp),
    ("LstStr", LstStr),
    ("MessageOnly", MessageOnly),
    ("MissingFile", MissingFile),
    ("NVCFReq", NVCFReq),
    ("PaginationInfo", PaginationInfo),
    ("ParameterRangeSchema", ParameterRangeSchema),
    ("PublishModel", PublishModel),
    ("Stats", Stats),
    ("ValidationDetails", ValidationDetails),
    ("WorkspaceBackupReq", WorkspaceBackupReq),
    ("WorkspaceListRsp", WorkspaceListRsp),
    ("WorkspaceReq", WorkspaceReq),
    ("WorkspaceRsp", WorkspaceRsp)
]

__all__ = [
    'auth_bp_v2',
    'workspaces_bp_v2',
    'datasets_bp_v2',
    'jobs_bp_v2',
    'health_bp_v2',
    'admin_bp_v2',
    'inference_microservices_bp_v2',
    'automl_params_bp_v2',
    'V2_SCHEMAS'
]
