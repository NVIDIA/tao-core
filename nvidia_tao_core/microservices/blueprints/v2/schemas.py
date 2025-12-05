# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the License);
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

"""Marshmallow schemas for API."""

import re
import sys
import math
from datetime import datetime
from marshmallow import Schema, fields, EXCLUDE, validates_schema, ValidationError, validate
from marshmallow_enum import EnumField, Enum
from marshmallow_oneofschema import OneOfSchema

from nvidia_tao_core.microservices.enum_constants import (
    ActionEnum,
    DatasetFormat,
    DatasetType,
    ExperimentNetworkArch,
    Metrics,
    BaseExperimentTask,
    BaseExperimentDomain,
    BaseExperimentBackboneType,
    BaseExperimentBackboneClass,
    BaseExperimentLicense,
    _get_dynamic_metric_patterns
)


def sys_int_format():
    """Get integer format based on system."""
    if sys.maxsize > 2**31 - 1:
        return "int64"
    return "int32"


def validate_endpoint_url(url):
    """Custom URL validator that accepts internal hostnames and services.

    This validator is more lenient than marshmallow's default URL validator,
    specifically allowing internal hostnames like 'seaweedfs-s3', 'localhost',
    IP addresses, and service names common in containerized environments.
    """
    if not url:
        return True  # allow_none=True is handled by the field

    # Basic URL structure validation using regex
    # This pattern allows for:
    # - http/https protocols
    # - hostnames with hyphens, underscores, alphanumeric characters
    # - IP addresses
    # - ports
    # - paths, query strings, fragments
    url_pattern = re.compile(
        r'^https?://'  # http or https protocol
        r'(?:'
        r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-_]*[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-_]*[a-zA-Z0-9])?'  # hostname
        r'|'
        r'[a-zA-Z0-9](?:[a-zA-Z0-9-_]*[a-zA-Z0-9])?'  # simple hostname (like 'seaweedfs-s3')
        r'|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # IPv4
        r'|'
        r'\[[0-9a-fA-F:]+\]'  # IPv6
        r')'
        r'(?::\d+)?'  # optional port
        r'(?:/[^\s]*)?$',  # optional path
        re.IGNORECASE
    )

    if not url_pattern.match(url):
        raise ValidationError('Invalid URL format.')

    return True


class DateTimeField(fields.DateTime):
    """Field for handling datetime objects.

    This field is used to handle datetime objects in the API.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, datetime):
            return value
        return super()._deserialize(value, attr, data, **kwargs)


class EnumFieldPrefix(fields.Field):
    """Enum field override for Metrics"""

    def __init__(self, enum, *args, **kwargs):
        """Init function of class"""
        self.enum = enum
        super().__init__(*args, **kwargs)

    def _deserialize(self, value, attr, data, **kwargs):
        if value in self.enum._value2member_map_:
            return value
        # Check for best_ prefixed values
        if value.startswith('best_'):
            base_value = value[5:]
            if base_value in self.enum._value2member_map_:
                return value

        # Check against dynamic metric patterns for networks like sparse4d
        if self._validate_dynamic_metric(value):
            return value

        raise ValidationError(f"Invalid value '{value}' for enum '{self.enum.__name__}'")

    def _validate_dynamic_metric(self, value: str) -> bool:
        """Validate value against dynamic metric patterns."""
        patterns = _get_dynamic_metric_patterns()
        for pattern in patterns:
            try:
                if re.match(pattern, value):
                    return True
            except re.error:
                # Skip invalid regex patterns
                continue
        return False

    def _serialize(self, value, attr, obj, **kwargs):
        return value

# ============================================================================
# ENUMS
# ============================================================================


class JobKindEnum(Enum):
    """Class defining job kind enum"""

    dataset = 'dataset'
    experiment = 'experiment'


class JobStatusEnum(Enum):
    """Class defining job status enum"""

    Done = 'Done'
    Running = 'Running'
    Error = 'Error'
    Pending = 'Pending'
    Canceled = 'Canceled'
    Canceling = 'Canceling'
    Pausing = 'Pausing'
    Paused = 'Paused'
    Resuming = 'Resuming'


class PullStatus(Enum):
    """Class defining artifact upload/download status"""

    starting = "starting"
    in_progress = "in_progress"
    pull_complete = "pull_complete"
    invalid_pull = "invalid_pull"


class BulkOpsStatus(Enum):
    """Class defining bulk operation status enum"""

    success = "success"
    failed = "failed"


class AllowedDockerEnvVariables(Enum):
    """Allowed docker environment variables while launching DNN containers"""

    HF_TOKEN = "HF_TOKEN"

    WANDB_API_KEY = "WANDB_API_KEY"
    WANDB_BASE_URL = "WANDB_BASE_URL"
    WANDB_USERNAME = "WANDB_USERNAME"
    WANDB_ENTITY = "WANDB_ENTITY"
    WANDB_PROJECT = "WANDB_PROJECT"
    WANDB_INSECURE_LOGGING = "WANDB_INSECURE_LOGGING"

    CLEARML_WEB_HOST = "CLEARML_WEB_HOST"
    CLEARML_API_HOST = "CLEARML_API_HOST"
    CLEARML_FILES_HOST = "CLEARML_FILES_HOST"
    CLEARML_API_ACCESS_KEY = "CLEARML_API_ACCESS_KEY"
    CLEARML_API_SECRET_KEY = "CLEARML_API_SECRET_KEY"

    CLOUD_BASED = "CLOUD_BASED"
    NVCF_HELM = "NVCF_HELM"
    TELEMETRY_OPT_OUT = "TELEMETRY_OPT_OUT"
    TAO_API_KEY = "TAO_API_KEY"
    TAO_USER_KEY = "TAO_USER_KEY"
    TAO_ADMIN_KEY = "TAO_ADMIN_KEY"
    TAO_API_SERVER = "TAO_API_SERVER"
    TAO_LOGGING_SERVER_URL = "TAO_LOGGING_SERVER_URL"
    RECURSIVE_DATASET_FILE_DOWNLOAD = "RECURSIVE_DATASET_FILE_DOWNLOAD"
    ORCHESTRATION_API_NETWORK = "ORCHESTRATION_API_NETWORK"
    ORCHESTRATION_API_ACTION = "ORCHESTRATION_API_ACTION"
    TAO_EXECUTION_BACKEND = "TAO_EXECUTION_BACKEND"
    AUTOML_EXPERIMENT_NUMBER = "AUTOML_EXPERIMENT_NUMBER"
    JOB_ID = "JOB_ID"
    TAO_API_JOB_ID = "TAO_API_JOB_ID"  # Automl brain job id
    RETAIN_CHECKPOINTS_FOR_RESUME = "RETAIN_CHECKPOINTS_FOR_RESUME"
    EARLY_STOP_EPOCH = "EARLY_STOP_EPOCH"

    TAO_TELEMETRY_SERVER = "TAO_TELEMETRY_SERVER"
    TAO_CLIENT_TYPE = "TAO_CLIENT_TYPE"  # Client type: container, api, cli, sdk, ui, etc.
    TAO_AUTOML_TRIGGERED = "TAO_AUTOML_TRIGGERED"  # Whether job is triggered by AutoML


class NVCFEndpoint(Enum):
    """Class defining action type enum"""

    login = 'login'
    org_gpu_types = 'org_gpu_types'
    workspace_retrieve_datasets = 'workspace_retrieve_datasets'
    list = 'list'
    retrieve = 'retrieve'
    delete = 'delete'
    bulk_delete = 'bulk_delete'
    create = 'create'
    update = 'update'
    partial_update = 'partial_update'
    specs_schema = 'specs_schema'
    job_run = 'job_run'
    job_retry = 'job_retry'
    job_list = 'job_list'
    job_retrieve = 'job_retrieve'
    job_schema = 'job_schema'
    job_logs = 'job_logs'
    job_cancel = 'job_cancel'
    job_delete = 'job_delete'
    job_download = 'job_download'
    job_pause = 'job_pause'
    jobs_cancel = 'jobs_cancel'
    bulk_cancel = 'bulk_cancel'
    job_resume = 'job_resume'
    automl_details = 'automl_details'
    get_epoch_numbers = 'get_epoch_numbers'
    model_publish = 'model_publish'
    remove_published_model = 'remove_published_model'
    status_update = 'status_update'
    log_update = 'log_update'
    container_job_run = 'container_job_run'
    container_job_status = 'container_job_status'


class CloudPullTypesEnum(Enum):
    """Class defining cloud pull types enum"""

    aws = 'aws'
    azure = 'azure'
    seaweedfs = 'seaweedfs'
    huggingface = 'huggingface'
    self_hosted = 'self_hosted'


class CloudFileType(Enum):
    """Class defining cloud file types enum"""

    file = "file"
    folder = "folder"


class DatasetIntentEnum(Enum):
    """Class defining dataset intent enum"""

    training = 'training'
    evaluation = 'evaluation'
    testing = 'testing'


class CheckpointChooseMethodEnum(Enum):
    """Class defining enum for methods of picking a trained checkpoint"""

    latest_model = 'latest_model'
    best_model = 'best_model'
    from_epoch_number = 'from_epoch_number'


class AutoMLAlgorithm(Enum):
    """Class defining automl algorithm enum"""

    bayesian = "bayesian"
    hyperband = "hyperband"


class SourceType(Enum):
    """Class defining source type enum for base experiments"""

    ngc = "ngc"
    huggingface = "huggingface"


# ============================================================================
# SCHEMAS
# ============================================================================


class MessageOnly(Schema):
    """Class defining dataset upload schema"""

    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))


class MissingFile(Schema):
    """Schema for individual missing file entries"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    path = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=50))
    regex = fields.Str(format="regex", regex=r'.*', allow_none=True)


class ValidationDetails(Schema):
    """Class defining dataset validation details schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    error_details = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    expected_structure = fields.Dict(
        keys=fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100)),
        values=fields.Raw(),
        validate=validate.Length(max=sys.maxsize)
    )
    actual_structure = fields.List(fields.Str(format="regex", regex=r'.*'))
    missing_files = fields.List(fields.Nested(MissingFile))
    network_type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    dataset_format = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    dataset_intent = fields.List(fields.Str(format="regex", regex=r'.*'))


class ErrorRsp(Schema):
    """Class defining error response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    error_desc = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    error_code = fields.Int(
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format()
    )


class PaginationInfo(Schema):
    """Class defining pagination info schema"""

    total_records = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    total_pages = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_size = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_index = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())


class BulkOps(Schema):
    """Class defining bulk operation schema"""

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    status = EnumField(BulkOpsStatus)
    error_desc = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    error_code = fields.Int(
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format(),
        allow_none=True
    )


class BulkOpsRsp(Schema):
    """Class defining bulk operation response schema"""

    results = fields.List(
        fields.Nested(BulkOps, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )


class DetailedStatus(Schema):
    """Class defining Status schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    date = fields.Str(format="mm/dd/yyyy", validate=fields.validate.Length(max=26))
    time = fields.Str(format="hh:mm:ss", validate=fields.validate.Length(max=26))
    message = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=6400))
    status = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class Graph(Schema):
    """Class defining Graph schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    x_min = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format()
    )
    x_max = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format()
    )
    y_min = fields.Float(allow_none=True)
    y_max = fields.Float(allow_none=True)
    values = fields.Dict(keys=fields.Str(allow_none=True), values=fields.Float(allow_none=True))
    units = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class CategoryWise(Schema):
    """Class defining CategoryWise schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    category = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value = fields.Float(allow_none=True)


class Category(Schema):
    """Class defining Category schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    category_wise_values = fields.List(
        fields.Nested(CategoryWise, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )


class KPI(Schema):
    """Class defining KPI schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    values = fields.Dict(allow_none=True)


class CustomFloatField(fields.Float):
    """Class defining custom Float field allown NaN and Inf values in Marshmallow"""

    def _deserialize(self, value, attr, data, **kwargs):
        if value == "nan" or (isinstance(value, float) and math.isnan(value)):
            return float("nan")
        if value == "inf" or (isinstance(value, float) and math.isinf(value)):
            return float("inf")
        if value == "-inf" or (isinstance(value, float) and math.isinf(value)):
            return float("-inf")
        if value is None:
            return None
        return super()._deserialize(value, attr, data)


class AutoMLResults(Schema):
    """Class defining AutoML results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = EnumFieldPrefix(Metrics)
    value = CustomFloatField(allow_none=True)


class Stats(Schema):
    """Class defining results stats schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    value = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))


class JobSubset(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    status = EnumField(JobStatusEnum)
    action = EnumField(ActionEnum)
    eta = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Epoch must be non-negative."
    )
    max_epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Max epoch should be non negative."
    )
    detailed_status_message = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=6400),
        allow_none=True
    )


class JobResult(Schema):
    """Class defining job results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    # Metrics
    graphical = fields.List(
        fields.Nested(Graph, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    categorical = fields.List(
        fields.Nested(Category, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    kpi = fields.List(fields.Nested(KPI, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    # AutoML
    epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Epoch must be non-negative."
    )
    max_epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Max epoch should be non negative."
    )
    automl_brain_info = fields.List(
        fields.Nested(Stats, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    automl_result = fields.List(
        fields.Nested(AutoMLResults, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    # Timing
    time_per_epoch = fields.Str(
        allow_none=True,
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=sys.maxsize)
    )
    time_per_iter = fields.Str(
        allow_none=True,
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=sys.maxsize)
    )
    cur_iter = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )
    eta = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    # General
    detailed_status = fields.Nested(DetailedStatus, allow_none=True)
    key_metric = fields.Float(allow_none=True)
    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    # Specs (only populated for AutoML experiments)
    specs = fields.Raw(allow_none=True)


class LoginReq(Schema):
    """Class defining login request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    ngc_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    ngc_org_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    enable_telemetry = fields.Bool(default=False, allow_none=True)  # NVAIE requires disable telemetry by default


class LoginRsp(Schema):
    """Class defining login response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    token = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    user_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    user_email = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)


class NVCFReq(Schema):
    """Class defining login response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    ngc_org_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    api_endpoint = EnumField(NVCFEndpoint)
    kind = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    handler_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    is_base_experiment = fields.Bool()
    is_job = fields.Bool()
    job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    action = EnumField(ActionEnum)
    request_body = fields.Raw()
    ngc_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    is_json_request = fields.Bool()


class GpuDetails(Schema):
    """Class defining telemetry request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    cluster = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    node = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    gpu_type = fields.Str(validate=validate.Length(max=2048))
    instance_type = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    gpu_count = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    cpu_cores = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    system_memory = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    gpu_memory = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    regions = fields.List(fields.Str(validate=validate.Length(max=2048)), allow_none=True)
    storage = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    driver_version = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    max_limit = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    current_used = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    current_available = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)


class TelemetryReq(Schema):
    """Class defining telemetry request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    version = fields.Str()
    network = fields.Str()
    action = fields.Str()
    success = fields.Bool()
    gpu = fields.List(fields.Str())
    time_lapsed = fields.Int(allow_none=True)
    user_error = fields.Bool(allow_none=True)
    client_type = fields.Str(allow_none=True)  # Client type: container, api, cli, sdk, ui, etc.
    automl_triggered = fields.Bool(allow_none=True)  # Whether job is triggered by AutoML


class AWSCloudPull(Schema):
    """Class defining AWS Cloud pull schema"""

    access_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    secret_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    endpoint_url = fields.Str(validate=[validate_endpoint_url, validate.Length(max=2048)], allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)
    cloud_type = fields.Constant(CloudPullTypesEnum.aws.value)


class SeaweedfsCloudPull(AWSCloudPull):
    """Class defining Seaweed Cloud pull schema"""

    cloud_type = fields.Constant(CloudPullTypesEnum.seaweedfs.value)


class AzureCloudPull(Schema):
    """Class defining Azure Cloud pull schema"""

    account_name = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    access_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    endpoint_url = fields.Str(validate=[validate_endpoint_url, validate.Length(max=2048)], allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)
    cloud_type = fields.Constant(CloudPullTypesEnum.azure.value)


class HuggingFaceCloudPull(Schema):
    """Class defining Hugging Face Cloud pull schema"""

    token = fields.Str(validate=validate.Length(max=2048))
    cloud_type = fields.Constant(CloudPullTypesEnum.huggingface.value)


class CloudSpecificDetails(OneOfSchema):
    """Class defining a polymorphic cloud specific details schema"""

    type_schemas = {
        "aws": AWSCloudPull,
        "azure": AzureCloudPull,
        "huggingface": HuggingFaceCloudPull,
        "seaweedfs": SeaweedfsCloudPull,
    }
    type_field = "cloud_type"

    def get_obj_type(self, obj):
        """Determine the schema to use based on the properties of the Python object"""
        cloud_type = obj.get("cloud_type")
        if cloud_type in [e.value for e in CloudPullTypesEnum]:
            return cloud_type
        raise fields.ValidationError(f"Invalid cloud type: {cloud_type}")


class WorkspaceReq(Schema):
    """Class defining Cloud Workspace request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    cloud_type = EnumField(CloudPullTypesEnum, allow_none=False)
    cloud_specific_details = fields.Nested(CloudSpecificDetails, allow_none=False)

    @validates_schema
    def validate_cloud_specific_details(self, data, **kwargs):
        """Return schema based on cloud_type and validate credentials"""
        cloud_type = data.get('cloud_type')

        if cloud_type:
            # First, validate the schema structure
            if cloud_type == CloudPullTypesEnum.aws:
                schema = AWSCloudPull()
            elif cloud_type == CloudPullTypesEnum.azure:
                schema = AzureCloudPull()
            elif cloud_type == CloudPullTypesEnum.seaweedfs:
                schema = AWSCloudPull()
            elif cloud_type == CloudPullTypesEnum.huggingface:
                schema = HuggingFaceCloudPull()
            else:
                schema = Schema()

            try:
                # Validate schema structure
                schema.load(data.get('cloud_specific_details', {}), unknown=EXCLUDE)
            except ValidationError:
                # Re-raise ValidationError as-is
                raise
            except Exception as e:
                raise fields.ValidationError(str(e))


class WorkspaceBackupReq(Schema):
    """Class defining workspace backup schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    backup_file_name = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    workspace_metadata = fields.Nested(WorkspaceReq, allow_none=False)


class WorkspaceRsp(Schema):
    """Class defining Cloud pull schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    cloud_type = EnumField(CloudPullTypesEnum, allow_none=False)


class WorkspaceListRsp(Schema):
    """Class defining workspace list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    workspaces = fields.List(fields.Nested(WorkspaceRsp), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfo, allow_none=True)


class DatasetPathLst(Schema):
    """Class defining dataset actions schema"""

    dataset_paths = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        ),
        validate=validate.Length(max=sys.maxsize)
    )


class DatasetActions(Schema):
    """Class defining dataset actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class LstStr(Schema):
    """Class defining dataset actions schema"""

    dataset_formats = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        ),
        validate=validate.Length(max=sys.maxsize)
    )
    accepted_dataset_intents = fields.List(
        EnumField(DatasetIntentEnum),
        allow_none=True,
        validate=validate.Length(max=3)
    )


class DatasetReq(Schema):
    """Class defining dataset request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048))
    type = EnumField(DatasetType)
    format = EnumField(DatasetFormat)
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    url = fields.URL(validate=fields.validate.Length(max=2048))  # For HuggingFace and Self_hosted
    cloud_file_path = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)
    use_for = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=3))
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )
    skip_validation = fields.Bool(allow_none=True)
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )


class DatasetJob(Schema):
    """Class defining dataset job result total schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    job_details = fields.Dict(
        keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        values=fields.Nested(JobResult),
        validate=validate.Length(max=sys.maxsize)
    )
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )


class DatasetRsp(Schema):
    """Class defining dataset response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("user_id", "docker_env_vars", "client_id", "client_secret", "filters")
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    type = EnumField(DatasetType)
    format = EnumField(DatasetFormat)
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    url = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)  # For HuggingFace and Self_hosted
    cloud_file_path = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.Dict(
        keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        values=fields.Nested(JobSubset),
        validate=validate.Length(max=sys.maxsize)
    )
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)
    use_for = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=3))
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )
    skip_validation = fields.Bool(allow_none=True)
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    validation_details = fields.Nested(ValidationDetails, allow_none=True)
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )


class DatasetListRsp(Schema):
    """Class defining dataset list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    datasets = fields.List(fields.Nested(DatasetRsp), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfo, allow_none=True)


class ExperimentActions(Schema):
    """Class defining job actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class PublishModel(Schema):
    """Class defining Publish model schema"""

    display_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    team_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    # format, framework, precision - to be determined by backend


class JobResume(Schema):
    """Class defining job resume request schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    specs = fields.Raw(allow_none=True)


class ParameterRangeSchema(Schema):
    """Schema for parameter attributes (used for both default and custom)"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    parameter = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=500),
        required=False  # Not required when used as nested schema
    )
    default_value = fields.Raw(allow_none=True)  # Only used for default section
    valid_min = fields.Raw(allow_none=True)  # Can be float or list of floats
    valid_max = fields.Raw(allow_none=True)  # Can be float or list of floats
    valid_options = fields.List(fields.Raw(), allow_none=True)
    option_weights = fields.List(fields.Float(), allow_none=True)  # Weights for valid_options
    math_cond = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    depends_on = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    parent_param = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)


class AutoMLParameterDetail(Schema):
    """Class defining individual parameter detail schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameter = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value_type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    default = fields.Nested(ParameterRangeSchema)
    custom = fields.Nested(ParameterRangeSchema, allow_none=True)


class AutoMLParameterDetailsRsp(Schema):
    """Class defining response schema for getting parameter details"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameter_details = fields.List(fields.Nested(AutoMLParameterDetail), validate=validate.Length(max=sys.maxsize))


class AutoMLUpdateParameterRangesReq(Schema):
    """Class defining request schema for updating parameter ranges"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), required=True)
    network_arch = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), required=True)
    parameter_ranges = fields.List(
        fields.Nested(ParameterRangeSchema),
        validate=validate.Length(min=1, max=sys.maxsize),
        required=True
    )


class AutoML(Schema):
    """Class defining automl parameters in a schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    automl_enabled = fields.Bool(allow_none=True)
    automl_algorithm = EnumField(AutoMLAlgorithm, allow_none=True)
    automl_max_recommendations = fields.Int(
        format="int64",
        validate=validate.Range(min=0, max=sys.maxsize),
        allow_none=True
    )
    automl_delete_intermediate_ckpt = fields.Bool(allow_none=True)
    override_automl_disabled_params = fields.Bool(allow_none=True)
    automl_R = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_nu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    epoch_multiplier = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    automl_hyperparameters = fields.Str(
        format="regex",
        regex=r'\[.*\]',
        validate=fields.validate.Length(max=5000),
        allow_none=True
    )
    automl_range_override = fields.List(
        fields.Nested(ParameterRangeSchema),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )


class BaseExperimentMetadata(Schema):
    """Class defining base experiment metadata schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    task = EnumField(BaseExperimentTask, by_value=True, allow_none=True)
    domain = EnumField(BaseExperimentDomain, by_value=True, allow_none=True)
    backbone_type = EnumField(BaseExperimentBackboneType, by_value=True, allow_none=True)
    backbone_class = EnumField(BaseExperimentBackboneClass, by_value=True, allow_none=True)
    num_parameters = fields.Str(
        format="regex",
        regex=r'^\d+(\.\d+)?M$',
        validate=fields.validate.Length(max=10),
        allow_none=True
    )
    accuracy = fields.Str(
        format="regex",
        regex=r'^\d{1,3}(\.\d+)?%$',
        validate=fields.validate.Length(max=10),
        allow_none=True
    )
    license = EnumField(BaseExperimentLicense, by_value=True, allow_none=True)
    model_card_link = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    is_backbone = fields.Bool()
    is_trainable = fields.Bool()
    spec_file_present = fields.Bool()
    specs = fields.Raw(allow_none=True)


class InferenceMicroserviceReq(Schema):
    """Class defining inference microservice request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    parent_job_id = fields.Str(
        format="uuid",
        validate=fields.validate.Length(max=36),
        description="Parent job ID",
        example="12345678-1234-1234-1234-123456789012",
        allow_none=True,
        required=False
    )
    kind = EnumField(
        JobKindEnum,
        description="Job kind",
        example="experiment",
        allow_none=True,
        required=False
    )
    model_path = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Path to the model",
        example="/workspace/model",
        allow_none=True,
        required=False
    )
    enable_lora = fields.Bool(
        description="Enable LoRA for inference",
        default=False
    )
    base_model_path = fields.Str(
        description="Base model path (e.g., hf_model://nvidia/Cosmos-Reason1-7B)",
        required=False,
        allow_none=True
    )
    docker_image = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Docker image for inference",
        example="nvcr.io/nvidia/vila-inference:latest"
    )
    gpu_type = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="GPU type",
        example="h100"
    )
    num_gpus = fields.Int(
        format="int64",
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        description="Number of GPUs required",
        example=1
    )
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(
        keys=fields.Str(
            format="regex",
            regex=(
                r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$'
            ),
            validate=fields.validate.Length(max=100),
            allow_none=True
        ),
        values=fields.Int(
            format="int64",
            validate=validate.Range(min=0, max=sys.maxsize),
            allow_none=True
        )
    )
    network_arch = EnumField(ExperimentNetworkArch, allow_none=False)


class InferenceMicroserviceRsp(Schema):
    """Class defining inference microservice response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    job_id = fields.Str(
        format="uuid",
        validate=fields.validate.Length(max=36),
        allow_none=True,
        description="Unique job ID for this Inference Microservice"
    )
    status = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Service status"
    )
    message = fields.Str(
        validate=fields.validate.Length(max=2048),
        description="Success message"
    )


class InferenceReq(Schema):
    """Class defining inference request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    input = fields.List(
        fields.Str(
            description="Base64-encoded images/videos with data URI format (data:image/jpeg;base64,...)",
            required=False
        ),
        allow_none=True
    )
    media = fields.Str(
        description="Cloud path to media file (e.g., aws://bucket/path/to/video.mp4)",
        required=False,
        allow_none=True
    )
    model = fields.Str(
        description="Model identifier (e.g. nvidia/nvdino-v2)",
        required=False,
        allow_none=True
    )
    prompt = fields.Str(
        description="Text prompt for VLM inference",
        required=False,
        allow_none=True,
        default=""
    )
    enable_lora = fields.Bool(
        description="Enable LoRA for inference",
        required=False,
        allow_none=True
    )
    base_model_path = fields.Str(
        description="Base model path (e.g., hf_model://nvidia/Cosmos-Reason1-7B)",
        required=False,
        allow_none=True
    )


class LoadAirgappedExperimentsReq(Schema):
    """Class defining load airgapped models request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    workspace_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    models_base_dir = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )


class LoadAirgappedExperimentsRsp(Schema):
    """Class defining load airgapped models response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    success = fields.Bool()
    message = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048)
    )
    experiments_loaded = fields.Int(
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )
    experiments_failed = fields.Int(
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )


class DatasetJobReq(Schema):
    """Class defining an dataset job request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=False, required=True)
    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    kind = fields.Constant(JobKindEnum.dataset.value)
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )


class ExperimentJobReq(Schema):
    """Class defining an experiment job request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Model version description - not changing variable name for backward compatibility
    model_description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Description common to all versions of models
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    ngc_path = fields.Str(
        format="regex",
        regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$',
        validate=fields.validate.Length(max=250)
    )
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    sha256_digest = fields.Dict(allow_none=True)
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=100),
        allow_none=True
    )
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(
        keys=fields.Str(
            format="regex",
            regex=(
                r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$'
            ),
            validate=fields.validate.Length(max=100),
            allow_none=True
        ),
        values=fields.Int(
            format="int64",
            validate=validate.Range(min=0, max=sys.maxsize),
            allow_none=True
        )
    )
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArch)
    base_experiment_ids = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=2)
    )
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    train_datasets = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=sys.maxsize)
    )
    read_only = fields.Bool()
    public = fields.Bool()
    automl_settings = fields.Nested(AutoML, allow_none=True)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    realtime_infer = fields.Bool(default=False)
    model_params = fields.Dict(allow_none=True)
    bundle_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    realtime_infer_request_timeout = fields.Int(
        format="int64",
        validate=validate.Range(min=0, max=sys.maxsize),
        allow_none=True
    )
    experiment_actions = fields.List(
        fields.Nested(ExperimentActions, allow_none=True),
        validate=validate.Length(max=sys.maxsize)
    )
    tensorboard_enabled = fields.Bool(allow_none=True)
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )
    retry_experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    kind = fields.Constant(JobKindEnum.experiment.value)


class JobReq(OneOfSchema):
    """Class defining a polymorphic job request schema"""

    type_schemas = {
        "dataset": DatasetJobReq,
        "experiment": ExperimentJobReq
    }
    type_field = "kind"

    def get_obj_type(self, obj):
        """Determine the schema to use based on the properties of the Python object"""
        kind = obj.get("kind")
        if kind in JobKindEnum:
            return kind
        raise fields.ValidationError(f"Invalid job kind: {kind}")


class DatasetJobRsp(Schema):
    """Class defining dataset job response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    job_details = fields.Dict(
        keys=fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        values=fields.Nested(JobResult),
        validate=validate.Length(max=sys.maxsize)
    )
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )
    kind = fields.Constant(JobKindEnum.dataset.value)
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment_ids = fields.List(
        fields.Str(format="uuid", validate=fields.validate.Length(max=36)),
        validate=validate.Length(max=2)
    )


class ExperimentJobRsp(Schema):
    """Class defining experiment job response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        load_only = ("user_id", "docker_env_vars", "realtime_infer_endpoint", "realtime_infer_model_name")
        unknown = EXCLUDE

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    created_on = DateTimeField(metadata={"maxLength": 24})
    last_modified = DateTimeField(metadata={"maxLength": 24})
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Model version description - not changing variable name for backward compatibility
    model_description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )  # Description common to all versions of models
    version = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    logo = fields.URL(validate=fields.validate.Length(max=2048), allow_none=True)
    ngc_path = fields.Str(
        format="regex",
        regex=r'^\w+(/[\w-]+)?/[\w-]+:[\w.-]+$',
        validate=fields.validate.Length(max=250)
    )
    workspace = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    sha256_digest = fields.Dict(allow_none=True)
    base_experiment_pull_complete = EnumField(PullStatus)
    additional_id_info = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=100),
        allow_none=True
    )
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    checkpoint_choose_method = EnumField(CheckpointChooseMethodEnum)
    checkpoint_epoch_number = fields.Dict(
        keys=fields.Str(
            format="regex",
            regex=(
                r'(from_epoch_number|latest_model|best_model)_[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}$'
            ),
            validate=fields.validate.Length(max=100),
            allow_none=True
        ),
        values=fields.Int(
            format="int64",
            validate=validate.Range(min=0, max=sys.maxsize),
            allow_none=True
        )
    )
    encryption_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    network_arch = EnumField(ExperimentNetworkArch)
    base_experiment_ids = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=2)
    )
    dataset_type = EnumField(DatasetType)
    dataset_formats = fields.List(EnumField(DatasetFormat), allow_none=True, validate=validate.Length(max=sys.maxsize))
    accepted_dataset_intents = fields.List(
        EnumField(DatasetIntentEnum, allow_none=True),
        validate=validate.Length(max=sys.maxsize)
    )
    eval_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    inference_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    calibration_dataset = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    train_datasets = fields.List(
        fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=sys.maxsize)
    )
    read_only = fields.Bool()
    public = fields.Bool()
    actions = fields.List(EnumField(ActionEnum), allow_none=True, validate=validate.Length(max=sys.maxsize))
    jobs = fields.Dict(
        keys=fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        values=fields.Nested(JobSubset),
        validate=validate.Length(max=sys.maxsize)
    )
    status = EnumField(JobStatusEnum)
    all_jobs_cancel_status = EnumField(JobStatusEnum, allow_none=True)
    automl_settings = fields.Nested(AutoML)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    realtime_infer = fields.Bool(allow_none=True)
    realtime_infer_support = fields.Bool()
    realtime_infer_endpoint = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )
    realtime_infer_model_name = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000),
        allow_none=True
    )
    model_params = fields.Dict(allow_none=True)
    realtime_infer_request_timeout = fields.Int(
        format="int64",
        validate=validate.Range(min=0, max=86400),
        allow_none=True
    )
    bundle_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    base_experiment_metadata = fields.Nested(BaseExperimentMetadata, allow_none=True)
    source_type = EnumField(SourceType, allow_none=True)
    experiment_actions = fields.List(
        fields.Nested(ExperimentActions, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    tensorboard_enabled = fields.Bool(default=False)
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )
    authorized_party_nca_id = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=2048),
        allow_none=True
    )
    parent_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    status = EnumField(JobStatusEnum)
    job_details = fields.Dict(
        keys=fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        values=fields.Nested(JobResult),
        validate=validate.Length(max=sys.maxsize)
    )
    sync = fields.Bool()
    specs = fields.Raw(allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    epoch_numbers = fields.List(
        fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize)),
        validate=validate.Length(max=sys.maxsize),
        allow_none=True
    )


class JobRsp(OneOfSchema):
    """Class defining a polymorphic job response schema"""

    type_schemas = {
        "dataset": DatasetJobRsp,
        "experiment": ExperimentJobRsp
    }
    type_field = "kind"

    def get_obj_type(self, obj):
        """Determine the schema to use based on the properties of the Python object"""
        kind = obj.get("kind")
        if kind in JobKindEnum:
            return kind
        raise fields.ValidationError(f"Invalid job kind: {kind}")


class JobListRsp(Schema):
    """Class defining job list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    jobs = fields.List(
        fields.Nested(JobRsp),
        validate=validate.Length(max=sys.maxsize)
    )
    pagination_info = fields.Nested(PaginationInfo, allow_none=True)
