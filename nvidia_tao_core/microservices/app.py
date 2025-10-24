#!/usr/bin/env python3

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

"""API modules defining schemas and endpoints"""
import ast

import pkg_resources
import bson
import sys
import uuid
import math
import json
import shutil
import os
import re
import requests
import traceback
import logging

from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from apispec_webframeworks.flask import FlaskPlugin
from flask import Flask, request, jsonify, make_response, render_template, send_from_directory, send_file
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from requests_toolbelt.multipart.encoder import MultipartEncoder
from marshmallow import Schema, fields, exceptions, validate, validates_schema, ValidationError, EXCLUDE
from marshmallow_enum import EnumField, Enum

from nvidia_tao_core.microservices.filter_utils import filtering, pagination
from nvidia_tao_core.microservices.auth_utils import credentials, authentication, access_control, metrics
from nvidia_tao_core.microservices.health_utils import health_check
from nvidia_tao_core.microservices.handlers.inference_microservice_handler import InferenceMicroserviceHandler
from nvidia_tao_core.microservices.handlers.mongo_handler import MongoHandler
from nvidia_tao_core.microservices.app_handlers.mongo_handler import MongoBackupHandler
from nvidia_tao_core.microservices.constants import AIRGAP_DEFAULT_USER
from nvidia_tao_core.microservices.enum_constants import (
    ActionEnum,
    DatasetFormat,
    DatasetType,
    ExperimentNetworkArch,
    ContainerNetworkArch,
    Metrics,
    BaseExperimentTask,
    BaseExperimentDomain,
    BaseExperimentBackboneType,
    BaseExperimentBackboneClass,
    BaseExperimentLicense,
    _get_dynamic_metric_patterns
)
from nvidia_tao_core.microservices.handlers.app_handler import AppHandler as app_handler
from nvidia_tao_core.microservices.handlers.container_handler import ContainerJobHandler as container_handler
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    resolve_metadata,
    get_root,
    get_metrics,
    set_metrics
)
from nvidia_tao_core.microservices.handlers.utilities import validate_uuid, send_microservice_request
from nvidia_tao_core.microservices.utils import (
    is_pvc_space_free,
    safe_load_file,
    log_monitor,
    log_api_error,
    DataMonitorLogTypeEnum
)
from nvidia_tao_core.microservices.job_utils.workflow import Workflow

from werkzeug.exceptions import HTTPException
from werkzeug.middleware.profiler import ProfilerMiddleware
from datetime import datetime
from functools import wraps

flask_plugin = FlaskPlugin()
marshmallow_plugin = MarshmallowPlugin()

TIMEOUT = 240

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

NAMESPACE = os.getenv("NAMESPACE", "default")

#
# Utils
#


def sys_int_format():
    """Get integer format based on system."""
    if sys.maxsize > 2**31 - 1:
        return "int64"
    return "int32"


def disk_space_check(f):
    """Decorator to check disk space for API endpoints"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        threshold_bytes = 100 * 1024 * 1024

        pvc_free_space, pvc_free_bytes = is_pvc_space_free(threshold_bytes)
        msg = (f"PVC free space remaining is {pvc_free_bytes} bytes "
               f"which is less than {threshold_bytes} bytes")
        if not pvc_free_space:
            return make_response(
                jsonify({
                    'error': f'Disk space is nearly full. {msg}. Delete appropriate experiments/datasets'
                }),
                500
            )

        return f(*args, **kwargs)

    return decorated_function


#
# Create an APISpec
#

try:
    tao_version = pkg_resources.get_distribution('nvidia_tao_core').version
except Exception:
    tao_version = os.getenv('TAO_VERSION', '6.0.0')

spec = APISpec(
    title='NVIDIA TAO API',
    version=tao_version,
    openapi_version='3.0.3',
    info={"description": 'NVIDIA TAO (Train, Adapt, Optimize) API document'},
    tags=[
        {"name": 'AUTHENTICATION', "description": 'Endpoints related to User Authentication'},
        {"name": 'DATASET', "description": 'Endpoints related to Datasets'},
        {"name": 'EXPERIMENT', "description": 'Endpoints related to Experiments'},
        {"name": "nSpectId",
         "description": "NSPECT-1T59-RTYH",
         "externalDocs": {
             "url": "https://nspect.nvidia.com/review?id=NSPECT-1T59-RTYH"
         }}
    ],
    plugins=[flask_plugin, marshmallow_plugin],
    security=[{"bearer-token": []}],
)

api_key_scheme = {"type": "apiKey", "in": "header", "name": "ngc_key"}
jwt_scheme = {"type": "http", "scheme": "bearer", "bearerFormat": "JWT", "description": "RFC8725 Compliant JWT"}

spec.components.security_scheme("api-key", api_key_scheme)
spec.components.security_scheme("bearer-token", jwt_scheme)

spec.components.header("X-RateLimit-Limit", {
    "description": "The number of allowed requests in the current period",
    "schema": {
        "type": "integer",
        "format": sys_int_format(),
        "minimum": -sys.maxsize - 1,
        "maximum": sys.maxsize,
    }
})
spec.components.header("Access-Control-Allow-Origin", {
    "description": "Origins that are allowed to share response",
    "schema": {
        "type": "string",
        "format": "regex",
        "maxLength": sys.maxsize,
    }
})


#
# Enum stuff for APISpecs
#
def enum_to_properties(self, field, **kwargs):
    """Add an OpenAPI extension for marshmallow_enum.EnumField instances"""
    if isinstance(field, EnumField):
        return {'type': 'string', 'enum': [m.name for m in field.enum]}
    return {}


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


marshmallow_plugin.converter.add_attribute_function(enum_to_properties)


#
# Global schemas and enums
#
class MessageOnlySchema(Schema):
    """Class defining dataset upload schema"""

    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))


class MissingFileSchema(Schema):
    """Schema for individual missing file entries"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE

    path = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=50))
    regex = fields.Str(format="regex", regex=r'.*', allow_none=True)


class ValidationDetailsSchema(Schema):
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
    missing_files = fields.List(fields.Nested(MissingFileSchema))
    network_type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    dataset_format = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    dataset_intent = fields.List(fields.Str(format="regex", regex=r'.*'))


class ErrorRspSchema(Schema):
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


class PaginationInfoSchema(Schema):
    """Class defining pagination info schema"""

    total_records = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    total_pages = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_size = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())
    page_index = fields.Int(validate=fields.validate.Range(min=0, max=sys.maxsize), format=sys_int_format())


#
# Flask app
#

class CustomProfilerMiddleware(ProfilerMiddleware):
    """Class defining custom middleware to exclude health related endpoints from profiling"""

    def __call__(self, environ, start_response):
        """Wrapper around ProfilerMiddleware to only perform profiling for non health related API requests"""
        if '/api/v1/health' in environ['PATH_INFO']:
            return self._app(environ, start_response)
        return super().__call__(environ, start_response)


app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
csrf = CSRFProtect()
csrf.init_app(app)
app.json.sort_keys = False
app.config['TRAP_HTTP_EXCEPTIONS'] = True
if os.getenv("PROFILER", "FALSE") == "True":
    app.config["PROFILE"] = True
    app.wsgi_app = CustomProfilerMiddleware(
        app.wsgi_app,
        stream=sys.stderr,
        sort_by=('cumtime',),
        restrictions=[50],
    )
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["10000/hour"],
    headers_enabled=True,
    storage_uri="memory://",
)


@app.errorhandler(HTTPException)
def handle_exception(e):
    """Return JSON instead of HTML for HTTP errors."""
    # start with the correct headers and status code from the error
    response = e.get_response()
    # replace the body with JSON
    response.data = json.dumps({
        "code": e.code,
        "name": e.name,
        "description": e.description,
    })
    response.content_type = "application/json"
    return response


@app.errorhandler(exceptions.ValidationError)
def handle_validation_exception(e):
    """Return 400 bad request for validation exceptions"""
    metadata = {"error_desc": str(e)}
    schema = ErrorRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    return response


# Define enum and schema common to Dataset and Experiment Api

class BulkOpsStatus(Enum):
    """Class defining bulk operation status enum"""

    success = "success"
    failed = "failed"


class BulkOpsSchema(Schema):
    """Class defining bulk operation schema"""

    id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    status = EnumField(BulkOpsStatus)
    error_desc = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    error_code = fields.Int(
        validate=fields.validate.Range(min=-sys.maxsize - 1, max=sys.maxsize),
        format=sys_int_format(),
        allow_none=True
    )


class BulkOpsRspSchema(Schema):
    """Class defining bulk operation response schema"""

    results = fields.List(
        fields.Nested(BulkOpsSchema, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )


#
# JobResultSchema
#
class DetailedStatusSchema(Schema):
    """Class defining Status schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    date = fields.Str(format="mm/dd/yyyy", validate=fields.validate.Length(max=26))
    time = fields.Str(format="hh:mm:ss", validate=fields.validate.Length(max=26))
    message = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=6400))
    status = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))


class GraphSchema(Schema):
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


class CategoryWiseSchema(Schema):
    """Class defining CategoryWise schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    category = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value = fields.Float(allow_none=True)


class CategorySchema(Schema):
    """Class defining Category schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    metric = EnumFieldPrefix(Metrics)
    category_wise_values = fields.List(
        fields.Nested(CategoryWiseSchema, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )


class KPISchema(Schema):
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


class AutoMLResultsSchema(Schema):
    """Class defining AutoML results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = EnumFieldPrefix(Metrics)
    value = CustomFloatField(allow_none=True)


class AutoMLResultsDetailedSchema(Schema):
    """Class defining AutoML detailed results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    current_experiment_id = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )
    best_experiment_id = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format()
    )
    metric = EnumFieldPrefix(Metrics)
    experiments = fields.Raw()


class StatsSchema(Schema):
    """Class defining results stats schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    metric = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    value = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))


class JobSubsetSchema(Schema):
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
        validate=fields.validate.Range(min=-1, max=sys.maxsize),
        format=sys_int_format(),
        error="Epoch should be larger than -1. With -1 meaning non-valid."
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


class JobResultSchema(Schema):
    """Class defining job results schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    # Metrics
    graphical = fields.List(
        fields.Nested(GraphSchema, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    categorical = fields.List(
        fields.Nested(CategorySchema, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    kpi = fields.List(fields.Nested(KPISchema, allow_none=True), validate=fields.validate.Length(max=sys.maxsize))
    # AutoML
    epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=-1, max=sys.maxsize),
        format=sys_int_format(),
        error="Epoch should be larger than -1. With -1 meaning non-valid."
    )
    max_epoch = fields.Int(
        allow_none=True,
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        format=sys_int_format(),
        error="Max epoch should be non negative."
    )
    automl_brain_info = fields.List(
        fields.Nested(StatsSchema, allow_none=True),
        validate=fields.validate.Length(max=sys.maxsize)
    )
    automl_result = fields.List(
        fields.Nested(AutoMLResultsSchema, allow_none=True),
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
    detailed_status = fields.Nested(DetailedStatusSchema, allow_none=True)
    key_metric = fields.Float(allow_none=True)
    message = fields.Str(allow_none=True, format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))


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
    AUTOML_EXPERIMENT_NUMBER = "AUTOML_EXPERIMENT_NUMBER"
    JOB_ID = "JOB_ID"
    TAO_API_JOB_ID = "TAO_API_JOB_ID"  # Automl brain job id


#
# AUTHENTICATION API
#
class LoginReqSchema(Schema):
    """Class defining login request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    ngc_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    ngc_org_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
    enable_telemetry = fields.Bool(default=False, allow_none=True)  # NVAIE requires disable telemetry by default


class LoginRspSchema(Schema):
    """Class defining login response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    token = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=sys.maxsize))
    user_name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    user_email = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)


#
# NVCF Super-Endpoint API
#
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


class NVCFReqSchema(Schema):
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


@app.route('/api/v1/orgs/<org_name>/super_endpoint', methods=['POST'])
@disk_space_check
def super_endpoint(org_name):
    """NVCF Super endpoint

    ---
    post:
      tags:
      - ORGS
      summary: Retrieve available GPU types based on the backend during deployment
      description: Retrieve available GPU types based on the backend during deployment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      responses:
        200:
          description: Returned the intented endpoints response
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Returned the intented endpoints response
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        schema = NVCFReqSchema()
        request_metadata = schema.dump(schema.load(request.get_json(force=True)))

        api_endpoint = request_metadata.get("api_endpoint")
        logger.info("Internal api endpoint to be called is %s", api_endpoint)
        kind = request_metadata.get("kind")
        handler_id = request_metadata.get("handler_id")
        is_base_experiment = request_metadata.get("is_base_experiment", False)
        is_job = request_metadata.get("is_job", False)
        job_id = request_metadata.get("job_id")
        action = request_metadata.get("action")
        request_body = request_metadata.get("request_body")
        ngc_key = request_metadata.get("ngc_key", "")
        is_json_request = request_metadata.get("is_json_request", False)

        workspace_only = False
        dataset_only = False
        experiment_only = False

        url = "http://localhost:8000/api/v1"

        if api_endpoint == "login":
            endpoint = f"{url}/login"
            request_type = "POST"

        elif api_endpoint == "org_gpu_types":
            endpoint = f"{url}/orgs/{org_name}:gpu_types"
            request_type = "GET"

        elif api_endpoint == "retrieve_datasets":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}:get_datasets"
            request_type = "GET"
            workspace_only = True

        elif api_endpoint == "get_dataset_formats":
            endpoint = f"{url}/orgs/{org_name}/{kind}:get_formats"
            request_type = "GET"
            dataset_only = True

        elif api_endpoint == "list":
            endpoint = f"{url}/orgs/{org_name}/{kind}"
            if is_base_experiment:
                endpoint = f"{url}/orgs/{org_name}/experiments:base"
            request_type = "GET"

        elif api_endpoint == "retrieve":
            endpoint = f"{url}/orgs/{org_name}/{kind}"
            if handler_id:
                endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}"
                if is_job:
                    endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}"
            request_type = "GET"

        elif api_endpoint == "delete":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}"
            request_type = "DELETE"

        elif api_endpoint == "bulk_delete":
            endpoint = f"{url}/orgs/{org_name}/{kind}"
            request_type = "DELETE"

        elif api_endpoint == "create":
            endpoint = f"{url}/orgs/{org_name}/{kind}"
            request_type = "POST"

        elif api_endpoint == "update":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}"
            request_type = "PUT"

        elif api_endpoint == "partial_update":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}"
            request_type = "PATCH"

        elif api_endpoint == "specs_schema":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/specs/{action}/schema"
            if is_base_experiment:
                endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/specs/{action}/schema:base"
            if is_job:
                endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}/schema"
            request_type = "GET"

        elif api_endpoint == "job_run":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs"
            request_type = "POST"

        elif api_endpoint == "container_job_run":
            endpoint = f"{url}/internal/container_job"
            request_type = "POST"

        elif api_endpoint == "container_job_status":
            endpoint = f"{url}/internal/container_job:status"
            request_body = {"results_dir": request_body.get("specs", {}).get("results_dir")}
            request_type = "GET"

        elif api_endpoint == "job_retry":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:retry"
            request_type = "POST"

        elif api_endpoint == "job_logs":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}/logs"
            request_type = "GET"

        elif api_endpoint == "job_cancel":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}:cancel_all_jobs"
            if is_job:
                endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:cancel"
            request_type = "POST"

        elif api_endpoint == "job_download":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:download"
            request_type = "GET"

        elif api_endpoint == "job_pause":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:pause"
            experiment_only = True
            request_type = "POST"

        elif api_endpoint == "job_resume":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:resume"
            experiment_only = True
            request_type = "POST"

        elif api_endpoint == "automl_details":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:automl_details"
            experiment_only = True
            request_type = "GET"

        elif api_endpoint == "get_epoch_numbers":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:get_epoch_numbers"
            experiment_only = True
            request_type = "GET"

        elif api_endpoint == "model_publish":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:publish_model"
            experiment_only = True
            request_type = "POST"

        elif api_endpoint == "remove_published_model":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:remove_published_model"
            experiment_only = True
            request_type = "DELETE"

        elif api_endpoint == "status_update":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:status_update"
            request_type = "POST"

        elif api_endpoint == "log_update":
            endpoint = f"{url}/orgs/{org_name}/{kind}/{handler_id}/jobs/{job_id}:log_update"
            request_type = "POST"

        else:
            metadata = {"error_desc": "Requested endpoint not present", "error_code": 1}
            schema = ErrorRspSchema()
            return make_response(jsonify(schema.dump(schema.load(metadata))), 404)

        headers = {}
        if os.getenv("HOST_PLATFORM", "") == "NVCF":
            headers['Authorization'] = f"Bearer {dict(request.headers).get('Authorization', ngc_key)}"
        else:
            headers['Authorization'] = dict(request.headers).get('Authorization', ngc_key)
        request_methods = {
            "GET": requests.get,
            "POST": requests.post,
            "PUT": requests.put,
            "PATCH": requests.patch,
            "DELETE": requests.delete
        }

        try:
            if dataset_only and ("experiments" in endpoint or "workspaces" in endpoint):
                raise ValueError(f"Endpoint {endpoint} only for datasets")
            if experiment_only and ("datasets" in endpoint or "workspaces" in endpoint):
                raise ValueError(f"Endpoint {endpoint} only for experiments")
            if workspace_only and ("datasets" in endpoint or "experiments" in endpoint):
                raise ValueError(f"Endpoint {endpoint} only for workspaces")

            if request_type in request_methods:
                response = request_methods[request_type](
                    endpoint,
                    headers=headers,
                    data=request_body if request_type in ("POST", "PATCH", "PUT") and (not is_json_request) else None,
                    json=request_body if request_type in ("POST", "PATCH", "PUT") and is_json_request else None,
                    params=request_body if request_type == "GET" else None,
                    timeout=TIMEOUT
                )
                response.raise_for_status()  # Checks for HTTP errors
                return jsonify(response.json()), response.status_code
            return jsonify({"error": "Unsupported request type"}), 400
        except requests.exceptions.RequestException as e:
            logger.error("Error in internal request: %s", e)
            return jsonify({"error": "Failed to process the request"}), 500
    except Exception as e:
        logger.error("Error in processing request: %s", e)
        return jsonify({"error": "Error in processing request"}), 500


@app.route('/api/v1/login', methods=['POST'])
@disk_space_check
def login():
    """User Login or Exchange username for user_id for air-gapped mode.

    ---
    post:
      tags:
      - AUTHENTICATION
      summary: Authenticate user with NGC credentials and set telemetry preferences
      description: |
        Authenticates a user using their NGC API key and organization name.
        Returns JWT token and user credentials upon successful authentication.
        The token can be used for subsequent API requests.
      security:
        - api-key: []
      requestBody:
        content:
          application/json:
            schema: LoginReqSchema
        description: |
          Login credentials including:
          - ngc_key: NGC API key for authentication
          - ngc_org_name: Organization name in NGC
          - enable_telemetry: Optional telemetry preference (default: False)
        required: true
      responses:
        200:
          description: Successfully authenticated. Returns user credentials and JWT token.
          content:
            application/json:
              schema: LoginRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        401:
          description: Authentication failed due to invalid credentials or permissions
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # air-gapped mode skips NGC authentication and return user_id directly
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        try:
            # Get username from request or use default
            request_data = request.get_json(force=True)
            username = request_data.get("username", AIRGAP_DEFAULT_USER)

            # Create UUID mapping of username
            user_id = str(uuid.uuid5(uuid.UUID(int=0), username))

            # Store user mapping in MongoDB for consistency
            mongo = MongoHandler("tao", "users")
            mongo.upsert({"id": user_id}, {"user_name": username, "id": user_id})

            logger.info("Airgapped mode login: Mapped username '%s' to user_id '%s'", username, user_id)
            return make_response(jsonify({"user_id": user_id}), 200)
        except Exception as e:
            logger.error("Airgapped mode login failed: %s", str(e))
            metadata = {"error_desc": "Login failed: " + str(e), "error_code": 1}
            schema = ErrorRspSchema()
            return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    # Regular NGC authentication flow
    schema = LoginReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    key = request_dict.get('ngc_key', 'invalid_key')
    org_name = request_dict.get('ngc_org_name', '')
    enable_telemetry = request_dict.get('enable_telemetry', None)

    creds, err = credentials.get_from_ngc(key, org_name, enable_telemetry)
    if err:
        logger.warning("Unauthorized: %s", err)
        metadata = {"error_desc": "Unauthorized: " + err, "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 401)
    schema = LoginRspSchema()
    schema_dict = schema.dump(schema.load(creds))
    return make_response(jsonify(schema_dict), 200)


ingress_enabled = os.getenv("INGRESSENABLED", "false") == "true"


# Internal endpoint for ingress controller to check authentication
@app.route('/api/v1/auth', methods=['GET'])
@disk_space_check
def auth():
    """authentication endpoint"""
    # Skip ngc authentication for air-gapped environments
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        try:
            # Get user_id from Authorization header
            user_id = None
            auth_header = request.headers.get('Authorization', '')
            if auth_header:
                user_id = auth_header.removeprefix("Bearer ").strip()

            # fall back to anonymous user
            if not user_id:
                user_id = str(uuid.uuid5(uuid.UUID(int=0), AIRGAP_DEFAULT_USER))

            logger.info("Airgapped mode auth with user_id '%s'", user_id)
            return make_response(jsonify({'user_id': user_id}), 200)
        except Exception as e:
            logger.error("Airgapped mode auth failed: %s", str(e))
            return make_response(jsonify({'error': str(e)}), 400)

    # retrieve jwt from headers
    token = ''
    url = request.headers.get('X-Original-Url', '') if ingress_enabled else request.path
    logger.info('URL: %s', url)
    method = request.headers.get('X-Original-Method', '') if ingress_enabled else request.method
    logger.info('Method: %s', method)
    # bypass authentication for http OPTIONS requests
    if method == 'OPTIONS':
        return make_response(jsonify({}), 200)
    # retrieve authorization token
    authorization = request.headers.get('Authorization', '')
    authorization_parts = authorization.split()
    if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'bearer':
        token = authorization_parts[1]
    if os.getenv("HOST_PLATFORM", "") == "NVCF":
        schema = NVCFReqSchema()
        try:
            request_metadata = schema.dump(schema.load(request.get_json(force=True)))
        except Exception:
            logger.error("Validation of schema failed")
            metadata = {"error_desc": "Validation of schema failed", "error_code": 2}
            schema = ErrorRspSchema()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
            return response
        token = request_metadata.get("ngc_key", "")

    # if token is not found, try to get it from basic auth for special endpoints
    if not token:
        logger.warning("token cannot be obtained")
        if len(authorization_parts) == 2 and authorization_parts[0].lower() == 'basic':
            basic_auth = request.authorization
            if basic_auth:
                # status callback: service to service authentication
                if basic_auth.username == '$oauthtoken':
                    try:
                        org_name, key = basic_auth.password.split(",")
                    except Exception as e:
                        logger.error("Exception thrown in auth: %s", str(e))
                        metadata = {
                            "error_desc": "Basic auth password not in the format of org_name,ngc_personal_key",
                            "error_code": 1
                        }
                        schema = ErrorRspSchema()
                        response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
                        return response
                    creds, err = credentials.get_from_ngc(key, org_name, True)
                    if 'token' in creds:
                        token = creds['token']
                # special metrics case
                elif basic_auth.username == '$metricstoken' and url.split('/', 3)[-1] == 'api/v1/metrics':
                    key = basic_auth.password
                    if metrics.validate(key):
                        return make_response(jsonify({}), 200)
                    metadata = {"error_desc": "wrong metrics key", "error_code": 1}
                    schema = ErrorRspSchema()
                    response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
                    return response

    # if token is still not found, return 401
    if not token:
        schema = ErrorRspSchema()
        metadata = {"error_desc": "Unauthorized: missing token", "error_code": 1}
        rsp = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
        return rsp

    logger.info('Token: ...%s', token[-10:])
    # authentication
    user_id, org_name, err = authentication.validate(url, token)
    log_content = f"user_id:{user_id}, org_name:{org_name}, method:{method}, url:{url}"
    log_monitor(log_type=DataMonitorLogTypeEnum.api, log_content=log_content)
    if err:
        logger.warning("Unauthorized: %s", err)
        metadata = {"error_desc": str(err), "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 401)
        return response
    # access control
    err = access_control.validate(user_id, org_name, url, token)
    if err:
        logger.warning("Forbidden: %s", err)
        metadata = {"error_desc": str(err), "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 403)
        return response
    return make_response(jsonify({'user_id': user_id}), 200)


class ContainerJobSchema(Schema):
    """Class defining NVCF request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    neural_network_name = EnumField(ContainerNetworkArch)
    action_name = EnumField(ActionEnum)
    specs = fields.Raw()
    cloud_metadata = fields.Raw()
    ngc_key = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    docker_env_vars = fields.Dict(
        keys=EnumField(AllowedDockerEnvVariables),
        values=fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500),
            allow_none=True
        )
    )
    statefulset_replicas = fields.Int(
        format="int64",
        validate=fields.validate.Range(min=0, max=sys.maxsize),
        allow_none=True
    )


@app.route('/api/v1/internal/container_job', methods=['POST'])
@disk_space_check
def container_job_run():
    """Run Job within container.

    ---
    post:
      tags:
        - INTERNAL
      summary: Run Container Job
      description:
        Starts a job within a container asynchronously and returns immediately with job ID.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ContainerJobSchema'
      responses:
        200:
          description: The container job was successfully launched.
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    description: The ID of the launched job
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request payload or job execution failed.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRspSchema'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or dataset not found.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRspSchema'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        500:
          description: Internal server error encountered while processing the job.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRspSchema'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    input_schema = ContainerJobSchema()
    job_dict = input_schema.dump(input_schema.load(request.get_json(force=True)))
    try:
        if "job_id" not in job_dict:
            job_dict["job_id"] = str(uuid.uuid4())

        statefulset_replicas = job_dict.get("statefulset_replicas")
        if statefulset_replicas:  # proxy requests to all replicas from master
            for replica_index in range(1, statefulset_replicas):
                send_microservice_request(
                    api_endpoint="post_action",
                    network=job_dict["neural_network_name"],
                    action=job_dict["action_name"],
                    cloud_metadata=job_dict["cloud_metadata"],
                    specs=job_dict["specs"],
                    docker_env_vars=job_dict["docker_env_vars"],
                    job_id=job_dict["job_id"],
                    statefulset_replica_index=replica_index,
                    statefulset_replicas=statefulset_replicas
                )

        job_id = container_handler.entrypoint_wrapper(job_dict)
        if job_id:
            return make_response(jsonify({'job_id': job_id}), 200)
        metadata = {"error": "Failed to launch job", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)
    except Exception as err:
        logger.error("Error in container_job_run: %s", str(traceback.format_exc()))
        metadata = {"error": str(err), "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)


class ContainerJobStatusSchema(Schema):
    """Class defining Get Job Status Response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    status = EnumField(JobStatusEnum)


@app.route('/api/v1/internal/container_job:status', methods=['GET'])
@disk_space_check
def container_job_status():
    """Get status of job running inside container.

    ---
    get:
      tags:
        - INTERNAL
      summary: Get Status of Container Job
      description:
        Retrieves the current status of a job running inside a container. The response
        includes whether the job is pending, in progress, completed, or failed.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                specs:
                  type: object
                  description: Specification details required to check job status.
      responses:
        200:
          description: Job status retrieved successfully.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/MessageOnlySchema'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request payload or unable to determine job status.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRspSchema'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or dataset not found.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRspSchema'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        500:
          description: Internal server error encountered while retrieving job status.
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorRspSchema'
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        response_code = 400
        # For GET requests, data should be in query parameters
        results_dir = request.args.get("results_dir")
        status = container_handler.get_current_job_status(results_dir)
        if status:
            response_code = 200
        schema = ContainerJobStatusSchema()
        schema_dict = schema.dump(schema.load({"status": status}))
        return make_response(jsonify(schema_dict), response_code)
    except Exception as err:
        logger.error("Error in container_job_status: %s", str(traceback.format_exc()))
        metadata = {"error": str(err), "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)


@app.before_request
def authenticate_without_ingress():
    """Authentication endpoint if ingress-nginx is not enabled"""
    skip_api_endpoints = ['/health', '/liveness', '/swagger', '/login', '/auth',
                          '/redoc', '/version', '/rapipdf', '/container_job',
                          '/openapi', '/version', '/tao_api_notebooks']
    if ingress_enabled or any(endpoint in request.path for endpoint in skip_api_endpoints):
        return None
    if "super_endpoint" in request.path:
        request_body = request.get_json(force=True)
        if "container_job" in request_body.get("api_endpoint") or "status_update" in request_body.get("api_endpoint"):
            logger.info("skipping authentication")
            return None
    logger.info("authenticate without ingress, auth being called now for %s", request.path)
    auth_response = auth()
    if auth_response.status_code == 200:
        return None
    logger.warning("authenticate failed")
    return auth_response

#
# ORGS API
#


class GpuDetailsSchema(Schema):
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


@app.route('/api/v1/orgs/<org_name>:gpu_types', methods=['GET'])
@disk_space_check
def org_gpu_types(org_name):
    """Retrieve available GPU type.

    ---
    get:
      tags:
      - ORGS
      summary: Retrieve available GPU types based on the backend during deployment
      description: Retrieve available GPU types based on the backend during deployment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      responses:
        200:
          description: Returned the gpu_types available
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: GPU types can't be retrieved for deployed Backend
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.get_gpu_types(user_id, org_name)
    # Get schema
    schema = GpuDetailsSchema()
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


#
# Metrics API
#

class TelemetryReqSchema(Schema):
    """Class defining telemetry request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    version = fields.Str()
    network = fields.Str()
    action = fields.Str()
    success = fields.Bool()
    gpu = fields.List(fields.Str())
    time_lapsed = fields.Int()


@app.route('/api/v1/metrics', methods=['POST'])
def metrics_upsert():
    """Report execution of new action.

    ---
    post:
        tags:
        - TELEMETRY
        summary: Report execution of new action
        description: Post anonymous metrics to NVIDIA Kratos
        requestBody:
            content:
                application/json:
                    schema: TelemetryReqSchema
                    description: Report new action, network and gpu list
                    required: true
        responses:
            200:
                description: Sucessfully reported execution of new action
            400:
                description: Bad request, see reply body for details
                content:
                    application/json:
                        schema: ErrorRspSchema
    """
    now = old_now = datetime.now()

    # get action report

    try:
        data = TelemetryReqSchema().load(request.get_json(force=True))
    except Exception as e:
        logger.error("Exception thrown in metrics_upsert: %s", str(e))
        return make_response(jsonify({}), 400)

    # update metrics.json

    metrics = get_metrics()
    if not metrics:
        metrics = safe_load_file(os.path.join(get_root(), 'metrics.json'))
        if not metrics:
            metadata = {
                "error_desc": "Metrics.json file not exists or can not be updated now, please try again later.",
                "error_code": 503
            }
            schema = ErrorRspSchema()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
            return response

    old_now = datetime.fromisoformat(metrics.get('last_updated', now.isoformat()))
    version = re.sub("[^a-zA-Z0-9]", "_", data.get('version', 'unknown')).lower()
    action = re.sub("[^a-zA-Z0-9]", "_", data.get('action', 'unknown')).lower()
    network = re.sub("[^a-zA-Z0-9]", "_", data.get('network', 'unknown')).lower()
    success = data.get('success', False)
    time_lapsed = data.get('time_lapsed', 0)
    gpus = data.get('gpu', ['unknown'])
    if success:
        metrics[f'total_action_{action}_pass'] = metrics.get(f'total_action_{action}_pass', 0) + 1
    else:
        metrics[f'total_action_{action}_fail'] = metrics.get(f'total_action_{action}_fail', 0) + 1
    metrics[f'version_{version}_action_{action}'] = metrics.get(f'version_{version}_action_{action}', 0) + 1
    metrics[f'network_{network}_action_{action}'] = metrics.get(f'network_{network}_action_{action}', 0) + 1
    metrics['time_lapsed_today'] = metrics.get('time_lapsed_today', 0) + time_lapsed
    if now.strftime("%d") != old_now.strftime("%d"):
        metrics['time_lapsed_today'] = time_lapsed
    for gpu in gpus:
        gpu = re.sub("[^a-zA-Z0-9]", "_", gpu).lower()
        metrics[f'gpu_{gpu}_action_{action}'] = metrics.get(f'gpu_{gpu}_action_{action}', 0) + 1
    metrics['last_updated'] = now.isoformat()

    def sanitize_gpu_name(gpu_name):
        # Convert to uppercase first, then replace all non-alphanumeric characters with _
        return re.sub("[^a-zA-Z0-9]", "_", gpu_name.upper())

    def create_gpu_identifier(gpu_list):
        # Count occurrences of each GPU type (case insensitive)
        gpu_counts = {}
        for gpu in map(sanitize_gpu_name, gpu_list):
            gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1

        # Format as "gpu_count_gpu1_count_gpu2_count..."
        gpu_parts = [f"{gpu}_{count}" for gpu, count in sorted(gpu_counts.items())]
        return f"{len(gpu_list)}_{'_'.join(gpu_parts)}"

    # Build metric name with all attributes
    status = "pass" if success else "fail"
    metric_components = [
        "network", network,
        "action", action,
        "version", version,
        "status", status,
        "gpu", create_gpu_identifier(gpus)
    ]
    full_metric_name = "_".join(metric_components)

    # Update metric counter
    metrics[full_metric_name] = metrics.get(full_metric_name, 0) + 1

    set_metrics(metrics)

    # success

    return make_response(bson.json_util.dumps(metrics), 201)


#
# WORKSPACE API
#


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


class CloudPullTypesEnum(Enum):
    """Class defining cloud pull types enum"""

    aws = 'aws'
    azure = 'azure'
    seaweedfs = 'seaweedfs'
    huggingface = 'huggingface'
    self_hosted = 'self_hosted'


class AWSCloudPullSchema(Schema):
    """Class defining AWS Cloud pull schema"""

    access_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    secret_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    endpoint_url = fields.Str(validate=[validate_endpoint_url, validate.Length(max=2048)], allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)


class AzureCloudPullSchema(Schema):
    """Class defining Azure Cloud pull schema"""

    account_name = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    access_key = fields.Str(required=True, validate=validate.Length(min=1, max=2048))
    cloud_region = fields.Str(validate=validate.Length(max=2048), allow_none=True)
    endpoint_url = fields.Str(validate=[validate_endpoint_url, validate.Length(max=2048)], allow_none=True)
    cloud_bucket_name = fields.Str(validate=validate.Length(min=1, max=2048), allow_none=True)


class HuggingFaceCloudPullSchema(Schema):
    """Class defining Hugging Face Cloud pull schema"""

    token = fields.Str(validate=validate.Length(max=2048))


class CloudFileType(Enum):
    """Class defining cloud file types enum"""

    file = "file"
    folder = "folder"


class WorkspaceReqSchema(Schema):
    """Class defining Cloud Workspace request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    version = fields.Str(format="regex", regex=r'^\d+\.\d+\.\d+$', validate=fields.validate.Length(max=10))
    cloud_type = EnumField(CloudPullTypesEnum, allow_none=False)
    cloud_specific_details = fields.Field(allow_none=False)

    @validates_schema
    def validate_cloud_specific_details(self, data, **kwargs):
        """Return schema based on cloud_type and validate credentials"""
        cloud_type = data.get('cloud_type')

        if cloud_type:
            # First, validate the schema structure
            if cloud_type == CloudPullTypesEnum.aws:
                schema = AWSCloudPullSchema()
            elif cloud_type == CloudPullTypesEnum.azure:
                schema = AzureCloudPullSchema()
            elif cloud_type == CloudPullTypesEnum.seaweedfs:
                schema = AWSCloudPullSchema()
            elif cloud_type == CloudPullTypesEnum.huggingface:
                schema = HuggingFaceCloudPullSchema()
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


class DateTimeField(fields.DateTime):
    """Field for handling datetime objects.

    This field is used to handle datetime objects in the API.
    """

    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, datetime):
            return value
        return super()._deserialize(value, attr, data, **kwargs)


class WorkspaceBackupReqSchema(Schema):
    """Class defining workspace backup schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    backup_file_name = fields.Str(validate=validate.Length(max=2048), allow_none=True)


class WorkspaceRspSchema(Schema):
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


class WorkspaceListRspSchema(Schema):
    """Class defining workspace list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    workspaces = fields.List(fields.Nested(WorkspaceRspSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allow_none=True)


class DatasetPathLstSchema(Schema):
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


@app.route('/api/v1/orgs/<org_name>/workspaces', methods=['GET'])
@disk_space_check
def workspace_list(org_name):
    """List workspaces.

    ---
    get:
      tags:
      - WORKSPACE
      summary: List workspaces
      description: Returns the list of workspaces
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Optional name filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: format
        in: query
        description: Optional format filter
        required: false
        schema:
          type: string
          enum: ["monai", "unet", "custom" ]
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: [ "object_detection", "segmentation", "image_classification" ]
      responses:
        200:
          description: Returned list of workspaces
          content:
            application/json:
              schema:
                type: array
                items: WorkspaceRspSchema
                maxItems: 2147483647
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    workspaces = app_handler.list_workspaces(user_id, org_name)
    filtered_workspaces = filtering.apply(request.args, workspaces)
    paginated_workspaces = pagination.apply(request.args, filtered_workspaces)
    metadata = {"workspaces": paginated_workspaces}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_workspaces),
            "total_pages": math.ceil(len(filtered_workspaces) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = WorkspaceListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['GET'])
@disk_space_check
def workspace_retrieve(org_name, workspace_id):
    """Retrieve Workspace.

    ---
    get:
      tags:
      - WORKSPACE
      summary: Retrieve Workspace
      description: Returns the Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.retrieve_workspace(user_id, org_name, workspace_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>:get_datasets', methods=['GET'])
@disk_space_check
def workspace_retrieve_datasets(org_name, workspace_id):
    """Retrieve Datasets from Workspace.

    ---
    get:
      tags:
      - WORKSPACE
      summary: Retrieve datasets from Workspace
      description: Returns the datasets matched with the request body args inside the Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned list of dataset paths within Workspace
          content:
            application/json:
              schema: DatasetPathLstSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    dataset_type = request.args.get("dataset_type", None)
    dataset_format = request.args.get("dataset_format", None)
    dataset_intention = request.args.getlist("dataset_intention")
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.retrieve_cloud_datasets(
        user_id,
        org_name,
        workspace_id,
        dataset_type,
        dataset_format,
        dataset_intention
    )
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetPathLstSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['DELETE'])
@disk_space_check
def workspace_delete(org_name, workspace_id):
    """Delete Workspace.

    ---
    delete:
      tags:
      - WORKSPACE
      summary: Delete Workspace
      description: Cancels all related running jobs and returns the deleted Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Deleted Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_workspace(org_name, workspace_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces', methods=['DELETE'])
@disk_space_check
def bulk_workspace_delete(org_name):
    """Bulk Delete Workspaces.

    ---
    delete:
      tags:
      - WORKSPACE
      summary: Delete multiple Workspaces
      description: Cancels all related running jobs and returns the status of deleted Workspaces
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                workspace_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Deleted Workspaces status
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: One or more Workspaces not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get workspace IDs from request body
    data = request.get_json()
    workspace_ids = data.get('workspace_ids')

    if not workspace_ids or not isinstance(workspace_ids, list):
        metadata = {"error_desc": "Invalid workspace IDs", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for workspace_id in workspace_ids:
        message = validate_uuid(workspace_id=workspace_id)
        if message:
            metadata = {"id": workspace_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the workspace
        response = app_handler.delete_workspace(org_name, workspace_id)
        if response.code == 200:
            results.append({"id": workspace_id, "status": "success"})
        else:
            results.append({"id": workspace_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all workspaces
    schema = BulkOpsRspSchema()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/orgs/<org_name>/workspaces', methods=['POST'])
@disk_space_check
def workspace_create(org_name):
    """Create new Workspace.

    ---
    post:
      tags:
      - WORKSPACE
      summary: Create new Workspace
      description: Returns the new Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema: WorkspaceReqSchema
            examples:
              aws_example:
                summary: Example with AWS cloud details
                value:
                  cloud_details:
                    cloud_type: aws
                    cloud_specific_details:
                      access_key: my_access_key
                      secret_key: my_secret_key
                      cloud_region: us-west-1
                      cloud_bucket_name: my_bucket_name
              azure_example:
                summary: Example with Azure cloud details
                value:
                  cloud_details:
                    cloud_type: azure
                    cloud_specific_details:
                      access_key: my_access_key
                      account_name: my_account_name
                      cloud_bucket_name: my_container_name
              huggingface_example:
                summary: Example with Hugging Face cloud details
                value:
                  cloud_details:
                    cloud_type: huggingface
                    cloud_specific_details:
                      token: my_token
        description: Initial metadata for new Workspace (type and format required)
        required: true
      responses:
        200:
          description: Retuned the new Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = WorkspaceReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = app_handler.create_workspace(user_id, org_name, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['PUT'])
@disk_space_check
def workspace_update(org_name, workspace_id):
    """Update Workspace.

    ---
    put:
      tags:
      - WORKSPACE
      summary: Update Workspace
      description: Returns the updated Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: WorkspaceReqSchema
        description: Updated metadata for Workspace
        required: true
      responses:
        200:
          description: Returned the updated Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = WorkspaceReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.update_workspace(user_id, org_name, workspace_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces/<workspace_id>', methods=['PATCH'])
@disk_space_check
def workspace_partial_update(org_name, workspace_id):
    """Partial update Workspace.

    ---
    patch:
      tags:
      - WORKSPACE
      summary: Partial update Workspace
      description: Returns the updated Workspace
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: workspace_id
        in: path
        description: ID of Workspace to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: WorkspaceReqSchema
        description: Updated metadata for Workspace
        required: true
      responses:
        200:
          description: Returned the updated Workspace
          content:
            application/json:
              schema: WorkspaceRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(workspace_id=workspace_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = WorkspaceRspSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.update_workspace(user_id, org_name, workspace_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = WorkspaceRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/workspaces:backup', methods=['POST'])
@disk_space_check
def workspace_backup(org_name):
    """Backup MongoDB data using workspace metadata.

    ---
    post:
      tags:
      - WORKSPACES
      summary: Backup MongoDB data using workspace metadata
      description: Backs up all MongoDB databases using provided workspace cloud credentials
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                workspace_metadata:
                  type: object
                  description: Workspace metadata containing cloud credentials
                  required: true
                backup_file_name:
                  type: string
                  description: Optional backup file name
              required:
                - workspace_metadata
      responses:
        200:
          description: Backup successful
          content:
            application/json:
              schema: MessageOnlySchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        request_data = request.get_json(force=True)
        workspace_metadata = request_data.get("workspace_metadata")
        backup_file_name = request_data.get("backup_file_name", "mongodb_backup.tar.gz")
        schema = WorkspaceReqSchema()
        workspace_metadata = schema.dump(schema.load(workspace_metadata))

        if not workspace_metadata:
            metadata = {"error_desc": "workspace_metadata is required", "error_code": 1}
            schema = ErrorRspSchema()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
            return response

        # Perform backup
        response = MongoBackupHandler.backup_for_workspace(workspace_metadata, backup_file_name)

        # Return response
        if response.code == 200:
            schema = MessageOnlySchema()
        else:
            schema = ErrorRspSchema()

        schema_dict = schema.dump(schema.load(response.data))
        return make_response(jsonify(schema_dict), response.code)

    except Exception as e:
        metadata = {"error_desc": f"Error in MongoDB backup: {str(e)}", "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response


@app.route('/api/v1/orgs/<org_name>/workspaces:restore', methods=['POST'])
@disk_space_check
def workspace_restore(org_name):
    """Restore MongoDB data using workspace metadata.

    ---
    post:
      tags:
      - WORKSPACES
      summary: Restore MongoDB data using workspace metadata
      description: Restores all MongoDB databases using provided workspace cloud credentials
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                workspace_metadata:
                  type: object
                  description: Workspace metadata containing cloud credentials
                  required: true
                backup_file_name:
                  type: string
                  description: Optional backup file name to restore from
              required:
                - workspace_metadata
      responses:
        200:
          description: Restore successful
          content:
            application/json:
              schema: MessageOnlySchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    try:
        request_data = request.get_json(force=True)
        workspace_metadata = request_data.get("workspace_metadata")
        schema = WorkspaceReqSchema()
        workspace_metadata = schema.dump(schema.load(workspace_metadata))
        backup_file_name = request_data.get("backup_file_name", "mongodb_backup.tar.gz")

        if not workspace_metadata:
            metadata = {"error_desc": "workspace_metadata is required", "error_code": 1}
            schema = ErrorRspSchema()
            response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
            return response

        # Perform restore
        response = MongoBackupHandler.restore_for_workspace(workspace_metadata, backup_file_name)

        # Return response
        if response.code == 200:
            schema = MessageOnlySchema()
        else:
            schema = ErrorRspSchema()

        schema_dict = schema.dump(schema.load(response.data))
        return make_response(jsonify(schema_dict), response.code)

    except Exception as e:
        metadata = {"error_desc": f"Error in MongoDB restore: {str(e)}", "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

#
# DATASET API
#


class DatasetActions(Schema):
    """Class defining dataset actions schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    action = EnumField(ActionEnum)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    specs = fields.Raw()
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class DatasetIntentEnum(Enum):
    """Class defining dataset intent enum"""

    training = 'training'
    evaluation = 'evaluation'
    testing = 'testing'


class LstStrSchema(Schema):
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


class DatasetReqSchema(Schema):
    """Class defining dataset request schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    shared = fields.Bool(allow_none=False)
    user_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36))
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
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
    base_experiment = fields.List(
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


class DatasetJobSchema(Schema):
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
        values=fields.Nested(JobResultSchema),
        validate=validate.Length(max=sys.maxsize)
    )
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    dataset_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class DatasetRspSchema(Schema):
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
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000))
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
        values=fields.Nested(JobSubsetSchema),
        validate=validate.Length(max=sys.maxsize)
    )
    client_url = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_id = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    client_secret = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    filters = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=2048), allow_none=True)
    status = EnumField(PullStatus)
    use_for = fields.List(EnumField(DatasetIntentEnum), allow_none=True, validate=validate.Length(max=3))
    base_experiment_pull_complete = EnumField(PullStatus)
    base_experiment = fields.List(
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
    validation_details = fields.Nested(ValidationDetailsSchema, allow_none=True)


class DatasetListRspSchema(Schema):
    """Class defining dataset list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    datasets = fields.List(fields.Nested(DatasetRspSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


class DatasetJobListSchema(Schema):
    """Class defining dataset list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    jobs = fields.List(fields.Nested(DatasetJobSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


@app.route('/api/v1/orgs/<org_name>/datasets:get_formats', methods=['GET'])
def get_dataset_formats(org_name):
    """Get dataset formats supported.

    ---
    post:
        tags:
        - DATASET
        summary: Given dataset type return dataset formats or return all formats
        description: Given dataset type return dataset formats or return all formats
        parameters:
        - name: org_name
          in: path
          description: Org Name
          required: true
          schema:
            type: string
            maxLength: 255
            pattern: '^[a-zA-Z0-9_-]+$'
        responses:
          200:
            description: Returns a list of dataset formats supported
            content:
              application/json:
                schema: LstStrSchema
            headers:
              Access-Control-Allow-Origin:
                $ref: '#/components/headers/Access-Control-Allow-Origin'
              X-RateLimit-Limit:
                $ref: '#/components/headers/X-RateLimit-Limit'
          404:
            description: Bad request, see reply body for details
            content:
              application/json:
                schema: ErrorRspSchema
            headers:
              Access-Control-Allow-Origin:
                $ref: '#/components/headers/Access-Control-Allow-Origin'
              X-RateLimit-Limit:
                $ref: '#/components/headers/X-RateLimit-Limit'
    """
    dataset_type = str(request.args.get('dataset_type', ''))
    # Get response
    response = app_handler.get_dataset_formats(dataset_type)
    # Get schema
    schema = None
    if response.code == 200:
        schema = LstStrSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets', methods=['GET'])
@disk_space_check
def dataset_list(org_name):
    """List Datasets.

    ---
    get:
      tags:
      - DATASET
      summary: List all accessible datasets
      description: |
        Returns a list of datasets that the authenticated user can access.
        Results can be filtered and paginated using query parameters.
        This includes:
        - Datasets owned by the user
        - Datasets shared with the user
        - Public datasets
      parameters:
      - name: org_name
        in: path
        description: Organization name to list datasets from
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: skip
        in: query
        description: Number of records to skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Maximum number of records to return per page
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Sort order for the results
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Filter datasets by name (case-sensitive partial match)
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: format
        in: query
        description: Filter datasets by their format type
        required: false
        schema:
          type: string
          enum: [
              "kitti", "pascal_voc", "raw", "coco_raw", "unet", "coco", "lprnet", "train", "test",
              "default", "custom", "classification_pyt", "visual_changenet_segment",
              "visual_changenet_classify"
          ]
      - name: type
        in: query
        description: Filter datasets by their primary type
        required: false
        schema:
          type: string
          enum: [
              "object_detection", "segmentation", "image_classification", "character_recognition",
              "action_recognition", "pointpillars", "pose_classification", "ml_recog", "ocdnet", "ocrnet",
              "optical_inspection", "re_identification", "centerpose"
          ]
      responses:
        200:
          description: Successfully retrieved list of accessible datasets
          content:
            application/json:
              schema: DatasetListRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    datasets = app_handler.list_datasets(user_id, org_name)
    filtered_datasets = filtering.apply(request.args, datasets)
    paginated_datasets = pagination.apply(request.args, filtered_datasets)
    metadata = {"datasets": paginated_datasets}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_datasets),
            "total_pages": math.ceil(len(filtered_datasets) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = DatasetListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['GET'])
@disk_space_check
def dataset_retrieve(org_name, dataset_id):
    """Retrieve Dataset.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve details of a specific dataset
      description: |
        Returns detailed information about a specific dataset including:
        - Basic metadata (name, description, creation date)
        - Dataset format and type
        - Access permissions
        - Associated jobs and their status
        - Available actions
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the dataset
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: Unique identifier of the dataset to retrieve
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully retrieved dataset details
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset not found or user lacks access permissions
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_dataset(org_name, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        # Check if this is a dataset validation error that should include structured details
        if (response.code == 404 and
                isinstance(response.data, dict) and
                response.data.get("validation_details")):
            # Use DatasetRspSchema for validation errors to include structured details
            schema = DatasetRspSchema()
        else:
            # Use ErrorRspSchema for other error types
            schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['DELETE'])
@disk_space_check
def dataset_delete(org_name, dataset_id):
    """Delete Dataset.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete a specific dataset
      description: |
        Deletes a dataset and its associated resources. The operation will:
        - Remove dataset files and metadata
        - Update user permissions

        Deletion is only allowed if:
        - User has write permissions
        - Dataset is not public
        - Dataset is not read-only
        - Dataset is not in use by any experiments
        - No running jobs are using the dataset
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the dataset
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: Unique identifier of the dataset to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Dataset successfully deleted
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Dataset cannot be deleted due to active usage or permissions
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset not found or user lacks delete permissions
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_dataset(org_name, dataset_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets', methods=['POST'])
@disk_space_check
def dataset_create(org_name):
    """Create new Dataset.

    ---
    post:
      tags:
      - DATASET
      summary: Create new Dataset
      description: Returns the new Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema: DatasetReqSchema
        description: Initial metadata for new Dataset (type and format required)
        required: true
      responses:
        200:
          description: Returned the new Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = app_handler.create_dataset(user_id, org_name, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 200:
        ds_format = request_dict.get("format", "")
        log_type = (DataMonitorLogTypeEnum.medical_dataset
                    if ds_format == "monai"
                    else DataMonitorLogTypeEnum.tao_dataset)
        log_api_error(user_id, org_name, schema_dict, log_type, action="creation")

    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['PUT'])
@disk_space_check
def dataset_update(org_name, dataset_id):
    """Update Dataset.

    ---
    put:
      tags:
      - DATASET
      summary: Update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetReqSchema
        description: Updated metadata for Dataset
        required: true
      responses:
        200:
          description: Returned the updated Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_dataset(org_name, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>', methods=['PATCH'])
@disk_space_check
def dataset_partial_update(org_name, dataset_id):
    """Partial update Dataset.

    ---
    patch:
      tags:
      - DATASET
      summary: Partial update Dataset
      description: Returns the updated Dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetReqSchema
        description: Updated metadata for Dataset
        required: true
      responses:
        200:
          description: Returned the updated Dataset
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = DatasetReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_dataset(org_name, dataset_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def dataset_specs_schema(org_name, dataset_id, action):
    """Retrieve Specs schema.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Specs schema
      description: Returns the Specs schema for a given action
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [
              "dataset_convert", "convert", "kmeans", "augment", "train",
              "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference",
              "annotation", "analyze", "validate", "auto_label", "calibration_tensorfile"
          ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.get_spec_schema(user_id, org_name, dataset_id, action, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['POST'])
@disk_space_check
def dataset_job_run(org_name, dataset_id):
    """Run Dataset Jobs.

    ---
    post:
      tags:
      - DATASET
      summary: Run Dataset Jobs
      description: |
        Asynchronously starts a dataset action and returns corresponding Job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the requested action is supported
        - Validates the provided specs match the action schema
        - Creates a new job with the provided parameters
        - Queues the job for execution
        - Returns the Job ID for tracking and retrieval
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: DatasetActions
      responses:
        200:
          description: Returned the Job ID corresponding to requested Dataset Action
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = DatasetActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('parent_job_id', None)
    if requested_job:
        requested_job = str(requested_job)
    requested_action = request_schema_data.get('action', "")
    specs = request_schema_data.get('specs', {})
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    platform_id = request_schema_data.get('platform_id', None)
    # Get response
    response = app_handler.job_run(
        org_name, dataset_id, requested_job, requested_action, "dataset",
        specs=specs, name=name, description=description, num_gpu=num_gpu,
        platform_id=platform_id
    )
    handler_metadata = resolve_metadata("dataset", dataset_id)
    dataset_format = handler_metadata.get("format")
    # Get schema
    if response.code == 200:
        # MONAI dataset jobs are sync jobs and the response should be returned directly.
        if dataset_format == "monai":
            return make_response(jsonify(response.data), response.code)
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:retry', methods=['POST'])
@disk_space_check
def dataset_job_retry(org_name, dataset_id, job_id):
    """Retry Dataset Jobs.

    ---
    post:
      tags:
      - DATASET
      summary: Retry Dataset Jobs
      description: |
        Asynchronously retries a dataset action and returns corresponding Job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists and is retryable
        - Creates a new job with the same parameters as the original job
        - Queues the job for execution
        - Returns the new Job ID for tracking and retrieval
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Job ID corresponding to requested Dataset Action
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_retry(org_name, dataset_id, "dataset", job_id)
    handler_metadata = resolve_metadata("dataset", dataset_id)
    dataset_format = handler_metadata.get("format")
    # Get schema
    if response.code == 200:
        # MONAI dataset jobs are sync jobs and the response should be returned directly.
        if dataset_format == "monai":
            return make_response(jsonify(response.data), response.code)
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['GET'])
@disk_space_check
def dataset_job_list(org_name, dataset_id):
    """List Jobs for Dataset.

    ---
    get:
      tags:
      - DATASET
      summary: List Jobs for Dataset
      description: |
        Returns the list of Jobs for a given dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Retrieves the list of jobs from storage
        - Applies pagination and filtering based on query parameters
        - Returns the filtered and paginated list of jobs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          pattern: '.*'
          maxLength: 36
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending" ]
      responses:
        200:
          description: Returned list of Jobs
          content:
            application/json:
              schema: DatasetJobListSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Dataset not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=None if dataset_id in ("*", "all") else dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = app_handler.job_list(user_id, org_name, dataset_id, "dataset")
    # Get schema
    if response.code == 200:
        filtered_jobs = filtering.apply(request.args, response.data)
        paginated_jobs = pagination.apply(request.args, filtered_jobs)
        metadata = {"jobs": paginated_jobs}
        # Pagination
        skip = request.args.get("skip", None)
        size = request.args.get("size", None)
        if skip is not None and size is not None:
            skip = int(skip)
            size = int(size)
            metadata["pagination_info"] = {
                "total_records": len(filtered_jobs),
                "total_pages": math.ceil(len(filtered_jobs) / size),
                "page_size": size,
                "page_index": skip // size,
            }
        schema = DatasetJobListSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))))
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>/schema', methods=['GET'])
@disk_space_check
def dataset_job_schema(org_name, dataset_id, job_id):
    """Retrieve Schema for a job.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Schema for a job
      description: |
        Returns the Specs schema for a given job. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the schema for the job's action
        - Returns the schema
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for JOB
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.get_spec_schema_for_job(user_id, org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def dataset_job_retrieve(org_name, dataset_id, job_id):
    """Retrieve Job for Dataset.

    ---
    get:
      tags:
      - DATASET
      summary: Retrieve Job for Dataset
      description: |
        Returns the Job for a given dataset and job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the job from storage
        - Returns the job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job
          content:
            application/json:
              schema: DatasetJobSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    return_specs = ast.literal_eval(request.args.get('return_specs', "False"))
    response = app_handler.job_retrieve(org_name, dataset_id, job_id, "dataset", return_specs=return_specs)
    # Get schema
    schema = None
    if response.code == 200:
        schema = DatasetJobSchema()
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:status_update', methods=['POST'])
@disk_space_check
def dataset_job_status_update(org_name, dataset_id, job_id):
    """Update Job status for Dataset.

    ---
    post:
      tags:
      - DATASET
      summary: Update status of a dataset job
      description: |
        Updates the status of a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Updates the job status based on provided data
        - Persists status changes to storage
        - Triggers any necessary status-based workflows
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              description: Status update data including new status and any additional metadata
      responses:
        200:
          description: Job status successfully updated
          content:
            application/json:
              schema: DatasetJobSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid status update request (e.g. invalid status value, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_status_update(org_name, dataset_id, job_id, "dataset", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:log_update', methods=['POST'])
@disk_space_check
def dataset_job_log_update(org_name, dataset_id, job_id):
    """Update Job log for Dataset.

    ---
    post:
      tags:
      - DATASET
      summary: Update log of a dataset job
      description: |
        Updates the log of a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Appends the provided log data to the job's log
        - Persists log changes to storage
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              description: Log update data including new log entries
      responses:
        200:
          description: Job log successfully updated
          content:
            application/json:
              schema: DatasetJobSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid log update request (e.g. missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_log_update(org_name, dataset_id, job_id, "dataset", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>/logs', methods=['GET'])
def dataset_job_logs(org_name, dataset_id, job_id):
    """Get realtime dataset job logs.

    ---
    get:
      tags:
      - DATASET
      summary: Get Job logs for Dataset
      description: |
        Returns the job logs for a given dataset and job ID. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the job logs from storage
        - Returns the job logs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Logs
          content:
            text/plain:
              example: "Execution status: PASS"
          headers:
            Access-Control-Allow-Origin:
               $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job not exist or logs not found.
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
               $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_job_logs(org_name, dataset_id, job_id, "dataset")
    if response.code == 200:
        response = make_response(response.data, 200)
        response.mimetype = 'text/plain'
        return response
    # Handle errors
    schema = ErrorRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(response.data))), 400)
    return response


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def dataset_job_cancel(org_name, dataset_id, job_id):
    """Cancel Dataset Job.

    ---
    post:
      tags:
      - DATASET
      summary: Cancel Dataset Job
      description: |
        Cancels a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists and is cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested cancelation of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_cancel(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets', methods=['DELETE'])
@disk_space_check
def bulk_dataset_delete(org_name):
    """Bulk Delete Datasets.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete multiple Datasets
      description: |
        Deletes multiple datasets and their associated resources. This endpoint:
        - Validates the datasets exist and user has access
        - Validates the datasets are not public
        - Validates the datasets are not read-only
        - Validates the datasets are not in use by any experiments
        - Validates no running jobs are using the datasets
        - Deletes the dataset files and metadata
        - Updates user permissions
        - Returns the status for each dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                dataset_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Deleted Datasets status
          content:
            application/json:
              schema: DatasetRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset IDs)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: One or more Datasets not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get dataset IDs from request body
    data = request.get_json()
    dataset_ids = data.get('dataset_ids')

    if not dataset_ids or not isinstance(dataset_ids, list):
        metadata = {"error_desc": "Invalid dataset IDs", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for dataset_id in dataset_ids:
        message = validate_uuid(dataset_id=dataset_id)
        if message:
            metadata = {"id": dataset_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the dataset
        response = app_handler.delete_dataset(org_name, dataset_id)
        if response.code == 200:
            results.append({"id": dataset_id, "status": "success"})
        else:
            results.append({"id": dataset_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all datasets
    schema = BulkOpsRspSchema()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs', methods=['DELETE'])
@disk_space_check
def bulk_dataset_job_delete(org_name, dataset_id):
    """Bulk Delete Dataset Jobs.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete multiple Dataset Jobs
      description: |
        Deletes multiple jobs within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the jobs exist and are deletable
        - Deletes the job files and metadata
        - Updates job status to 'deleted'
        - Returns the status for each job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                job_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Successfully requested deletion of specified Job IDs
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job IDs)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Jobs not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get job IDs from request body
    data = request.get_json()
    job_ids = data.get('job_ids')

    if not job_ids or not isinstance(job_ids, list):
        metadata = {"error_desc": "Invalid job IDs", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for job_id in job_ids:
        message = validate_uuid(job_id=job_id)
        if message:
            metadata = {"id": job_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the job
        response = app_handler.job_delete(org_name, dataset_id, job_id, "dataset")
        if response.code == 200:
            results.append({"id": job_id, "status": "success"})
        else:
            results.append({"id": job_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all jobs
    schema = BulkOpsRspSchema()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def dataset_job_delete(org_name, dataset_id, job_id):
    """Delete Dataset Job.

    ---
    delete:
      tags:
      - DATASET
      summary: Delete Dataset Job
      description: |
        Deletes a specific job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists and is deletable
        - Deletes the job files and metadata
        - Updates job status to 'deleted'
        - Returns the deletion status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested deletion of specified Job ID
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_delete(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def dataset_job_files_list(org_name, dataset_id, job_id):
    """List Job Files.

    ---
    get:
      tags:
      - DATASET
      summary: List Job Files
          description: |
        Lists the files produced by a given job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Retrieves the list of files from storage
        - Returns the list of files
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Files
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
                  maxLength: 1000
                  maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_list_files(org_name, dataset_id, job_id, "dataset")
    # Get schema
    if response.code == 200:
        if isinstance(response.data, list) and (all(isinstance(f, str) for f in response.data) or response.data == []):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:download_selective_files', methods=['GET'])
@disk_space_check
def dataset_job_download_selective_files(org_name, dataset_id, job_id):
    """Download selective Job Artifacts.

    ---
    get:
      tags:
      - DATASET
      summary: Download selective Job Artifacts
      description: |
        Downloads selective artifacts produced by a given job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Validates the requested files exist
        - Downloads the requested files
        - Returns the downloaded files as a tarball
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 5000
                maxLength: 5000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID, file list)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_lists = request.args.getlist('file_lists')
    tar_files = ast.literal_eval(request.args.get('tar_files', "True"))
    if not file_lists:
        return make_response(jsonify("No files passed in list format to download or"), 400)
    # Get response
    response = app_handler.job_download(
        org_name,
        dataset_id,
        job_id,
        "dataset",
        file_lists=file_lists,
        tar_files=tar_files
    )
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def dataset_job_download(org_name, dataset_id, job_id):
    """Download Job Artifacts.

    ---
    get:
      tags:
      - DATASET
      summary: Download Job Artifacts
      description: |
        Downloads all artifacts produced by a given job within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the job exists
        - Downloads all job files
        - Returns the downloaded files as a tarball
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID of Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_download(org_name, dataset_id, job_id, "dataset")
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/datasets:cancel_all_jobs', methods=['POST'])
@disk_space_check
def bulk_dataset_jobs_cancel(org_name):
    """Cancel all jobs within multiple datasets.

    ---
    post:
      tags:
      - DATASET
      summary: Cancel all Jobs under multiple datasets
      description: |
        Cancels all jobs within multiple datasets. This endpoint:
        - Validates the datasets exist and user has access
        - Validates the jobs exist and are cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status for each dataset
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                dataset_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Successfully canceled all jobs under the specified datasets
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset IDs)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Datasets or Jobs not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get dataset IDs from the request body
    data = request.get_json()
    dataset_ids = data.get('dataset_ids')

    if not dataset_ids or not isinstance(dataset_ids, list):
        metadata = {"error_desc": "Invalid dataset IDs", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    for dataset_id in dataset_ids:
        message = validate_uuid(dataset_id=dataset_id)
        if message:
            results.append({"id": dataset_id, "error_desc": message, "error_code": 1})
            continue

        # Cancel all jobs for each dataset
        response = app_handler.all_job_cancel(user_id, org_name, dataset_id, "dataset")
        if response.code == 200:
            results.append({"id": dataset_id, "status": "success"})
        else:
            results.append({"id": dataset_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for each dataset's job cancellation
    schema = BulkOpsRspSchema()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/orgs/<org_name>/datasets/<dataset_id>:cancel_all_jobs', methods=['POST'])
@disk_space_check
def dataset_jobs_cancel(org_name, dataset_id):
    """Cancel all jobs within dataset (or pause training).

    ---
    post:
      tags:
      - DATASET
      summary: Cancel all Jobs under dataset
      description: |
        Cancels all jobs within a dataset. This endpoint:
        - Validates the dataset exists and user has access
        - Validates the jobs exist and are cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: dataset_id
        in: path
        description: ID for Dataset
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully canceled all jobs under datasets
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid dataset ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Dataset or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(dataset_id=dataset_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.all_job_cancel(user_id, org_name, dataset_id, "dataset")
    # Get schema
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


#
# EXPERIMENT API
#
class LstIntSchema(Schema):
    """Class defining dataset actions schema"""

    data = fields.List(
        fields.Int(
            format="int64",
            validate=validate.Range(min=0, max=sys.maxsize),
            allow_none=True
        ),
        validate=validate.Length(max=sys.maxsize)
    )


class ExperimentActions(Schema):
    """Class defining experiment actions schema"""

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


class JobResumeSchema(Schema):
    """Class defining job resume request schema"""

    parent_job_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    specs = fields.Raw(allow_none=True)


class CheckpointChooseMethodEnum(Enum):
    """Class defining enum for methods of picking a trained checkpoint"""

    latest_model = 'latest_model'
    best_model = 'best_model'
    from_epoch_number = 'from_epoch_number'


class ExperimentTypeEnum(Enum):
    """Class defining type of experiment"""

    vision = 'vision'
    medical = 'medical'
    maxine = 'maxine'


class ExperimentExportTypeEnum(Enum):
    """Class defining model export type"""

    tao = 'tao'
    monai_bundle = 'monai_bundle'


class AutoMLAlgorithm(Enum):
    """Class defining automl algorithm enum"""

    bayesian = "bayesian"
    hyperband = "hyperband"


class AutoMLSchema(Schema):
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


class BaseExperimentMetadataSchema(Schema):
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


class ExperimentReqSchema(Schema):
    """Class defining experiment request schema"""

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
        validate=fields.validate.Length(max=1000)
    )  # Model version description - not changing variable name for backward compatability
    model_description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000)
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
    base_experiment = fields.List(
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
    automl_settings = fields.Nested(AutoMLSchema, allow_none=True)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    type = EnumField(ExperimentTypeEnum, default=ExperimentTypeEnum.vision)
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


class ExperimentJobSchema(Schema):
    """Class defining experiment job schema"""

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
        keys=fields.Str(
            format="uuid",
            validate=fields.validate.Length(max=36)
        ),
        values=fields.Nested(JobResultSchema),
        validate=validate.Length(max=sys.maxsize)
    )
    sync = fields.Bool()
    specs = fields.Raw(allow_none=True)
    name = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    description = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=1000), allow_none=True)
    num_gpu = fields.Int(format="int64", validate=validate.Range(min=0, max=sys.maxsize), allow_none=True)
    platform_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)
    experiment_id = fields.Str(format="uuid", validate=fields.validate.Length(max=36), allow_none=True)


class SourceType(Enum):
    """Class defining source type enum for base experiments"""

    ngc = "ngc"
    huggingface = "huggingface"


class ExperimentRspSchema(Schema):
    """Class defining experiment response schema"""

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
        validate=fields.validate.Length(max=1000)
    )  # Model version description - not changing variable name for backward compatability
    model_description = fields.Str(
        format="regex",
        regex=r'.*',
        validate=fields.validate.Length(max=1000)
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
    base_experiment = fields.List(
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
        values=fields.Nested(JobSubsetSchema),
        validate=validate.Length(max=sys.maxsize)
    )
    status = EnumField(JobStatusEnum)
    all_jobs_cancel_status = EnumField(JobStatusEnum, allow_none=True)
    automl_settings = fields.Nested(AutoMLSchema)
    metric = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    type = EnumField(ExperimentTypeEnum, default=ExperimentTypeEnum.vision, allow_none=True)
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
    base_experiment_metadata = fields.Nested(BaseExperimentMetadataSchema, allow_none=True)
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


class ExperimentTagListSchema(Schema):
    """Class defining experiment tags list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
    tags = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=36)
        ),
        validate=validate.Length(max=16)
    )


class ExperimentListRspSchema(Schema):
    """Class defining experiment list response schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    experiments = fields.List(
        fields.Nested(ExperimentRspSchema),
        validate=validate.Length(max=sys.maxsize)
    )
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


class ExperimentJobListSchema(Schema):
    """Class defining experiment job list schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    jobs = fields.List(fields.Nested(ExperimentJobSchema), validate=validate.Length(max=sys.maxsize))
    pagination_info = fields.Nested(PaginationInfoSchema, allowed_none=True)


class ExperimentDownloadSchema(Schema):
    """Class defining experiment artifacts download schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    export_type = EnumField(ExperimentExportTypeEnum)


class LoadAirgappedExperimentsReqSchema(Schema):
    """Class defining load airgapped experiments request schema"""

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


class LoadAirgappedExperimentsRspSchema(Schema):
    """Class defining load airgapped experiments response schema"""

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


@app.route('/api/v1/orgs/<org_name>/experiments', methods=['GET'])
@disk_space_check
def experiment_list(org_name):
    """List Experiments.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Experiments
      description: Returns the list of Experiments
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Optional name filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: ["vision", "medical"]
      - name: network_arch
        in: query
        description: Optional network architecture filter
        required: false
        schema:
          type: string
          enum: [
              "action_recognition",
              "classification_pyt",
              "mal",
              "ml_recog",
              "ocdnet",
              "ocrnet",
              "optical_inspection",
              "pointpillars",
              "pose_classification",
              "re_identification",
              "deformable_detr",
              "dino",
              "segformer",
              "visual_changenet_classify",
              "visual_changenet_segment",
              "centerpose"
          ]
      - name: read_only
        in: query
        description: Optional read_only filter
        required: false
        allowEmptyValue: true
        schema:
          type: boolean
      - name: user_only
        in: query
        description: Optional filter to select user owned experiments only
        required: false
        schema:
          type: boolean
      - name: tag
        in: query
        description: Optional tag filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      responses:
        200:
          description: Returned the list of Experiments
          content:
            application/json:
              schema: ExperimentListRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_only = str(request.args.get('user_only', None)) in {'True', 'yes', 'y', 'true', 't', '1', 'on'}
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    experiments = app_handler.list_experiments(user_id, org_name, user_only)
    filtered_experiments = filtering.apply(request.args, experiments)
    paginated_experiments = pagination.apply(request.args, filtered_experiments)
    metadata = {"experiments": paginated_experiments}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_experiments),
            "total_pages": math.ceil(len(filtered_experiments) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = ExperimentListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/experiments:get_tags', methods=['GET'])
@disk_space_check
def experiment_tags_list(org_name):
    """Retrieve All Unique Experiment Tags.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve all unique experiment tags
      description: Returns all unique experiment tags
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      responses:
        200:
          description: Returned the unique experiment tags list
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    experiments = app_handler.list_experiments(user_id, org_name, user_only=True)
    tags = [tag for exp in experiments for tag in exp.get('tags', [])]
    unique_tags = list({t.lower(): t for t in tags}.values())
    metadata = {"tags": unique_tags}
    schema = ExperimentTagListSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/experiments:base', methods=['GET'])
@disk_space_check
def base_experiment_list(org_name):
    """List Base Experiments.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Experiments that can be used for transfer learning
      description: Returns the list of models published in NGC public catalog and private org's model registry
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending", "name-descending", "name-ascending" ]
      - name: name
        in: query
        description: Optional name filter
        required: false
        schema:
          type: string
          maxLength: 5000
          pattern: '.*'
      - name: type
        in: query
        description: Optional type filter
        required: false
        schema:
          type: string
          enum: ["vision", "medical"]
      - name: network_arch
        in: query
        description: Optional network architecture filter
        required: false
        schema:
          type: string
          enum: [
              "action_recognition",
              "classification_pyt",
              "mal",
              "ml_recog",
              "ocdnet",
              "ocrnet",
              "optical_inspection",
              "pointpillars",
              "pose_classification",
              "re_identification",
              "deformable_detr",
              "dino",
              "segformer",
              "visual_changenet_classify",
              "visual_changenet_segment",
              "centerpose"
          ]
      - name: read_only
        in: query
        description: Optional read_only filter
        required: false
        allowEmptyValue: true
        schema:
          type: boolean
      responses:
        200:
          description: Returned the list of Experiments
          content:
            application/json:
              schema: ExperimentListRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    experiments = app_handler.list_base_experiments(user_id, org_name)
    filtered_experiments = filtering.apply(request.args, experiments)
    paginated_experiments = pagination.apply(request.args, filtered_experiments)
    metadata = {"experiments": paginated_experiments}
    # Pagination
    skip = request.args.get("skip", None)
    size = request.args.get("size", None)
    if skip is not None and size is not None:
        skip = int(skip)
        size = int(size)
        metadata["pagination_info"] = {
            "total_records": len(filtered_experiments),
            "total_pages": math.ceil(len(filtered_experiments) / size),
            "page_size": size,
            "page_index": skip // size,
        }
    schema = ExperimentListRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(metadata))))
    return response


@app.route('/api/v1/orgs/<org_name>/experiments:load_airgapped', methods=['POST'])
@disk_space_check
def load_airgapped_experiments(org_name):
    """Load Airgapped Experiments.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Load base experiments from airgapped cloud storage
      description: Loads base experiment metadata from airgapped cloud storage using workspace credentials
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        required: true
        content:
          application/json:
            schema: LoadAirgappedExperimentsReqSchema
      responses:
        200:
          description: Successfully loaded airgapped experiments
          content:
            application/json:
              schema: LoadAirgappedExperimentsRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request
          content:
            application/json:
              schema: ErrorRspSchema
        404:
          description: Workspace not found
          content:
            application/json:
              schema: ErrorRspSchema
        500:
          description: Internal server error
          content:
            application/json:
              schema: ErrorRspSchema
    """
    message = validate_uuid(workspace_id=request.get_json().get('workspace_id'))
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    schema = LoadAirgappedExperimentsReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))

    # Authenticate user
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    # Get response from handler
    response = app_handler.load_airgapped_experiments(
        user_id,
        org_name,
        request_dict['workspace_id']
    )

    # Get appropriate schema based on response code
    schema = None
    if response.code == 200:
        schema = LoadAirgappedExperimentsRspSchema()
    else:
        schema = ErrorRspSchema()

    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['GET'])
@disk_space_check
def experiment_retrieve(org_name, experiment_id):
    """Retrieve Experiment.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Experiment
      description: Returns the Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to return
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.retrieve_experiment(org_name, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments', methods=['DELETE'])
@disk_space_check
def bulk_experiment_delete(org_name):
    """Bulk Delete Experiments.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete multiple Experiments
      description: Cancels all related running jobs and returns the status of deleted Experiments
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                experiment_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Deleted Experiments status
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: One or more Experiments not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get experiment IDs from request body
    data = request.get_json()
    experiment_ids = data.get('experiment_ids')

    if not experiment_ids or not isinstance(experiment_ids, list):
        metadata = {"error_desc": "Invalid experiment IDs", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for experiment_id in experiment_ids:
        message = validate_uuid(experiment_id=experiment_id)
        if message:
            metadata = {"id": experiment_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the experiment
        response = app_handler.delete_experiment(org_name, experiment_id)
        if response.code == 200:
            results.append({"id": experiment_id, "status": "success"})
        else:
            results.append({"id": experiment_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all experiments
    schema = BulkOpsRspSchema()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['DELETE'])
@disk_space_check
def experiment_delete(org_name, experiment_id):
    """Delete Experiment.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment
      description: Cancels all related running jobs and returns the deleted Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to delete
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the deleted Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.delete_experiment(org_name, experiment_id)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments', methods=['POST'])
@disk_space_check
def experiment_create(org_name):
    """Create new Experiment.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Create new Experiment
      description: Returns the new Experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema: ExperimentReqSchema
        description: Initial metadata for new Experiment (base_experiment or network_arch required)
        required: true
      responses:
        200:
          description: Returned the new Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Bad request, see reply body for details
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    # Get response
    response = app_handler.create_experiment(user_id, org_name, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 200:
        mdl_nw = request_dict.get("network_arch", None)
        is_medical = isinstance(mdl_nw, str) and mdl_nw.startswith("monai_")
        log_type = DataMonitorLogTypeEnum.medical_experiment if is_medical else DataMonitorLogTypeEnum.tao_experiment
        log_api_error(user_id, org_name, schema_dict, log_type, action="creation")

    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['PUT'])
@disk_space_check
def experiment_update(org_name, experiment_id):
    """Update Experiment.

    ---
    put:
      tags:
      - EXPERIMENT
      summary: Update Experiment
      description: |
        Updates an existing experiment with new metadata. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the provided metadata matches the schema
        - Updates the experiment metadata in storage
        - Returns the updated experiment metadata
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentReqSchema
        description: Updated metadata for Experiment
        required: true
      responses:
        200:
          description: Returned the updated Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_experiment(org_name, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>', methods=['PATCH'])
@disk_space_check
def experiment_partial_update(org_name, experiment_id):
    """Partial update Experiment.

    ---
    patch:
      tags:
      - EXPERIMENT
      summary: Partial update Experiment
      description: |
        Partially updates an existing experiment with new metadata. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the provided metadata matches the schema
        - Updates the experiment metadata in storage
        - Returns the updated experiment metadata
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentReqSchema
        description: Updated metadata for Experiment
        required: true
      responses:
        200:
          description: Returned the updated Experiment
          content:
            application/json:
              schema: ExperimentRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    schema = ExperimentReqSchema()
    request_dict = schema.dump(schema.load(request.get_json(force=True)))
    # Get response
    response = app_handler.update_experiment(org_name, experiment_id, request_dict)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentRspSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def specs_schema_without_handler_id(org_name, action):
    """Retrieve Specs schema.

    ---
    get:
      summary: Retrieve Specs schema without experiment or dataset id
      description: |
        Returns the Specs schema for a given action. This endpoint:
        - Validates the action is supported
        - Retrieves the schema for the action
        - Returns the schema
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [
            "dataset_convert", "convert", "kmeans", "augment", "train",
            "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference",
            "annotation", "analyze", "validate", "auto_label", "calibration_tensorfile"
          ]
      responses:
        200:
          description: Returned the Specs schema for given action and network
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get response
    network = request.args.get('network')
    dataset_format = request.args.get('format')
    train_datasets = request.args.getlist('train_datasets')

    response = app_handler.get_spec_schema_without_handler_id(org_name, network, dataset_format, action, train_datasets)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/specs/<action>/schema', methods=['GET'])
@disk_space_check
def experiment_specs_schema(org_name, experiment_id, action):
    """Retrieve Specs schema.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Specs schema
      description: |
        Returns the Specs schema for a given action and experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the action is supported
        - Retrieves the schema for the action
        - Returns the schema
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [
            "dataset_convert", "convert", "kmeans", "augment", "train",
            "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec", "inference",
            "annotation", "analyze", "validate", "auto_label", "calibration_tensorfile"
          ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.get_spec_schema(user_id, org_name, experiment_id, action, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/specs/<action>/schema:base', methods=['GET'])
@disk_space_check
def base_experiment_specs_schema(org_name, experiment_id, action):
    """Retrieve Base Experiment Specs schema.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Base Experiment Specs schema
      description: |
        Returns the Specs schema for a given action of the base experiment. This endpoint:
        - Validates the base experiment exists and user has access
        - Validates the action is supported
        - Retrieves the schema for the action
        - Returns the schema
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Base Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: action
        in: path
        description: Action name
        required: true
        schema:
          type: string
          enum: [
            "train", "evaluate", "prune", "retrain", "export", "gen_trt_engine", "trtexec",
            "inference", "auto_label"
          ]
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Action not found or Base spec file not present
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
               $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_base_experiment_spec_schema(experiment_id, action)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['POST'])
@disk_space_check
def experiment_job_run(org_name, experiment_id):
    """Run Experiment Jobs.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Run Experiment Jobs
      description: |
        Asynchronously starts a Experiment Action and returns corresponding Job ID. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the requested action is supported
        - Validates the provided specs match the action schema
        - Creates a new job with the provided parameters
        - Queues the job for execution
        - Returns the Job ID for tracking and retrieval
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: ExperimentActions
      responses:
        200:
          description: Returned the Job ID corresponding to requested Experiment Action
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = ExperimentActions()
    request_schema_data = schema.dump(schema.load(request_data))
    requested_job = request_schema_data.get('parent_job_id', None)
    if requested_job:
        requested_job = str(requested_job)
    requested_action = request_schema_data.get('action', None)
    if not requested_action:
        metadata = {"error_desc": "Action is required to run job", "error_code": 400}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    specs = request_schema_data.get('specs', {})
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    platform_id = request_schema_data.get('platform_id', None)
    if isinstance(specs, dict) and "cluster" in specs:
        metadata = {"error_desc": "cluster is an invalid spec", "error_code": 3}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_run(
        org_name, experiment_id, requested_job, requested_action, "experiment",
        specs=specs, name=name, description=description, num_gpu=num_gpu,
        platform_id=platform_id
    )
    # Get schema
    schema = None
    if response.code == 200:
        if hasattr(response, "attachment_key") and response.attachment_key:
            try:
                output_path = response.data[response.attachment_key]
                all_files = [
                    os.path.join(dirpath, f)
                    for dirpath, dirnames, filenames in os.walk(output_path)
                    for f in filenames
                ]
                files_dict = {}
                for f in all_files:
                    with open(f, "rb") as file:
                        files_dict[os.path.relpath(f, output_path)] = file.read()
                multipart_data = MultipartEncoder(fields=files_dict)
                send_file_response = make_response(multipart_data.to_string())
                send_file_response.headers["Content-Type"] = multipart_data.content_type
                # send_file sets correct response code as 200, should convert back to 200
                if send_file_response.status_code == 200:
                    send_file_response.status_code = response.code
                    # remove sent file as it's useless now
                    shutil.rmtree(response.data[response.attachment_key], ignore_errors=True)
                return send_file_response
            except Exception as e:
                # get user_id for more information
                handler_metadata = resolve_metadata("experiment", experiment_id)
                user_id = handler_metadata.get("user_id")
                logger.error(
                    f"respond attached data for org: {org_name} experiment: {experiment_id} "
                    f"user: {user_id} failed, got error: {e}"
                )
                metadata = {"error_desc": "respond attached data failed", "error_code": 2}
                schema = ErrorRspSchema()
                response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
                return response
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 200:
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            is_medical = handler_metadata.get("type").lower() == "medical"
            user_id = handler_metadata.get("user_id", None)
            if user_id:
                log_type = DataMonitorLogTypeEnum.medical_job if is_medical else DataMonitorLogTypeEnum.tao_job
                log_api_error(user_id, org_name, schema_dict, log_type, action="creation")
        except Exception as e:
            logger.error(f"Exception thrown in experiment_job_run is {str(e)}")
            log_monitor(DataMonitorLogTypeEnum.api, "Cannot parse experiment info for job.")

    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:retry', methods=['POST'])
@disk_space_check
def experiment_job_retry(org_name, experiment_id, job_id):
    """Retry Experiment Job.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Retry Experiment Jobs
      description: |
        Asynchronously retries a Experiment Action and returns corresponding Job ID. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is retryable
        - Creates a new job with the same parameters as the original job
        - Queues the job for execution
        - Returns the new Job ID for tracking and retrieval
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Job ID corresponding to requested Experiment Action
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    # Get response
    response = app_handler.job_retry(org_name, experiment_id, "experiment", job_id)
    # Get schema
    schema = None
    if response.code == 200:
        if isinstance(response.data, str) and not validate_uuid(response.data):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: invalid job IDs", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    if response.code != 200:
        try:
            handler_metadata = resolve_metadata("experiment", experiment_id)
            is_medical = handler_metadata.get("type").lower() == "medical"
            user_id = handler_metadata.get("user_id", None)
            if user_id:
                log_type = DataMonitorLogTypeEnum.medical_job if is_medical else DataMonitorLogTypeEnum.tao_job
                log_api_error(user_id, org_name, schema_dict, log_type, action="creation")
        except Exception as e:
            logger.error(f"Exception thrown in experiment_job_retry is {str(e)}")
            log_monitor(DataMonitorLogTypeEnum.api, "Cannot parse experiment info for job.")

    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:publish_model', methods=['POST'])
@disk_space_check
def experiment_model_publish(org_name, experiment_id, job_id):
    """Publish models to NGC.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Publish models to NGC
      description: |
        Publishes models to NGC private registry. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is publishable
        - Validates the provided metadata matches the schema
        - Publishes the model to NGC with the provided metadata
        - Returns a success message
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: PublishModel
      responses:
        200:
          description: String message for successful upload
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        schema = MessageOnlySchema()
        message = "Cannot publish model in air-gapped mode."
        response = make_response(jsonify(schema.dump({"message": message})), 400)
        return response

    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    display_name = request_schema_data.get('display_name', '')
    description = request_schema_data.get('description', '')
    team_name = request_schema_data.get('team_name', '')
    # Get response
    response = app_handler.publish_model(
        org_name,
        team_name,
        experiment_id,
        job_id,
        display_name=display_name,
        description=description
    )
    # Get schema
    schema_dict = None

    if response.code == 200:
        schema = MessageOnlySchema()
        logger.info("Returning success response: %s", response.data)
        schema_dict = schema.dump({"message": "Published model into requested org"})
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:get_epoch_numbers', methods=['GET'])
@disk_space_check
def experiment_job_get_epoch_numbers(org_name, experiment_id, job_id):
    """Get the epoch numbers for the checkpoints present for this job.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Get epoch numbers present for this job
      description: |
        Retrieves the epoch numbers for the checkpoints present for this job. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves the list of epoch numbers from storage
        - Returns the list of epoch numbers
      parameters:
      - name: org_name
        in: path
        description: Organization name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: List of epoch numbers
        content:
          application/json:
              schema: LstIntSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.job_get_epoch_numbers(user_id, org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema_dict = None
    if response.code == 200:
        schema = LstIntSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route(
    '/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:remove_published_model',
    methods=['DELETE']
)
@disk_space_check
def experiment_remove_published_model(org_name, experiment_id, job_id):
    """Remove published models from NGC.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Remove publish models from NGC
      description: |
        Removes models from NGC private registry. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is publishable
        - Validates the provided metadata matches the schema
        - Removes the model from NGC
        - Returns a success message
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: PublishModel
      responses:
        200:
          description: String message for successfull deletion
          content:
            application/json:
              schema:
                type: string
                format: uuid
                maxLength: 36
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    if os.getenv("AIRGAPPED_MODE", "false").lower() == "true":
        schema = MessageOnlySchema()
        message = "Cannot remove published model in air-gapped mode."
        response = make_response(jsonify(schema.dump({"message": message})), 400)
        return response

    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.args.to_dict()
    schema = PublishModel()
    request_schema_data = schema.dump(schema.load(request_data))
    team_name = request_schema_data.get('team_name', '')
    # Get response
    response = app_handler.remove_published_model(org_name, team_name, experiment_id, job_id)
    # Get schema
    schema_dict = None

    if response.code == 200:
        schema = MessageOnlySchema()
        logger.info("Returning success response")
        schema_dict = schema.dump({"message": "Removed model"})
    else:
        schema = ErrorRspSchema()
        # Load metadata in schema and return
        schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/schema', methods=['GET'])
@disk_space_check
def experiment_job_schema(org_name, experiment_id, job_id):
    """Retrieve Schema for a job.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Schema for a job
      description: |
        Returns the Specs schema for a given job. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves the schema for the job's action
        - Returns the schema
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for JOB
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned the Specs schema for given action
          content:
            application/json:
              schema:
                type: object
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Dataset or Action not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.get_spec_schema_for_job(user_id, org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify(response.data), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['GET'])
@disk_space_check
def experiment_job_list(org_name, experiment_id):
    """List Jobs for Experiment.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Jobs for Experiment
      description: |
        Returns the list of Jobs for a given experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Retrieves the list of jobs from storage
        - Applies pagination and filtering based on query parameters
        - Returns the filtered and paginated list of jobs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          pattern: '.*'
          maxLength: 36
      - name: skip
        in: query
        description: Optional skip for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: size
        in: query
        description: Optional size for pagination
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      - name: sort
        in: query
        description: Optional sort
        required: false
        schema:
          type: string
          enum: ["date-descending", "date-ascending" ]
      responses:
        200:
          description: Returned list of Jobs
          content:
            application/json:
              schema: ExperimentJobListSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User or Experiment not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=None if experiment_id in ("*", "all") else experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    # Get response
    response = app_handler.job_list(user_id, org_name, experiment_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        filtered_jobs = filtering.apply(request.args, response.data)
        paginated_jobs = pagination.apply(request.args, filtered_jobs)
        metadata = {"jobs": paginated_jobs}
        # Pagination
        skip = request.args.get("skip", None)
        size = request.args.get("size", None)
        if skip is not None and size is not None:
            skip = int(skip)
            size = int(size)
            metadata["pagination_info"] = {
                "total_records": len(filtered_jobs),
                "total_pages": math.ceil(len(filtered_jobs) / size),
                "page_size": size,
                "page_index": skip // size,
            }
        schema = ExperimentJobListSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))))
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>', methods=['GET'])
@disk_space_check
def experiment_job_retrieve(org_name, experiment_id, job_id):
    """Retrieve Job for Experiment.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve Job for Experiment
      description: |
        Returns the Job for a given experiment and job ID. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves the job from storage
        - Returns the job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job
          content:
            application/json:
              schema: ExperimentJobSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    return_specs = ast.literal_eval(request.args.get('return_specs', "False"))
    response = app_handler.job_retrieve(org_name, experiment_id, job_id, "experiment", return_specs=return_specs)
    # Get schema
    schema = None
    if response.code == 200:
        schema = ExperimentJobSchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/logs', methods=['GET'])
def experiment_job_logs(org_name, experiment_id, job_id):
    """Get realtime job logs. AutoML train job will return current recommendation's experiment log.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Get Job logs for Experiment
      description: |
        Returns the job logs for a given experiment and job ID. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves the job logs from storage
        - Returns the job logs
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: automl_experiment_index
        in: query
        description: Optional filter to retrieve logs from specific autoML experiment
        required: false
        schema:
          type: integer
          format: int32
          minimum: 0
          maximum: 2147483647
      responses:
        200:
          description: Returned Job Logs
          content:
            text/plain:
              example: "Execution status: PASS"
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Job not exist or logs not found.
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.get_job_logs(
        org_name,
        experiment_id,
        job_id,
        "experiment",
        request.args.get('automl_experiment_index', None)
    )
    if response.code == 200:
        response = make_response(response.data, 200)
        response.mimetype = 'text/plain'
        return response
    # Handle errors
    schema = ErrorRspSchema()
    response = make_response(jsonify(schema.dump(schema.load(response.data))), 400)
    return response


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:automl_details', methods=['GET'])
@disk_space_check
def experiment_job_automl_details(org_name, experiment_id, job_id):
    """Retrieve AutoML details.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Retrieve usable AutoML details
      description: |
        Retrieves usable AutoML details for a given experiment and job ID. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves the AutoML details from storage
        - Returns the AutoML details
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
                maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    response = app_handler.automl_details(org_name, experiment_id, job_id)
    # Get schema
    schema = AutoMLResultsDetailedSchema()
    if response.code == 200:
        if isinstance(response.data, dict) or response.data == []:
            response = make_response(jsonify(schema.dump(schema.load(response.data))), response.code)
            return response
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:status_update', methods=['POST'])
@disk_space_check
def experiment_job_status_update(org_name, experiment_id, job_id):
    """Update Job status for Experiment.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Update status of an experiment job
      description: |
        Updates the status of a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Updates the job status based on provided data
        - Persists status changes to storage
        - Triggers any necessary status-based workflows
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the experiment
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Unique identifier of the experiment containing the job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Unique identifier of the job to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              description: Status update data including new status and any additional metadata
      responses:
        200:
          description: Job status successfully updated
          content:
            application/json:
              schema: ExperimentJobSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid status update request (e.g. invalid status value, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Experiment or job not found, or user lacks permission to update
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_status_update(org_name, experiment_id, job_id, "experiment", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:log_update', methods=['POST'])
@disk_space_check
def experiment_job_log_update(org_name, experiment_id, job_id):
    """Update Job log for Experiment.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Update log of an experiment job
      description: |
        Updates the log of a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Updates the job log based on provided data
      parameters:
      - name: org_name
        in: path
        description: Organization name owning the experiment
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Unique identifier of the experiment containing the job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Unique identifier of the job to update
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              description: Log update data including
      responses:
        200:
          description: Job logs successfully updated
          content:
            application/json:
              schema: ExperimentJobSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid log update request (e.g. invalid log value, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: Experiment or job not found, or user lacks permission to update
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    callback_data = request.json
    # Get response
    response = app_handler.job_log_update(org_name, experiment_id, job_id, "experiment", callback_data=callback_data)
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments:cancel_all_jobs', methods=['POST'])
@disk_space_check
def bulk_experiment_jobs_cancel(org_name):
    """Cancel all jobs within multiple experiments.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel all Jobs under multiple experiments
      description: |
        Cancels all jobs within multiple experiments. This endpoint:
        - Validates the experiments exist and user has access
        - Validates the jobs exist and are cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status for each experiment
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                experiment_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Successfully canceled all jobs under the specified experiments
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment IDs)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiments or Jobs not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get experiment IDs from the request body
    data = request.get_json()
    experiment_ids = data.get('experiment_ids')

    if not experiment_ids or not isinstance(experiment_ids, list):
        metadata = {"error_desc": "Invalid experiment IDs", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)

    for experiment_id in experiment_ids:
        message = validate_uuid(experiment_id=experiment_id)
        if message:
            results.append({"id": experiment_id, "error_desc": message, "error_code": 1})
            continue

        # Cancel all jobs for each experiment
        response = app_handler.all_job_cancel(user_id, org_name, experiment_id, "experiment")
        if response.code == 200:
            results.append({"id": experiment_id, "status": "success"})
        else:
            results.append({"id": experiment_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for each experiment's job cancellation
    schema = BulkOpsRspSchema()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>:cancel_all_jobs', methods=['POST'])
@disk_space_check
def experiment_jobs_cancel(org_name, experiment_id):
    """Cancel all jobs within experiment (or pause training).

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel all Jobs under experiment
      description: |
        Cancels all jobs within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the jobs exist and are cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully canceled all jobs under experiments
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
    response = app_handler.all_job_cancel(user_id, org_name, experiment_id, "experiment")
    # Get schema
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:pause', methods=['POST'])
@disk_space_check
def experiment_job_pause(org_name, experiment_id, job_id):
    """Pause Experiment Job (only for training).

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Pause Experiment Job - only for training
      description: |
        Pauses a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is pausable
        - Updates the job status to 'paused'
        - Persists status changes to storage
        - Triggers any necessary pause workflows
        - Returns the pause status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested training pause of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_pause(org_name, experiment_id, job_id, "experiment")
    # Get schema
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:cancel', methods=['POST'])
@disk_space_check
def experiment_job_cancel(org_name, experiment_id, job_id):
    """Cancel Experiment Job (or pause training).

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Cancel Experiment Job or pause training
      description: |
        Cancels a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is cancellable
        - Updates the job status to 'cancelled'
        - Persists status changes to storage
        - Triggers any necessary cancellation workflows
        - Returns the cancellation status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested cancelation or training pause of specified Job ID
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_cancel(org_name, experiment_id, job_id, "experiment")
    # Get schema
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs', methods=['DELETE'])
@disk_space_check
def bulk_experiment_job_delete(org_name, experiment_id):
    """Bulk Delete Experiment Jobs.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete multiple Experiment Jobs
      description: |
        Deletes multiple jobs within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the jobs exist and are deletable
        - Deletes the job files and metadata
        - Updates job status to 'deleted'
        - Returns the deletion status for each job
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                job_ids:
                  type: array
                  items:
                    type: string
                    format: uuid
                    maxLength: 36
                  maxItems: 2147483647
      responses:
        200:
          description: Successfully requested deletion of specified Job IDs
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job IDs)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment, or Jobs not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    # Get job IDs from request body
    data = request.get_json()
    job_ids = data.get('job_ids')

    if not job_ids or not isinstance(job_ids, list):
        metadata = {"error_desc": "Invalid job IDs", "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

    results = []
    for job_id in job_ids:
        message = validate_uuid(job_id=job_id)
        if message:
            metadata = {"id": job_id, "error_desc": message, "error_code": 1}
            results.append(metadata)
            continue

        # Attempt to delete the job
        response = app_handler.job_delete(org_name, experiment_id, job_id, "experiment")
        if response.code == 200:
            results.append({"id": job_id, "status": "success"})
        else:
            results.append({"id": job_id, "status": "failed", "error_desc": response.data.get("error_desc", "")})

    # Return status for all jobs
    schema = BulkOpsRspSchema()
    schema_dict = schema.dump(schema.load({"results": results}))
    return make_response(jsonify(schema_dict), 200)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>', methods=['DELETE'])
@disk_space_check
def experiment_job_delete(org_name, experiment_id, job_id):
    """Delete Experiment Job.

    ---
    delete:
      tags:
      - EXPERIMENT
      summary: Delete Experiment Job
      description: |
        Deletes a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is deletable
        - Deletes the job files and metadata
        - Updates job status to 'deleted'
        - Returns the deletion status
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Successfully requested deletion of specified Job ID
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_delete(org_name, experiment_id, job_id, "experiment")
    # Get schema
    schema = None
    if response.code == 200:
        return make_response(jsonify({}), response.code)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:resume', methods=['POST'])
@disk_space_check
def experiment_job_resume(org_name, experiment_id, job_id):
    """Resume Experiment Job - train/retrain only.

    ---
    post:
      tags:
      - EXPERIMENT
      summary: Resume Experiment Job
      description: |
        Resumes a specific job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists and is resumable
        - Validates the provided metadata matches the schema
        - Updates the job metadata
        - Queues the job for execution
        - Returns the new Job ID for tracking and retrieval
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID for Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: ID for Job
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema: JobResumeSchema
        description: Adjustable metadata for the resumed job.
        required: false
      responses:
        200:
          description: Successfully requested resume of specified Job ID (asynchronous)
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID, missing required fields)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True).copy()
    schema = JobResumeSchema()
    request_schema_data = schema.dump(schema.load(request_data))
    parent_job_id = request_schema_data.get('parent_job_id', None)
    name = request_schema_data.get('name', '')
    description = request_schema_data.get('description', '')
    num_gpu = request_schema_data.get('num_gpu', -1)
    platform_id = request_schema_data.get('platform_id', None)
    if parent_job_id:
        parent_job_id = str(parent_job_id)
    specs = request_schema_data.get('specs', {})
    # Get response
    response = app_handler.resume_experiment_job(
        org_name,
        experiment_id,
        job_id,
        "experiment",
        parent_job_id,
        specs=specs,
        name=name,
        description=description,
        num_gpu=num_gpu,
        platform_id=platform_id
    )
    # Get schema
    if response.code == 200:
        schema = MessageOnlySchema()
    else:
        schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:download', methods=['GET'])
@disk_space_check
def experiment_job_download(org_name, experiment_id, job_id):
    """Download Job Artifacts.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download Job Artifacts
      description: |
        Downloads the artifacts produced by a given job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Validates the requested export type is supported
        - Downloads the job artifacts
        - Returns the downloaded artifacts as a tarball
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
                maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    request_data = request.get_json(force=True, silent=True)
    request_data = {} if request_data is None else request_data
    try:
        request_schema_data = ExperimentDownloadSchema().load(request_data)
    except exceptions.ValidationError as err:
        metadata = {"error_desc": str(err)}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 404)
        return response
    export_type = request_schema_data.get("export_type", ExperimentExportTypeEnum.tao)
    # Get response
    response = app_handler.job_download(org_name, experiment_id, job_id, "experiment", export_type=export_type.name)
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:list_files', methods=['GET'])
@disk_space_check
def experiment_job_files_list(org_name, experiment_id, job_id):
    """List Job Files.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: List Job Files
          description: |
        Lists the files produced by a given job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Retrieves the list of files from storage
        - Returns the list of files
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Files
          content:
            application/json:
              schema:
                type: array
                items:
                  type: string
                  maxLength: 1000
                  maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    # Get response
    response = app_handler.job_list_files(org_name, experiment_id, job_id, "experiment")
    # Get schema
    if response.code == 200:
        if isinstance(response.data, list) and (all(isinstance(f, str) for f in response.data) or response.data == []):
            return make_response(jsonify(response.data), response.code)
        metadata = {"error_desc": "internal error: file list invalid", "error_code": 2}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 500)
        return response
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


@app.route(
    '/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>:download_selective_files',
    methods=['GET']
)
@disk_space_check
def experiment_job_download_selective_files(org_name, experiment_id, job_id):
    """Download selective Job Artifacts.

    ---
    get:
      tags:
      - EXPERIMENT
      summary: Download selective Job Artifacts
      description: |
        Downloads selective artifacts produced by a given job within an experiment. This endpoint:
        - Validates the experiment exists and user has access
        - Validates the job exists
        - Validates the requested files exist
        - Downloads the requested files
        - Returns the downloaded files as a tarball
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: ID of Experiment
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Returned Job Artifacts
          content:
            application/octet-stream:
              schema:
                type: string
                format: binary
                maxLength: 1000
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        400:
          description: Invalid request (e.g. invalid experiment ID, job ID, file list)
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
        404:
          description: User, Experiment or Job not found
          content:
            application/json:
              schema: ErrorRspSchema
          headers:
            Access-Control-Allow-Origin:
              $ref: '#/components/headers/Access-Control-Allow-Origin'
            X-RateLimit-Limit:
              $ref: '#/components/headers/X-RateLimit-Limit'
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response
    file_lists = request.args.getlist('file_lists')
    best_model = ast.literal_eval(request.args.get('best_model', "False"))
    latest_model = ast.literal_eval(request.args.get('latest_model', "False"))
    tar_files = ast.literal_eval(request.args.get('tar_files', "True"))
    if not (file_lists or best_model or latest_model):
        return make_response(
            jsonify("No files passed in list format to download or, best_model or latest_model is not enabled"),
            400
        )
    # Get response
    response = app_handler.job_download(
        org_name,
        experiment_id,
        job_id,
        "experiment",
        file_lists=file_lists,
        best_model=best_model,
        latest_model=latest_model,
        tar_files=tar_files
    )
    # Get schema
    schema = None
    if response.code == 200:
        file_path = response.data  # Response is assumed to have the file path
        file_dir = "/".join(file_path.split("/")[:-1])
        file_name = file_path.split("/")[-1]  # infer the name
        return send_from_directory(file_dir, file_name, as_attachment=True)
    schema = ErrorRspSchema()
    # Load metadata in schema and return
    schema_dict = schema.dump(schema.load(response.data))
    return make_response(jsonify(schema_dict), response.code)


# Inference Microservice Endpoints

@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/inference_microservice/start', methods=['POST'])
@disk_space_check
def inference_microservice_start(org_name, experiment_id):
    """Start a new Inference Microservice and return job_id.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Start Inference Microservice
      description: |
        Creates a new Inference Microservice job and starts the StatefulSet. Returns job_id for subsequent operations.
        - Creates a new unique job_id
        - Starts Inference Microservice StatefulSet microservice
        - Returns job_id for file uploads and inference requests
      parameters:
        - name: org_name
          in: path
          required: true
          description: Name of the organization
          schema:
            type: string
        - name: experiment_id
          in: path
          required: true
          description: Experiment ID
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                model_path:
                  type: string
                  description: Path to the model
                  example: "/workspace/model"
                docker_image:
                  type: string
                  description: Docker image for inference
                  example: "nvcr.io/nvidia/vila-inference:latest"
                gpu_type:
                  type: string
                  description: GPU type required
                  example: "H100"
                num_gpus:
                  type: integer
                  description: Number of GPUs required
                  example: 1
              required:
                - model_path
      responses:
        200:
          description: Inference Microservice started successfully
          content:
            application/json:
              schema:
                type: object
                properties:
                  job_id:
                    type: string
                    description: Unique job ID for this Inference Microservice
                  status:
                    type: string
                    description: Service status
                  message:
                    type: string
                    description: Success message
        400:
          description: Bad request
        500:
          description: Internal server error
    """
    try:
        request_data = request.get_json(force=True)

        # Generate unique job_id
        job_id = str(uuid.uuid4())

        # Create job configuration
        response = InferenceMicroserviceHandler.start_inference_microservice(
            org_name, experiment_id, job_id, request_data
        )

        return make_response(jsonify(response.data), response.code)

    except Exception as err:
        logger.error("Error in inference_microservice_start: %s", str(traceback.format_exc()))
        return make_response(jsonify({
            'error': str(err),
            'error_code': 1
        }), 500)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/inference_microservice/inference',
           methods=['POST'])
@disk_space_check
def inference_microservice_inference(org_name, experiment_id, job_id):
    """Make an inference request to a running Inference Microservice.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Make Inference Microservice inference request
      description: |
        Sends inference request to a running Inference Microservice. This endpoint:
        - Validates the Inference Microservice is running
        - Processes prompts and images for Inference Microservice inference
        - Returns generated content from the Inference Microservice model
        - Supports both single and batch inference requests
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Experiment ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Inference Microservice Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                input:
                  type: array
                  items:
                    type: string
                  description: Base64-encoded images/videos with data URI format (data:image/jpeg;base64,...)
                model:
                  type: string
                  description: Model identifier (e.g. nvidia/nvdino-v2)
                prompt:
                  type: string
                  description: Text prompt for Inference Microservice inference
                  default: ""
              required: [input, model]
      responses:
        200:
          description: Inference completed successfully
        400:
          description: Invalid request parameters
        404:
          description: Inference Microservice not found
        500:
          description: Inference request failed
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    try:
        request_data = request.get_json()

        if not request_data:
            metadata = {"error_desc": "Input data is required", "error_code": 1}
            schema = ErrorRspSchema()
            return make_response(jsonify(schema.dump(schema.load(metadata))), 400)

        # Process Inference Microservice inference via direct StatefulSet call
        result = InferenceMicroserviceHandler.process_inference_microservice_request_direct(job_id, request_data)

        return make_response(jsonify(result), 200 if result.get("status") != "error" else 500)

    except Exception as e:
        logger.error("Error processing Inference Microservice inference request: %s", str(e))
        metadata = {"error_desc": str(e), "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/inference_microservice/status',
           methods=['GET'])
@disk_space_check
def inference_microservice_status(org_name, experiment_id, job_id):  # noqa: D214
    """Get status of a Inference Microservice.

    ---
    get:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Get Inference Microservice status
      description: |
        Returns the status of a Inference Microservice.
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Experiment ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Inference Microservice Job ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Service status retrieved successfully
        404:
          description: Inference Microservice not found
        500:
          description: Failed to get service status
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    try:
        # Get Inference Microservice service status directly
        result = InferenceMicroserviceHandler.get_inference_microservice_status_direct(job_id)
        return make_response(jsonify(result), 200 if result.get("status") != "error" else 500)

    except Exception as e:
        logger.error("Error getting Inference Microservice status: %s", str(e))
        metadata = {"error_desc": str(e), "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)


@app.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>/jobs/<job_id>/inference_microservice/stop',
           methods=['POST'])
@disk_space_check
def stop_inference_microservice(org_name, experiment_id, job_id):  # noqa: D214
    """Stop a Inference Microservice.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Stop Inference Microservice
      description: |
        Stops a running Inference Microservice.
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
      - name: experiment_id
        in: path
        description: Experiment ID
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      - name: job_id
        in: path
        description: Inference Microservice Job ID to stop
        required: true
        schema:
          type: string
          format: uuid
          maxLength: 36
      responses:
        200:
          description: Inference Microservice stopped successfully
        400:
          description: Invalid request parameters
        404:
          description: Inference Microservice not found
        500:
          description: Failed to stop service
    """
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        metadata = {"error_desc": message, "error_code": 1}
        schema = ErrorRspSchema()
        response = make_response(jsonify(schema.dump(schema.load(metadata))), 400)
        return response

    try:
        # Stop the Inference Microservice
        result = InferenceMicroserviceHandler.stop_inference_microservice(job_id)

        # Update job status if stopped successfully
        if result.code == 200:
            from nvidia_tao_core.microservices.handlers.stateless_handlers import update_job_status
            update_job_status(
                experiment_id,
                job_id,
                status="Done",
                kind="experiments"
            )

        return make_response(jsonify(result.data), result.code)

    except Exception as e:
        logger.error("Error stopping Inference Microservice: %s", str(e))
        metadata = {"error_desc": str(e), "error_code": 1}
        schema = ErrorRspSchema()
        return make_response(jsonify(schema.dump(schema.load(metadata))), 500)


#
# HEALTH API
#
@app.route('/api/v1/health', methods=['GET'])
def api_health():
    """api health endpoint"""
    return make_response(jsonify(['liveness', 'readiness']))


@app.route('/api/v1/health/liveness', methods=['GET'])
@disk_space_check
def liveness():
    """api liveness endpoint"""
    try:
        live_state = health_check.check_logging()
        if live_state:
            return make_response(jsonify("OK"), 200)
    except Exception as e:
        logger.error("Exception thrown in liveness: %s", str(e))
        logger.error("liveness error: %s", traceback.format_exc())
    return make_response(jsonify("Error"), 400)


@app.route('/api/v1/health/readiness', methods=['GET'])
@disk_space_check
def readiness():
    """api readiness endpoint"""
    try:
        if health_check.check_logging():
            ready_state = True
            if os.getenv("BACKEND"):
                if not (health_check.check_k8s() and Workflow.healthy()):
                    ready_state = False
            if ready_state:
                return make_response(jsonify("OK"), 200)
    except Exception as e:
        logger.error("Exception thrown in readiness: %s", str(e))
        logger.error("readiness error: %s", traceback.format_exc())
    return make_response(jsonify("Error"), 400)


#
# BASIC API
#
@app.route('/', methods=['GET'])
@disk_space_check
def root():
    """api root endpoint"""
    return make_response(jsonify([
        'api',
        'openapi.yaml',
        'openapi.json',
        'rapipdf',
        'redoc',
        'swagger',
        'version',
        'tao_api_notebooks.zip'
    ]))


@app.route('/api', methods=['GET'])
def version_list():
    """version list endpoint"""
    return make_response(jsonify(['v1']))


@app.route('/api/v1', methods=['GET'])
def version_v1():
    """version endpoint"""
    return make_response(jsonify(['login', 'user', 'auth', 'health']))


@app.route('/api/v1/orgs', methods=['GET'])
def user_list():
    """user list endpoint"""
    error = {"error_desc": "Listing orgs is not authorized: Missing Org Name", "error_code": 1}
    schema = ErrorRspSchema()
    return make_response(jsonify(schema.dump(schema.load(error))), 403)


@app.route('/api/v1/orgs/<org_name>', methods=['GET'])
@disk_space_check
def user(org_name):
    """user endpoint"""
    return make_response(jsonify(['dataset', 'experiment']))


@app.route('/openapi.yaml', methods=['GET'])
def openapi_yaml():
    """openapi_yaml endpoint"""
    r = make_response(spec.to_yaml())
    r.mimetype = 'text/x-yaml'
    return r


@app.route('/openapi.json', methods=['GET'])
def openapi_json():
    """openapi_json endpoint"""
    r = make_response(jsonify(spec.to_dict()))
    r.mimetype = 'application/json'
    return r


@app.route('/rapipdf', methods=['GET'])
def rapipdf():
    """rapipdf endpoint"""
    return render_template('rapipdf.html')


@app.route('/redoc', methods=['GET'])
def redoc():
    """redoc endpoint"""
    return render_template('redoc.html')


@app.route('/swagger', methods=['GET'])
def swagger():
    """swagger endpoint"""
    return render_template('swagger.html')


@app.route('/version', methods=['GET'])
def version():
    """version endpoint"""
    git_branch = os.environ.get('GIT_BRANCH', 'unknown')
    git_commit_sha = os.environ.get('GIT_COMMIT_SHA', 'unknown')
    git_commit_time = os.environ.get('GIT_COMMIT_TIME', 'unknown')
    version = {'version': tao_version, 'branch': git_branch, 'sha': git_commit_sha, 'time': git_commit_time}
    r = make_response(jsonify(version))
    r.mimetype = 'application/json'
    return r


@app.route('/tao_api_notebooks.zip', methods=['GET'])
@disk_space_check
def download_folder():
    """Download notebooks endpoint"""
    # Create a temporary zip file containing the folder
    shutil.make_archive("/tmp/tao_api_notebooks", 'zip', "/shared/notebooks/")

    # Send the zip file for download
    return send_file(
        "/tmp/tao_api_notebooks.zip",
        as_attachment=True,
        download_name="tao_api_notebooks.zip"
    )


#
# End of APIs
#

with app.test_request_context():
    spec.path(view=super_endpoint)
    spec.path(view=login)
    spec.path(view=org_gpu_types)
    spec.path(view=workspace_list)
    spec.path(view=workspace_retrieve)
    spec.path(view=workspace_retrieve_datasets)
    spec.path(view=workspace_delete)
    spec.path(view=bulk_workspace_delete)
    spec.path(view=workspace_create)
    spec.path(view=workspace_update)
    spec.path(view=workspace_partial_update)
    spec.path(view=workspace_backup)
    spec.path(view=workspace_restore)
    spec.path(view=get_dataset_formats)
    spec.path(view=dataset_list)
    spec.path(view=dataset_retrieve)
    spec.path(view=dataset_delete)
    spec.path(view=bulk_dataset_delete)
    spec.path(view=dataset_create)
    spec.path(view=dataset_update)
    spec.path(view=dataset_partial_update)
    spec.path(view=dataset_specs_schema)
    spec.path(view=dataset_job_run)
    spec.path(view=dataset_job_retry)
    spec.path(view=dataset_job_list)
    spec.path(view=dataset_job_retrieve)
    spec.path(view=dataset_job_schema)
    spec.path(view=dataset_job_logs)
    spec.path(view=dataset_job_cancel)
    spec.path(view=dataset_jobs_cancel)
    spec.path(view=bulk_dataset_jobs_cancel)
    spec.path(view=bulk_dataset_job_delete)
    spec.path(view=dataset_job_delete)
    spec.path(view=dataset_job_download)
    spec.path(view=experiment_list)
    spec.path(view=base_experiment_list)
    spec.path(view=experiment_retrieve)
    spec.path(view=experiment_delete)
    spec.path(view=bulk_experiment_delete)
    spec.path(view=experiment_create)
    spec.path(view=experiment_update)
    spec.path(view=experiment_partial_update)
    spec.path(view=experiment_specs_schema)
    spec.path(view=base_experiment_specs_schema)
    spec.path(view=experiment_job_run)
    spec.path(view=experiment_job_retry)
    spec.path(view=experiment_job_get_epoch_numbers)
    spec.path(view=experiment_model_publish)
    spec.path(view=experiment_remove_published_model)
    spec.path(view=experiment_job_schema)
    spec.path(view=experiment_job_list)
    spec.path(view=experiment_job_retrieve)
    spec.path(view=experiment_job_logs)
    spec.path(view=experiment_job_automl_details)
    spec.path(view=experiment_job_pause)
    spec.path(view=experiment_jobs_cancel)
    spec.path(view=bulk_experiment_jobs_cancel)
    spec.path(view=experiment_job_cancel)
    spec.path(view=bulk_experiment_job_delete)
    spec.path(view=experiment_job_delete)
    spec.path(view=experiment_job_resume)
    spec.path(view=experiment_job_download)
    spec.path(view=inference_microservice_start)
    spec.path(view=inference_microservice_inference)
    spec.path(view=inference_microservice_status)
    spec.path(view=stop_inference_microservice)


def main():
    """Main function"""
    if os.getenv("DEV_MODE", "False").lower() in ("true", "1"):
        app.run(host="0.0.0.0", port=8008)
    else:
        app.run()


if __name__ == '__main__':
    main()
