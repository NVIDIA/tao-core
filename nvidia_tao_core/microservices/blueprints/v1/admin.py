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

"""Admin blueprint for v1 API - documentation, metrics, and utility endpoints."""

import os
import bson
import shutil
import logging
import requests
from flask import Blueprint, jsonify, make_response, render_template, send_file, request
from flask import current_app

from nvidia_tao_core.microservices.decorators import disk_space_check
from .schemas import ErrorRsp, NVCFReq, TelemetryReq
from nvidia_tao_core.microservices.config import get_tao_version
from nvidia_tao_core.microservices.utils.stateless_handler_utils import set_metrics, get_metrics, get_root
from nvidia_tao_core.microservices.utils.core_utils import safe_load_file
from nvidia_tao_core.telemetry.processor import MetricProcessor

TIMEOUT = 240
logger = logging.getLogger(__name__)

# v1 Admin Blueprint - URL prefix will be set during registration (/api/v1)
admin_bp_v1 = Blueprint('admin_v1', __name__, template_folder='templates')

tao_version = get_tao_version()


@admin_bp_v1.route('/', methods=['GET'])
def version_v1():
    """api v1 root endpoint"""
    return make_response(jsonify([
        'login',
        'auth',
        'health',
        'openapi.yaml',
        'openapi.json',
        'rapipdf',
        'redoc',
        'swagger',
        'version',
        'tao_api_notebooks.zip',
        'orgs',
        'metrics'
    ]))


@admin_bp_v1.route('/orgs', methods=['GET'])
def user_list():
    """user list endpoint"""
    error = {"error_desc": "Listing orgs is not authorized: Missing Org Name", "error_code": 1}
    schema = ErrorRsp()
    return make_response(jsonify(schema.dump(schema.load(error))), 403)


@admin_bp_v1.route('/orgs/<org_name>', methods=['GET'])
@disk_space_check
def user(org_name):
    """user endpoint"""
    return make_response(jsonify(['dataset', 'experiment']))


@admin_bp_v1.route('/openapi.yaml', methods=['GET'])
def openapi_yaml():
    """openapi_yaml endpoint"""
    spec = current_app.spec_v1
    r = make_response(spec.to_yaml())
    r.mimetype = 'text/x-yaml'
    return r


@admin_bp_v1.route('/openapi.json', methods=['GET'])
def openapi_json():
    """openapi_json endpoint"""
    spec = current_app.spec_v1
    r = make_response(jsonify(spec.to_dict()))
    r.mimetype = 'application/json'
    return r


@admin_bp_v1.route('/rapipdf', methods=['GET'])
def rapipdf():
    """rapipdf endpoint"""
    return render_template('v1/rapipdf.html')


@admin_bp_v1.route('/redoc', methods=['GET'])
def redoc():
    """redoc endpoint"""
    return render_template('v1/redoc.html')


@admin_bp_v1.route('/swagger', methods=['GET'])
def swagger():
    """swagger endpoint"""
    logger.info(f"Received swagger request for v1 API, root_path={admin_bp_v1.root_path}")
    return render_template('v1/swagger.html')


@admin_bp_v1.route('/version', methods=['GET'])
def version():
    """version endpoint"""
    git_branch = os.environ.get('GIT_BRANCH', 'unknown')
    git_commit_sha = os.environ.get('GIT_COMMIT_SHA', 'unknown')
    git_commit_time = os.environ.get('GIT_COMMIT_TIME', 'unknown')
    version = {'version': tao_version, 'branch': git_branch, 'sha': git_commit_sha, 'time': git_commit_time}
    r = make_response(jsonify(version))
    r.mimetype = 'application/json'
    return r


@admin_bp_v1.route('/tao_api_notebooks.zip', methods=['GET'])
@disk_space_check
def download_folder():
    """Download notebooks endpoint"""
    # Create a temporary zip file containing the folder
    shutil.make_archive("/tmp/tao_api_notebooks", 'zip', "/shared/notebooks/")

    # Return the zip file as a downloadable attachment
    return send_file("/tmp/tao_api_notebooks.zip", as_attachment=True,
                     download_name="tao_api_notebooks.zip", mimetype='application/zip')


@admin_bp_v1.route('/metrics', methods=['POST'])
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
                    schema: TelemetryReq
                    description: Report new action, network and gpu list
                    required: true
        responses:
            201:
                description: Sucessfully reported execution of new action
    """
    # Validate and load telemetry data
    try:
        raw_data = TelemetryReq().load(request.get_json(force=True))
    except Exception as e:
        logger.error("Exception thrown in metrics_upsert: %s", str(e))
        return make_response(jsonify({}), 400)

    # Load existing metrics
    metrics = get_metrics()
    if not metrics:
        metrics = safe_load_file(os.path.join(get_root(), 'metrics.json'))
        if not metrics:
            # Warning: No historical metrics data found, starting with new record.
            logger.warning("No existing metrics history found; starting new metrics record.")
            metrics = {}  # Start a new, empty metrics dict

    # Process metrics using the extensible MetricProcessor
    # This orchestrator handles all metric building using configured builders
    processor = MetricProcessor()
    metrics = processor.process(metrics, raw_data)

    # Persist metrics
    set_metrics(metrics)

    return make_response(bson.json_util.dumps(metrics), 201)


@admin_bp_v1.route('/orgs/<org_name>/super_endpoint', methods=['POST'])
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
        schema = NVCFReq()
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
            schema = ErrorRsp()
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
