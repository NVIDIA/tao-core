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

"""Inference Microservice blueprint for API v2 inference management endpoints."""

import logging
import traceback
import uuid
from flask import Blueprint, request, jsonify, make_response

from nvidia_tao_core.microservices.decorators import disk_space_check
from nvidia_tao_core.microservices.utils.auth_utils import authentication
from nvidia_tao_core.microservices.handlers.experiment_handler import ExperimentHandler
from nvidia_tao_core.microservices.handlers.inference_microservice_handler import InferenceMicroserviceHandler
from nvidia_tao_core.microservices.utils.handler_utils import validate_uuid
from .schemas import (
    ErrorRsp,
    InferenceReq,
    InferenceMicroserviceReq,
    InferenceMicroserviceRsp
)

logger = logging.getLogger(__name__)

# v2 Inference Microservice Blueprint - URL prefix will be set during registration
inference_microservices_bp_v2 = Blueprint('inference_microservices_v2', __name__, template_folder='templates')


@inference_microservices_bp_v2.route('/orgs/<org_name>/inference_microservices:start', methods=['POST'])
@disk_space_check
def inference_microservice_start(org_name):
    """Start a new Inference Microservice and return job_id.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Start Inference Microservice
      description: Creates a new Inference Microservice job and starts the StatefulSet
      parameters:
        - name: org_name
          in: path
          required: true
          description: Name of the organization
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema: InferenceMicroserviceReq
      responses:
        201:
          description: Inference Microservice started successfully
          content:
            application/json:
              schema: InferenceMicroserviceRsp
        400:
          description: Bad request
          content:
            application/json:
              schema: ErrorRsp
        500:
          description: Internal server error
          content:
            application/json:
              schema: ErrorRsp
    """
    try:
        request_data = request.get_json(force=True)

        # Generate unique job_id
        experiment_id = job_id = str(uuid.uuid4())

        # Validate request with InferenceMicroserviceReq schema
        schema = InferenceMicroserviceReq()
        validated_data = schema.load(request_data)
        # Create experiment request dict with only the fields needed
        network_arch = validated_data.get("network_arch")
        experiment_request = {
            "network_arch": network_arch.value if hasattr(network_arch, 'value') else network_arch,
        }
        if validated_data.get("workspace"):
            experiment_request["workspace"] = validated_data.get("workspace")
        user_id = authentication.get_user_id(request.headers.get('Authorization', ''), org_name)
        experiment_response = ExperimentHandler.create_experiment(
            user_id, org_name, experiment_request, experiment_id=experiment_id)
        if experiment_response.code != 200:
            schema = ErrorRsp()
            metadata = {"error_desc": 'Failed to start Inference Microservice', "error_code": 1}
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 500)

        # Pass validated data to inference microservice handler
        response = InferenceMicroserviceHandler.start_inference_microservice(
            org_name, experiment_id, job_id, validated_data
        )

        if response.code == 200:
            metadata = {
                'job_id': job_id,
                'status': 'starting',
                'message': f'Inference Microservice started with job_id: {job_id}'
            }
            schema = InferenceMicroserviceRsp()
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 201)
        schema = ErrorRsp()
        metadata = {"error_desc": response.data['error_desc'], "error_code": response.data['error_code']}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)

    except Exception as err:
        logger.error("Error in inference_microservice_start: %s", str(traceback.format_exc()))
        schema = ErrorRsp()
        metadata = {"error_desc": str(err), "error_code": 1}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)


@inference_microservices_bp_v2.route('/orgs/<org_name>/inference_microservices/<job_id>:inference', methods=['POST'])
@disk_space_check
def inference_microservice_inference(org_name, job_id):
    """Make an inference request to a running Inference Microservice.

    ---
    post:
      tags:
      - INFERENCE_MICROSERVICE
      summary: Make Inference Microservice inference request
      description: Sends inference request to a running Inference Microservice
      parameters:
      - name: org_name
        in: path
        description: Org Name
        required: true
        schema:
          type: string
          maxLength: 255
          pattern: '^[a-zA-Z0-9_-]+$'
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
            schema: InferenceReq
      responses:
        200:
          description: Inference completed successfully
        400:
          description: Invalid request parameters
          content:
            application/json:
              schema: ErrorRsp
        404:
          description: Inference Microservice not found
          content:
            application/json:
              schema: ErrorRsp
        500:
          description: Inference request failed
          content:
            application/json:
              schema: ErrorRsp
    """
    experiment_id = job_id
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        schema = ErrorRsp()
        metadata = {"error_desc": message, "error_code": 1}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)

    try:
        request_data = request.get_json(force=True)
        if not request_data:
            schema = ErrorRsp()
            metadata = {"error_desc": "Input data is required", "error_code": 2}
            schema_dict = schema.dump(schema.load(metadata))
            return make_response(jsonify(schema_dict), 400)

        schema = InferenceReq()
        request_dict = schema.dump(schema.load(request_data))

        # Process Inference Microservice inference via direct StatefulSet call
        result = InferenceMicroserviceHandler.process_inference_microservice_request_direct(job_id, request_dict)
        if result.get("status") != "error":
            return make_response(jsonify(result), 200)
        schema = ErrorRsp()
        metadata = {"error_desc": result, "error_code": 3}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)

    except Exception as e:
        logger.error("Error processing Inference Microservice inference request: %s", str(e))
        schema = ErrorRsp()
        metadata = {"error_desc": str(e), "error_code": 1}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)


@inference_microservices_bp_v2.route('/orgs/<org_name>/inference_microservices/<job_id>:status', methods=['GET'])
@disk_space_check
def inference_microservice_status(org_name, job_id):  # noqa: D214
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
          content:
            application/json:
              schema: ErrorRsp
        500:
          description: Failed to get service status
          content:
            application/json:
              schema: ErrorRsp
    """
    experiment_id = job_id
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        schema = ErrorRsp()
        metadata = {"error_desc": message, "error_code": 1}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)

    try:
        # Get Inference Microservice service status directly
        result = InferenceMicroserviceHandler.get_inference_microservice_status_direct(job_id)
        if result.get("status") != "error":
            return make_response(jsonify(result), 200)
        schema = ErrorRsp()
        metadata = {"error_desc": result, "error_code": 2}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)

    except Exception as e:
        logger.error("Error getting Inference Microservice status: %s", str(e))
        schema = ErrorRsp()
        metadata = {"error_desc": str(e), "error_code": 3}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)


@inference_microservices_bp_v2.route('/orgs/<org_name>/inference_microservices/<job_id>:stop', methods=['DELETE'])
@disk_space_check
def stop_inference_microservice(org_name, job_id):  # noqa: D214
    """Stop a Inference Microservice.

    ---
    delete:
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
          content:
            application/json:
              schema: ErrorRsp
        404:
          description: Inference Microservice not found
          content:
            application/json:
              schema: ErrorRsp
        500:
          description: Failed to stop service
          content:
            application/json:
              schema: ErrorRsp
    """
    experiment_id = job_id
    message = validate_uuid(experiment_id=experiment_id, job_id=job_id)
    if message:
        schema = ErrorRsp()
        metadata = {"error_desc": message, "error_code": 1}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 400)

    try:
        # Stop the Inference Microservice
        result = InferenceMicroserviceHandler.stop_inference_microservice(job_id)

        # Update job status if stopped successfully
        if result.code == 200:
            from nvidia_tao_core.microservices.utils.stateless_handler_utils import update_job_status
            update_job_status(
                experiment_id,
                job_id,
                status="Done",
                kind="experiments"
            )

        return make_response(jsonify(result.data), result.code)

    except Exception as e:
        logger.error("Error stopping Inference Microservice: %s", str(e))
        schema = ErrorRsp()
        metadata = {"error_desc": str(e), "error_code": 1}
        schema_dict = schema.dump(schema.load(metadata))
        return make_response(jsonify(schema_dict), 500)
