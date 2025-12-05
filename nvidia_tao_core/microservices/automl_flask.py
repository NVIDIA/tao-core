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

"""AutoML Flask Routes - Manage parameter ranges and details"""
import sys
import logging
import traceback
from flask import Blueprint, request, jsonify
from marshmallow import Schema, fields, validate, EXCLUDE

from nvidia_tao_core.microservices.automl.params import flatten_properties
from nvidia_tao_core.microservices.handlers.stateless_handlers import (
    get_automl_custom_param_ranges,
    save_automl_custom_param_ranges,
    get_handler_metadata
)
from nvidia_tao_core.microservices.handlers.utilities import validate_uuid
from nvidia_tao_core.scripts.generate_schema import generate_schema
from nvidia_tao_core.microservices.utils import get_microservices_network_and_action
from nvidia_tao_core.microservices.enum_constants import ExperimentNetworkArch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create blueprint for automl params routes
automl_params_bp = Blueprint('automl_params', __name__)


class ParameterDetailsReqSchema(Schema):
    """Class defining request schema for getting parameter details"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameters = fields.List(
        fields.Str(
            format="regex",
            regex=r'.*',
            validate=fields.validate.Length(max=500)
        ),
        validate=validate.Length(min=1, max=sys.maxsize),
        required=True
    )


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
    math_cond = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100), allow_none=True)
    depends_on = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)
    parent_param = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500), allow_none=True)


class ParameterDetailSchema(Schema):
    """Class defining individual parameter detail schema"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameter = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=500))
    value_type = fields.Str(format="regex", regex=r'.*', validate=fields.validate.Length(max=100))
    default = fields.Nested(ParameterRangeSchema)
    custom = fields.Nested(ParameterRangeSchema, allow_none=True)


class ParameterDetailsRspSchema(Schema):
    """Class defining response schema for getting parameter details"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameter_details = fields.List(fields.Nested(ParameterDetailSchema), validate=validate.Length(max=sys.maxsize))


class UpdateParameterRangesReqSchema(Schema):
    """Class defining request schema for updating parameter ranges"""

    class Meta:
        """Class enabling sorting field values by the order in which they are declared"""

        ordered = True
        unknown = EXCLUDE
    parameter_ranges = fields.List(
        fields.Nested(ParameterRangeSchema),
        validate=validate.Length(min=1, max=sys.maxsize),
        required=True
    )


@automl_params_bp.route('/api/v1/orgs/<org_name>/experiments/<experiment_id>:get_automl_param_details', methods=['GET'])
def get_automl_param_details(org_name, experiment_id):
    """Get detailed information about AutoML parameters including valid ranges, options, and math conditions.

    Args:
        org_name: Organization name
        experiment_id: Experiment ID
        Query params:
            parameters: Comma-separated list of parameter names (e.g., ?parameters=train.num_epochs,train.learning_rate)

    Returns:
        Parameter details including default ranges and any custom ranges set by the user
    """
    try:
        # Validate experiment_id is a valid UUID
        validation_error = validate_uuid(experiment_id=experiment_id)
        if validation_error:
            return jsonify({"error_desc": validation_error, "error_code": 1}), 400

        # Get parameters from query string
        parameters_query = request.args.get('parameters', '')
        if not parameters_query:
            return jsonify({"error_desc": "Missing 'parameters' query parameter", "error_code": 1}), 400

        # Split comma-separated parameters
        parameter_names = [p.strip() for p in parameters_query.split(',') if p.strip()]

        if not parameter_names:
            return jsonify({"error_desc": "No parameters provided", "error_code": 1}), 400

        # Get experiment metadata to determine network architecture
        handler_metadata = get_handler_metadata(experiment_id, "experiments")
        if not handler_metadata:
            return jsonify({
                "error_desc": f"Experiment {experiment_id} not found",
                "error_code": 1
            }), 404

        network_arch_enum = handler_metadata.get("network_arch")
        if not network_arch_enum:
            return jsonify({
                "error_desc": "Network architecture not found in experiment metadata",
                "error_code": 1
            }), 400

        # Convert enum to string if needed
        if hasattr(network_arch_enum, 'value'):
            network_arch = network_arch_enum.value
        else:
            network_arch = str(network_arch_enum)

        # Get the actual network name from the enum mapping
        try:
            # Try to get the network name from ExperimentNetworkArch enum
            network_arch_obj = ExperimentNetworkArch(network_arch)
            network_name, _ = get_microservices_network_and_action(network_arch_obj.value, "train")
        except (ValueError, AttributeError):
            network_name = network_arch

        logger.info(f"Getting parameter details for network: {network_name}, parameters: {parameter_names}")

        # Generate schema for the network
        try:
            json_schema = generate_schema(network_name, "train")
        except Exception as e:
            logger.error(f"Error generating schema for network: {network_name}")
            return jsonify({
                "error_desc": f"Unable to generate schema for network: {network_name}. Error: {str(e)}",
                "error_code": 1
            }), 400

        # Flatten the schema properties to get parameter details
        format_json_schema = flatten_properties(json_schema["properties"])

        # Get custom parameter ranges if they exist
        custom_ranges = get_automl_custom_param_ranges(experiment_id)

        # Build response for requested parameters
        parameter_details = []
        for param_name in parameter_names:
            if param_name in format_json_schema:
                param_info = format_json_schema[param_name]

                # Build the parameter detail object
                # Convert empty strings to None for numeric fields
                valid_min = param_info.get("valid_min", None)
                valid_max = param_info.get("valid_max", None)
                if valid_min == "":
                    valid_min = None
                if valid_max == "":
                    valid_max = None

                # Build default section from schema
                default_section = {
                    "default_value": param_info.get("default_value", None),
                    "valid_min": valid_min,
                    "valid_max": valid_max,
                    "valid_options": param_info.get("valid_options", []),
                    "math_cond": param_info.get("math_cond", None),
                    "depends_on": param_info.get("depends_on", None),
                    "parent_param": param_info.get("parent_param", None)
                }

                # Build custom section from user overrides
                custom_section = None
                if param_name in custom_ranges:
                    custom_section = {
                        "valid_min": custom_ranges[param_name].get("valid_min"),
                        "valid_max": custom_ranges[param_name].get("valid_max"),
                        "valid_options": custom_ranges[param_name].get("valid_options"),
                        "depends_on": custom_ranges[param_name].get("depends_on"),
                        "math_cond": custom_ranges[param_name].get("math_cond"),
                        "parent_param": custom_ranges[param_name].get("parent_param")
                    }

                param_detail = {
                    "parameter": param_name,
                    "value_type": param_info.get("value_type", ""),
                    "default": default_section,
                    "custom": custom_section
                }

                parameter_details.append(param_detail)
            else:
                logger.warning(f"Parameter '{param_name}' not found in schema for network '{network_name}'")

        if not parameter_details:
            return jsonify({
                "error_desc": f"None of the requested parameters were found in the schema for network '{network_name}'",
                "error_code": 1
            }), 404

        # Validate and serialize response
        rsp_schema = ParameterDetailsRspSchema()
        response_data = {"parameter_details": parameter_details}
        serialized = rsp_schema.dump(response_data)

        return jsonify(serialized), 200

    except Exception as e:
        logger.error(f"Error in get_automl_param_details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error_desc": f"Internal server error: {str(e)}",
            "error_code": 1
        }), 500


@automl_params_bp.route(
    '/api/v1/orgs/<org_name>/experiments/<experiment_id>:update_automl_param_ranges',
    methods=['POST', 'PATCH']
)
def update_automl_param_ranges(org_name, experiment_id):
    """Update custom valid ranges for AutoML parameters.

    Args:
        org_name: Organization name
        experiment_id: Experiment ID

    Returns:
        Success message
    """
    try:
        # Validate experiment_id is a valid UUID
        validation_error = validate_uuid(experiment_id=experiment_id)
        if validation_error:
            return jsonify({"error_desc": validation_error, "error_code": 1}), 400

        # Check if request has JSON content-type
        if not request.is_json:
            return jsonify({
                "error_desc": "Request Content-Type must be 'application/json'",
                "error_code": 1
            }), 400

        # Validate request body
        req_schema = UpdateParameterRangesReqSchema()
        errors = req_schema.validate(request.json)
        if errors:
            return jsonify({
                "error_desc": f"Invalid request: {errors}",
                "error_code": 1
            }), 400

        request_data = request.json
        parameter_ranges = request_data.get("parameter_ranges", [])

        if not parameter_ranges:
            return jsonify({"error_desc": "No parameter ranges provided", "error_code": 1}), 400

        # Get experiment metadata to verify it exists
        handler_metadata = get_handler_metadata(experiment_id, "experiments")
        if not handler_metadata:
            return jsonify({"error_desc": f"Experiment {experiment_id} not found", "error_code": 1}), 404

        network_arch_enum = handler_metadata.get("network_arch")
        if not network_arch_enum:
            return jsonify({
                "error_desc": "Network architecture not found in experiment metadata",
                "error_code": 1
            }), 400

        # Convert enum to string if needed
        if hasattr(network_arch_enum, 'value'):
            network_arch = network_arch_enum.value
        else:
            network_arch = str(network_arch_enum)

        # Get the actual network name from the enum mapping
        try:
            network_arch_obj = ExperimentNetworkArch(network_arch)
            network_name, _ = get_microservices_network_and_action(network_arch_obj.value, "train")
        except (ValueError, AttributeError):
            network_name = network_arch

        logger.info(f"Updating parameter ranges for network: {network_name}, experiment: {experiment_id}")

        # Generate schema to validate parameters exist
        try:
            json_schema = generate_schema(network_name, "train")
        except Exception as e:
            logger.error(f"Error generating schema for network: {network_name}")
            return jsonify({
                "error_desc": f"Unable to generate schema for network: {network_name}. Error: {str(e)}",
                "error_code": 1
            }), 400

        format_json_schema = flatten_properties(json_schema["properties"])

        # Validate that all parameters exist in the schema and ranges are valid
        validated_ranges = {}
        errors = []
        for param_range in parameter_ranges:
            param_name = param_range.get("parameter")
            custom_min = param_range.get("valid_min")
            custom_max = param_range.get("valid_max")
            custom_options = param_range.get("valid_options")
            custom_depends_on = param_range.get("depends_on")
            custom_math_cond = param_range.get("math_cond")
            custom_parent_param = param_range.get("parent_param")

            # Check if parameter exists in schema
            if param_name not in format_json_schema:
                errors.append(f"Parameter '{param_name}' not found in schema for network '{network_name}'")
                continue

            param_info = format_json_schema[param_name]
            schema_min = param_info.get("valid_min")
            schema_max = param_info.get("valid_max")

            # Validate custom ranges (min/max)
            if custom_min is not None and custom_max is not None:
                # Handle both scalar and list types
                if isinstance(custom_min, list) and isinstance(custom_max, list):
                    # Validate list ranges element-wise
                    if len(custom_min) != len(custom_max):
                        errors.append(
                            f"Parameter '{param_name}': valid_min and valid_max "
                            "must have same length"
                        )
                        continue

                    for i, (min_val, max_val) in enumerate(zip(custom_min, custom_max)):
                        if min_val >= max_val:
                            errors.append(
                                f"Parameter '{param_name}': valid_min[{i}] must be less "
                                f"than valid_max[{i}]"
                            )
                            continue

                        # Validate against schema bounds if they exist and are lists
                        if isinstance(schema_min, list) and len(schema_min) > i:
                            if (schema_min[i] is not None and schema_min[i] != "" and
                                    min_val < schema_min[i]):
                                errors.append(
                                    f"Parameter '{param_name}': valid_min[{i}] ({min_val}) cannot be less than "
                                    f"schema minimum ({schema_min[i]})"
                                )
                                continue

                        if isinstance(schema_max, list) and len(schema_max) > i:
                            if (schema_max[i] is not None and schema_max[i] != "" and
                                    max_val > schema_max[i]):
                                errors.append(
                                    f"Parameter '{param_name}': valid_max[{i}] ({max_val}) "
                                    f"cannot be greater than schema maximum ({schema_max[i]})"
                                )
                                continue
                else:
                    # Scalar validation
                    if custom_min >= custom_max:
                        errors.append(
                            f"Parameter '{param_name}': valid_min must be less than "
                            "valid_max"
                        )
                        continue

                    # Validate against schema bounds if they exist
                    if (schema_min is not None and schema_min != "" and
                            custom_min < schema_min):
                        errors.append(
                            f"Parameter '{param_name}': valid_min ({custom_min}) cannot be less than "
                            f"schema minimum ({schema_min})"
                        )
                        continue

                    if (schema_max is not None and schema_max != "" and
                            custom_max > schema_max):
                        errors.append(
                            f"Parameter '{param_name}': valid_max ({custom_max}) cannot be greater than "
                            f"schema maximum ({schema_max})"
                        )
                        continue

            # Validate valid_options if provided
            if custom_options is not None:
                schema_options = param_info.get("valid_options", [])
                if schema_options:
                    # Ensure custom options are a subset of schema options
                    invalid_options = [opt for opt in custom_options if opt not in schema_options]
                    if invalid_options:
                        errors.append(
                            f"Parameter '{param_name}': valid_options {invalid_options} are not in "
                            f"schema options {schema_options}"
                        )
                        continue

            validated_ranges[param_name] = {
                "valid_min": custom_min,
                "valid_max": custom_max,
                "valid_options": custom_options,
                "depends_on": custom_depends_on,
                "math_cond": custom_math_cond,
                "parent_param": custom_parent_param
            }

        if errors:
            return jsonify({
                "error_desc": f"Validation errors: {'; '.join(errors)}",
                "error_code": 1
            }), 400

        # Save the custom ranges
        save_automl_custom_param_ranges(experiment_id, validated_ranges)

        logger.info(f"Successfully updated {len(validated_ranges)} parameter range(s) for experiment {experiment_id}")

        return jsonify({
            "message": f"Successfully updated parameter ranges for {len(validated_ranges)} parameter(s)",
            "updated_parameters": list(validated_ranges.keys())
        }), 200

    except Exception as e:
        logger.error(f"Error in update_automl_param_ranges: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error_desc": f"Internal server error: {str(e)}",
            "error_code": 1
        }), 500
