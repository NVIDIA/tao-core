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

"""AutoML algorithm's Base Class"""
import math
import numpy as np
import random

from nvidia_tao_core.microservices.automl.utils import fix_input_dimension
from nvidia_tao_core.microservices.network_utils import network_constants
from nvidia_tao_core.microservices.network_utils import automl_helper
from nvidia_tao_core.microservices.handlers.stateless_handlers import get_job_specs


class AutoMLAlgorithmBase:
    """AutoML algorithms base class"""

    def __init__(self, job_context, root, network, parameters):
        """AutoML algorithm Base class"""
        self.job_context = job_context
        self.root = root
        self.handler_root = "/".join(self.root.split("/")[0:-2])
        self.network = network
        self.parameters = parameters
        self.parent_params = {}
        self.default_train_spec = get_job_specs(self.job_context.id)
        self.default_train_spec_flattened = {}

    def generate_automl_param_rec_value(self, parameter_config):
        """Generate a random value for the parameter passed"""
        parameter_name = parameter_config.get("parameter")
        data_type = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if data_type in ("int", "integer"):
            if parameter_name == "augmentation_config.preprocessing.output_image_height":
                if "model_config.input_image_config.size_height_width.height" in self.parent_params.keys():
                    return self.parent_params["model_config.input_image_config.size_height_width.height"]
            if parameter_name == "augmentation_config.preprocessing.output_image_width":
                if "model_config.input_image_config.size_height_width.width" in self.parent_params.keys():
                    return self.parent_params["model_config.input_image_config.size_height_width.width"]

            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return int(default_value)
            if (type(v_min) is not str and math.isnan(v_min)) or (type(v_max) is not str and math.isnan(v_max)):
                return int(default_value)

            v_min = int(v_min)
            if (type(v_max) is not str and math.isinf(v_max)) or v_max == "inf":
                v_max = int(default_value)
            else:
                v_max = int(v_max)
            random_int = np.random.randint(v_min, v_max + 1)

            if math_cond and type(math_cond) is str:
                factor = int(math_cond.split(" ")[1])
                random_int = fix_input_dimension(random_int, factor)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if (isinstance(parent_param, str) and parent_param != "nan" and parent_param == "TRUE") or (
                    isinstance(parent_param, bool) and parent_param
                ):
                    self.parent_params[parameter_name] = random_int

            return random_int

        if data_type == "bool":
            return np.random.randint(0, 2) == 1

        if data_type == "ordered_int":
            if parameter_config.get("valid_options", "") == "":
                return default_value
            valid_values = parameter_config.get("valid_options")
            sample = int(np.random.choice(valid_values))
            return sample

        if data_type in ("categorical", "ordered"):
            if parameter_config.get("valid_options", "") == "":
                return default_value
            valid_values = parameter_config.get("valid_options")
            sample = np.random.choice(valid_values)
            return sample

        if "list_1_" in data_type:
            if data_type == "list_1_backbone":
                # List needed in the form of consective numbers [1,2,3,4,5],
                # where the continuous numbers are decided by dependent parameters
                # Get backbone constant name from network_utils
                backbone_parameter = network_constants.backbone_mapper.get(self.network, "")
                backbone = self.parent_params.get(
                    backbone_parameter,
                    self.default_train_spec_flattened.get(backbone_parameter, None)
                )
                # Get the bounds from automl_helper
                bound_start, bound_end = (
                    automl_helper.automl_list_helper.get(self.network, {})
                    .get(data_type, {})
                    .get(parameter_name, {})
                    .get(backbone, {})
                )
            elif data_type == "list_1_normal":
                bound_start, bound_end = (
                    automl_helper.automl_list_helper.get(self.network, {})
                    .get(data_type, {})
                    .get(parameter_name, {})
                )
            else:
                return []
            # Generate two random numbers within the bounds
            random_number1 = random.randint(bound_start, bound_end)
            random_number2 = random.randint(bound_start, bound_end)
            # Make sure the numbers are in ascending order
            bound_start = min(random_number1, random_number2)
            bound_end = max(random_number1, random_number2)
            # Create a list of consecutive numbers between start_number and end_number
            automl_suggested_value = list(range(bound_start, bound_end + 1))
            return automl_suggested_value

        if data_type in ("list_2", "list_3"):
            automl_suggested_value = []
            bound_type, dependent_parameter = (
                automl_helper.automl_list_helper.get(self.network, {})
                .get(data_type, {})
                .get(parameter_name, {})
            )
            bound_value = self.parent_params.get(
                dependent_parameter,
                self.default_train_spec_flattened.get(dependent_parameter, None)
            )
            if not bound_value:
                if bound_type == "img_size":
                    bound_value = 1080  # Default value considering a HD image
                elif bound_type == "lr_steps":
                    bound_value = 50  # Default value of 50 epochs
                else:
                    return []

            # List needed in the form of multiple numbers operated with bounds
            if data_type == "list_2":
                # Generate a random number between 3 and 6 (inclusive)
                num_random_numbers = random.randint(3, 6)
                # Generate a list of random numbers
                if bound_type == "lr_steps":
                    automl_suggested_value = [random.randint(1, bound_value) for _ in range(num_random_numbers)]
                    return sorted(automl_suggested_value)
                if bound_type == "img_size":
                    # Calculate the range of valid multiples of 16
                    min_multiple = max(bound_value // 2, 16)
                    min_multiple -= min_multiple % 16  # Ensure min_multiple is a multiple of 16
                    max_multiple = bound_value - (bound_value % 16)
                    # Calculate the number of valid multiples of 16 within the range
                    num_multiples = ((max_multiple - min_multiple) // 16) + 1
                    # Generate random multiples of 16
                    automl_suggested_value = [
                        min_multiple + 16 * random.randint(0, num_multiples - 1)
                        for _ in range(num_random_numbers)
                    ]
                    return sorted(automl_suggested_value)
                return []

            # List needed in the form of pair of same numbers lke [15,15]
            if data_type == "list_3":
                if bound_type == "img_size":
                    min_value = bound_value // 100  # 1/100th of the bound value
                    max_value = bound_value // 10   # 1/10th of the bound value
                    # Generate a random integer within the specified range
                    random_integer = random.randint(min_value, max_value)
                    if self.network == "ml_recog":
                        # For ml_recog, the random integer needs to be a odd number
                        if random_integer % 2 == 0:
                            random_integer += 1
                    automl_suggested_value = [random_integer, random_integer]
                    return automl_suggested_value
                return []
        return default_value
