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

"""Hyperband AutoML algorithm modules"""
import numpy as np
import math
import logging
import os

from nvidia_tao_core.microservices.utils.automl_utils import (
    ResumeRecommendation, JobStates, get_valid_range, clamp_value
)
from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase
from nvidia_tao_core.microservices.utils.handler_utils import get_flatten_specs
from nvidia_tao_core.microservices.utils.stateless_handler_utils import (
    save_job_specs,
    get_job_specs,
    save_automl_brain_info,
    get_automl_brain_info
)
from nvidia_tao_core.microservices.automl import network_utils

# Configure logging
TAO_LOG_LEVEL = os.getenv('TAO_LOG_LEVEL', 'INFO').upper()
tao_log_level = getattr(logging, TAO_LOG_LEVEL, logging.INFO)
logging.basicConfig(
    level=logging.WARNING,  # Root logger: suppress third-party DEBUG logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('nvidia_tao_core').setLevel(tao_log_level)
logger = logging.getLogger(__name__)


class HyperBand(AutoMLAlgorithmBase):
    """Hyperband AutoML algorithm class"""

    def __init__(self, job_context, root, network, parameters, R, nu, epoch_multiplier):
        """Initialize the Hyperband algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters (epoch params will be filtered out)
            R: the maximum amount of resource that can be allocated to a single configuration
            nu: an input that controls the proportion of configurations discarded in each round of SuccessiveHalving
            epoch_multiplier: multiplying factor for epochs
        """
        super().__init__(job_context, root, network, parameters)
        self.epoch_multiplier = int(epoch_multiplier)
        self.ni = {}
        self.ri = {}
        self.brackets_and_sh_sequence(R, nu)
        self.epoch_number = 0
        self.resume_epoch_number = 0
        # State variables
        self.bracket = "0"  # Bracket
        self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        self.sh_iter = 0  # SH iteration
        self.experiments_considered = []
        self.expt_iter = 0  # Recommendations within the SH
        self.complete = False
        self.reverse_sort = True
        logger.info(f"Hyperband initialized with R={R}, nu={nu}, epoch_multiplier={self.epoch_multiplier}")

    def brackets_and_sh_sequence(self, R, nu):
        """Generate ni,ri arrays based on R and nu values"""
        smax = int(np.log(R) / np.log(nu))
        logger.info(f"Hyperband bracket calculation: R={R}, nu={nu}, smax={smax}")
        for itr, s in enumerate(range(smax, 0, -1)):
            self.ni[str(itr)] = []
            self.ri[str(itr)] = []
            n = int(math.ceil(int((smax + 1) / (s + 1)) * (nu**s)))
            r = int(R / (nu**s))
            logger.info(f"  Bracket {itr} (s={s}): initial n={n}, r={r}")
            for s_idx in range(s + 1):
                ni = int(n * (nu**(-s_idx)))
                ri = int(r * (nu**s_idx))
                self.ni[str(itr)].append(ni)
                self.ri[str(itr)].append(ri)
            logger.info(f"  Bracket {itr} final: ni={self.ni[str(itr)]}, ri={self.ri[str(itr)]}")
        logger.info(f"All brackets: ni={self.ni}, ri={self.ri}")

    def _is_epoch_parameter(self, param_name):
        """Check if a parameter controls epoch count

        Args:
            param_name: Flattened parameter name (e.g., "train_config.num_epochs")

        Returns:
            bool: True if parameter controls epoch count
        """
        # Simple epoch parameter names at root level
        if param_name in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
            return True

        # Split the parameter name to check nested paths
        parts = param_name.split(".")

        # Check if any part is in epoch-related config sections
        has_training_config = any(
            p in ("training_config", "train_config", "train") for p in parts
        )
        has_epoch_key = any(
            p in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch", "max_epochs")
            for p in parts
        )

        return has_training_config and has_epoch_key

    def override_num_epochs(self, num_epochs):
        """Override num epochs parameter in train spec file

        This ensures bracket-based epoch control by updating all epoch-related
        parameters in the spec, regardless of automl_params settings.
        """
        spec = get_job_specs(self.job_context.id)
        for key1 in spec:
            if key1 in ("training_config", "train_config", "train"):
                for key2 in spec[key1]:
                    if key2 in ("num_epochs", "epochs", "n_epochs", "max_iters", "epoch"):
                        spec[key1][key2] = num_epochs
                    elif key2 in ("train_config"):
                        for key3 in spec[key1][key2]:
                            if key3 == "runner":
                                for key4 in spec[key1][key2][key3]:
                                    if key4 == "max_epochs":
                                        spec[key1][key2][key3][key4] = num_epochs
            elif key1 in ("num_epochs"):
                spec[key1] = num_epochs
        save_job_specs(self.job_context.id, spec)

    def generate_automl_param_rec_value(self, parameter_config):
        """Generate a random value for the parameter passed"""
        parameter_name = parameter_config.get("parameter")

        # Apply custom overrides if provided
        if self.custom_ranges and parameter_name in self.custom_ranges:
            for override_key, override_value in self.custom_ranges[parameter_name].items():
                if override_value is not None:
                    parameter_config[override_key] = override_value

        tp = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if tp == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return float(default_value)
            if (type(v_min) is not str and math.isnan(v_min)) or (type(v_max) is not str and math.isnan(v_max)):
                return float(default_value)
            v_min, v_max = get_valid_range(parameter_config, self.parent_params, self.custom_ranges)

            # Apply math condition if specified
            # Skip relational constraints (like "> depends_on") as they're handled in base class
            if math_cond and type(math_cond) is str and "depends_on" not in math_cond:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
                    if operator == "^":
                        # Use helper function for power constraints with equal priority
                        fallback = np.random.uniform(low=v_min, high=v_max)
                        fallback = clamp_value(fallback, v_min, v_max)
                        random_float = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        # Regular sampling for non-power constraints
                        random_float = np.random.uniform(low=v_min, high=v_max)
                        random_float = clamp_value(random_float, v_min, v_max)
            else:
                # No math condition, regular sampling
                random_float = np.random.uniform(low=v_min, high=v_max)
                random_float = clamp_value(random_float, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if ((type(parent_param) is str and parent_param != "nan" and parent_param == "TRUE") or
                        (type(parent_param) is bool and parent_param)):
                    self.parent_params[parameter_config.get("parameter")] = random_float

            # Apply network-specific parameter logic
            return network_utils.apply_network_specific_param_logic(
                network=self.network,
                data_type=tp,
                parameter_name=parameter_name,
                value=random_float,
                v_max=v_max,
                default_train_spec=self.default_train_spec,
                parent_params=self.parent_params
            )

        return super().generate_automl_param_rec_value(parameter_config)

    def save_state(self):
        """Save the Hyperband algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["bracket"] = self.bracket
        state_dict["sh_iter"] = self.sh_iter
        state_dict["expt_iter"] = self.expt_iter
        state_dict["complete"] = self.complete
        state_dict["epoch_number"] = self.epoch_number
        state_dict["resume_epoch_number"] = self.resume_epoch_number
        state_dict["epoch_multiplier"] = self.epoch_multiplier
        state_dict["ni"] = self.ni
        state_dict["ri"] = self.ri

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(job_context, root, network, parameters, R, nu, epoch_multiplier):
        """Load the Hyperband algorithm related variables to brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return HyperBand(job_context, root, network, parameters, R, nu, epoch_multiplier)

        brain = HyperBand(job_context, root, network, parameters, R, nu, epoch_multiplier)
        # Load state (Remember everything)
        brain.bracket = json_loaded["bracket"]  # Bracket
        brain.sh_iter = json_loaded["sh_iter"]  # SH iteration
        brain.expt_iter = json_loaded["expt_iter"]  # Recommendations within the SH
        brain.complete = json_loaded["complete"]
        brain.epoch_number = json_loaded["epoch_number"]
        brain.resume_epoch_number = json_loaded["resume_epoch_number"]

        return brain

    def _generate_one_recommendation(self, history):
        """Updates the counter variables and performs successive halving"""
        if self.complete:
            return None

        num = self.ni[self.bracket][self.sh_iter]
        if self.expt_iter == num:
            self.expt_iter = 0
            self.sh_iter += 1
        if self.sh_iter == len(self.ni[self.bracket]):
            self.sh_iter = 0
            self.bracket = str(int(self.bracket) + 1)
            if self.bracket in self.ri.keys():
                self.override_num_epochs(self.ri[self.bracket][-1] * self.epoch_multiplier)
        if int(self.bracket) > int(max(list(self.ni.keys()), key=int)):
            self.complete = True
            return None

        if self.sh_iter == 0:
            specs = self._generate_random_parameters()
            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            final_epoch = self.ri[self.bracket][-1] * self.epoch_multiplier
            self.override_num_epochs(final_epoch)
            logger.info(
                f"Hyperband: New experiment in bracket {self.bracket}, SH iter {self.sh_iter}, "
                f"will_pause_at_epoch={self.epoch_number}, spec_total_epochs={final_epoch}"
            )
            to_return = specs
        else:
            # Do successive halving on the last bracket
            # Here, we are sloppy in defining the window, but we assume runs that run for more epochs will be better
            # We take history[-bracket_size:] and prune this at every SH step
            lower = -1 * self.ni.get(self.bracket, [0])[0]

            self.resume_epoch_number = int(self.ri[self.bracket][self.sh_iter - 1] * self.epoch_multiplier)
            if self.expt_iter == 0:
                if self.sh_iter == 1:
                    self.experiments_considered = sorted(
                        history[lower:],
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]
                else:
                    for experiment in self.experiments_considered:
                        experiment.result = history[experiment.id].result
                    self.experiments_considered = sorted(
                        self.experiments_considered,
                        key=lambda rec: rec.result,
                        reverse=self.reverse_sort
                    )[0:self.ni[self.bracket][self.sh_iter]]

            self.epoch_number = self.ri[self.bracket][self.sh_iter] * self.epoch_multiplier
            final_epoch = self.ri[self.bracket][-1] * self.epoch_multiplier
            self.override_num_epochs(final_epoch)
            logger.info(
                f"Hyperband: Resume experiment in bracket {self.bracket}, SH iter {self.sh_iter}, "
                f"will_run_to_epoch={self.epoch_number}, spec_total_epochs={final_epoch}, "
                f"resume_from={self.resume_epoch_number}"
            )
            resumerec = ResumeRecommendation(
                self.experiments_considered[self.expt_iter].id,
                self.experiments_considered[self.expt_iter].specs
            )
            to_return = resumerec
        self.expt_iter += 1

        return to_return

    def done(self):
        """Return if Hyperband algorithm is complete or not"""
        return self.complete

    def _generate_random_parameters(self):
        """Generates random parameter values for a recommendation

        Note: Epoch-controlling parameters are skipped because Hyperband controls
        epochs via bracket configuration. The actual epoch count is set by
        override_num_epochs() based on ri[bracket][sh_iter] * epoch_multiplier.
        """
        hyperparam_dict = {}
        for param in self.parameters:
            name = param["parameter"]
            # Skip epoch parameters in Hyperband - epochs are controlled by brackets
            if self._is_epoch_parameter(name):
                logger.info(f"Skipping epoch parameter in Hyperband (bracket-controlled): {name}")
                continue
            rec = self.generate_automl_param_rec_value(param)
            logger.info(f"Generated random parameter in hyperband: {name} = {rec}")
            hyperparam_dict[name] = rec
        return hyperparam_dict

    def generate_recommendations(self, history):
        """Generates recommendations for the controller to run"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)
        if history == []:
            rec1 = self._generate_one_recommendation(history)
            assert type(rec1) is dict, f"Recommendation must be a dictionary, got {type(rec1)}"
            self.track_id = 0
            return [rec1]

        if history[self.track_id].status not in [JobStates.success, JobStates.failure]:
            return []
        rec = self._generate_one_recommendation(history)
        if type(rec) is dict:
            self.track_id = len(history)
            return [rec]
        if type(rec) is ResumeRecommendation:
            self.track_id = rec.id
            return [rec]
        return []
