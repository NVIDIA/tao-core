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

"""Bayesian AutoML algorithm modules"""
import numpy as np
import os
import math
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from scipy.stats import norm
from scipy.optimize import minimize

from nvidia_tao_core.microservices.automl.utils import JobStates, get_valid_range, clamp_value
from nvidia_tao_core.microservices.automl.automl_algorithm_base import AutoMLAlgorithmBase
from nvidia_tao_core.microservices.handlers.utilities import get_total_epochs, get_flatten_specs
from nvidia_tao_core.microservices.handlers.stateless_handlers import save_automl_brain_info, get_automl_brain_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Bayesian(AutoMLAlgorithmBase):
    """Bayesian AutoML algorithm class"""

    def __init__(self, job_context, root, network, parameters):
        """Initialize the Bayesian algorithm class

        Args:
            root: handler root
            network: model we are running AutoML on
            parameters: automl sweepable parameters
        """
        super().__init__(job_context, root, network, parameters)
        length_scale = [1.0] * len(self.parameters)
        m52 = ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5)
        # m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) # is another option
        self.gp = GaussianProcessRegressor(
            kernel=m52,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=10,
            random_state=95051
        )
        # The following 2 need to be stored
        self.Xs = []
        self.ys = []

        self.xi = 0.01
        self.num_restarts = 5

        self.num_epochs_per_experiment = get_total_epochs(job_context, os.path.join(self.handler_root, "specs"))

    def generate_automl_param_rec_value(self, parameter_config, suggestion):
        """Convert 0 to 1 GP prediction into a possible value"""
        parameter_name = parameter_config.get("parameter")
        data_type = parameter_config.get("value_type")
        default_value = parameter_config.get("default_value", None)
        math_cond = parameter_config.get("math_cond", None)
        parent_param = parameter_config.get("parent_param", None)

        if data_type == "float":
            v_min = parameter_config.get("valid_min", "")
            v_max = parameter_config.get("valid_max", "")
            if v_min == "" or v_max == "":
                return float(default_value)
            if (type(v_min) is not str and math.isnan(v_min)) or (type(v_max) is not str and math.isnan(v_max)):
                return float(default_value)

            v_min, v_max = get_valid_range(parameter_config, self.parent_params)

            # Apply math condition if specified
            if math_cond and type(math_cond) is str:
                parts = math_cond.split(" ")
                if len(parts) >= 2:
                    operator = parts[0]
                    factor = int(float(parts[1]))
                    if operator == "^":
                        # Use helper function for power constraints with equal priority
                        normalized = suggestion * (v_max - v_min) + v_min
                        fallback = clamp_value(normalized, v_min, v_max)
                        quantized = float(self._apply_power_constraint_with_equal_priority(
                            v_min, v_max, factor, fallback))
                    else:
                        # Regular sampling for non-power constraints
                        normalized = suggestion * (v_max - v_min) + v_min
                        quantized = clamp_value(normalized, v_min, v_max)
            else:
                # No math condition, regular sampling
                normalized = suggestion * (v_max - v_min) + v_min
                quantized = clamp_value(normalized, v_min, v_max)

            if not (type(parent_param) is float and math.isnan(parent_param)):
                if (isinstance(parent_param, str) and parent_param != "nan" and parent_param == "TRUE") or (
                    isinstance(parent_param, bool) and parent_param
                ):
                    self.parent_params[parameter_name] = quantized
            return quantized

        return super().generate_automl_param_rec_value(parameter_config)

    def save_state(self):
        """Save the Bayesian algorithm related variables to brain metadata"""
        state_dict = {}
        state_dict["Xs"] = np.array(self.Xs).tolist()  # List of np arrays
        state_dict["ys"] = np.array(self.ys).tolist()  # List

        save_automl_brain_info(self.job_context.id, state_dict)

    @staticmethod
    def load_state(job_context, root, network, parameters):
        """Load the Bayesian algorithm related variables to brain metadata"""
        json_loaded = get_automl_brain_info(job_context.id)
        if not json_loaded:
            return Bayesian(job_context, root, network, parameters)

        Xs = []
        for x in json_loaded["Xs"]:
            Xs.append(np.array(x))
        ys = json_loaded["ys"]
        bayesian = Bayesian(job_context, root, network, parameters)
        # Load state (Remember everything)
        bayesian.Xs = Xs
        bayesian.ys = ys

        len_y = len(ys)
        if Xs and ys:
            bayesian.gp.fit(np.array(Xs[:len_y]), np.array(ys))

        return bayesian

    def generate_recommendations(self, history):
        """Generates parameter values and appends to recommendations"""
        get_flatten_specs(self.default_train_spec, self.default_train_spec_flattened)
        if history == []:
            # default recommendation => random points
            # TODO: In production, this must be default values for a baseline
            suggestions = np.random.rand(len(self.parameters))
            self.Xs.append(suggestions)
            recommendations = []
            for param_dict, suggestion in zip(self.parameters, suggestions):
                recommendation_value = self.generate_automl_param_rec_value(param_dict, suggestion)
                logger.info(f"Recommendation param: {param_dict['parameter']} value: {recommendation_value}")
                recommendations.append(recommendation_value)
            return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]
        # This function will be called every 5 seconds or so.
        # If no change in history, dont give a recommendation
        # ie - wait for previous recommendation to finish
        if history[-1].status not in [JobStates.success, JobStates.failure]:
            return []

        # Update the GP based on results
        self.ys.append(history[-1].result)
        self.update_gp()

        # Generate one recommendation
        # Generate "suggestions" which are in [0.0, 1.0] by optimizing EI
        suggestions = self.optimize_ei()  # length = len(self.parameters), np.array type
        self.Xs.append(suggestions)
        # Convert the suggestions to recommendations based on parameter type
        # Assume one:one mapping between self.parameters and suggestions
        recommendations = []
        assert len(self.parameters) == len(suggestions), (
            f"Number of parameters ({len(self.parameters)}) does not match "
            f"number of suggestions ({len(suggestions)})"
        )
        for param_dict, suggestion in zip(self.parameters, suggestions):
            recommendation_value = self.generate_automl_param_rec_value(param_dict, suggestion)
            logger.info(f"Recommendation param: {param_dict['parameter']} value: {recommendation_value}")
            recommendations.append(recommendation_value)

        return [dict(zip([param["parameter"] for param in self.parameters], recommendations))]

    def update_gp(self):
        """Update gausian regressor parameters"""
        Xs_npy = np.array(self.Xs)
        ys_npy = np.array(self.ys)
        self.gp.fit(Xs_npy, ys_npy)

    def optimize_ei(self):
        """Optmize expected improvement functions"""
        best_ei = 1.0
        best_x = None

        dim = len(self.Xs[0])
        bounds = [(0, 1)] * len(self.parameters)

        for _ in range(self.num_restarts):
            x0 = np.random.rand(dim)
            res = minimize(self._expected_improvement, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_ei:
                best_ei = res.fun
                best_x = res.x
        return best_x.reshape(-1)

    """
    Used from:
    http://krasserm.github.io/2018/03/21/bayesian-optimization/
    """
    def _expected_improvement(self, X, xi=0.01):
        """Calculate the expected improvement at points X based on existing samples.

        Args:
            X: Points at which EI shall be calculated (m x d)
            xi: Exploitation-exploration trade-off parameter

        Returns:
            float: Expected improvements at points X
        """
        X = X.reshape(1, -1)

        mu, sigma = self.gp.predict(X, return_std=True)
        mu_sample = self.gp.predict(np.array(self.Xs))

        sigma = sigma.reshape(-1, 1)
        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [1]
        mu_sample_opt = np.max(mu_sample)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - self.xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return -1 * ei[0, 0]
