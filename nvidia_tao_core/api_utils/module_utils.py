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

"""Module utils"""

import toml
import importlib
import pkg_resources

entrypoint_paths = {
    "nvidia_tao_pytorch": "nvidia_tao_pytorch.core.entrypoint",
    "nvidia_tao_deploy": "nvidia_tao_deploy.cv.common.entrypoint.entrypoint_hydra",
    "nvidia_tao_tf2": "nvidia_tao_tf2.common.entrypoint.entrypoint",
    "nvidia_tao_ds": "nvidia_tao_ds.core.entrypoint.entrypoint",
    "maxine_eye_contact": "maxine_eye_contact.entrypoint.maxine_eye_contact"
}

entry_points = [
    p for p in pkg_resources.iter_entry_points('console_scripts')
    if p.module_name.split('.')[0] in entrypoint_paths.keys()
]


def get_entry_points():
    """Return the entrypoints present"""
    eps = [ep.name for ep in entry_points]
    toml_path = "/home/pyproject.toml"
    with open(toml_path, "r", encoding="utf-8") as f:
        toml_data = toml.load(f)
    for ep_name in toml_data.get("project", {}).get("scripts", {}).keys():
        model = ep_name.split("-")
        if len(model) > 1:
            model = model[0]
        eps.append(model)
    eps = list(set(eps))
    return eps


def get_neural_network_actions(neural_network_name):
    """Return the valid neural network actions"""
    for ep in entry_points:
        if ep.name == neural_network_name:
            module = importlib.import_module(ep.module_name)
            actions = module.get_subtask_list()
            return module, actions
    actions = {}
    for ep in pkg_resources.iter_entry_points('console_scripts'):
        if neural_network_name in ep.name:
            model_action = ep.name.split("-")
            if len(model_action) > 1:
                actions[model_action[1]] = 1
    return neural_network_name, actions


def get_entry_point_module_mapping(neural_network_name):
    """Construct and return a dictionary for entrypoints"""
    entrypoints = {
        ep.name: ep.module_name for ep in entry_points
    }
    for ep in pkg_resources.iter_entry_points('console_scripts'):
        if neural_network_name in ep.name:
            model = ep.name.split("-")
            if len(model) > 1:
                model = model[0]
            entrypoints[model] = ep.module_name
    return entrypoints
