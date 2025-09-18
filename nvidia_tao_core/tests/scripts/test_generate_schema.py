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

"""Api tests to generate schema for networks"""

import pytest

from nvidia_tao_core.microservices.constants import TAO_NETWORKS
from nvidia_tao_core.microservices.enum_constants import _get_network_architectures
from nvidia_tao_core.microservices.utils import get_microservices_network_and_action
from nvidia_tao_core.scripts.generate_schema import generate_schema

EXCLUDED_KEYWORDS = [
    'maxine', 'monai', 'vlm', 'segmentation',
    'image_classification', 'character_recognition', 'object_detection'
]
config_networks = [
    network for network in _get_network_architectures()
    if not any(keyword in network for keyword in EXCLUDED_KEYWORDS)
]
constant_networks = [
    network for network in TAO_NETWORKS
    if not any(keyword in network for keyword in EXCLUDED_KEYWORDS)
]


TEST_ACTIONS = ["train", "evaluate", "distill", "export", "gen_trt_engine", "inference"]


@pytest.mark.parametrize("network", config_networks)
@pytest.mark.parametrize("action", TEST_ACTIONS)
def test_networks_from_enum(network, action):
    """Test schema from api network_arch enum with specific actions"""
    network_arch, _ = get_microservices_network_and_action(network, action)
    schema = generate_schema(network_arch, action)
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "default" in schema


@pytest.mark.parametrize("network", constant_networks)
@pytest.mark.parametrize("action", TEST_ACTIONS)
def test_networks_from_constants(network, action):
    """Test schema from TAO_NETWORKS constant with specific actions"""
    network_arch, _ = get_microservices_network_and_action(network, action)
    schema = generate_schema(network_arch, action)
    assert isinstance(schema, dict)
    assert "properties" in schema
    assert "default" in schema
