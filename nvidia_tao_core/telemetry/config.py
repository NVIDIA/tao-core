# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Configuration for telemetry metric attributes.

To add a new telemetry attribute:
1. Add a new MetricAttribute to METRIC_ATTRIBUTES list
2. Update TelemetryData TypedDict in types.py if needed
3. That's it! The rest is handled automatically by the framework.

Processing rules:
- STRING attributes are automatically sanitized (lowercase, special chars removed)
- BOOLEAN, INTEGER, LIST attributes are used as-is
"""

from typing import Dict, List

from nvidia_tao_core.telemetry.types import AttributeType, MetricAttribute


# Metric Attribute Registry
# To add a new attribute, simply add it to this list
METRIC_ATTRIBUTES: List[MetricAttribute] = [
    MetricAttribute(
        name='version',
        raw_key='version',
        attr_type=AttributeType.STRING,
        default='unknown',
        metric_order=3
    ),
    MetricAttribute(
        name='action',
        raw_key='action',
        attr_type=AttributeType.STRING,
        default='unknown',
        metric_order=2
    ),
    MetricAttribute(
        name='network',
        raw_key='network',
        attr_type=AttributeType.STRING,
        default='unknown',
        metric_order=1
    ),
    MetricAttribute(
        name='success',
        raw_key='success',
        attr_type=AttributeType.BOOLEAN,
        default=False,
        metric_order=4,
        include_in_comprehensive=False  # Transformed to 'status'
    ),
    MetricAttribute(
        name='user_error',
        raw_key='user_error',
        attr_type=AttributeType.BOOLEAN,
        default=False,
        metric_order=5,
        include_in_comprehensive=False
    ),
    MetricAttribute(
        name='time_lapsed',
        raw_key='time_lapsed',
        attr_type=AttributeType.INTEGER,
        default=0,
        include_in_comprehensive=False
    ),
    MetricAttribute(
        name='gpus',
        raw_key='gpu',
        attr_type=AttributeType.LIST,
        default=['unknown'],
        include_in_comprehensive=False  # Uses special GPU identifier
    ),
    MetricAttribute(
        name='client_type',
        raw_key='client_type',
        attr_type=AttributeType.STRING,
        default='container',
        metric_order=6,
        include_in_comprehensive=False  # Used only in labeled metrics
    ),
    MetricAttribute(
        name='automl_triggered',
        raw_key='automl_triggered',
        attr_type=AttributeType.BOOLEAN,
        default=False,
        include_in_comprehensive=False  # Used only in labeled metrics
    ),
]


def get_attribute_map() -> Dict[str, MetricAttribute]:
    """Get mapping from attribute name to MetricAttribute.

    Returns:
        Dictionary mapping attribute names to their configurations
    """
    return {attr.name: attr for attr in METRIC_ATTRIBUTES}


def get_raw_key_map() -> Dict[str, MetricAttribute]:
    """Get mapping from raw key to MetricAttribute.

    Returns:
        Dictionary mapping raw data keys to their configurations
    """
    return {attr.raw_key: attr for attr in METRIC_ATTRIBUTES}
