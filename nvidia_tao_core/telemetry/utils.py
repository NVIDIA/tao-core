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

"""Utility functions for telemetry data processing."""

import re
from typing import Any, Dict, List

from nvidia_tao_core.telemetry.config import METRIC_ATTRIBUTES
from nvidia_tao_core.telemetry.types import AttributeType, TelemetryData


# Constants
TELEMETRY_SANITIZE_PATTERN = r"[^a-zA-Z0-9]"


def sanitize_field_value(value: Any, uppercase: bool = False) -> str:
    """Sanitize field value for use in metric names.

    Args:
        value: The value to sanitize
        uppercase: Whether to convert to uppercase

    Returns:
        Sanitized string value
    """
    sanitized = re.sub(TELEMETRY_SANITIZE_PATTERN, "_", str(value))
    return sanitized.upper() if uppercase else sanitized.lower()


def create_gpu_identifier(gpu_list: List[str]) -> str:
    """Create a unique identifier for a list of GPUs.

    Args:
        gpu_list: List of GPU names

    Returns:
        String identifier in format: count_GPU1_count_GPU2_count...

    Example:
        >>> create_gpu_identifier(['NVIDIA A100', 'NVIDIA A100', 'NVIDIA V100'])
        '3_NVIDIA_A100_2_NVIDIA_V100_1'
    """
    gpu_counts: Dict[str, int] = {}
    for gpu in gpu_list:
        sanitized_gpu = sanitize_field_value(gpu, uppercase=True)
        gpu_counts[sanitized_gpu] = gpu_counts.get(sanitized_gpu, 0) + 1

    gpu_parts = [f"{gpu}_{count}" for gpu, count in sorted(gpu_counts.items())]
    return f"{len(gpu_list)}_{'_'.join(gpu_parts)}"


def extract_telemetry_data(raw_data: Dict[str, Any]) -> TelemetryData:
    """Extract and normalize telemetry data from request using attribute configuration.

    This function is configuration-driven - it uses the METRIC_ATTRIBUTES registry
    to determine how to process each field. To add a new field, simply add it to
    the METRIC_ATTRIBUTES list in config.py.

    Args:
        raw_data: Raw telemetry data from request

    Returns:
        Dictionary with normalized telemetry fields

    Example:
        >>> raw = {'action': 'train', 'network': 'ResNet-50', 'success': True}
        >>> data = extract_telemetry_data(raw)
        >>> data['action']
        'train'
        >>> data['network']
        'resnet_50'
    """
    result: Dict[str, Any] = {}

    for attr in METRIC_ATTRIBUTES:
        raw_value = raw_data.get(attr.raw_key, attr.default)

        # STRING types are sanitized to lowercase; other types used as-is
        if attr.attr_type == AttributeType.STRING:
            value = sanitize_field_value(raw_value)
        else:
            value = raw_value

        result[attr.name] = value

    return result  # type: ignore[return-value]
