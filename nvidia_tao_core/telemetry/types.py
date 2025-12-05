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

"""Type definitions for telemetry system."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, TypedDict


class TelemetryData(TypedDict, total=False):
    """Normalized telemetry data structure.

    Attributes:
        version: TAO toolkit version
        action: Action being performed (train, evaluate, export, etc.)
        network: Network architecture being used
        success: Whether the operation succeeded
        user_error: Whether the error was user-caused
        time_lapsed: Time taken for operation in seconds
        gpus: List of GPU names
        client_type: Client type (container, api, cli, sdk, ui, etc.)
        automl_triggered: Whether job is triggered by AutoML
    """

    version: str
    action: str
    network: str
    success: bool
    user_error: bool
    time_lapsed: int
    gpus: List[str]
    client_type: str
    automl_triggered: bool


class AttributeType(Enum):
    """Types of telemetry attributes."""

    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    LIST = "list"


@dataclass
class MetricAttribute:
    """Configuration for a telemetry attribute.

    Attributes:
        name: Internal name of the attribute
        raw_key: Key in the raw input data
        attr_type: Type of the attribute (determines if sanitization is needed)
        default: Default value if missing
        include_in_comprehensive: Whether to include in comprehensive metric name
        metric_order: Order in comprehensive metric name (lower = earlier)

    Note:
        Sanitization is automatic based on attr_type:
        - STRING: sanitized (special chars removed, lowercase)
        - BOOLEAN, INTEGER, LIST: used as-is
    """

    name: str
    raw_key: str
    attr_type: AttributeType
    default: Any
    include_in_comprehensive: bool = True
    metric_order: int = 100
