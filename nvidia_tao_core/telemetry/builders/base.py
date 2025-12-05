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

"""Base class for metric builders."""

from abc import ABC, abstractmethod
from typing import Any, Dict

from nvidia_tao_core.telemetry.types import TelemetryData


class MetricBuilder(ABC):
    """Abstract base class for metric builders.

    Metric builders are responsible for generating specific types of metrics
    from telemetry data. This allows for pluggable metric generation strategies.

    To create a new metric builder:
    1. Subclass MetricBuilder
    2. Implement the build() method
    3. Register it with MetricProcessor

    Example:
        class CustomMetricsBuilder(MetricBuilder):
            def build(self, metrics, telemetry_data, context):
                # Your custom logic here
                metrics['custom_metric'] = compute_value(telemetry_data)
    """

    @abstractmethod
    def build(
        self,
        metrics: Dict[str, Any],
        telemetry_data: TelemetryData,
        context: Dict[str, Any]
    ) -> None:
        """Build and update metrics.

        Args:
            metrics: Metrics dictionary to update (modified in place)
            telemetry_data: Normalized telemetry data
            context: Additional context (e.g., timestamps, configuration)
        """
        pass
