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

"""Comprehensive metrics builder with all attributes."""

from typing import Any, Dict, List

from nvidia_tao_core.telemetry.builders.base import MetricBuilder
from nvidia_tao_core.telemetry.config import METRIC_ATTRIBUTES
from nvidia_tao_core.telemetry.types import AttributeType, TelemetryData
from nvidia_tao_core.telemetry.utils import create_gpu_identifier


class ComprehensiveMetricsBuilder(MetricBuilder):
    """Builder for comprehensive metrics with all attributes.

    Generates a single metric name that includes all configured attributes,
    making it easy to slice and dice metrics by any dimension.

    Example metric:
        network_resnet50_action_train_version_5_3_0_status_pass_gpu_2_NVIDIA_A100_2

    This allows analytics systems to filter and aggregate by any attribute
    without needing separate metric counters for each combination.
    """

    def build(
        self,
        metrics: Dict[str, Any],
        telemetry_data: TelemetryData,
        context: Dict[str, Any]
    ) -> None:
        """Build comprehensive metric name with all attributes.

        Args:
            metrics: Metrics dictionary to update
            telemetry_data: Normalized telemetry data
            context: Additional context (unused by this builder)
        """
        metric_name = self._build_metric_name(telemetry_data)
        metrics[metric_name] = metrics.get(metric_name, 0) + 1

    def _build_metric_name(self, telemetry_data: TelemetryData) -> str:
        """Build comprehensive metric name using configured attributes.

        This method is configuration-driven - attributes are included based on
        their settings in METRIC_ATTRIBUTES. Adding a new attribute to the
        comprehensive metric name requires no code changes here.

        The order is maintained for backward compatibility:
        network, action, version, status, gpu

        Args:
            telemetry_data: Normalized telemetry data

        Returns:
            Comprehensive metric name string
        """
        components: List[str] = []

        # Add status (derived from success field) - order matters for backward compatibility
        status = "pass" if telemetry_data.get('success', False) else "fail"

        # Add GPU identifier (special handling)
        gpu_identifier = create_gpu_identifier(telemetry_data.get('gpus', ['unknown']))

        # Collect attributes that should be included in comprehensive metric
        # Sort by metric_order to maintain consistent ordering
        attrs_to_include = [
            (attr, telemetry_data.get(attr.name))
            for attr in sorted(METRIC_ATTRIBUTES, key=lambda a: a.metric_order)
            if attr.include_in_comprehensive and attr.name in telemetry_data
        ]

        # Build components in order, inserting status at the right position
        # Status should come after version (order=3)
        STATUS_ORDER = 4

        for attr, value in attrs_to_include:
            # Insert status before elements with order >= STATUS_ORDER
            if attr.metric_order >= STATUS_ORDER and not any(
                comp == "status" for comp in components
            ):
                components.extend(["status", status])

            if attr.attr_type == AttributeType.BOOLEAN:
                components.extend([attr.name, str(value).lower()])
            else:
                components.extend([attr.name, str(value)])

        # If status wasn't inserted (no attributes with order >= STATUS_ORDER), add it now
        if "status" not in components:
            components.extend(["status", status])

        # Add GPU identifier at the end
        components.extend(["gpu", gpu_identifier])

        return "_".join(components)
