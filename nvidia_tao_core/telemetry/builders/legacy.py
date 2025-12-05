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

"""Legacy metrics builder for backward compatibility."""

from typing import Any, Dict

from nvidia_tao_core.telemetry.builders.base import MetricBuilder
from nvidia_tao_core.telemetry.types import TelemetryData
from nvidia_tao_core.telemetry.utils import sanitize_field_value


class LegacyMetricsBuilder(MetricBuilder):
    """Builder for legacy metric counters (backward compatibility).

    Generates metrics like:
    - total_action_{action}_{pass/fail}
    - version_{version}_action_{action}
    - network_{network}_action_{action}
    - gpu_{gpu}_action_{action}

    These metrics are maintained for backward compatibility with existing
    dashboards and monitoring systems.
    """

    def build(
        self,
        metrics: Dict[str, Any],
        telemetry_data: TelemetryData,
        context: Dict[str, Any]
    ) -> None:
        """Build legacy metrics for backward compatibility.

        Args:
            metrics: Metrics dictionary to update
            telemetry_data: Normalized telemetry data
            context: Additional context (unused by this builder)
        """
        action = telemetry_data['action']
        version = telemetry_data['version']
        network = telemetry_data['network']
        success = telemetry_data['success']

        # Update pass/fail counters
        status_key = f'total_action_{action}_{"pass" if success else "fail"}'
        metrics[status_key] = metrics.get(status_key, 0) + 1

        # Update version-action counters
        version_key = f'version_{version}_action_{action}'
        metrics[version_key] = metrics.get(version_key, 0) + 1

        # Update network-action counters
        network_key = f'network_{network}_action_{action}'
        metrics[network_key] = metrics.get(network_key, 0) + 1

        # Update per-GPU counters
        for gpu in telemetry_data['gpus']:
            sanitized_gpu = sanitize_field_value(gpu)
            gpu_key = f'gpu_{sanitized_gpu}_action_{action}'
            metrics[gpu_key] = metrics.get(gpu_key, 0) + 1
