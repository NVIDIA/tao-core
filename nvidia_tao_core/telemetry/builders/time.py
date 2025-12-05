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

"""Time-based metrics builder."""

from datetime import datetime
from typing import Any, Dict

from nvidia_tao_core.telemetry.builders.base import MetricBuilder
from nvidia_tao_core.telemetry.types import TelemetryData


class TimeMetricsBuilder(MetricBuilder):
    """Builder for time-based metrics.

    Handles daily accumulation of time_lapsed, resetting on day boundary.
    This is useful for tracking daily usage patterns and resource consumption.
    """

    def build(
        self,
        metrics: Dict[str, Any],
        telemetry_data: TelemetryData,
        context: Dict[str, Any]
    ) -> None:
        """Build time-based metrics.

        Accumulates time_lapsed for the current day, resetting when the day changes.

        Args:
            metrics: Metrics dictionary to update
            telemetry_data: Normalized telemetry data
            context: Additional context including 'now' and 'old_now' timestamps
        """
        time_lapsed = telemetry_data.get('time_lapsed', 0)
        now = context.get('now', datetime.now())
        old_now = context.get('old_now', now)

        # Reset daily counter if day changed
        if now.strftime("%d") != old_now.strftime("%d"):
            metrics['time_lapsed_today'] = time_lapsed
        else:
            metrics['time_lapsed_today'] = metrics.get('time_lapsed_today', 0) + time_lapsed
