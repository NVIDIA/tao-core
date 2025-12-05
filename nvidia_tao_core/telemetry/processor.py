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

"""Metric processor for orchestrating telemetry metric building."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from nvidia_tao_core.telemetry.builders import (
    ComprehensiveMetricsBuilder,
    LabeledMetricsBuilder,
    LegacyMetricsBuilder,
    MetricBuilder,
    TimeMetricsBuilder,
)
from nvidia_tao_core.telemetry.utils import extract_telemetry_data


class MetricProcessor:
    """Orchestrates metric building using configured builders.

    This class coordinates the different metric builders to process telemetry
    data and update metrics. New metric types can be added by registering
    additional builders.

    Example:
        >>> processor = MetricProcessor()
        >>> processor.add_builder(CustomMetricsBuilder())
        >>> metrics = {}
        >>> raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}
        >>> updated_metrics = processor.process(metrics, raw_data)

    Custom builders:
        >>> class AlertMetricsBuilder(MetricBuilder):
        ...     def build(self, metrics, telemetry_data, context):
        ...         if not telemetry_data['success']:
        ...             metrics['failures_today'] = metrics.get('failures_today', 0) + 1
        >>>
        >>> processor = MetricProcessor()
        >>> processor.add_builder(AlertMetricsBuilder())
    """

    def __init__(self, builders: Optional[List[MetricBuilder]] = None):
        """Initialize processor with builders.

        Args:
            builders: List of metric builders to use. If None, uses default builders
                     (Legacy, Comprehensive, Time, and Labeled builders).
        """
        if builders is None:
            # Default builders - includes both old (backward compat) and new (labeled)
            self.builders = [
                LegacyMetricsBuilder(),           # Keep for old dashboards
                ComprehensiveMetricsBuilder(),    # Keep for old dashboards
                TimeMetricsBuilder(),             # Keep for time-based metrics
                LabeledMetricsBuilder(),          # NEW: Prometheus-style labeled metrics
            ]
        else:
            self.builders = builders

    def add_builder(self, builder: MetricBuilder) -> None:
        """Add a new metric builder.

        Builders are executed in the order they are added.

        Args:
            builder: Metric builder to add
        """
        self.builders.append(builder)

    def process(
        self,
        metrics: Dict[str, Any],
        raw_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process telemetry data and update metrics.

        This is the main entry point for processing telemetry data. It:
        1. Extracts and normalizes the raw data
        2. Runs all registered builders
        3. Updates timestamps

        Args:
            metrics: Existing metrics dictionary to update
            raw_data: Raw telemetry data from request
            context: Additional context (timestamps, configuration, etc.)

        Returns:
            Updated metrics dictionary
        """
        if context is None:
            context = {}

        # Ensure timestamps are in context
        if 'now' not in context:
            context['now'] = datetime.now()
        if 'old_now' not in context:
            old_now_iso = metrics.get('last_updated')
            context['old_now'] = (
                datetime.fromisoformat(old_now_iso) if old_now_iso
                else context['now']
            )

        # Extract and normalize telemetry data
        telemetry_data = extract_telemetry_data(raw_data)

        # Run all builders
        for builder in self.builders:
            builder.build(metrics, telemetry_data, context)

        # Update timestamp
        metrics['last_updated'] = context['now'].isoformat()

        return metrics
