# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for MetricProcessor."""

from datetime import datetime
from unittest.mock import Mock
from nvidia_tao_core.telemetry.processor import MetricProcessor
from nvidia_tao_core.telemetry.builders import (
    MetricBuilder,
    LegacyMetricsBuilder,
    ComprehensiveMetricsBuilder,
    TimeMetricsBuilder,
    LabeledMetricsBuilder
)


class TestMetricProcessorInitialization:
    """Test cases for MetricProcessor initialization."""

    def test_default_initialization(self):
        """Test that processor initializes with default builders."""
        processor = MetricProcessor()

        assert len(processor.builders) == 4
        assert isinstance(processor.builders[0], LegacyMetricsBuilder)
        assert isinstance(processor.builders[1], ComprehensiveMetricsBuilder)
        assert isinstance(processor.builders[2], TimeMetricsBuilder)
        assert isinstance(processor.builders[3], LabeledMetricsBuilder)

    def test_custom_builders_initialization(self):
        """Test initialization with custom builders."""
        custom_builder = LegacyMetricsBuilder()
        processor = MetricProcessor(builders=[custom_builder])

        assert len(processor.builders) == 1
        assert processor.builders[0] is custom_builder

    def test_empty_builders_initialization(self):
        """Test initialization with empty builders list."""
        processor = MetricProcessor(builders=[])

        assert len(processor.builders) == 0


class TestMetricProcessorAddBuilder:
    """Test cases for add_builder method."""

    def test_add_single_builder(self):
        """Test adding a single builder."""
        processor = MetricProcessor(builders=[])
        builder = LegacyMetricsBuilder()

        processor.add_builder(builder)

        assert len(processor.builders) == 1
        assert processor.builders[0] is builder

    def test_add_multiple_builders(self):
        """Test adding multiple builders."""
        processor = MetricProcessor(builders=[])
        builder1 = LegacyMetricsBuilder()
        builder2 = ComprehensiveMetricsBuilder()

        processor.add_builder(builder1)
        processor.add_builder(builder2)

        assert len(processor.builders) == 2
        assert processor.builders[0] is builder1
        assert processor.builders[1] is builder2

    def test_add_builder_to_existing(self):
        """Test adding builder to processor with default builders."""
        processor = MetricProcessor()
        initial_count = len(processor.builders)

        custom_builder = LegacyMetricsBuilder()
        processor.add_builder(custom_builder)

        assert len(processor.builders) == initial_count + 1
        assert processor.builders[-1] is custom_builder


class TestMetricProcessorProcess:
    """Test cases for process method."""

    def test_process_basic(self):
        """Test basic metric processing."""
        processor = MetricProcessor()
        metrics = {}
        raw_data = {
            'version': '5.3.0',
            'action': 'train',
            'network': 'ResNet-50',
            'success': True,
            'user_error': False,
            'time_lapsed': 3600,
            'gpu': ['NVIDIA A100']
        }

        result = processor.process(metrics, raw_data)

        # Should have metrics from all default builders
        assert 'total_action_train_pass' in result  # Legacy
        assert any('network_resnet_50' in k for k in result.keys())  # Comprehensive
        assert 'time_lapsed_today' in result  # Time
        assert 'last_updated' in result  # Timestamp

    def test_process_updates_last_updated(self):
        """Test that process updates last_updated timestamp."""
        processor = MetricProcessor()
        metrics = {}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}

        result = processor.process(metrics, raw_data)

        assert 'last_updated' in result
        # Should be a valid ISO format datetime
        datetime.fromisoformat(result['last_updated'])

    def test_process_with_existing_metrics(self):
        """Test processing with existing metrics."""
        processor = MetricProcessor()
        metrics = {
            'total_action_train_pass': 5,
            'last_updated': '2025-01-01T00:00:00'
        }
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}

        result = processor.process(metrics, raw_data)

        # Should increment existing metric
        assert result['total_action_train_pass'] == 6

    def test_process_with_custom_context(self):
        """Test processing with custom context."""
        processor = MetricProcessor()
        metrics = {}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True, 'time_lapsed': 100}

        custom_now = datetime(2025, 1, 15, 12, 0)
        context = {'now': custom_now}

        result = processor.process(metrics, raw_data, context)

        assert result['last_updated'] == custom_now.isoformat()

    def test_process_creates_context_if_missing(self):
        """Test that process creates context timestamps if not provided."""
        processor = MetricProcessor()
        metrics = {}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}

        result = processor.process(metrics, raw_data, context=None)

        assert 'last_updated' in result

    def test_process_extracts_old_now_from_metrics(self):
        """Test that old_now is extracted from metrics' last_updated."""
        processor = MetricProcessor()
        old_timestamp = '2025-01-01T00:00:00'
        metrics = {'last_updated': old_timestamp}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True, 'time_lapsed': 100}

        # Mock a builder to capture the context
        mock_builder = Mock(spec=MetricBuilder)
        processor.builders = [mock_builder]

        processor.process(metrics, raw_data)

        # Verify builder was called
        mock_builder.build.assert_called_once()
        call_context = mock_builder.build.call_args[0][2]

        assert 'old_now' in call_context
        assert call_context['old_now'] == datetime.fromisoformat(old_timestamp)

    def test_process_with_empty_raw_data(self):
        """Test processing with empty raw data (uses defaults)."""
        processor = MetricProcessor()
        metrics = {}
        raw_data = {}

        result = processor.process(metrics, raw_data)

        # Should still create metrics with defaults
        assert 'last_updated' in result
        assert len(result) > 1  # More than just last_updated

    def test_process_calls_all_builders(self):
        """Test that process calls all registered builders."""
        mock_builder1 = Mock(spec=MetricBuilder)
        mock_builder2 = Mock(spec=MetricBuilder)
        mock_builder3 = Mock(spec=MetricBuilder)

        processor = MetricProcessor(builders=[mock_builder1, mock_builder2, mock_builder3])
        metrics = {}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}

        processor.process(metrics, raw_data)

        mock_builder1.build.assert_called_once()
        mock_builder2.build.assert_called_once()
        mock_builder3.build.assert_called_once()

    def test_process_passes_same_telemetry_data_to_all_builders(self):
        """Test that all builders receive the same telemetry data."""
        mock_builder1 = Mock(spec=MetricBuilder)
        mock_builder2 = Mock(spec=MetricBuilder)

        processor = MetricProcessor(builders=[mock_builder1, mock_builder2])
        metrics = {}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}

        processor.process(metrics, raw_data)

        # Extract telemetry_data passed to each builder
        telemetry_data1 = mock_builder1.build.call_args[0][1]
        telemetry_data2 = mock_builder2.build.call_args[0][1]

        # Should be the same data
        assert telemetry_data1 == telemetry_data2

    def test_process_modifies_metrics_in_place(self):
        """Test that process modifies the metrics dict in place."""
        processor = MetricProcessor()
        metrics = {}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}

        result = processor.process(metrics, raw_data)

        # Result should be the same object as input metrics
        assert result is metrics


class TestMetricProcessorIntegration:
    """Integration tests for MetricProcessor."""

    def test_full_processing_pipeline(self):
        """Test complete processing pipeline with real data."""
        processor = MetricProcessor()
        metrics = {}

        # Simulate multiple telemetry events
        events = [
            {'version': '5.3.0', 'network': 'resnet50', 'action': 'train',
             'success': True, 'gpu': ['NVIDIA A100', 'NVIDIA A100'], 'time_lapsed': 3600},
            {'version': '5.3.0', 'network': 'resnet50', 'action': 'train',
             'success': False, 'gpu': ['NVIDIA V100'], 'time_lapsed': 1800},
            {'version': '5.3.0', 'network': 'yolov4', 'action': 'evaluate',
             'success': True, 'gpu': ['NVIDIA A100'], 'time_lapsed': 900},
        ]

        for event in events:
            metrics = processor.process(metrics, event)

        # Verify metrics were accumulated
        assert metrics['total_action_train_pass'] == 1
        assert metrics['total_action_train_fail'] == 1
        assert metrics['total_action_evaluate_pass'] == 1
        assert 'time_lapsed_today' in metrics
        assert metrics['time_lapsed_today'] == 6300  # Sum of all time_lapsed

    def test_custom_builder_integration(self):
        """Test integration with a custom builder."""
        class CustomCounterBuilder(MetricBuilder):
            def build(self, metrics, telemetry_data, context):
                metrics['custom_event_count'] = metrics.get('custom_event_count', 0) + 1

        processor = MetricProcessor()
        processor.add_builder(CustomCounterBuilder())

        metrics = {}
        raw_data = {'action': 'train', 'network': 'resnet50', 'success': True}

        # Process twice
        processor.process(metrics, raw_data)
        processor.process(metrics, raw_data)

        assert metrics['custom_event_count'] == 2
