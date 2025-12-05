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

"""Unit tests for telemetry metric builders."""

import pytest
from datetime import datetime
from nvidia_tao_core.telemetry.builders import (
    MetricBuilder,
    LegacyMetricsBuilder,
    ComprehensiveMetricsBuilder,
    TimeMetricsBuilder
)
from nvidia_tao_core.telemetry.types import TelemetryData


class TestMetricBuilderInterface:
    """Test cases for MetricBuilder abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that MetricBuilder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricBuilder()  # type: ignore

    def test_subclass_must_implement_build(self):
        """Test that subclasses must implement build method."""
        class IncompleteBuilder(MetricBuilder):
            pass

        with pytest.raises(TypeError):
            IncompleteBuilder()  # type: ignore


class TestLegacyMetricsBuilder:
    """Test cases for LegacyMetricsBuilder."""

    def test_build_pass_metrics(self):
        """Test building legacy metrics for successful action."""
        builder = LegacyMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['nvidia_a100'],
            'user_error': False,
            'time_lapsed': 0
        }

        builder.build(metrics, telemetry_data, {})

        assert metrics['total_action_train_pass'] == 1
        assert metrics['version_5_3_0_action_train'] == 1
        assert metrics['network_resnet50_action_train'] == 1
        assert metrics['gpu_nvidia_a100_action_train'] == 1

    def test_build_fail_metrics(self):
        """Test building legacy metrics for failed action."""
        builder = LegacyMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'export',
            'network': 'yolov4',
            'success': False,
            'gpus': ['nvidia_v100'],
            'user_error': False,
            'time_lapsed': 0
        }

        builder.build(metrics, telemetry_data, {})

        assert metrics['total_action_export_fail'] == 1
        assert metrics['version_5_3_0_action_export'] == 1
        assert metrics['network_yolov4_action_export'] == 1
        assert metrics['gpu_nvidia_v100_action_export'] == 1

    def test_incremental_updates(self):
        """Test that metrics increment correctly on multiple builds."""
        builder = LegacyMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['nvidia_a100'],
            'user_error': False,
            'time_lapsed': 0
        }

        # Build 3 times
        for _ in range(3):
            builder.build(metrics, telemetry_data, {})

        assert metrics['total_action_train_pass'] == 3
        assert metrics['version_5_3_0_action_train'] == 3
        assert metrics['network_resnet50_action_train'] == 3
        assert metrics['gpu_nvidia_a100_action_train'] == 3

    def test_multiple_gpus(self):
        """Test building metrics with multiple GPUs."""
        builder = LegacyMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['nvidia_a100', 'nvidia_v100', 'nvidia_a100'],
            'user_error': False,
            'time_lapsed': 0
        }

        builder.build(metrics, telemetry_data, {})

        # Each GPU should be counted separately
        assert metrics['gpu_nvidia_a100_action_train'] == 2
        assert metrics['gpu_nvidia_v100_action_train'] == 1


class TestComprehensiveMetricsBuilder:
    """Test cases for ComprehensiveMetricsBuilder."""

    def test_build_comprehensive_metric_pass(self):
        """Test building comprehensive metric for successful action."""
        builder = ComprehensiveMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA A100', 'NVIDIA A100'],
            'time_lapsed': 0
        }

        builder.build(metrics, telemetry_data, {})

        expected_key = "network_resnet50_action_train_version_5_3_0_status_pass_gpu_2_NVIDIA_A100_2"
        assert expected_key in metrics
        assert metrics[expected_key] == 1

    def test_build_comprehensive_metric_fail(self):
        """Test building comprehensive metric for failed action."""
        builder = ComprehensiveMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '6_0_0',
            'action': 'export',
            'network': 'yolov4',
            'success': False,
            'user_error': True,
            'gpus': ['NVIDIA V100'],
            'time_lapsed': 0
        }

        builder.build(metrics, telemetry_data, {})

        expected_key = "network_yolov4_action_export_version_6_0_0_status_fail_gpu_1_NVIDIA_V100_1"
        assert expected_key in metrics
        assert metrics[expected_key] == 1

    def test_comprehensive_metric_increments(self):
        """Test that comprehensive metrics increment correctly."""
        builder = ComprehensiveMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA A100'],
            'time_lapsed': 0
        }

        # Build twice with same data
        builder.build(metrics, telemetry_data, {})
        builder.build(metrics, telemetry_data, {})

        # Should have incremented
        keys = list(metrics.keys())
        assert len(keys) == 1
        assert metrics[keys[0]] == 2

    def test_comprehensive_metric_with_mixed_gpus(self):
        """Test comprehensive metric with mixed GPU types."""
        builder = ComprehensiveMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA A100', 'NVIDIA V100', 'NVIDIA A100'],
            'time_lapsed': 0
        }

        builder.build(metrics, telemetry_data, {})

        # Find the comprehensive metric
        comprehensive_keys = [k for k in metrics.keys() if k.startswith('network_resnet50')]
        assert len(comprehensive_keys) == 1

        metric_name = comprehensive_keys[0]
        assert "gpu_3_" in metric_name
        assert "NVIDIA_A100_2" in metric_name
        assert "NVIDIA_V100_1" in metric_name

    def test_metric_name_ordering(self):
        """Test that comprehensive metric components are in correct order."""
        builder = ComprehensiveMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA A100'],
            'time_lapsed': 0
        }

        builder.build(metrics, telemetry_data, {})

        metric_name = list(metrics.keys())[0]
        parts = metric_name.split('_')

        # Find indices of key components
        network_idx = parts.index('network')
        action_idx = parts.index('action')
        version_idx = parts.index('version')
        status_idx = parts.index('status')
        gpu_idx = parts.index('gpu')

        # Check ordering: network < action < version < status < gpu
        assert network_idx < action_idx < version_idx < status_idx < gpu_idx


class TestTimeMetricsBuilder:
    """Test cases for TimeMetricsBuilder."""

    def test_build_time_metrics_same_day(self):
        """Test time metrics accumulation on the same day."""
        builder = TimeMetricsBuilder()
        metrics = {'time_lapsed_today': 100}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['NVIDIA A100'],
            'user_error': False,
            'time_lapsed': 50
        }

        now = datetime(2025, 1, 15, 14, 30)
        old_now = datetime(2025, 1, 15, 10, 0)
        context = {'now': now, 'old_now': old_now}

        builder.build(metrics, telemetry_data, context)

        assert metrics['time_lapsed_today'] == 150  # 100 + 50

    def test_build_time_metrics_new_day(self):
        """Test that time metrics reset on new day."""
        builder = TimeMetricsBuilder()
        metrics = {'time_lapsed_today': 1000}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['NVIDIA A100'],
            'user_error': False,
            'time_lapsed': 50
        }

        now = datetime(2025, 1, 16, 1, 0)
        old_now = datetime(2025, 1, 15, 23, 0)
        context = {'now': now, 'old_now': old_now}

        builder.build(metrics, telemetry_data, context)

        assert metrics['time_lapsed_today'] == 50  # Reset to new day's value

    def test_build_time_metrics_first_entry(self):
        """Test time metrics with first entry (no existing time_lapsed_today)."""
        builder = TimeMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['NVIDIA A100'],
            'user_error': False,
            'time_lapsed': 100
        }

        now = datetime(2025, 1, 15, 14, 30)
        old_now = datetime(2025, 1, 15, 10, 0)
        context = {'now': now, 'old_now': old_now}

        builder.build(metrics, telemetry_data, context)

        assert metrics['time_lapsed_today'] == 100

    def test_build_time_metrics_zero_time(self):
        """Test time metrics with zero time elapsed."""
        builder = TimeMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['NVIDIA A100'],
            'user_error': False,
            'time_lapsed': 0
        }

        now = datetime(2025, 1, 15, 14, 30)
        old_now = datetime(2025, 1, 15, 10, 0)
        context = {'now': now, 'old_now': old_now}

        builder.build(metrics, telemetry_data, context)

        assert metrics['time_lapsed_today'] == 0

    def test_build_without_context_timestamps(self):
        """Test that time builder handles missing context timestamps."""
        builder = TimeMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'gpus': ['NVIDIA A100'],
            'user_error': False,
            'time_lapsed': 100
        }

        # Context without timestamps - should use current time
        context = {}

        builder.build(metrics, telemetry_data, context)

        assert 'time_lapsed_today' in metrics
        assert metrics['time_lapsed_today'] == 100


class TestBuilderIntegration:
    """Integration tests for using multiple builders together."""

    def test_all_builders_together(self):
        """Test using all builders together on same data."""
        legacy_builder = LegacyMetricsBuilder()
        comprehensive_builder = ComprehensiveMetricsBuilder()
        time_builder = TimeMetricsBuilder()

        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA A100'],
            'time_lapsed': 100
        }
        context = {'now': datetime.now(), 'old_now': datetime.now()}

        legacy_builder.build(metrics, telemetry_data, context)
        comprehensive_builder.build(metrics, telemetry_data, context)
        time_builder.build(metrics, telemetry_data, context)

        # Should have metrics from all builders
        assert 'total_action_train_pass' in metrics  # Legacy
        assert any('network_resnet50' in k for k in metrics.keys())  # Comprehensive
        assert 'time_lapsed_today' in metrics  # Time
