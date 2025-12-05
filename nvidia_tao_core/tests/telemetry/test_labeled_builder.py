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

"""Unit tests for LabeledMetricsBuilder."""

from nvidia_tao_core.telemetry.builders.labeled import LabeledMetricsBuilder
from nvidia_tao_core.telemetry.types import TelemetryData


class TestLabeledMetricsBuilder:
    """Test cases for LabeledMetricsBuilder."""

    def test_build_job_total_metric(self):
        """Test building tao_job_total metric with labels."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100', 'NVIDIA_A100'],
            'time_lapsed': 3600,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Check job total metric exists with primary_gpu and gpu_count
        expected_key = (
            'tao_job_total{tao_action="train",tao_automl_triggered="false",'
            'tao_client_type="container",tao_gpu_count="2",'
            'tao_network="resnet50",tao_primary_gpu="A100",'
            'tao_status="pass",tao_user_error="false",tao_version="5_3_0"}'
        )
        assert expected_key in metrics
        assert metrics[expected_key] == 1

    def test_build_duration_sum_metric(self):
        """Test building tao_job_duration_sum metric."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 3600,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Check duration sum metric (same labels as tao_job_total)
        expected_key = (
            'tao_job_duration_sum{tao_action="train",tao_automl_triggered="false",'
            'tao_client_type="container",tao_gpu_count="1",'
            'tao_network="resnet50",tao_primary_gpu="A100",'
            'tao_status="pass",tao_user_error="false",tao_version="5_3_0"}'
        )
        assert expected_key in metrics
        assert metrics[expected_key] == 3600

    def test_build_gpu_time_sum_metric(self):
        """Test building tao_job_gpu_time_sum metric."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100', 'NVIDIA_A100'],
            'time_lapsed': 3600,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Check GPU-time sum metric (gpu_count × duration = 2 × 3600 = 7200)
        expected_key = (
            'tao_job_gpu_time_sum{tao_action="train",tao_automl_triggered="false",'
            'tao_client_type="container",tao_gpu_count="2",'
            'tao_network="resnet50",tao_primary_gpu="A100",'
            'tao_status="pass",tao_user_error="false",tao_version="5_3_0"}'
        )
        assert expected_key in metrics
        assert metrics[expected_key] == 7200  # 2 GPUs × 3600 seconds

    def test_duration_sum_accumulates(self):
        """Test that duration sum accumulates across multiple jobs with same labels."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 3600,
            'client_type': 'container',
            'automl_triggered': False
        }

        # Build first job
        builder.build(metrics, telemetry_data, {})

        # Build second job with same labels but different duration
        telemetry_data['time_lapsed'] = 1800
        builder.build(metrics, telemetry_data, {})

        # Duration sum should accumulate
        duration_key = [
            k for k in metrics.keys() if k.startswith('tao_job_duration_sum')
        ][0]
        assert metrics[duration_key] == 5400  # 3600 + 1800

        # GPU-time sum should also accumulate (1 GPU each job)
        gpu_time_key = [
            k for k in metrics.keys() if k.startswith('tao_job_gpu_time_sum')
        ][0]
        assert metrics[gpu_time_key] == 5400  # (1×3600) + (1×1800)

        # Job total should also increment
        job_key = [
            k for k in metrics.keys() if k.startswith('tao_job_total')
        ][0]
        assert metrics[job_key] == 2

    def test_build_gpu_total_metric(self):
        """Test building tao_job_gpu_total metric."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100', 'NVIDIA_A100'],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Check GPU total metric (only gpu_type label)
        expected_key = 'tao_job_gpu_total{tao_gpu_type="A100"}'
        assert expected_key in metrics
        assert metrics[expected_key] == 2  # Two GPUs

    def test_extract_gpu_type(self):
        """Test GPU type extraction from various formats."""
        builder = LabeledMetricsBuilder()

        # Datacenter GPUs
        assert builder._extract_gpu_type('NVIDIA_A100_40GB') == 'A100'
        assert builder._extract_gpu_type('NVIDIA_A100') == 'A100'
        assert builder._extract_gpu_type('nvidia_v100') == 'V100'
        assert builder._extract_gpu_type('H100') == 'H100'
        assert builder._extract_gpu_type('NVIDIA_V100_32GB') == 'V100'
        assert builder._extract_gpu_type('NVIDIA H200 80GB') == 'H200'
        assert builder._extract_gpu_type('NVIDIA_H100_80GB') == 'H100'
        assert builder._extract_gpu_type('NVIDIA_A40') == 'A40'
        assert builder._extract_gpu_type('NVIDIA_A30') == 'A30'

        # Grace Blackwell / Grace Hopper
        assert builder._extract_gpu_type('NVIDIA GB200') == 'GB200'
        assert builder._extract_gpu_type('NVIDIA GB300') == 'GB300'
        assert builder._extract_gpu_type('NVIDIA GH200') == 'GH200'

        # RTX series (consumer)
        assert builder._extract_gpu_type('GeForce RTX 4090') == 'RTX4090'
        assert builder._extract_gpu_type('GeForce RTX 5090') == 'RTX5090'
        assert builder._extract_gpu_type('NVIDIA RTX 3080 Ti') == 'RTX3080'

        # RTX Pro / Ada (various formats)
        assert builder._extract_gpu_type('NVIDIA RTX A6000') == 'RTXA6000'
        assert builder._extract_gpu_type('NVIDIA-RTX-A6000') == 'RTXA6000'  # With dashes
        assert builder._extract_gpu_type('NVIDIA RTX 6000 Ada') == 'RTX6000'
        assert builder._extract_gpu_type('NVIDIA-RTX-6000-Ada') == 'RTX6000'  # With dashes

        # L-series
        assert builder._extract_gpu_type('NVIDIA L40S') == 'L40S'
        assert builder._extract_gpu_type('NVIDIA L40') == 'L40'
        assert builder._extract_gpu_type('NVIDIA L4') == 'L4'

        # Jetson
        assert builder._extract_gpu_type('Jetson AGX Orin') == 'ORIN'
        assert builder._extract_gpu_type('Jetson Orin Nano') == 'ORIN'

        # Older GPUs
        assert builder._extract_gpu_type('NVIDIA T4') == 'T4'
        assert builder._extract_gpu_type('Tesla P100') == 'P100'

    def test_extract_primary_gpu_homogeneous(self):
        """Test primary GPU extraction for homogeneous GPU jobs."""
        builder = LabeledMetricsBuilder()

        # Single type
        assert builder._extract_primary_gpu(
            ['NVIDIA_A100', 'NVIDIA_A100', 'NVIDIA_A100']
        ) == 'A100'
        assert builder._extract_primary_gpu(['NVIDIA_V100']) == 'V100'

    def test_extract_primary_gpu_mixed_prioritizes_newer(self):
        """Test that primary GPU prioritizes newer GPUs in mixed jobs."""
        builder = LabeledMetricsBuilder()

        # A100 is newer than V100 → should return A100
        assert builder._extract_primary_gpu(
            ['NVIDIA_A100', 'NVIDIA_V100', 'NVIDIA_V100']
        ) == 'A100'

        # H100 is newest → should return H100
        assert builder._extract_primary_gpu(
            ['NVIDIA_V100', 'NVIDIA_A100', 'NVIDIA_H100']
        ) == 'H100'

        # Even if A100 is more common, H100 is newer
        assert builder._extract_primary_gpu(
            ['NVIDIA_A100', 'NVIDIA_A100', 'NVIDIA_A100', 'NVIDIA_H100']
        ) == 'H100'

    def test_extract_primary_gpu_priority_order(self):
        """Test GPU priority order from newest to oldest based on compute capability."""
        builder = LabeledMetricsBuilder()

        # GB200 > GH200 > H200
        assert builder._extract_primary_gpu(
            ['NVIDIA_GB200', 'NVIDIA_GH200']
        ) == 'GB200'
        assert builder._extract_primary_gpu(
            ['NVIDIA_GH200', 'NVIDIA_H200']
        ) == 'GH200'

        # H200 > H100
        assert builder._extract_primary_gpu(
            ['NVIDIA_H200', 'NVIDIA_H100']
        ) == 'H200'

        # H100 > L40S (Hopper > Ada datacenter)
        assert builder._extract_primary_gpu(
            ['NVIDIA_H100', 'NVIDIA_L40S']
        ) == 'H100'

        # L40S > A100 (Ada > Ampere datacenter)
        assert builder._extract_primary_gpu(
            ['NVIDIA_L40S', 'NVIDIA_A100']
        ) == 'L40S'

        # A100 > V100 (Ampere > Volta)
        assert builder._extract_primary_gpu(
            ['NVIDIA_A100', 'NVIDIA_V100']
        ) == 'A100'

        # A100 > A30 (within same generation)
        assert builder._extract_primary_gpu(
            ['NVIDIA_A100', 'NVIDIA_A30']
        ) == 'A100'

        # RTX 5090 > RTX 4090 (newer generation)
        assert builder._extract_primary_gpu(
            ['GeForce_RTX_5090', 'GeForce_RTX_4090']
        ) == 'RTX5090'

        # RTX 4090 > RTX 3090
        assert builder._extract_primary_gpu(
            ['GeForce_RTX_4090', 'GeForce_RTX_3090']
        ) == 'RTX4090'

    def test_status_derived_from_success(self):
        """Test that status label is derived from success field."""
        builder = LabeledMetricsBuilder()
        metrics = {}

        # Test success=True → status=pass
        telemetry_data_pass: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data_pass, {})
        assert any('tao_status="pass"' in k for k in metrics.keys())

        # Test success=False → status=fail
        metrics = {}
        telemetry_data_fail: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': False,
            'user_error': True,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data_fail, {})
        assert any('tao_status="fail"' in k for k in metrics.keys())

    def test_gpu_count_label(self):
        """Test that gpu_count label is added."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': [
                'NVIDIA_A100', 'NVIDIA_A100', 'NVIDIA_A100', 'NVIDIA_A100'
            ],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Should have gpu_count="4"
        job_keys = [
            k for k in metrics.keys() if k.startswith('tao_job_total')
        ]
        assert len(job_keys) == 1
        assert 'tao_gpu_count="4"' in job_keys[0]

    def test_gpu_time_calculation(self):
        """Test that GPU-time is calculated correctly (gpu_count × duration)."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100', 'NVIDIA_A100', 'NVIDIA_A100'],  # 3 GPUs
            'time_lapsed': 3600  # 1 hour
        }

        builder.build(metrics, telemetry_data, {})

        # GPU-time = 3 GPUs × 3600 seconds = 10800 GPU-seconds
        gpu_time_keys = [
            k for k in metrics.keys() if k.startswith('tao_job_gpu_time_sum')
        ]
        assert len(gpu_time_keys) == 1
        assert metrics[gpu_time_keys[0]] == 10800

    def test_label_sorting(self):
        """Test that labels are sorted alphabetically in metric key."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Find the job total metric
        job_total_keys = [
            k for k in metrics.keys() if k.startswith('tao_job_total')
        ]
        assert len(job_total_keys) == 1

        key = job_total_keys[0]
        # Labels should be alphabetically sorted:
        # tao_action, tao_automl_triggered, tao_client_type, tao_gpu_count,
        # tao_network, tao_primary_gpu, tao_status, tao_user_error, tao_version
        assert key.index('tao_action') < key.index('tao_automl_triggered')
        assert key.index('tao_automl_triggered') < key.index('tao_client_type')
        assert key.index('tao_client_type') < key.index('tao_gpu_count')
        assert key.index('tao_gpu_count') < key.index('tao_network')
        assert key.index('tao_network') < key.index('tao_primary_gpu')
        assert key.index('tao_primary_gpu') < key.index('tao_status')
        assert key.index('tao_status') < key.index('tao_user_error')
        assert key.index('tao_user_error') < key.index('tao_version')

    def test_incremental_updates(self):
        """Test that labeled metrics increment correctly."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 1800,
            'client_type': 'container',
            'automl_triggered': False
        }

        # Build 3 times
        for _ in range(3):
            builder.build(metrics, telemetry_data, {})

        # Job total should increment
        job_total_key = [
            k for k in metrics.keys() if k.startswith('tao_job_total')
        ][0]
        assert metrics[job_total_key] == 3

        # Duration sum should accumulate (counter behavior)
        duration_key = [
            k for k in metrics.keys() if k.startswith('tao_job_duration_sum')
        ][0]
        assert metrics[duration_key] == 5400  # 1800 * 3

    def test_different_gpu_types(self):
        """Test metrics with different GPU types."""
        builder = LabeledMetricsBuilder()
        metrics = {}

        # A100 job
        telemetry_data_a100: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100', 'NVIDIA_A100'],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }
        builder.build(metrics, telemetry_data_a100, {})

        # V100 job
        telemetry_data_v100: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_V100'],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }
        builder.build(metrics, telemetry_data_v100, {})

        # Should have separate metrics for different primary GPUs
        a100_key = [
            k for k in metrics.keys()
            if 'tao_primary_gpu="A100"' in k and k.startswith('tao_job_total')
        ]
        v100_key = [
            k for k in metrics.keys()
            if 'tao_primary_gpu="V100"' in k and k.startswith('tao_job_total')
        ]

        assert len(a100_key) == 1
        assert len(v100_key) == 1
        assert metrics[a100_key[0]] == 1
        assert metrics[v100_key[0]] == 1

        # GPU total should track all GPUs by type
        assert 'tao_job_gpu_total{tao_gpu_type="A100"}' in metrics
        assert metrics['tao_job_gpu_total{tao_gpu_type="A100"}'] == 2  # Two A100s
        assert 'tao_job_gpu_total{tao_gpu_type="V100"}' in metrics
        assert metrics['tao_job_gpu_total{tao_gpu_type="V100"}'] == 1  # One V100

    def test_mixed_gpu_job(self):
        """Test that mixed-GPU jobs prioritize newer GPU and track all types."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100', 'NVIDIA_A100', 'NVIDIA_V100'],  # Mixed!
            'time_lapsed': 3600,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Primary GPU should be A100 (newer than V100)
        job_keys = [
            k for k in metrics.keys() if k.startswith('tao_job_total')
        ]
        assert len(job_keys) == 1
        assert 'tao_primary_gpu="A100"' in job_keys[0]
        assert 'tao_gpu_count="3"' in job_keys[0]

        # But GPU total should track BOTH types
        assert metrics['tao_job_gpu_total{tao_gpu_type="A100"}'] == 2
        assert metrics['tao_job_gpu_total{tao_gpu_type="V100"}'] == 1

        # GPU-time should be 3 × 3600 = 10800
        gpu_time_keys = [
            k for k in metrics.keys() if k.startswith('tao_job_gpu_time_sum')
        ]
        assert metrics[gpu_time_keys[0]] == 10800

    def test_no_duration_when_zero(self):
        """Test that duration metric is not created when time_lapsed is 0."""
        builder = LabeledMetricsBuilder()
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 0,
            'client_type': 'container',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data, {})

        # Duration sum metric should not be created
        duration_metrics = [
            k for k in metrics.keys() if k.startswith('tao_job_duration_sum')
        ]
        assert len(duration_metrics) == 0

    def test_average_duration_calculation(self):
        """Test that average duration can be calculated from sum and total."""
        builder = LabeledMetricsBuilder()
        metrics = {}

        # Job 1: 3600 seconds
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 3600,
            'client_type': 'container',
            'automl_triggered': False
        }
        builder.build(metrics, telemetry_data, {})

        # Job 2: 1800 seconds (same labels)
        telemetry_data['time_lapsed'] = 1800
        builder.build(metrics, telemetry_data, {})

        # Get the metrics
        duration_sum = [
            v for k, v in metrics.items() if k.startswith('tao_job_duration_sum')
        ][0]
        job_total = [
            v for k, v in metrics.items() if k.startswith('tao_job_total')
        ][0]

        # Calculate average
        average_duration = duration_sum / job_total
        assert average_duration == 2700  # (3600 + 1800) / 2

    def test_label_escaping(self):
        """Test that label values are properly quoted."""
        builder = LabeledMetricsBuilder()

        labels = {'tao_network': 'resnet50', 'tao_action': 'train'}
        key = builder._build_metric_key('tao_job_total', labels)

        # Should have quotes around values
        assert 'tao_action="train"' in key
        assert 'tao_network="resnet50"' in key
        assert key.startswith('tao_job_total{')
        assert key.endswith('}')

    def test_client_type_and_automl_labels(self):
        """Test that client_type and automl_triggered labels are correctly added."""
        builder = LabeledMetricsBuilder()

        # Test with API client and AutoML triggered
        metrics = {}
        telemetry_data: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 3600,
            'client_type': 'api',
            'automl_triggered': True
        }

        builder.build(metrics, telemetry_data, {})

        # Check that client_type and automl_triggered are in labels
        expected_key = (
            'tao_job_total{tao_action="train",tao_automl_triggered="true",'
            'tao_client_type="api",tao_gpu_count="1",'
            'tao_network="resnet50",tao_primary_gpu="A100",'
            'tao_status="pass",tao_user_error="false",tao_version="5_3_0"}'
        )
        assert expected_key in metrics
        assert metrics[expected_key] == 1

        # Test with default values (container client, no AutoML)
        metrics = {}
        telemetry_data_default: TelemetryData = {
            'version': '5_3_0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'gpus': ['NVIDIA_A100'],
            'time_lapsed': 3600,
            'client_type': 'container',  # Explicit default
            'automl_triggered': False  # Explicit default
        }

        builder.build(metrics, telemetry_data_default, {})

        # Should default to container and false
        expected_key_default = (
            'tao_job_total{tao_action="train",tao_automl_triggered="false",'
            'tao_client_type="container",tao_gpu_count="1",'
            'tao_network="resnet50",tao_primary_gpu="A100",'
            'tao_status="pass",tao_user_error="false",tao_version="5_3_0"}'
        )
        assert expected_key_default in metrics
        assert metrics[expected_key_default] == 1

        # Test with CLI client
        metrics = {}
        telemetry_data_cli: TelemetryData = {
            'version': '6_0_0',
            'action': 'evaluate',
            'network': 'dino',
            'success': False,
            'user_error': True,
            'gpus': ['NVIDIA_H100'],
            'client_type': 'cli',
            'automl_triggered': False
        }

        builder.build(metrics, telemetry_data_cli, {})

        # Check CLI client type
        expected_key_cli = (
            'tao_job_total{tao_action="evaluate",tao_automl_triggered="false",'
            'tao_client_type="cli",tao_gpu_count="1",'
            'tao_network="dino",tao_primary_gpu="H100",'
            'tao_status="fail",tao_user_error="true",tao_version="6_0_0"}'
        )
        assert expected_key_cli in metrics
        assert metrics[expected_key_cli] == 1


class TestIntegrationWithOtherBuilders:
    """Test that LabeledMetricsBuilder works alongside other builders."""

    def test_all_builders_together(self):
        """Test using all builders together produces both old and new formats."""
        from nvidia_tao_core.telemetry.processor import MetricProcessor

        processor = MetricProcessor()  # Uses default builders (including labeled)
        metrics = {}

        raw_data = {
            'version': '5.3.0',
            'network': 'ResNet-50',
            'action': 'train',
            'success': True,
            'user_error': False,
            'time_lapsed': 3600,
            'gpu': ['NVIDIA A100', 'NVIDIA A100']
        }

        result = processor.process(metrics, raw_data)

        # Should have legacy metrics
        assert 'total_action_train_pass' in result
        assert 'version_5_3_0_action_train' in result

        # Should have comprehensive metric
        comprehensive_keys = [
            k for k in result.keys() if k.startswith('network_resnet_50')
        ]
        assert len(comprehensive_keys) >= 1

        # Should have labeled metrics
        labeled_keys = [
            k for k in result.keys() if k.startswith('tao_job_total{')
        ]
        assert len(labeled_keys) >= 1

        # Should have time metric
        assert 'time_lapsed_today' in result

        # Should have timestamp
        assert 'last_updated' in result
