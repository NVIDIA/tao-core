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

"""Labeled metrics builder using Prometheus-style labels.

This builder creates metrics with labels (key-value pairs) instead of
encoding all attributes in the metric name. This is the recommended
Prometheus approach for multi-dimensional data.

Benefits:
- Shorter metric names
- Flexible querying in Grafana/PromQL
- Better performance in Prometheus
- Easy aggregation and filtering
- Scalable as attributes grow

Example metrics:
    tao_job_total{
        tao_network="resnet50",
        tao_action="train",
        tao_version="5_3_0",
        tao_status="pass",
        tao_primary_gpu="A100",
        tao_gpu_count="2"
    } = 1
    tao_job_duration_sum{
        tao_network="resnet50",
        tao_action="train",
        tao_version="5_3_0",
        tao_status="pass",
        tao_primary_gpu="A100",
        tao_gpu_count="2"
    } = 3600
    tao_job_gpu_time_sum{
        tao_network="resnet50",
        tao_action="train",
        tao_version="5_3_0",
        tao_status="pass",
        tao_primary_gpu="A100",
        tao_gpu_count="2"
    } = 7200
    tao_job_gpu_total{tao_gpu_type="A100"} = 2
"""

from typing import Any, Dict

from nvidia_tao_core.telemetry.builders.base import MetricBuilder
from nvidia_tao_core.telemetry.config import METRIC_ATTRIBUTES
from nvidia_tao_core.telemetry.types import AttributeType, TelemetryData


class LabeledMetricsBuilder(MetricBuilder):
    """Builder for labeled metrics using Prometheus-style labels.

    Creates metrics with a simple base name and labels for each dimension,
    following Prometheus best practices.

    Metrics generated:
    - tao_job_total: Counter of jobs with labels (includes primary_gpu, gpu_count)
    - tao_job_duration_sum: Sum of job durations (use with tao_job_total to calculate average)
    - tao_job_gpu_time_sum: Sum of GPU-seconds (gpu_count × duration, for GPU-hour calculations)
    - tao_job_gpu_total: Total GPU usage by GPU type (tracks all GPUs including mixed jobs)
    """

    def __init__(self):
        """Initialize the labeled metrics builder."""
        # Base metric names following Prometheus naming conventions
        self.metric_names = {
            'counter': 'tao_job_total',
            'duration_sum': 'tao_job_duration_sum',
            'gpu_time_sum': 'tao_job_gpu_time_sum',
            'gpu_total': 'tao_job_gpu_total'
        }

        # GPU priority order (newest to oldest) based on compute capability
        # Reference: https://developer.nvidia.com/cuda-gpus
        # Used to select primary GPU in mixed-GPU jobs
        self.gpu_priority = [
            # Compute Capability 12.0 - Blackwell Consumer/Pro
            'RTX5090', 'RTX5080', 'RTX5070', 'RTX5060', 'RTX5050',
            'RTX6000', 'RTX5000', 'RTX4500', 'RTX4000', 'RTX2000',  # RTX PRO Blackwell

            # Compute Capability 11.0 - Jetson Thor
            'T5000', 'T4000',

            # Compute Capability 10.3 - Grace Blackwell Ultra
            'GB300', 'B300',

            # Compute Capability 10.0 - Grace Blackwell
            'GB200', 'B200',

            # Compute Capability 9.0 - Grace Hopper / Hopper
            'GH200', 'H200', 'H100',

            # Compute Capability 8.9 - Ada Lovelace (Data Center & Consumer)
            'L40S', 'L40', 'L4',
            'RTX4090', 'RTX4080', 'RTX4070', 'RTX4060', 'RTX4050',  # GeForce RTX 40 series
            'RTXA6000', 'RTXA5000', 'RTXA4500', 'RTXA4000', 'RTXA2000',  # RTX Ada Pro

            # Compute Capability 8.7 - Jetson Orin
            'ORIN',

            # Compute Capability 8.6 - Ampere (Consumer/Workstation)
            'A40', 'A16', 'A10', 'A2',
            'RTX3090', 'RTX3080', 'RTX3070', 'RTX3060', 'RTX3050',  # GeForce RTX 30 series

            # Compute Capability 8.0 - Ampere (Data Center)
            'A100', 'A30',

            # Compute Capability 7.5 - Turing
            'T4', 'T2000', 'T1200', 'T1000', 'T600', 'T500', 'T400',
            'RTX2080', 'RTX2070', 'RTX2060',  # GeForce RTX 20 series
            'QUADRO8000', 'QUADRO6000', 'QUADRO5000', 'QUADRO4000',  # Quadro RTX

            # Compute Capability 7.0 - Volta
            'V100', 'TITAN',

            # Compute Capability 6.x - Pascal
            'P100', 'P40', 'P6', 'P4',
            'GTX1080', 'GTX1070', 'GTX1060', 'GTX1050',

            # Older (in case encountered)
            'K80', 'K40', 'K20', 'M60', 'M40',
        ]

    def build(
        self,
        metrics: Dict[str, Any],
        telemetry_data: TelemetryData,
        context: Dict[str, Any]
    ) -> None:
        """Build labeled metrics from telemetry data.

        Args:
            metrics: Metrics dictionary to update
            telemetry_data: Normalized telemetry data
            context: Additional context
        """
        # Extract GPU information
        gpus = telemetry_data.get('gpus', [])
        gpu_count = len(gpus)
        time_lapsed = telemetry_data.get('time_lapsed', 0)

        # Extract labels from telemetry data (includes primary_gpu and gpu_count)
        labels = self._build_labels(telemetry_data, gpus)

        # 1. Job counter - main metric
        counter_key = self._build_metric_key(self.metric_names['counter'], labels)
        metrics[counter_key] = metrics.get(counter_key, 0) + 1

        # 2. Duration sum (accumulates across all jobs with same labels)
        if time_lapsed > 0:
            # Use same labels as job counter for easy division (average = sum / count)
            duration_sum_key = self._build_metric_key(
                self.metric_names['duration_sum'],
                labels
            )
            # Accumulate duration (counter behavior)
            metrics[duration_sum_key] = metrics.get(duration_sum_key, 0) + time_lapsed

            # 3. GPU-time sum (gpu_count × duration, pre-calculated for GPU-hour calculations)
            gpu_time = gpu_count * time_lapsed  # GPU-seconds
            gpu_time_sum_key = self._build_metric_key(
                self.metric_names['gpu_time_sum'],
                labels
            )
            metrics[gpu_time_sum_key] = metrics.get(gpu_time_sum_key, 0) + gpu_time

        # 4. GPU total counter (per GPU type, tracks all GPUs including mixed jobs)
        gpu_type_counts = self._count_gpu_types(gpus)
        for gpu_type, count in gpu_type_counts.items():
            gpu_total_key = self._build_metric_key(
                self.metric_names['gpu_total'],
                {'tao_gpu_type': gpu_type}
            )
            metrics[gpu_total_key] = metrics.get(gpu_total_key, 0) + count

    def _build_labels(self, telemetry_data: TelemetryData, gpus: list) -> Dict[str, str]:
        """Build label dictionary from telemetry data.

        Args:
            telemetry_data: Normalized telemetry data
            gpus: List of GPU names (passed separately for efficiency)

        Returns:
            Dictionary of label key-value pairs
        """
        labels: Dict[str, str] = {}

        # Add standard labels from configured attributes
        for attr in METRIC_ATTRIBUTES:
            if attr.name not in telemetry_data:
                continue

            # Skip fields that are handled separately or are metric values, not labels
            if attr.name in ('success', 'time_lapsed'):
                continue

            # Convert attribute to label
            value = telemetry_data[attr.name]

            if attr.attr_type == AttributeType.BOOLEAN:
                labels[f'tao_{attr.name}'] = str(value).lower()
            elif attr.attr_type == AttributeType.LIST:
                # Lists handled separately (e.g., GPUs)
                continue
            else:
                labels[f'tao_{attr.name}'] = str(value)

        # Add derived status label (from success field)
        success = telemetry_data.get('success', False)
        labels['tao_status'] = 'pass' if success else 'fail'

        # Add GPU labels
        if gpus:
            # Primary GPU (most recent/modern, or most common if same generation)
            primary_gpu = self._extract_primary_gpu(gpus)
            labels['tao_primary_gpu'] = primary_gpu

            # GPU count
            labels['tao_gpu_count'] = str(len(gpus))
        else:
            labels['tao_primary_gpu'] = 'unknown'
            labels['tao_gpu_count'] = '0'

        return labels

    def _extract_primary_gpu(self, gpus: list) -> str:
        """Extract primary GPU from list, prioritizing newer GPUs.

        For mixed-GPU jobs (e.g., 2x A100 + 1x V100), selects the newest GPU type.
        For homogeneous jobs, returns that type.

        Priority order (by compute capability):
        - CC 12.0: RTX 5090/5080 (Blackwell consumer), RTX PRO 6000 Blackwell
        - CC 11.0: Jetson T5000/T4000
        - CC 10.3: GB300, B300
        - CC 10.0: GB200, B200
        - CC 9.0: GH200, H200, H100 (Hopper)
        - CC 8.9: L40S, L40, L4, RTX 4090/4080 (Ada), RTX A6000 Ada
        - CC 8.7: Jetson Orin
        - CC 8.6: A40, A10, RTX 3090/3080 (Ampere consumer)
        - CC 8.0: A100, A30 (Ampere datacenter)
        - CC 7.5: T4, RTX 2080 (Turing)
        - CC 7.0: V100 (Volta)
        - CC 6.x: P100 (Pascal)

        Reference: https://developer.nvidia.com/cuda-gpus

        Args:
            gpus: List of GPU names

        Returns:
            Primary GPU type (newest available in the job)

        Examples:
            >>> _extract_primary_gpu(['NVIDIA A100', 'NVIDIA A100'])
            'A100'
            >>> _extract_primary_gpu(['NVIDIA A100', 'NVIDIA V100', 'NVIDIA A100'])
            'A100'  # A100 is newer than V100
            >>> _extract_primary_gpu(['NVIDIA H100', 'NVIDIA A100'])
            'H100'  # H100 is newest
        """
        if not gpus:
            return 'unknown'

        # Extract all GPU types
        gpu_types = [self._extract_gpu_type(gpu) for gpu in gpus]
        unique_types = set(gpu_types)

        # Select based on priority (newest first)
        for priority_gpu in self.gpu_priority:
            if priority_gpu in unique_types:
                return priority_gpu

        # If no match in priority list, use most common
        # (Fallback for unknown GPU types)
        gpu_counts = {}
        for gpu_type in gpu_types:
            gpu_counts[gpu_type] = gpu_counts.get(gpu_type, 0) + 1

        # Return most common (or first if tied)
        primary = max(gpu_counts.items(), key=lambda x: (x[1], x[0]))[0]
        return primary

    def _count_gpu_types(self, gpus: list) -> Dict[str, int]:
        """Count each GPU type in the list.

        Args:
            gpus: List of GPU names (e.g., ['NVIDIA A100', 'NVIDIA A100', 'NVIDIA V100'])

        Returns:
            Dictionary mapping GPU type to count (e.g., {'A100': 2, 'V100': 1})
        """
        gpu_counts: Dict[str, int] = {}
        for gpu in gpus:
            gpu_type = self._extract_gpu_type(gpu)
            gpu_counts[gpu_type] = gpu_counts.get(gpu_type, 0) + 1
        return gpu_counts

    def _extract_gpu_type(self, gpu_name: str) -> str:
        """Extract GPU type from GPU name.

        Extracts the model identifier from full GPU name, handling various formats
        including datacenter GPUs, consumer GPUs, RTX series, and Jetson devices.

        Args:
            gpu_name: Full GPU name (e.g., 'NVIDIA A100 40GB', 'GeForce RTX 4090', 'Jetson Orin')

        Returns:
            Simplified GPU type (e.g., 'A100', 'RTX4090', 'GH200', 'ORIN')

        Examples:
            >>> _extract_gpu_type('NVIDIA A100 40GB')
            'A100'
            >>> _extract_gpu_type('GeForce RTX 4090')
            'RTX4090'
            >>> _extract_gpu_type('NVIDIA GH200')
            'GH200'
            >>> _extract_gpu_type('NVIDIA RTX A6000')
            'RTXA6000'
            >>> _extract_gpu_type('Jetson AGX Orin')
            'ORIN'
            >>> _extract_gpu_type('NVIDIA L40S')
            'L40S'
        """
        # Normalize: uppercase, replace spaces and dashes with underscores
        gpu_upper = str(gpu_name).upper().replace(' ', '_').replace('-', '_')

        # Special handling for specific GPU families
        if 'SPARK' in gpu_upper:
            return 'SPARK'
        # Grace Blackwell / Grace Hopper (GB/GH prefix)
        for model in ['GB300', 'GB200', 'GB100', 'GH200', 'GH100']:
            if model in gpu_upper:
                return model

        # Blackwell / Hopper (B200, B300, H200, H100)
        for model in ['B300', 'B200', 'B100', 'H200', 'H100']:
            if model in gpu_upper:
                return model

        # RTX series (consumer and pro)
        # RTX 5090, RTX 4090, RTX A6000, etc.
        if 'RTX' in gpu_upper:
            # Extract RTX model
            parts = gpu_upper.split('_')
            for i, part in enumerate(parts):
                if part == 'RTX' and i + 1 < len(parts):
                    # Next part is the model (e.g., '5090', 'A6000', '4090')
                    model = parts[i + 1]
                    # Remove 'TI' suffix if present
                    model = model.replace('TI', '')
                    # Remove 'ADA' suffix if present (e.g., "RTX 6000 Ada")
                    model = model.replace('ADA', '')
                    return f'RTX{model}'

        # Jetson and embedded series (Orin, Xavier, Nano, Thor, etc.)
        if 'JETSON' in gpu_upper or 'THOR' in gpu_upper or 'ORIN' in gpu_upper:
            # Jetson T5000, T4000 (e.g., Jetson Thor models)
            for part in gpu_upper.split('_'):
                if part.startswith('T') and any(c.isdigit() for c in part):
                    return part
            if 'THOR' in gpu_upper:
                return 'THOR'
            if 'ORIN' in gpu_upper:
                return 'ORIN'

        # L-series (L40S, L40, L4)
        if 'L40S' in gpu_upper:
            return 'L40S'
        if 'L40' in gpu_upper:
            return 'L40'
        if 'L4' in gpu_upper:
            return 'L4'

        # Standard datacenter/consumer GPUs (A100, V100, T4, P100, etc.)
        # Find part with both letters and numbers
        parts = gpu_upper.split('_')
        for part in parts:
            # Skip common prefixes
            if part in ['NVIDIA', 'GEFORCE', 'QUADRO', 'TESLA', 'JETSON', 'TITAN']:
                continue

            # Look for model identifier (has both letters and numbers)
            if any(c.isdigit() for c in part) and any(c.isalpha() for c in part):
                # Found model like A100, V100, T4, P100, K80, etc.
                return part

        # If still no match, return first non-common part
        for part in parts:
            if part and part not in ['NVIDIA', 'GEFORCE', 'QUADRO', 'TESLA']:
                return part

        # Fallback to sanitized original name
        return gpu_upper.replace('NVIDIA_', '').replace('GEFORCE_', '')

    def _build_metric_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """Build Prometheus-style metric key with labels.

        Creates metric key in format: metric_name{label1="value1",label2="value2"}
        Labels are sorted alphabetically for consistency.

        Args:
            metric_name: Base metric name (e.g., 'tao_job_total')
            labels: Dictionary of label key-value pairs

        Returns:
            Metric key string in Prometheus format

        Examples:
            >>> _build_metric_key("tao_job_total", {"tao_action": "train", "tao_network": "resnet50"})
            'tao_job_total{tao_action="train",tao_network="resnet50"}'
        """
        if not labels:
            return metric_name

        # Sort labels alphabetically for consistency
        sorted_labels = sorted(labels.items())
        label_str = ','.join(f'{k}="{v}"' for k, v in sorted_labels)

        return f'{metric_name}{{{label_str}}}'
