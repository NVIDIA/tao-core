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

"""Metric builders for processing telemetry data.

This package contains different metric builders that can be composed together
to generate various types of metrics from telemetry data.
"""

from nvidia_tao_core.telemetry.builders.base import MetricBuilder
from nvidia_tao_core.telemetry.builders.comprehensive import ComprehensiveMetricsBuilder
from nvidia_tao_core.telemetry.builders.labeled import LabeledMetricsBuilder
from nvidia_tao_core.telemetry.builders.legacy import LegacyMetricsBuilder
from nvidia_tao_core.telemetry.builders.time import TimeMetricsBuilder

__all__ = [
    'MetricBuilder',
    'LegacyMetricsBuilder',
    'ComprehensiveMetricsBuilder',
    'TimeMetricsBuilder',
    'LabeledMetricsBuilder',
]
