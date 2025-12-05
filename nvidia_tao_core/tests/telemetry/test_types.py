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

"""Unit tests for telemetry types module."""

from nvidia_tao_core.telemetry.types import AttributeType, MetricAttribute, TelemetryData


class TestAttributeType:
    """Test cases for AttributeType enum."""

    def test_attribute_types_exist(self):
        """Test that all expected attribute types exist."""
        assert AttributeType.STRING.value == "string"
        assert AttributeType.BOOLEAN.value == "boolean"
        assert AttributeType.INTEGER.value == "integer"
        assert AttributeType.LIST.value == "list"

    def test_attribute_type_values(self):
        """Test that attribute type values are strings."""
        for attr_type in AttributeType:
            assert isinstance(attr_type.value, str)


class TestMetricAttribute:
    """Test cases for MetricAttribute dataclass."""

    def test_metric_attribute_creation(self):
        """Test creating a MetricAttribute with all required fields."""
        attr = MetricAttribute(
            name='test_attr',
            raw_key='test',
            attr_type=AttributeType.STRING,
            default='default_value'
        )

        assert attr.name == 'test_attr'
        assert attr.raw_key == 'test'
        assert attr.attr_type == AttributeType.STRING
        assert attr.default == 'default_value'
        assert attr.include_in_comprehensive is True  # Default
        assert attr.metric_order == 100  # Default

    def test_metric_attribute_with_custom_fields(self):
        """Test creating a MetricAttribute with custom optional fields."""
        attr = MetricAttribute(
            name='custom',
            raw_key='custom_key',
            attr_type=AttributeType.BOOLEAN,
            default=False,
            include_in_comprehensive=False,
            metric_order=5
        )

        assert attr.include_in_comprehensive is False
        assert attr.metric_order == 5

    def test_metric_attribute_integer_type(self):
        """Test MetricAttribute with INTEGER type."""
        attr = MetricAttribute(
            name='count',
            raw_key='count',
            attr_type=AttributeType.INTEGER,
            default=0
        )

        assert attr.attr_type == AttributeType.INTEGER
        assert attr.default == 0

    def test_metric_attribute_list_type(self):
        """Test MetricAttribute with LIST type."""
        attr = MetricAttribute(
            name='items',
            raw_key='items',
            attr_type=AttributeType.LIST,
            default=[]
        )

        assert attr.attr_type == AttributeType.LIST
        assert attr.default == []


class TestTelemetryData:
    """Test cases for TelemetryData TypedDict."""

    def test_telemetry_data_complete(self):
        """Test creating TelemetryData with all fields."""
        data: TelemetryData = {
            'version': '5.3.0',
            'action': 'train',
            'network': 'resnet50',
            'success': True,
            'user_error': False,
            'time_lapsed': 3600,
            'gpus': ['NVIDIA A100']
        }

        assert data['version'] == '5.3.0'
        assert data['action'] == 'train'
        assert data['network'] == 'resnet50'
        assert data['success'] is True
        assert data['user_error'] is False
        assert data['time_lapsed'] == 3600
        assert data['gpus'] == ['NVIDIA A100']

    def test_telemetry_data_partial(self):
        """Test creating TelemetryData with partial fields (total=False)."""
        data: TelemetryData = {
            'action': 'evaluate',
            'network': 'yolov4'
        }

        assert data['action'] == 'evaluate'
        assert data['network'] == 'yolov4'
        assert 'version' not in data

    def test_telemetry_data_empty(self):
        """Test creating empty TelemetryData."""
        data: TelemetryData = {}
        assert len(data) == 0

    def test_telemetry_data_types(self):
        """Test that TelemetryData enforces expected types."""
        data: TelemetryData = {
            'version': '5.3.0',  # str
            'action': 'train',  # str
            'network': 'resnet50',  # str
            'success': True,  # bool
            'user_error': False,  # bool
            'time_lapsed': 3600,  # int
            'gpus': ['GPU1', 'GPU2']  # List[str]
        }

        assert isinstance(data['version'], str)
        assert isinstance(data['action'], str)
        assert isinstance(data['network'], str)
        assert isinstance(data['success'], bool)
        assert isinstance(data['user_error'], bool)
        assert isinstance(data['time_lapsed'], int)
        assert isinstance(data['gpus'], list)
