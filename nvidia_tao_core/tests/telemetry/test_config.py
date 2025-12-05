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

"""Unit tests for telemetry configuration module."""

from nvidia_tao_core.telemetry.config import (
    METRIC_ATTRIBUTES,
    get_attribute_map,
    get_raw_key_map
)
from nvidia_tao_core.telemetry.types import AttributeType, MetricAttribute


class TestMetricAttributes:
    """Test cases for METRIC_ATTRIBUTES configuration."""

    def test_metric_attributes_is_list(self):
        """Test that METRIC_ATTRIBUTES is a list."""
        assert isinstance(METRIC_ATTRIBUTES, list)
        assert len(METRIC_ATTRIBUTES) > 0

    def test_all_attributes_are_metric_attribute(self):
        """Test that all items in METRIC_ATTRIBUTES are MetricAttribute instances."""
        for attr in METRIC_ATTRIBUTES:
            assert isinstance(attr, MetricAttribute)

    def test_expected_attributes_exist(self):
        """Test that expected core attributes exist."""
        attr_names = {attr.name for attr in METRIC_ATTRIBUTES}

        expected_names = {'version', 'action', 'network', 'success', 'user_error', 'time_lapsed', 'gpus'}
        assert expected_names.issubset(attr_names), f"Missing attributes: {expected_names - attr_names}"

    def test_attribute_names_unique(self):
        """Test that all attribute names are unique."""
        names = [attr.name for attr in METRIC_ATTRIBUTES]
        assert len(names) == len(set(names)), "Duplicate attribute names found"

    def test_raw_keys_unique(self):
        """Test that all raw keys are unique."""
        raw_keys = [attr.raw_key for attr in METRIC_ATTRIBUTES]
        assert len(raw_keys) == len(set(raw_keys)), "Duplicate raw keys found"

    def test_string_attributes_configured(self):
        """Test that STRING type attributes are properly configured."""
        string_attrs = [attr for attr in METRIC_ATTRIBUTES if attr.attr_type == AttributeType.STRING]

        assert len(string_attrs) > 0, "No STRING attributes found"

        # All string attributes should have string defaults
        for attr in string_attrs:
            assert isinstance(attr.default, str)

    def test_boolean_attributes_configured(self):
        """Test that BOOLEAN type attributes are properly configured."""
        bool_attrs = [attr for attr in METRIC_ATTRIBUTES if attr.attr_type == AttributeType.BOOLEAN]

        assert len(bool_attrs) > 0, "No BOOLEAN attributes found"

        # All boolean attributes should have boolean defaults
        for attr in bool_attrs:
            assert isinstance(attr.default, bool)

    def test_integer_attributes_configured(self):
        """Test that INTEGER type attributes are properly configured."""
        int_attrs = [attr for attr in METRIC_ATTRIBUTES if attr.attr_type == AttributeType.INTEGER]

        assert len(int_attrs) > 0, "No INTEGER attributes found"

        # All integer attributes should have integer defaults
        for attr in int_attrs:
            assert isinstance(attr.default, int)

    def test_list_attributes_configured(self):
        """Test that LIST type attributes are properly configured."""
        list_attrs = [attr for attr in METRIC_ATTRIBUTES if attr.attr_type == AttributeType.LIST]

        assert len(list_attrs) > 0, "No LIST attributes found"

        # All list attributes should have list defaults
        for attr in list_attrs:
            assert isinstance(attr.default, list)

    def test_metric_order_values(self):
        """Test that metric_order values are valid."""
        for attr in METRIC_ATTRIBUTES:
            assert isinstance(attr.metric_order, int)
            assert attr.metric_order >= 0


class TestGetAttributeMap:
    """Test cases for get_attribute_map function."""

    def test_returns_dict(self):
        """Test that get_attribute_map returns a dictionary."""
        attr_map = get_attribute_map()
        assert isinstance(attr_map, dict)

    def test_maps_name_to_attribute(self):
        """Test that the map correctly maps names to MetricAttribute objects."""
        attr_map = get_attribute_map()

        for name, attr in attr_map.items():
            assert isinstance(attr, MetricAttribute)
            assert attr.name == name

    def test_contains_all_attributes(self):
        """Test that the map contains all attributes from METRIC_ATTRIBUTES."""
        attr_map = get_attribute_map()

        expected_names = {attr.name for attr in METRIC_ATTRIBUTES}
        actual_names = set(attr_map.keys())

        assert expected_names == actual_names

    def test_can_lookup_by_name(self):
        """Test that we can look up attributes by name."""
        attr_map = get_attribute_map()

        # Test a few known attributes
        if 'version' in attr_map:
            assert attr_map['version'].name == 'version'
        if 'action' in attr_map:
            assert attr_map['action'].name == 'action'


class TestGetRawKeyMap:
    """Test cases for get_raw_key_map function."""

    def test_returns_dict(self):
        """Test that get_raw_key_map returns a dictionary."""
        raw_key_map = get_raw_key_map()
        assert isinstance(raw_key_map, dict)

    def test_maps_raw_key_to_attribute(self):
        """Test that the map correctly maps raw keys to MetricAttribute objects."""
        raw_key_map = get_raw_key_map()

        for raw_key, attr in raw_key_map.items():
            assert isinstance(attr, MetricAttribute)
            assert attr.raw_key == raw_key

    def test_contains_all_attributes(self):
        """Test that the map contains all attributes from METRIC_ATTRIBUTES."""
        raw_key_map = get_raw_key_map()

        expected_keys = {attr.raw_key for attr in METRIC_ATTRIBUTES}
        actual_keys = set(raw_key_map.keys())

        assert expected_keys == actual_keys

    def test_can_lookup_by_raw_key(self):
        """Test that we can look up attributes by raw key."""
        raw_key_map = get_raw_key_map()

        # Test a few known attributes
        if 'version' in raw_key_map:
            assert raw_key_map['version'].raw_key == 'version'
        if 'gpu' in raw_key_map:
            assert raw_key_map['gpu'].raw_key == 'gpu'

    def test_raw_key_differs_from_name(self):
        """Test that we can handle cases where raw_key differs from name."""
        raw_key_map = get_raw_key_map()

        # 'gpu' raw_key maps to 'gpus' name
        if 'gpu' in raw_key_map:
            assert raw_key_map['gpu'].name == 'gpus'
            assert raw_key_map['gpu'].raw_key == 'gpu'


class TestConfigurationConsistency:
    """Test cases for overall configuration consistency."""

    def test_no_orphaned_attributes(self):
        """Test that all attributes are accessible from both maps."""
        attr_map = get_attribute_map()
        raw_key_map = get_raw_key_map()

        # Every attribute should be in both maps
        for attr in METRIC_ATTRIBUTES:
            assert attr.name in attr_map
            assert attr.raw_key in raw_key_map

    def test_comprehensive_metric_attributes_ordered(self):
        """Test that attributes included in comprehensive metrics have valid ordering."""
        comprehensive_attrs = [
            attr for attr in METRIC_ATTRIBUTES
            if attr.include_in_comprehensive
        ]

        # Should have at least some attributes for comprehensive metrics
        assert len(comprehensive_attrs) > 0

        # All should have valid metric_order
        for attr in comprehensive_attrs:
            assert isinstance(attr.metric_order, int)
