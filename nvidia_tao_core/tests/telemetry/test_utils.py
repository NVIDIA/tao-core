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

"""Unit tests for telemetry utility functions."""

from nvidia_tao_core.telemetry.utils import (
    sanitize_field_value,
    create_gpu_identifier,
    extract_telemetry_data,
    TELEMETRY_SANITIZE_PATTERN
)


class TestSanitizeFieldValue:
    """Test cases for sanitize_field_value function."""

    def test_sanitize_lowercase_default(self):
        """Test that sanitization converts to lowercase by default."""
        assert sanitize_field_value("ResNet-50") == "resnet_50"
        assert sanitize_field_value("YOLOv4") == "yolov4"
        assert sanitize_field_value("BERT") == "bert"

    def test_sanitize_special_characters(self):
        """Test that special characters are replaced with underscores."""
        assert sanitize_field_value("test-network") == "test_network"
        assert sanitize_field_value("v5.3.0") == "v5_3_0"
        assert sanitize_field_value("test@example#123") == "test_example_123"
        assert sanitize_field_value("network/model") == "network_model"

    def test_sanitize_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        assert sanitize_field_value("resnet50") == "resnet50"
        assert sanitize_field_value("model123") == "model123"
        assert sanitize_field_value("test123abc") == "test123abc"

    def test_sanitize_uppercase_option(self):
        """Test sanitization with uppercase=True."""
        assert sanitize_field_value("nvidia-a100", uppercase=True) == "NVIDIA_A100"
        assert sanitize_field_value("v100-32gb", uppercase=True) == "V100_32GB"
        assert sanitize_field_value("test", uppercase=True) == "TEST"

    def test_sanitize_empty_string(self):
        """Test sanitizing empty string."""
        assert sanitize_field_value("") == ""

    def test_sanitize_numeric_values(self):
        """Test sanitizing numeric values."""
        assert sanitize_field_value(123) == "123"
        assert sanitize_field_value(5.3) == "5_3"

    def test_sanitize_with_multiple_special_chars(self):
        """Test sanitizing strings with multiple consecutive special characters."""
        assert sanitize_field_value("test---name") == "test___name"
        assert sanitize_field_value("a@#$-b") == "a____b"


class TestCreateGpuIdentifier:
    """Test cases for create_gpu_identifier function."""

    def test_single_gpu(self):
        """Test GPU identifier with single GPU."""
        result = create_gpu_identifier(["NVIDIA A100"])
        assert result == "1_NVIDIA_A100_1"

    def test_multiple_same_gpus(self):
        """Test GPU identifier with multiple identical GPUs."""
        result = create_gpu_identifier(["NVIDIA A100", "NVIDIA A100", "NVIDIA A100"])
        assert result == "3_NVIDIA_A100_3"

    def test_mixed_gpu_types(self):
        """Test GPU identifier with mixed GPU types."""
        result = create_gpu_identifier(["NVIDIA A100", "NVIDIA V100", "NVIDIA A100"])
        # Total count is 3, A100 appears 2 times, V100 appears 1 time
        # Sorted alphabetically: NVIDIA_A100 comes before NVIDIA_V100
        assert "3_" in result
        assert "NVIDIA_A100_2" in result
        assert "NVIDIA_V100_1" in result

    def test_gpu_name_sanitization(self):
        """Test that GPU names are sanitized (uppercase, special chars removed)."""
        result = create_gpu_identifier(["nvidia-a100-40gb"])
        assert "NVIDIA_A100_40GB_1" in result

    def test_alphabetical_ordering(self):
        """Test that GPU types are sorted alphabetically."""
        result = create_gpu_identifier(["V100", "A100", "H100"])
        parts = result.split("_")
        # Should be: 3_A100_1_H100_1_V100_1
        assert parts[0] == "3"  # Total count
        # The GPU names should appear in alphabetical order
        gpu_names_in_result = [parts[i] for i in range(1, len(parts), 2)]
        assert gpu_names_in_result == sorted(gpu_names_in_result)

    def test_empty_gpu_list(self):
        """Test GPU identifier with empty list."""
        result = create_gpu_identifier([])
        assert result == "0_"

    def test_gpu_counts(self):
        """Test that GPU counts are correctly calculated."""
        result = create_gpu_identifier(["A100", "A100", "V100"])
        assert "3_A100_2" in result
        assert "V100_1" in result


class TestExtractTelemetryData:
    """Test cases for extract_telemetry_data function."""

    def test_extract_all_fields(self):
        """Test extracting all telemetry fields."""
        raw_data = {
            'version': '5.3.0',
            'action': 'train',
            'network': 'ResNet-50',
            'success': True,
            'user_error': False,
            'time_lapsed': 3600,
            'gpu': ['NVIDIA A100', 'NVIDIA V100']
        }

        result = extract_telemetry_data(raw_data)

        assert result['version'] == '5_3_0'  # Sanitized
        assert result['action'] == 'train'  # Sanitized (already lowercase)
        assert result['network'] == 'resnet_50'  # Sanitized
        assert result['success'] is True  # Not sanitized (boolean)
        assert result['user_error'] is False  # Not sanitized (boolean)
        assert result['time_lapsed'] == 3600  # Not sanitized (integer)
        assert result['gpus'] == ['NVIDIA A100', 'NVIDIA V100']  # Not sanitized (list)

    def test_extract_with_defaults(self):
        """Test that default values are used when fields are missing."""
        raw_data = {}

        result = extract_telemetry_data(raw_data)

        assert result['version'] == 'unknown'
        assert result['action'] == 'unknown'
        assert result['network'] == 'unknown'
        assert result['success'] is False
        assert result['user_error'] is False
        assert result['time_lapsed'] == 0
        assert result['gpus'] == ['unknown']

    def test_extract_partial_fields(self):
        """Test extracting with only some fields present."""
        raw_data = {
            'action': 'evaluate',
            'network': 'yolov4',
            'gpu': ['Tesla V100']
        }

        result = extract_telemetry_data(raw_data)

        assert result['action'] == 'evaluate'
        assert result['network'] == 'yolov4'
        assert result['gpus'] == ['Tesla V100']
        assert result['version'] == 'unknown'  # Default

    def test_string_sanitization(self):
        """Test that STRING type attributes are sanitized."""
        raw_data = {
            'version': '5.3.0',
            'action': 'Train-Model',
            'network': 'ResNet-50'
        }

        result = extract_telemetry_data(raw_data)

        # All strings should be sanitized
        assert result['version'] == '5_3_0'
        assert result['action'] == 'train_model'
        assert result['network'] == 'resnet_50'

    def test_non_string_preservation(self):
        """Test that non-STRING types are not sanitized."""
        raw_data = {
            'success': True,
            'user_error': False,
            'time_lapsed': 3600,
            'gpu': ['GPU-1', 'GPU-2']
        }

        result = extract_telemetry_data(raw_data)

        # Boolean, integer, and list should be preserved as-is
        assert result['success'] is True
        assert result['user_error'] is False
        assert result['time_lapsed'] == 3600
        assert result['gpus'] == ['GPU-1', 'GPU-2']  # List not sanitized

    def test_raw_key_mapping(self):
        """Test that raw keys are correctly mapped to attribute names."""
        raw_data = {
            'gpu': ['NVIDIA A100']  # raw_key is 'gpu', but name is 'gpus'
        }

        result = extract_telemetry_data(raw_data)

        assert 'gpus' in result
        assert result['gpus'] == ['NVIDIA A100']


class TestConstants:
    """Test cases for module constants."""

    def test_sanitize_pattern_exists(self):
        """Test that TELEMETRY_SANITIZE_PATTERN is defined."""
        assert TELEMETRY_SANITIZE_PATTERN is not None
        assert isinstance(TELEMETRY_SANITIZE_PATTERN, str)

    def test_sanitize_pattern_valid(self):
        """Test that the sanitize pattern is a valid regex."""
        import re
        # Should not raise an exception
        re.compile(TELEMETRY_SANITIZE_PATTERN)
