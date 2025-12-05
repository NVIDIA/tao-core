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

"""Unit tests for telemetry.send_telemetry_data function."""

import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_gpu_data():
    """Sample GPU data for testing."""
    return [
        {"name": "NVIDIA A100", "memory": "40GB"},
        {"name": "NVIDIA A100", "memory": "40GB"},
        {"name": "NVIDIA V100", "memory": "32GB"}
    ]


@pytest.fixture
def mock_metrics():
    """Mock metrics module for testing."""
    with patch('nvidia_tao_core.telemetry.telemetry.METRICS_MODULE_EXISTS', True):
        with patch('nvidia_tao_core.telemetry.telemetry.metrics') as mock:
            mock.report = MagicMock(return_value=None)
            yield mock


@pytest.fixture
def mock_logging():
    """Mock logging for testing."""
    with patch('nvidia_tao_core.telemetry.telemetry.logging') as mock:
        yield mock


class TestSendTelemetryData:
    """Test cases for send_telemetry_data function."""

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "yes"}, clear=False)
    def test_telemetry_opt_out_yes(self, mock_logging, sample_gpu_data):
        """Test that telemetry is skipped when opted out with 'yes'."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        # Verify opt-out message is logged
        mock_logging.info.assert_any_call("Opted out of telemetry reporting. Skipped.")

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "true"}, clear=False)
    def test_telemetry_opt_out_true(self, mock_logging, sample_gpu_data):
        """Test that telemetry is skipped when opted out with 'true'."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        mock_logging.info.assert_any_call("Opted out of telemetry reporting. Skipped.")

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "1"}, clear=False)
    def test_telemetry_opt_out_one(self, mock_logging, sample_gpu_data):
        """Test that telemetry is skipped when opted out with '1'."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        mock_logging.info.assert_any_call("Opted out of telemetry reporting. Skipped.")

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_sent_successfully(self, mock_metrics, mock_logging, sample_gpu_data):
        """Test successful telemetry reporting."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        # Mock successful response (None means success)
        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=2,
            time_lapsed=120,
            pass_status=True,
            user_error=False
        )

        # Verify metrics.report was called
        mock_metrics.report.assert_called_once()
        call_args = mock_metrics.report.call_args

        # Verify data structure
        assert call_args[1]['data']['network'] == "test_network"
        assert call_args[1]['data']['action'] == "train"
        assert call_args[1]['data']['gpu'] == ["NVIDIA A100", "NVIDIA A100"]
        assert call_args[1]['data']['success'] is True
        assert call_args[1]['data']['user_error'] is False
        assert call_args[1]['data']['time_lapsed'] == 120
        assert call_args[1]['data']['version'] == "5.3.0"

        # Verify success message is logged
        mock_logging.info.assert_any_call("Telemetry sent successfully.")

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_sent_with_failure_response(self, mock_metrics, mock_logging, sample_gpu_data):
        """Test telemetry reporting with failure response."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        # Mock error response
        mock_metrics.report.return_value = "error 500"

        send_telemetry_data(
            network="test_network",
            action="export",
            gpu_data=sample_gpu_data,
            num_gpus=1,
            pass_status=False,
            user_error=True
        )

        # Verify error message is logged
        mock_logging.info.assert_any_call("Failed with reponse: error 500")

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_without_time_lapsed(self, mock_metrics, sample_gpu_data):
        """Test telemetry without time_lapsed parameter."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="evaluate",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        # Verify time_lapsed is not in data when not provided
        call_args = mock_metrics.report.call_args
        assert 'time_lapsed' not in call_args[1]['data']

    @patch.dict(os.environ, {
        "TELEMETRY_OPT_OUT": "no",
        "TAO_TOOLKIT_VERSION": "6.0.0",
        "TAO_TELEMETRY_SERVER": "https://custom-server.com",
    }, clear=False)
    def test_telemetry_with_custom_env_vars(self, mock_metrics, sample_gpu_data):
        """Test telemetry with custom environment variables."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        call_args = mock_metrics.report.call_args

        # Verify custom version is used
        assert call_args[1]['data']['version'] == "6.0.0"

        # Verify custom server URL is used
        assert call_args[1]['base_url'] == "https://custom-server.com"

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_when_metrics_module_not_exists(self, mock_logging, sample_gpu_data):
        """Test telemetry when metrics module doesn't exist."""
        with patch('nvidia_tao_core.telemetry.telemetry.METRICS_MODULE_EXISTS', False):
            from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

            send_telemetry_data(
                network="test_network",
                action="train",
                gpu_data=sample_gpu_data,
                num_gpus=1
            )

            # Verify start and end messages are logged but no telemetry is sent
            mock_logging.info.assert_any_call("================> Start Reporting Telemetry <================")
            mock_logging.info.assert_any_call("================> End Reporting Telemetry <================")

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_gpu_data_slicing(self, mock_metrics, sample_gpu_data):
        """Test that GPU data is properly sliced based on num_gpus."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        # Request 2 GPUs from 3 available
        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=2
        )

        call_args = mock_metrics.report.call_args

        # Verify only first 2 GPU names are included
        assert call_args[1]['data']['gpu'] == ["NVIDIA A100", "NVIDIA A100"]
        assert len(call_args[1]['data']['gpu']) == 2

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "false"}, clear=False)
    def test_telemetry_opt_in_with_false(self, mock_metrics, sample_gpu_data):
        """Test that telemetry is sent when TELEMETRY_OPT_OUT is 'false'."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        # Verify metrics.report was called
        mock_metrics.report.assert_called_once()

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "0"}, clear=False)
    def test_telemetry_opt_in_with_zero(self, mock_metrics, sample_gpu_data):
        """Test that telemetry is sent when TELEMETRY_OPT_OUT is '0'."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        # Verify metrics.report was called
        mock_metrics.report.assert_called_once()

    @patch.dict(os.environ, {}, clear=False)
    def test_telemetry_default_opt_in(self, mock_metrics, sample_gpu_data):
        """Test that telemetry is sent by default when TELEMETRY_OPT_OUT is not set."""
        # Remove TELEMETRY_OPT_OUT if it exists
        os.environ.pop('TELEMETRY_OPT_OUT', None)

        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        # Verify metrics.report was called (default is opt-in)
        mock_metrics.report.assert_called_once()

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_with_zero_time_lapsed(self, mock_metrics, sample_gpu_data):
        """Test telemetry with time_lapsed=0 (should be included)."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1,
            time_lapsed=0
        )

        call_args = mock_metrics.report.call_args

        # Verify time_lapsed=0 is included
        assert call_args[1]['data']['time_lapsed'] == 0

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_logging_messages(self, mock_metrics, mock_logging, sample_gpu_data):
        """Test that appropriate logging messages are generated."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        # Verify all expected logging messages
        mock_logging.info.assert_any_call("================> Start Reporting Telemetry <================")
        mock_logging.info.assert_any_call("================> End Reporting Telemetry <================")

        # Verify "Sending..." message is logged
        calls = [str(call) for call in mock_logging.info.call_args_list]
        assert any("Sending" in str(call) for call in calls)

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_with_single_gpu(self, mock_metrics, sample_gpu_data):
        """Test telemetry with single GPU."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="test_network",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=1
        )

        call_args = mock_metrics.report.call_args

        # Verify only one GPU is included
        assert len(call_args[1]['data']['gpu']) == 1
        assert call_args[1]['data']['gpu'] == ["NVIDIA A100"]

    @patch.dict(os.environ, {"TELEMETRY_OPT_OUT": "no"}, clear=False)
    def test_telemetry_all_parameters(self, mock_metrics, sample_gpu_data):
        """Test telemetry with all parameters provided."""
        from nvidia_tao_core.telemetry.telemetry import send_telemetry_data

        mock_metrics.report.return_value = None

        send_telemetry_data(
            network="resnet50",
            action="train",
            gpu_data=sample_gpu_data,
            num_gpus=2,
            time_lapsed=3600,
            pass_status=True,
            user_error=False
        )

        call_args = mock_metrics.report.call_args
        data = call_args[1]['data']

        # Verify all parameters are correctly passed
        assert data['network'] == "resnet50"
        assert data['action'] == "train"
        assert data['gpu'] == ["NVIDIA A100", "NVIDIA A100"]
        assert data['success'] is True
        assert data['user_error'] is False
        assert data['time_lapsed'] == 3600
        assert 'version' in data
