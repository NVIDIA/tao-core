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

"""Unit tests for metrics API endpoint and utilities."""

import json
import pytest
from datetime import datetime
from unittest.mock import patch


class TestMetricsAPIEndpoint:
    """Test cases for /api/v1/metrics endpoint."""

    @pytest.fixture
    def client(self):
        """Create Flask test client."""
        # Import app here to avoid circular imports
        from nvidia_tao_core.microservices.app import app
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client

    @pytest.fixture(autouse=True)
    def mock_mongo(self):
        """Mock MongoDB connections to prevent hanging."""
        with patch('nvidia_tao_core.microservices.utils.stateless_handler_utils.MongoHandler'):
            yield

    @pytest.fixture
    def mock_metrics_storage(self):
        """Mock metrics storage."""
        with patch('nvidia_tao_core.microservices.blueprints.v1.admin.get_metrics') as mock_get:
            with patch('nvidia_tao_core.microservices.blueprints.v1.admin.set_metrics') as mock_set:
                mock_get.return_value = {
                    'last_updated': datetime(2025, 1, 15, 10, 0).isoformat()
                }
                yield mock_get, mock_set

    @pytest.fixture
    def mock_auth(self):
        """Mock authentication by patching os.getenv for INGRESSENABLED."""
        # Patch os.getenv in the auth module to return "true" for INGRESSENABLED
        # This is equivalent to the old: patch('nvidia_tao_core.microservices.app.ingress_enabled', True)
        with patch('nvidia_tao_core.microservices.blueprints.v1.auth.os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'INGRESSENABLED':
                    return 'true'
                return default
            mock_getenv.side_effect = getenv_side_effect
            yield

    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for metrics endpoint."""
        import base64
        credentials = base64.b64encode(b'$metricstoken:test_key').decode('utf-8')
        return {
            'Authorization': f'Basic {credentials}'
        }

    def test_metrics_endpoint_successful_request(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test successful metrics submission."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100', 'NVIDIA A100'],
            'time_lapsed': 120,
            'user_error': False
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201
        mock_set.assert_called_once()

    def test_metrics_endpoint_with_minimal_data(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with minimal required data."""
        _ = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100']
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

    def test_metrics_endpoint_failed_action(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission for failed action."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'yolov4',
            'action': 'export',
            'success': False,
            'gpu': ['NVIDIA V100'],
            'user_error': True
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify that set_metrics was called with updated metrics
        call_args = mock_set.call_args[0][0]
        assert 'total_action_export_fail' in call_args
        assert call_args['total_action_export_fail'] == 1

    def test_metrics_endpoint_multiple_submissions(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test multiple metrics submissions increment counters."""
        mock_get, mock_set = mock_metrics_storage

        # Set initial state
        mock_get.return_value = {
            'last_updated': datetime(2025, 1, 15, 10, 0).isoformat(),
            'total_action_train_pass': 5
        }

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100']
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify counter was incremented
        call_args = mock_set.call_args[0][0]
        assert call_args['total_action_train_pass'] == 6

    def test_metrics_endpoint_invalid_data(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with invalid data."""
        _ = mock_metrics_storage
        payload = {
            'invalid_field': 'value'
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_metrics_endpoint_empty_payload(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with empty payload."""
        _ = mock_metrics_storage
        response = client.post(
            '/api/v1/metrics',
            data=json.dumps({}),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

    def test_metrics_endpoint_malformed_json(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with malformed JSON."""
        _ = mock_metrics_storage
        response = client.post(
            '/api/v1/metrics',
            data='invalid json{',
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 400

    def test_metrics_endpoint_time_lapsed_accumulation(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test that time_lapsed accumulates correctly on same day."""
        mock_get, mock_set = mock_metrics_storage

        mock_get.return_value = {
            'last_updated': datetime.now().isoformat(),
            'time_lapsed_today': 100
        }

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100'],
            'time_lapsed': 50
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify time accumulated
        call_args = mock_set.call_args[0][0]
        assert call_args['time_lapsed_today'] == 150

    def test_metrics_endpoint_comprehensive_metric_created(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test that comprehensive metric name is created correctly."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '6.0.0',
            'network': 'efficientnet',
            'action': 'evaluate',
            'success': True,
            'gpu': ['NVIDIA H100'],
            'user_error': False
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify comprehensive metric was created
        call_args = mock_set.call_args[0][0]

        # Find the comprehensive metric (long key with all attributes)
        comprehensive_keys = [k for k in call_args.keys() if k.startswith('network_efficientnet_action_evaluate')]

        comprehensive_key = max(comprehensive_keys, key=len)

        assert 'network_efficientnet' in comprehensive_key
        assert 'action_evaluate' in comprehensive_key
        assert 'version_6_0_0' in comprehensive_key
        assert 'status_pass' in comprehensive_key

    def test_metrics_endpoint_with_special_characters(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with special characters in fields."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0-dev',
            'network': 'custom-net@v2',
            'action': 'fine-tune!',
            'success': True,
            'gpu': ['NVIDIA-A100-80GB']
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify special characters were sanitized
        call_args = mock_set.call_args[0][0]

        # Check that sanitized versions exist in keys
        assert any('custom_net_v2' in k for k in call_args.keys())
        assert any('fine_tune' in k for k in call_args.keys())

    def test_metrics_endpoint_last_updated_timestamp(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test that last_updated timestamp is set correctly."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100']
        }

        before_time = datetime.now()

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        after_time = datetime.now()

        assert response.status_code == 201

        # Verify timestamp was updated
        call_args = mock_set.call_args[0][0]
        timestamp = datetime.fromisoformat(call_args['last_updated'])

        assert before_time <= timestamp <= after_time

    def test_metrics_endpoint_without_auth(self, client, mock_metrics_storage):
        """Test metrics submission without authentication - succeeds as metrics endpoint is open."""
        _ = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100']
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Metrics endpoint is in admin blueprint which doesn't have auth requirement
        assert response.status_code == 201

    def test_metrics_endpoint_with_wrong_auth(self, client, mock_metrics_storage):
        """Test metrics submission with auth headers - succeeds as metrics endpoint is open."""
        _ = mock_metrics_storage

        import base64
        credentials = base64.b64encode(b'$metricstoken:wrong_key').decode('utf-8')
        headers = {'Authorization': f'Basic {credentials}'}

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100']
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=headers
        )

        # Metrics endpoint is in admin blueprint which doesn't have auth requirement
        assert response.status_code == 201

    def test_metrics_endpoint_user_error_true(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with user_error=True."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': False,
            'gpu': ['NVIDIA A100'],
            'user_error': True
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify comprehensive metric was created
        call_args = mock_set.call_args[0][0]
        comprehensive_keys = [k for k in call_args.keys() if 'status_fail' in k]
        assert len(comprehensive_keys) > 0

    def test_metrics_endpoint_user_error_false(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with user_error=False."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100'],
            'user_error': False
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify comprehensive metric was created
        call_args = mock_set.call_args[0][0]
        comprehensive_keys = [k for k in call_args.keys() if 'status_pass' in k]
        assert len(comprehensive_keys) > 0

    def test_metrics_endpoint_without_user_error_field(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission without user_error field (should default to False)."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100']
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify comprehensive metric was created
        call_args = mock_set.call_args[0][0]
        comprehensive_keys = [k for k in call_args.keys() if 'status_pass' in k]
        assert len(comprehensive_keys) > 0

    def test_metrics_endpoint_multiple_gpu_types(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission with multiple different GPU types."""
        _, mock_set = mock_metrics_storage

        payload = {
            'version': '5.3.0',
            'network': 'resnet50',
            'action': 'train',
            'success': True,
            'gpu': ['NVIDIA A100', 'NVIDIA V100', 'NVIDIA H100', 'NVIDIA A100']
        }

        response = client.post(
            '/api/v1/metrics',
            data=json.dumps(payload),
            content_type='application/json',
            headers=auth_headers
        )

        assert response.status_code == 201

        # Verify GPU counters were updated for each type
        call_args = mock_set.call_args[0][0]
        assert call_args['gpu_nvidia_a100_action_train'] == 2
        assert call_args['gpu_nvidia_v100_action_train'] == 1
        assert call_args['gpu_nvidia_h100_action_train'] == 1

        # Verify comprehensive metric includes all GPU types with counts
        call_args = mock_set.call_args[0][0]
        comprehensive_keys = [k for k in call_args.keys() if k.startswith('network_resnet50_action_train')]
        assert comprehensive_keys, "No comprehensive metric keys found"
        comp_key = max(comprehensive_keys, key=len)
        assert "gpu_4_" in comp_key
        assert "NVIDIA_A100_2" in comp_key
        assert "NVIDIA_V100_1" in comp_key
        assert "NVIDIA_H100_1" in comp_key

    def test_metrics_endpoint_different_actions(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission for different actions."""
        _, mock_set = mock_metrics_storage

        actions = ['train', 'evaluate', 'export', 'prune', 'inference']

        for action in actions:
            payload = {
                'version': '5.3.0',
                'network': 'resnet50',
                'action': action,
                'success': True,
                'gpu': ['NVIDIA A100']
            }

            response = client.post(
                '/api/v1/metrics',
                data=json.dumps(payload),
                content_type='application/json',
                headers=auth_headers
            )

            assert response.status_code == 201

        # Verify all actions were recorded
        call_args = mock_set.call_args[0][0]
        for action in actions:
            assert f'total_action_{action}_pass' in call_args
            assert call_args[f'total_action_{action}_pass'] == 1

    def test_metrics_endpoint_different_networks(self, client, mock_metrics_storage, mock_auth, auth_headers):
        """Test metrics submission for different networks."""
        _, mock_set = mock_metrics_storage

        networks = ['resnet50', 'yolov4', 'efficientnet', 'mask_rcnn']

        for network in networks:
            payload = {
                'version': '5.3.0',
                'network': network,
                'action': 'train',
                'success': True,
                'gpu': ['NVIDIA A100']
            }

            response = client.post(
                '/api/v1/metrics',
                data=json.dumps(payload),
                content_type='application/json',
                headers=auth_headers
            )

            assert response.status_code == 201

        # Verify all networks were recorded
        call_args = mock_set.call_args[0][0]
        for network in networks:
            assert f'network_{network}_action_train' in call_args
            assert call_args[f'network_{network}_action_train'] == 1
