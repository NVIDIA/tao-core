# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for job timeout monitoring feature in workflow.py"""

import os
from unittest.mock import Mock, patch
from datetime import datetime, timezone, timedelta

from nvidia_tao_core.microservices.job_utils.workflow import (
    get_last_status_timestamp,
    check_job_timeout,
    terminate_timed_out_job,
    check_for_timed_out_jobs
)


class TestGetLastStatusTimestamp:
    """Test get_last_status_timestamp function"""

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    def test_get_last_status_timestamp_with_valid_status(self, mock_get_dnn_status):
        """Test getting timestamp from valid status data"""
        job_id = "test-job-123"
        now = datetime.now(tz=timezone.utc)
        timestamp_str = now.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        mock_get_dnn_status.return_value = [
            {'timestamp': timestamp_str, 'message': 'Training started'},
            {'timestamp': (now - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 1'}
        ]

        result = get_last_status_timestamp(job_id, automl=False, experiment_number="0")

        assert result is not None
        assert isinstance(result, datetime)
        # Should return the most recent timestamp
        assert abs((result - now).total_seconds()) < 1

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    def test_get_last_status_timestamp_no_status(self, mock_get_dnn_status):
        """Test when no status data is available"""
        mock_get_dnn_status.return_value = None

        result = get_last_status_timestamp("test-job-123")

        assert result is None

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    def test_get_last_status_timestamp_empty_status(self, mock_get_dnn_status):
        """Test when status data is empty"""
        mock_get_dnn_status.return_value = []

        result = get_last_status_timestamp("test-job-123")

        assert result is None

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    def test_get_last_status_timestamp_multiple_formats(self, mock_get_dnn_status):
        """Test parsing multiple timestamp formats"""
        now = datetime.now(tz=timezone.utc)

        mock_get_dnn_status.return_value = [
            {'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Format 1'},
            {'timestamp': (now - timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%SZ'), 'message': 'Format 2'},
            {'timestamp': (now - timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%S.%f'), 'message': 'Format 3'}
        ]

        result = get_last_status_timestamp("test-job-123")

        assert result is not None
        assert abs((result - now).total_seconds()) < 1

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    def test_get_last_status_timestamp_automl_experiment(self, mock_get_dnn_status):
        """Test getting timestamp for AutoML experiment"""
        job_id = "automl-job-123"
        experiment_number = "5"
        now = datetime.now(tz=timezone.utc)

        mock_get_dnn_status.return_value = [
            {'timestamp': now.strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Training'}
        ]

        result = get_last_status_timestamp(job_id, automl=True, experiment_number=experiment_number)

        assert result is not None
        mock_get_dnn_status.assert_called_once_with(job_id, automl=True, experiment_number=experiment_number)


class TestCheckJobTimeout:
    """Test check_job_timeout function"""

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_last_status_timestamp')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.JOB_STATUS_TIMEOUT_MINUTES', 2)
    def test_check_job_timeout_not_timed_out(self, mock_get_timestamp, mock_get_metadata):
        """Test job that has not timed out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment'
        }

        # Job last updated 1 minute ago (within 2 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
        mock_get_timestamp.return_value = last_update
        mock_get_metadata.return_value = {'status': 'Running'}

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_last_status_timestamp')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.JOB_STATUS_TIMEOUT_MINUTES', 2)
    def test_check_job_timeout_timed_out(self, mock_get_timestamp, mock_get_metadata):
        """Test job that has timed out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False,
            'handler_id': 'handler-123',
            'kind': 'experiment'
        }

        # Job last updated 5 minutes ago (exceeds 2 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        mock_get_timestamp.return_value = last_update
        mock_get_metadata.return_value = {'status': 'Running'}

        result = check_job_timeout(job_info)

        assert result is True

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    def test_check_job_timeout_done_status(self, mock_get_metadata):
        """Test that completed jobs are not checked for timeout"""
        job_info = {
            'job_id': 'test-job-123',
            'is_automl': False
        }

        mock_get_metadata.return_value = {'status': 'Done'}

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    def test_check_job_timeout_error_status(self, mock_get_metadata):
        """Test that errored jobs are not checked for timeout"""
        job_info = {
            'job_id': 'test-job-123',
            'is_automl': False
        }

        mock_get_metadata.return_value = {'status': 'Error'}

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_automl_controller_info')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_last_status_timestamp')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.JOB_STATUS_TIMEOUT_MINUTES', 2)
    def test_check_job_timeout_automl_experiment(self, mock_get_timestamp, mock_get_controller):
        """Test timeout check for AutoML experiment"""
        job_id = "automl-job-123"
        experiment_number = "5"
        job_info = {
            'job_id': job_id,
            'is_automl': True,
            'experiment_number': experiment_number
        }

        # Mock controller info showing experiment is running
        mock_get_controller.return_value = [
            {'id': '5', 'status': 'running'},
            {'id': '6', 'status': 'pending'}
        ]

        # Experiment timed out (5 minutes exceeds 2 minute timeout)
        last_update = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        mock_get_timestamp.return_value = last_update

        result = check_job_timeout(job_info)

        assert result is True

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_automl_controller_info')
    def test_check_job_timeout_automl_experiment_completed(self, mock_get_controller):
        """Test that completed AutoML experiments are not checked"""
        job_info = {
            'job_id': 'automl-job-123',
            'is_automl': True,
            'experiment_number': '5'
        }

        mock_get_controller.return_value = [
            {'id': '5', 'status': 'done'}
        ]

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_last_status_timestamp')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.JOB_STATUS_TIMEOUT_MINUTES', 2)
    def test_check_job_timeout_no_timestamp_uses_last_modified(self, mock_get_timestamp, mock_get_metadata):
        """Test fallback to last_modified when no status timestamp exists"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False
        }

        # No status timestamp
        mock_get_timestamp.return_value = None

        # But has last_modified that's recent (within 2 minute timeout)
        last_modified = datetime.now(tz=timezone.utc) - timedelta(minutes=1)
        mock_get_metadata.return_value = {
            'status': 'Running',
            'last_modified': last_modified.isoformat()
        }

        result = check_job_timeout(job_info)

        assert result is False

    def test_check_job_timeout_missing_job_id(self):
        """Test handling of missing job_id"""
        job_info = {
            'is_automl': False
        }

        result = check_job_timeout(job_info)

        assert result is False


class TestTerminateTimedOutJob:
    """Test terminate_timed_out_job function"""

    @patch('nvidia_tao_core.microservices.job_utils.executor.statefulset_executor.StatefulSetExecutor')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.update_job_status')
    def test_terminate_timed_out_regular_job(self, mock_update_status, mock_executor_class):
        """Test terminating a timed out regular job"""
        job_id = "test-job-123"
        handler_id = "handler-123"
        job_info = {
            'job_id': job_id,
            'handler_id': handler_id,
            'kind': 'experiment',
            'is_automl': False
        }

        mock_executor = Mock()
        mock_executor.delete_statefulset.return_value = True
        mock_executor_class.return_value = mock_executor

        result = terminate_timed_out_job(job_info)

        assert result is True
        mock_update_status.assert_called_once_with(handler_id, job_id, status="Error", kind='experiment')
        mock_executor.delete_statefulset.assert_called_once_with(job_id, use_ngc=True)

    @patch('nvidia_tao_core.microservices.job_utils.executor.statefulset_executor.StatefulSetExecutor')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.save_automl_controller_info')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_automl_controller_info')
    def test_terminate_timed_out_automl_experiment(
        self, mock_get_controller, mock_save_controller, mock_executor_class
    ):
        """Test terminating a timed out AutoML experiment"""
        job_id = "automl-job-123"
        handler_id = "handler-123"
        experiment_number = "5"
        job_info = {
            'job_id': job_id,
            'handler_id': handler_id,
            'kind': 'experiment',
            'is_automl': True,
            'experiment_number': experiment_number
        }

        # Mock controller info
        mock_get_controller.return_value = [
            {'id': '5', 'status': 'running', 'message': 'Training'},
            {'id': '6', 'status': 'pending', 'message': 'Waiting'}
        ]

        mock_executor = Mock()
        mock_executor.delete_statefulset.return_value = True
        mock_executor_class.return_value = mock_executor

        result = terminate_timed_out_job(job_info)

        assert result is True

        # Verify controller info was updated
        call_args = mock_save_controller.call_args[0]
        assert call_args[0] == job_id
        updated_controller = call_args[1]

        # Find the experiment that should be marked as error
        experiment_5 = next(exp for exp in updated_controller if exp['id'] == '5')
        assert experiment_5['status'] == 'error'
        assert 'timeout' in experiment_5['message'].lower()

        # Verify StatefulSet was deleted
        mock_executor.delete_statefulset.assert_called()

    @patch('nvidia_tao_core.microservices.job_utils.executor.statefulset_executor.StatefulSetExecutor')
    def test_terminate_timed_out_job_missing_info(self, mock_executor_class):
        """Test handling of missing job information"""
        job_info = {
            'job_id': 'test-job-123',
            # Missing handler_id
            'is_automl': False
        }

        result = terminate_timed_out_job(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.job_utils.executor.statefulset_executor.StatefulSetExecutor')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.update_job_status')
    def test_terminate_timed_out_job_statefulset_deletion_fails(self, mock_update_status, mock_executor_class):
        """Test when StatefulSet deletion fails"""
        job_id = "test-job-123"
        handler_id = "handler-123"
        job_info = {
            'job_id': job_id,
            'handler_id': handler_id,
            'kind': 'experiment',
            'is_automl': False
        }

        mock_executor = Mock()
        mock_executor.delete_statefulset.return_value = False
        mock_executor_class.return_value = mock_executor

        result = terminate_timed_out_job(job_info)

        assert result is False
        # Status should still be updated even if deletion fails
        mock_update_status.assert_called_once()


class TestCheckForTimedOutJobs:
    """Test check_for_timed_out_jobs function"""

    @patch('nvidia_tao_core.microservices.job_utils.workflow.terminate_timed_out_job')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'true'})
    def test_check_for_timed_out_jobs_with_timeouts(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout, mock_terminate
    ):
        """Test checking for timed out jobs when some have timed out"""
        # Mock running jobs
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'},
            {'job_id': 'job-2', 'is_automl': False, 'handler_id': 'handler-2', 'kind': 'experiment'}
        ]

        # Mock running AutoML experiments
        mock_get_automl.return_value = [
            {
                'job_id': 'automl-1',
                'is_automl': True,
                'experiment_number': '3',
                'handler_id': 'handler-3',
                'kind': 'experiment'
            }
        ]

        # First job timed out, others did not
        mock_check_timeout.side_effect = [True, False, False]
        mock_terminate.return_value = True

        result = check_for_timed_out_jobs()

        assert len(result) == 1
        assert 'job-1' in result[0]
        mock_terminate.assert_called_once()

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'false'})
    def test_check_for_timed_out_jobs_monitoring_disabled(self, mock_get_jobs, mock_get_automl):
        """Test that timeout monitoring can be disabled"""
        result = check_for_timed_out_jobs()

        assert result == []
        # Should not even query for running jobs
        mock_get_jobs.assert_not_called()
        mock_get_automl.assert_not_called()

    @patch('nvidia_tao_core.microservices.job_utils.workflow.terminate_timed_out_job')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'true'})
    def test_check_for_timed_out_jobs_no_timeouts(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout, mock_terminate
    ):
        """Test when no jobs have timed out"""
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'}
        ]
        mock_get_automl.return_value = []

        # No jobs timed out
        mock_check_timeout.return_value = False

        result = check_for_timed_out_jobs()

        assert len(result) == 0
        mock_terminate.assert_not_called()

    @patch('nvidia_tao_core.microservices.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'true'})
    def test_check_for_timed_out_jobs_handles_exceptions(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout
    ):
        """Test that exceptions during timeout check don't crash the function"""
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'}
        ]
        mock_get_automl.return_value = []

        # Simulate an exception during timeout check
        mock_check_timeout.side_effect = Exception("Test exception")

        # Should not raise exception
        result = check_for_timed_out_jobs()

        assert len(result) == 0

    @patch('nvidia_tao_core.microservices.job_utils.workflow.terminate_timed_out_job')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.check_job_timeout')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_automl_experiments')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_jobs')
    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': '1'})
    def test_check_for_timed_out_jobs_with_both_regular_and_automl(
        self, mock_get_jobs, mock_get_automl, mock_check_timeout, mock_terminate
    ):
        """Test checking both regular jobs and AutoML experiments"""
        mock_get_jobs.return_value = [
            {'job_id': 'job-1', 'is_automl': False, 'handler_id': 'handler-1', 'kind': 'experiment'}
        ]
        mock_get_automl.return_value = [
            {
                'job_id': 'automl-1',
                'is_automl': True,
                'experiment_number': '3',
                'handler_id': 'handler-3',
                'kind': 'experiment'
            }
        ]

        # Both timed out
        mock_check_timeout.return_value = True
        mock_terminate.return_value = True

        result = check_for_timed_out_jobs()

        assert len(result) == 2
        assert mock_terminate.call_count == 2


class TestTimeoutResetOnRestart:
    """Test that timeout timer resets when jobs are restarted or resumed"""

    def test_resume_job_includes_delete_status_call(self):
        """Test that experiment_handler.py includes delete_dnn_status import and call"""
        # Verify that the resume function has the timeout reset code
        import inspect
        from nvidia_tao_core.microservices.app_handlers import experiment_handler

        # Check that delete_dnn_status is imported
        assert hasattr(experiment_handler, 'delete_dnn_status')

        # Check that resume_experiment_job exists
        assert hasattr(experiment_handler.ExperimentHandler, 'resume_experiment_job')

        # Verify the function source contains the delete_dnn_status call
        source = inspect.getsource(experiment_handler.ExperimentHandler.resume_experiment_job)
        assert 'delete_dnn_status' in source
        assert 'automl=False' in source

    def test_restart_threads_includes_delete_status_call(self):
        """Test that workflow.py restart_threads includes delete_dnn_status call"""
        # Verify that the restart_threads function has the timeout reset code
        import inspect
        from nvidia_tao_core.microservices.job_utils import workflow

        # Check that delete_dnn_status is imported
        assert hasattr(workflow, 'delete_dnn_status')

        # Check that restart_threads exists
        assert hasattr(workflow.Workflow, 'restart_threads')

        # Verify the function source contains the delete_dnn_status call
        source = inspect.getsource(workflow.Workflow.restart_threads)
        assert 'delete_dnn_status' in source
        assert 'automl=False' in source

    def test_automl_controller_includes_delete_status_call(self):
        """Test that AutoML controller includes delete_dnn_status for experiment restarts"""
        # This test verifies the existing AutoML behavior
        import inspect
        from nvidia_tao_core.microservices.automl import controller

        # Check that delete_dnn_status is imported
        assert hasattr(controller, 'delete_dnn_status')

        # Check that Controller class exists
        assert hasattr(controller, 'Controller')

        # Verify Controller source code contains delete_dnn_status usage
        source = inspect.getsource(controller.Controller)
        assert 'delete_dnn_status' in source


class TestTimeoutConfiguration:
    """Test timeout configuration and environment variable handling"""

    @patch.dict(os.environ, {'JOB_STATUS_TIMEOUT_MINUTES': '30'})
    def test_timeout_env_variable_custom_value(self):
        """Test that timeout can be configured via environment variable"""
        # Need to reload the module to pick up new env var
        import importlib
        from nvidia_tao_core.microservices import constants
        importlib.reload(constants)

        assert constants.JOB_STATUS_TIMEOUT_MINUTES == 30

    @patch.dict(os.environ, {}, clear=True)
    def test_timeout_env_variable_default_value(self):
        """Test default timeout value when env var is not set"""
        # Need to reload the module to pick up new env var
        import importlib
        from nvidia_tao_core.microservices import constants

        # Remove the env var if it exists
        if 'JOB_STATUS_TIMEOUT_MINUTES' in os.environ:
            del os.environ['JOB_STATUS_TIMEOUT_MINUTES']

        importlib.reload(constants)

        # Default should be 15 minutes
        assert constants.JOB_STATUS_TIMEOUT_MINUTES == 15

    @patch.dict(os.environ, {'JOB_TIMEOUT_MONITORING_ENABLED': 'false'})
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_all_running_jobs')
    def test_timeout_monitoring_can_be_disabled(self, mock_get_jobs):
        """Test that timeout monitoring can be completely disabled"""
        result = check_for_timed_out_jobs()

        assert result == []
        mock_get_jobs.assert_not_called()


class TestTimeoutWithStatusUpdates:
    """Test timeout behavior with various status update patterns"""

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.JOB_STATUS_TIMEOUT_MINUTES', 2)
    def test_timeout_with_continuous_updates(self, mock_get_status, mock_get_metadata):
        """Test that jobs with continuous updates don't time out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False
        }

        now = datetime.now(tz=timezone.utc)

        # Job has regular status updates (all within 2 minute window)
        mock_get_status.return_value = [
            {'timestamp': (now - timedelta(seconds=90)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 1'},
            {'timestamp': (now - timedelta(seconds=60)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 2'},
            {'timestamp': (now - timedelta(seconds=30)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 3'},
            {'timestamp': (now - timedelta(seconds=10)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 4'}
        ]
        mock_get_metadata.return_value = {'status': 'Running'}

        result = check_job_timeout(job_info)

        assert result is False

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.JOB_STATUS_TIMEOUT_MINUTES', 2)
    def test_timeout_with_stale_updates(self, mock_get_status, mock_get_metadata):
        """Test that jobs with stale updates time out"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False
        }

        now = datetime.now(tz=timezone.utc)

        # Job has old status updates, nothing recent (5 minutes ago exceeds 2 minute timeout)
        mock_get_status.return_value = [
            {'timestamp': (now - timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 1'},
            {'timestamp': (now - timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Epoch 2'}
        ]
        mock_get_metadata.return_value = {'status': 'Running'}

        result = check_job_timeout(job_info)

        assert result is True

    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_handler_job_metadata')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.get_dnn_status')
    @patch('nvidia_tao_core.microservices.job_utils.workflow.JOB_STATUS_TIMEOUT_MINUTES', 2)
    def test_timeout_boundary_condition(self, mock_get_status, mock_get_metadata):
        """Test timeout just below boundary (2 minutes)"""
        job_id = "test-job-123"
        job_info = {
            'job_id': job_id,
            'is_automl': False
        }

        now = datetime.now(tz=timezone.utc)

        # Job last updated 119 seconds ago (just under 2 minute timeout)
        mock_get_status.return_value = [
            {'timestamp': (now - timedelta(seconds=119)).strftime('%Y-%m-%dT%H:%M:%S.%fZ'), 'message': 'Training'}
        ]
        mock_get_metadata.return_value = {'status': 'Running'}

        result = check_job_timeout(job_info)

        # Just under 2 minutes, should not be timed out
        assert result is False
