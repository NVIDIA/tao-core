# Job Workflow Timeout Feature - Unit Tests

## Overview

This directory contains comprehensive unit tests for the **job timeout monitoring feature** introduced in TAO Core microservices. The timeout feature automatically terminates jobs that have stopped sending status updates for a configurable period (default: 15 minutes), preventing zombie jobs from consuming resources indefinitely.

## Test Coverage Summary

✅ **31 comprehensive unit tests** covering all aspects of the timeout feature  
⏱️ **Test execution time**: ~0.87 seconds  
🎯 **Test timeout value**: 2 minutes (for faster, clearer tests)  
📊 **Production default**: 15 minutes

## Test Files

### `test_workflow_timeout.py` (709 lines)

Comprehensive test suite organized into 7 test classes:

#### 1. **TestGetLastStatusTimestamp** (5 tests)
Tests for retrieving the most recent status update timestamp from job history.

- ✅ Valid status data with timestamps
- ✅ No status data available  
- ✅ Empty status history
- ✅ Multiple timestamp formats (ISO 8601 variants)
- ✅ AutoML experiment status tracking

**Key Feature**: Handles multiple ISO 8601 timestamp formats for robustness.

#### 2. **TestCheckJobTimeout** (8 tests)
Tests for determining if a job has exceeded the timeout threshold.

- ✅ Jobs within timeout window (not timed out)
- ✅ Jobs exceeding timeout window (timed out)
- ✅ Completed jobs (Done status - skipped)
- ✅ Error jobs (skipped)
- ✅ AutoML experiment timeout checking
- ✅ AutoML completed experiments (skipped)
- ✅ Fallback to `last_modified` when no status timestamps exist
- ✅ Missing job ID handling

**Key Feature**: Only checks jobs with "Running" or "Pending" status; completed jobs are never timed out.

#### 3. **TestTerminateTimedOutJob** (4 tests)
Tests for terminating jobs that have timed out.

- ✅ Regular job termination (status → "Error", StatefulSet deleted)
- ✅ AutoML experiment termination (experiment status → "error")
- ✅ Missing job information handling
- ✅ StatefulSet deletion failure handling

**Key Feature**: AutoML experiments update controller info and delete StatefulSets.

#### 4. **TestCheckForTimedOutJobs** (5 tests)
Tests for the main timeout monitoring loop.

- ✅ Detection and termination of multiple timed out jobs
- ✅ Timeout monitoring enable/disable via `JOB_TIMEOUT_MONITORING_ENABLED`
- ✅ No jobs timed out scenario
- ✅ Exception handling during timeout checks
- ✅ Mixed regular and AutoML job checking

**Key Feature**: Graceful error handling ensures one bad job doesn't crash the monitoring loop.

#### 5. **TestTimeoutResetOnRestart** (3 tests)
Tests verifying timeout timer resets when jobs are restarted or resumed.

- ✅ Resume job includes `delete_dnn_status` call
- ✅ Restart monitoring threads includes `delete_dnn_status` call  
- ✅ AutoML controller includes `delete_dnn_status` call

**Key Feature**: Uses source code inspection to verify the timeout reset logic is present without complex mocking.

#### 6. **TestTimeoutConfiguration** (3 tests)
Tests for timeout configuration via environment variables.

- ✅ Custom timeout value via `JOB_STATUS_TIMEOUT_MINUTES`
- ✅ Default timeout value (15 minutes)
- ✅ Monitoring enable/disable via `JOB_TIMEOUT_MONITORING_ENABLED`

**Key Feature**: Supports runtime configuration without code changes.

#### 7. **TestTimeoutWithStatusUpdates** (3 tests)
Tests for timeout behavior with various status update patterns.

- ✅ Continuous updates (should not timeout)
- ✅ Stale updates (should timeout)
- ✅ Boundary condition (just under threshold)

**Key Feature**: Tests real-world scenarios with different update frequencies.

## Running the Tests

### Run all timeout tests:
```bash
cd /localhome/local-rarunachalam/tao-core
python -m pytest nvidia_tao_core/tests/microservices/job_utils/test_workflow_timeout.py -v
```

### Run specific test class:
```bash
pytest nvidia_tao_core/tests/microservices/job_utils/test_workflow_timeout.py::TestCheckJobTimeout -v
```

### Run with coverage:
```bash
pytest nvidia_tao_core/tests/microservices/job_utils/test_workflow_timeout.py \
    --cov=nvidia_tao_core.microservices.job_utils.workflow \
    --cov=nvidia_tao_core.microservices.app_handlers.experiment_handler \
    --cov-report=html
```

### Run with markers:
```bash
pytest -m timeout -v
```

## Configuration

The timeout feature can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `JOB_STATUS_TIMEOUT_MINUTES` | `15` | Timeout duration in minutes |
| `JOB_TIMEOUT_MONITORING_ENABLED` | `"true"` | Enable/disable monitoring |

**Test Value**: Tests use `2` minutes for faster execution and clearer test values.

## Code Coverage

The test suite provides comprehensive coverage for:

### 1. Core Timeout Logic (`workflow.py`)
- `get_last_status_timestamp()` - Extract most recent status timestamp
- `check_job_timeout()` - Determine if job exceeded timeout
- `terminate_timed_out_job()` - Terminate timed out jobs
- `check_for_timed_out_jobs()` - Main monitoring loop

### 2. Timeout Reset on Restart
- `experiment_handler.py::resume_experiment_job()` - Regular job resume
- `workflow.py::Workflow.restart_threads()` - Monitoring thread restart
- `controller.py::Controller` - AutoML experiment restart

### 3. Edge Cases
- Missing data handling
- Exception handling
- Boundary conditions  
- Multiple timestamp formats
- Both regular and AutoML jobs
- Concurrent job monitoring

## Key Features Tested

### 1. Timeout Detection
Jobs are considered timed out when:
- Status is "Running" or "Pending"
- No status updates received for > `JOB_STATUS_TIMEOUT_MINUTES`
- For AutoML: experiment status is "running", "pending", or "started"

### 2. Timeout Action
When a job times out:
- **Regular jobs**: Status → "Error", StatefulSet deleted
- **AutoML**: Experiment status → "error" with timeout message, StatefulSet deleted

### 3. Timer Reset
Timeout timer resets (status history cleared) when:
- Job is explicitly resumed after being paused
- Monitoring thread is restarted after application restart
- AutoML experiment is restarted

### 4. Monitoring Loop
- Runs every 15 seconds as part of workflow scan loop
- Can be disabled via `JOB_TIMEOUT_MONITORING_ENABLED=false`
- Handles exceptions gracefully
- Checks both regular jobs and AutoML experiments

## Implementation Files

### Production Code
- `nvidia_tao_core/microservices/job_utils/workflow.py` - Core timeout logic
- `nvidia_tao_core/microservices/app_handlers/experiment_handler.py` - Resume job timeout reset
- `nvidia_tao_core/microservices/automl/controller.py` - AutoML timeout reset (existing)
- `nvidia_tao_core/microservices/constants.py` - Configuration constants

### Test Code
- `nvidia_tao_core/tests/microservices/job_utils/test_workflow_timeout.py` - All unit tests
- `nvidia_tao_core/tests/pytest.ini` - Pytest configuration (updated with timeout marker)

## Test Design Principles

### 1. **Fast Execution**
- All tests complete in < 1 second
- Uses 2-minute test timeout instead of 15-minute production timeout
- Minimal test setup/teardown

### 2. **Isolation**
- Extensive use of mocking to avoid external dependencies
- No MongoDB connections required
- No Kubernetes API calls required  
- No actual job termination

### 3. **Clarity**
- Clear test names describing what is being tested
- Comprehensive docstrings
- Easy-to-understand test values (1 minute, 5 minutes, etc.)

### 4. **Robustness**
- Tests handle timing edge cases
- Exception handling verified
- Multiple code paths tested

## Future Enhancements

Potential areas for additional testing:

1. **Integration Tests**
   - Real MongoDB connections
   - Real Kubernetes StatefulSet operations
   - End-to-end job lifecycle testing

2. **Performance Tests**
   - Load testing with many concurrent jobs
   - Monitoring loop performance under load
   - Scalability testing

3. **Stress Tests**
   - Rapid job creation/termination
   - Network failures during timeout checks
   - Database connection failures

4. **Additional Scenarios**
   - Multiple jobs timing out simultaneously
   - Jobs with intermittent status updates
   - Status update bursts after long silence

## Changelog

### Initial Implementation
- Added timeout monitoring feature to workflow.py
- Added timeout reset on job resume/restart
- Created comprehensive test suite with 31 tests
- Updated pytest.ini with timeout marker

## Contact

For questions or issues with the timeout feature or tests, please contact the TAO Core team.

