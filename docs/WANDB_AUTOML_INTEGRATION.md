# WandB AutoML Table Integration

## Overview

This integration adds comprehensive WandB tracking for AutoML experiments, logging all changing hyperparameters and metrics throughout the AutoML lifecycle.

## Implementation Details

### Changes Made

#### 1. `automl_start.py`
- Updated `automl_start()` to pass `parameter_names` (the list of hyperparameters being varied) to the Controller
- Both new Controller initialization and `load_state()` now receive `parameter_names`

#### 2. `controller.py`
Added WandB tracking capabilities to the AutoML Controller:

##### New Instance Variables:
- `self.parameter_names`: List of hyperparameter names being varied by AutoML
- `self.wandb_table`: WandB Table object for tracking experiments
- `self.wandb_initialized`: Flag indicating if WandB was successfully initialized

##### New Methods:

**`_initialize_wandb_for_automl()`**
- Retrieves `WANDB_API_KEY` from experiment metadata's `docker_env_vars`
- Logs into WandB using the API key
- Initializes a WandB run for the AutoML controller
- Groups all experiments under `self.wandb_group_name` (format: `automl_{job_id}`)
- Creates the tracking table

**`_create_wandb_table()`**
- Creates a WandB Table with columns:
  - `experiment_id`: Sequential ID of the recommendation (0, 1, 2, ...)
  - `job_id`: Unique job identifier
  - `status`: Current status (pending/running/success/failure)
  - `{metric_key}`: The optimization metric value (e.g., mAP, loss)
  - `best_epoch_number`: The epoch number with the best metric
  - All varying hyperparameters from `automl_hyperparameters`

**`_update_wandb_table()`**
- Updates the WandB table with the current state of all recommendations
- Called periodically throughout the AutoML lifecycle
- Recreates the table with all recommendations and their latest values
- Navigates nested hyperparameter specifications using dot notation

##### Integration Points:

1. **`start()`**: Initializes WandB at the beginning of the AutoML run
2. **`write_results()`**: Updates the WandB table periodically with latest metrics
3. **Completion/Error handlers**: Properly closes the WandB run and logs final table

## Usage

### Prerequisites

Set `WANDB_API_KEY` in the experiment's `docker_env_vars`:

```python
experiment_metadata["docker_env_vars"]["WANDB_API_KEY"] = "your_wandb_api_key"
```

### Table Contents

The logged WandB table (`automl_experiments`) contains:

| Column | Description | Example |
|--------|-------------|---------|
| experiment_id | Sequential experiment number | 0, 1, 2, ... |
| job_id | Unique job UUID | "abc-123-def-456" |
| status | Experiment status | "success", "failure", "pending" |
| {metric} | Optimization metric value | 0.85 (for mAP), 0.123 (for loss) |
| best_epoch_number | Best performing epoch | "epoch_10" |
| {param1} | First varying hyperparameter | 0.001 |
| {param2} | Second varying hyperparameter | 32 |
| ... | Additional hyperparameters | ... |

### Example

For an AutoML run optimizing learning rate and batch size for object detection:

```
experiment_id | job_id        | status  | mAP  | best_epoch_number | train.learning_rate | train.batch_size
--------------|---------------|---------|------|-------------------|---------------------|------------------
0             | uuid-1        | success | 0.75 | epoch_10          | 0.001               | 32
1             | uuid-2        | success | 0.82 | epoch_15          | 0.0005              | 64
2             | uuid-3        | failure | 0.00 | epoch_0           | 0.01                | 16
3             | uuid-4        | success | 0.89 | epoch_20          | 0.0008              | 48
```

## Benefits

1. **Centralized Tracking**: All AutoML experiments and their hyperparameters in one table
2. **Easy Comparison**: Compare hyperparameters and results across experiments
3. **Group Visualization**: All child experiment runs grouped together in WandB UI
4. **Historical Record**: Complete record of all tried hyperparameter combinations
5. **Progress Monitoring**: Real-time updates as experiments complete

## Notes

- WandB initialization is optional - if `WANDB_API_KEY` is not present, AutoML continues without WandB tracking
- All errors in WandB operations are caught and logged as warnings to prevent AutoML failure
- The table is updated periodically (every time `write_results()` is called)
- Child experiment runs are automatically grouped using `self.wandb_group_name`
- Nested hyperparameters (e.g., `train.optimizer.learning_rate`) are supported via dot notation

