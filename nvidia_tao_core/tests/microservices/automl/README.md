# AutoML Custom Parameter Ranges - Unit Tests

This directory contains comprehensive unit tests for the **Customizable AutoML Parameter Ranges** feature.

## Feature Overview

The customizable AutoML parameter ranges feature allows users to override default parameter ranges and options for AutoML experiments. This enables more fine-grained control over the hyperparameter search space.

## Test Files

### 1. `test_utils.py` (397 lines)
Tests for utility functions in `automl/utils.py`:

**TestGetValidRange:**
- `test_basic_range_no_custom_ranges` - Basic range without overrides
- `test_custom_min_override` - Custom minimum value override
- `test_custom_max_override` - Custom maximum value override
- `test_custom_min_and_max_override` - Both min and max overrides
- `test_inf_values_use_default` - Infinite values replaced with defaults
- `test_depends_on_greater_than` - Dependency with `>` operator
- `test_depends_on_greater_equal` - Dependency with `>=` operator
- `test_depends_on_less_than` - Dependency with `<` operator
- `test_depends_on_less_equal` - Dependency with `<=` operator
- `test_custom_depends_on_override` - Custom dependency override
- `test_depends_on_missing_parent` - Missing parent parameter handling
- `test_list_values_for_betas` - List values (optimizer betas)

**TestGetValidOptions:**
- `test_basic_options_no_custom_ranges` - Basic valid options
- `test_custom_options_override` - Custom options override
- `test_empty_valid_options` - Empty options handling
- `test_custom_options_with_integers` - Integer options
- `test_no_parameter_name` - Missing parameter name
- `test_custom_ranges_but_no_valid_options_key` - Partial custom ranges

**TestOtherUtilFunctions:**
- Tests for `clamp_value`, `fix_input_dimension`, and `fix_power_of_factor`

### 2. `test_stateless_handlers.py` (282 lines)
Tests for handler functions in `handlers/stateless_handlers.py`:

**TestGetAutoMLCustomParamRanges:**
- `test_get_custom_param_ranges_exists` - Retrieve existing custom ranges
- `test_get_custom_param_ranges_empty` - No custom ranges in experiment
- `test_get_custom_param_ranges_experiment_not_found` - Non-existent experiment
- `test_get_custom_param_ranges_with_various_types` - Various data types

**TestSaveAutoMLCustomParamRanges:**
- `test_save_custom_param_ranges` - Save basic custom ranges
- `test_save_empty_custom_param_ranges` - Save empty ranges
- `test_save_complex_custom_param_ranges` - Complex ranges with various types
- `test_save_and_get_roundtrip` - Save and retrieve roundtrip

### 3. `test_automl_algorithm_base.py` (487 lines)
Tests for base AutoML algorithm class in `automl/automl_algorithm_base.py`:

**TestAutoMLAlgorithmBaseInitialization:**
- `test_initialization_with_custom_ranges` - Load custom ranges during init
- `test_initialization_without_custom_ranges` - No custom ranges

**TestGenerateAutoMLParamRecValue:**
- `test_int_parameter_with_custom_min_max` - Integer with custom range
- `test_categorical_parameter_with_custom_options` - Categorical with custom options
- `test_ordered_int_parameter_with_custom_options` - Ordered int with custom options
- `test_bool_parameter_no_custom_override` - Boolean parameters
- `test_subset_list_parameter_with_custom_options` - Subset list with custom options
- `test_optional_list_parameter_with_custom_options` - Optional list with custom options
- `test_list_2_optimizer_betas_with_custom_ranges` - Optimizer betas with custom ranges

**TestApplyPowerConstraint:**
- `test_power_constraint_basic` - Basic power constraint
- `test_power_constraint_no_valid_powers` - No valid powers in range

### 4. `test_bayesian.py` (411 lines)
Tests for Bayesian AutoML algorithm in `automl/bayesian.py`:

**TestBayesianCustomRanges:**
- `test_initialization_with_custom_ranges` - Bayesian initialization with custom ranges
- `test_generate_param_rec_value_float_with_custom_range` - Float parameter generation
- `test_generate_param_rec_value_int_with_custom_range` - Integer parameter generation
- `test_generate_param_rec_value_categorical_with_custom_options` - Categorical parameter
- `test_multiple_parameters_with_different_custom_ranges` - Multiple parameters
- `test_custom_override_applied_before_processing` - Override application timing
- `test_no_custom_ranges_uses_schema_defaults` - Schema defaults fallback

**TestBayesianGetValidRangeIntegration:**
- `test_get_valid_range_called_with_custom_ranges` - Integration with get_valid_range

### 5. `test_hyperband.py` (523 lines)
Tests for HyperBand AutoML algorithm in `automl/hyperband.py`:

**TestHyperBandCustomRanges:**
- `test_initialization_with_custom_ranges` - HyperBand initialization with custom ranges
- `test_generate_param_rec_value_float_with_custom_range` - Float parameter generation
- `test_generate_param_rec_value_int_with_custom_range` - Integer parameter generation
- `test_generate_param_rec_value_categorical_with_custom_options` - Categorical parameter
- `test_multiple_parameters_with_different_custom_ranges` - Multiple parameters
- `test_custom_override_applied_before_processing` - Override application timing
- `test_no_custom_ranges_uses_schema_defaults` - Schema defaults fallback

**TestHyperBandGetValidRangeIntegration:**
- `test_get_valid_range_called_with_custom_ranges` - Integration with get_valid_range
- `test_ordered_int_parameter_with_custom_options` - Ordered int parameters

## Running the Tests

To run all AutoML tests:
```bash
cd /localhome/local-rarunachalam/tao-core
python -m pytest nvidia_tao_core/tests/microservices/automl/ -v
```

To run specific test files:
```bash
# Test utilities
python -m pytest nvidia_tao_core/tests/microservices/automl/test_utils.py -v

# Test handlers
python -m pytest nvidia_tao_core/tests/microservices/handlers/test_stateless_handlers.py -v

# Test base algorithm
python -m pytest nvidia_tao_core/tests/microservices/automl/test_automl_algorithm_base.py -v

# Test Bayesian
python -m pytest nvidia_tao_core/tests/microservices/automl/test_bayesian.py -v

# Test HyperBand
python -m pytest nvidia_tao_core/tests/microservices/automl/test_hyperband.py -v
```

To run with coverage:
```bash
python -m pytest nvidia_tao_core/tests/microservices/automl/ --cov=nvidia_tao_core.microservices.automl --cov-report=html
```

## Test Coverage

The test suite covers:
- ✅ Custom parameter range loading from experiments
- ✅ Custom parameter range application in all AutoML algorithms
- ✅ Valid range computation with custom overrides
- ✅ Valid options retrieval with custom overrides
- ✅ Parameter dependency handling with custom ranges
- ✅ Float, integer, categorical, and list parameter types
- ✅ Edge cases (missing data, empty ranges, invalid values)
- ✅ Integration between different components

## Key Testing Strategies

1. **Mocking**: Extensive use of `unittest.mock` to isolate components and avoid external dependencies (MongoDB, file system)
2. **Parametric Testing**: Multiple test cases for different data types and scenarios
3. **Edge Case Testing**: Tests for missing data, empty values, and boundary conditions
4. **Integration Testing**: Tests verifying correct interaction between components
5. **Randomness Testing**: Tests for random parameter generation verify values are within expected ranges

## Total Test Coverage

- **Total Lines**: 2,100 lines of test code
- **Total Test Cases**: 60+ individual test cases
- **Files Tested**: 5 source files
  - `automl/utils.py`
  - `automl/automl_algorithm_base.py`
  - `automl/bayesian.py`
  - `automl/hyperband.py`
  - `handlers/stateless_handlers.py`

