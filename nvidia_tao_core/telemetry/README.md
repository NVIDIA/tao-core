# Telemetry Module

A configuration-driven architecture for processing and aggregating telemetry metrics data.

## Architecture Overview

The telemetry module provides an extensible framework for collecting and processing metrics. It uses a builder pattern with configuration-driven attribute processing, making it easy to add new telemetry attributes or metric types without modifying core logic.

### Key Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Configuration-Driven**: Add new attributes by updating config, not code
3. **Extensible**: Add new metric types by implementing builders
4. **Type-Safe**: Comprehensive type annotations for better IDE support
5. **Testable**: Each component can be tested independently

### File Structure

```
telemetry/
├── types.py              # Type definitions and data structures
├── config.py             # Metric attribute configuration
├── utils.py              # Core utility functions
├── processor.py          # Main orchestrator
└── builders/             # Metric builder implementations
    ├── __init__.py       # Exports all builders
    ├── base.py           # Abstract base class
    ├── legacy.py         # Legacy metrics (backward compatibility)
    ├── comprehensive.py  # Comprehensive metrics with all attributes
    └── time.py           # Time-based metrics
```

### Architecture Components

#### 1. **Types & Configuration** (`types.py`, `config.py`)
- Define the data structures and attribute configurations
- Central registry for all telemetry attributes
- Easy to extend with new attributes

#### 2. **Utilities** (`utils.py`)
- Core processing functions (sanitization, GPU identifiers, data extraction)
- Reusable across all builders
- Configuration-aware

#### 3. **Builders** (`builders/`)
- Pluggable metric generators following a common interface
- Each builder focuses on one type of metric
- Easy to add custom builders

#### 4. **Processor** (`processor.py`)
- Orchestrates all builders
- Manages context and timestamps
- Provides simple API for metric processing

## Key Components

### 1. Type Definitions (`types.py`)

Defines the core data structures:

- **`TelemetryData`**: TypedDict for normalized telemetry data
- **`AttributeType`**: Enum for attribute types (STRING, BOOLEAN, INTEGER, LIST)
- **`MetricAttribute`**: Dataclass for attribute configuration

### 2. Configuration (`config.py`)

Central registry for metric attributes:

```python
METRIC_ATTRIBUTES = [
    MetricAttribute(
        name='version',
        raw_key='version',
        attr_type=AttributeType.STRING,  # Automatically sanitized
        default='unknown',
        metric_order=3
    ),
    MetricAttribute(
        name='user_error',
        raw_key='user_error',
        attr_type=AttributeType.BOOLEAN,  # Used as-is, no sanitization
        default=False,
        metric_order=5
    ),
    # ... more attributes
]
```

**To add a new attribute**, simply add it to the `METRIC_ATTRIBUTES` list!

**Note**: Processing is automatic based on `attr_type`:
- `STRING` → sanitized (special chars removed, converted to lowercase)
- `BOOLEAN`, `INTEGER`, `LIST` → used as-is (no transformation)

### 3. Core Utilities (`utils.py`)

Essential processing functions:

- `sanitize_field_value()`: Sanitize values for metric names
- `create_gpu_identifier()`: Create unique GPU identifiers
- `extract_telemetry_data()`: Extract and normalize raw telemetry data

### 4. Metric Builders (`builders/`)

Pluggable metric generators:

#### Base Class (`base.py`)
```python
class MetricBuilder(ABC):
    @abstractmethod
    def build(self, metrics, telemetry_data, context):
        """Build and update metrics."""
        pass
```

#### Built-in Builders:
- **`LegacyMetricsBuilder`**: Backward-compatible metrics
- **`ComprehensiveMetricsBuilder`**: All-in-one metric names
- **`TimeMetricsBuilder`**: Time-based accumulation

### 5. Processor (`processor.py`)

Orchestrates metric building:

```python
processor = MetricProcessor()
metrics = processor.process({}, raw_data)
```

## Usage Examples

### Basic Usage

```python
from nvidia_tao_core.telemetry import MetricProcessor

# Initialize processor (uses default builders)
processor = MetricProcessor()

# Process telemetry data
metrics = {}
raw_data = {
    'action': 'train',
    'network': 'ResNet-50',
    'version': '5.3.0',
    'success': True,
    'gpu': ['NVIDIA A100', 'NVIDIA A100']
}

updated_metrics = processor.process(metrics, raw_data)
```

### Adding a Custom Builder

```python
from nvidia_tao_core.telemetry import MetricBuilder, MetricProcessor

class AlertMetricsBuilder(MetricBuilder):
    """Track failures for alerting."""
    
    def build(self, metrics, telemetry_data, context):
        if not telemetry_data['success']:
            key = f"failure_{telemetry_data['action']}"
            metrics[key] = metrics.get(key, 0) + 1

# Use custom builder
processor = MetricProcessor()
processor.add_builder(AlertMetricsBuilder())
```

### Custom Builder Collection

```python
from nvidia_tao_core.telemetry import (
    MetricProcessor,
    ComprehensiveMetricsBuilder,
    TimeMetricsBuilder
)

# Create processor with only specific builders
processor = MetricProcessor(builders=[
    ComprehensiveMetricsBuilder(),
    TimeMetricsBuilder(),
    CustomMetricsBuilder()
])
```

### Adding a New Telemetry Attribute

**Step 1**: Add to `config.py`:
```python
MetricAttribute(
    name='framework',
    raw_key='framework',
    attr_type=AttributeType.STRING,  # Auto-sanitized to lowercase
    default='pytorch',
    metric_order=6
)
```

**Step 2**: Update `TelemetryData` in `types.py`:
```python
class TelemetryData(TypedDict, total=False):
    # ... existing fields
    framework: str  # Add new field
```

**Step 3**: Done! STRING types are auto-sanitized; other types used as-is.

## Benefits of This Architecture

### ✅ Separation of Concerns
- Types are separate from logic
- Configuration is isolated and easy to modify
- Each builder handles one responsibility

### ✅ Easy to Extend
- Add new attributes in one place (`config.py`)
- Add new metric types by creating a builder
- No need to modify core logic

### ✅ Testable
- Each component can be tested independently
- Mock builders for processor tests
- Mock data extraction for builder tests

### ✅ Maintainable
- Clear file organization
- Self-documenting structure
- Easy to find and modify code

### ✅ Backward Compatible
- `utils.py` re-exports everything
- Existing code continues to work
- Gradual migration path

## Testing

Each component should be tested independently:

```python
# Test a builder
from nvidia_tao_core.telemetry.builders import LegacyMetricsBuilder

def test_legacy_builder():
    builder = LegacyMetricsBuilder()
    metrics = {}
    telemetry_data = {
        'action': 'train',
        'version': '5_3_0',
        'network': 'resnet50',
        'success': True,
        'gpus': ['nvidia_a100']
    }
    builder.build(metrics, telemetry_data, {})
    
    assert 'total_action_train_pass' in metrics
    assert metrics['total_action_train_pass'] == 1
```

## Import Guide

Recommended imports for common use cases:

```python
# Main processor
from nvidia_tao_core.telemetry.processor import MetricProcessor

# Utility functions
from nvidia_tao_core.telemetry.utils import (
    extract_telemetry_data,
    sanitize_field_value,
    create_gpu_identifier
)

# For custom builders
from nvidia_tao_core.telemetry.builders import MetricBuilder
from nvidia_tao_core.telemetry.types import TelemetryData

# For configuration
from nvidia_tao_core.telemetry.config import METRIC_ATTRIBUTES
from nvidia_tao_core.telemetry.types import MetricAttribute, AttributeType
```

## Best Practices

1. **Adding Attributes**: Always add to `config.py` and update `types.py`
2. **Custom Builders**: Keep them focused on one metric type
3. **Testing**: Test each builder independently
4. **Documentation**: Document metric formats and purposes
5. **Backward Compatibility**: Use `utils.py` for re-exports

## Performance Considerations

- Builders run sequentially
- Each builder is O(1) for most operations
- GPU identifier creation is O(n log n) where n = GPU count
- Comprehensive metrics can create long key names

## Future Enhancements

Potential improvements:
- Async builder support
- Metric validation
- Builder dependencies/ordering
- Metric aggregation strategies
- Export to different formats (JSON, Prometheus, etc.)

