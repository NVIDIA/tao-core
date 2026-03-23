# TAO API v1 to v2 Migration Guide

## Overview

The TAO API v2 represents a significant architectural evolution from v1, designed to provide a more streamlined, consistent, and powerful interface for managing machine learning workflows. This guide provides comprehensive information for migrating from v1 to v2.

## Key Architectural Changes

### 1. **Unified Job-Centric Architecture**
- **v1**: Separate experiment and dataset job endpoints under their respective resources
- **v2**: Unified job management through a single `/jobs` endpoint
- **Impact**: Simplified API surface with consistent job operations regardless of job type

### 2. **First-Class Resource Endpoints**
- **v1**: Primary resources: Experiments, Workspaces, Datasets
- **v2**: Primary resources: **Jobs**, **Inference Microservices**, Workspaces, Datasets
- **Impact**: Jobs and Inference Microservices are now top-level resources with their own management endpoints

### 3. **Consolidated Job Creation**
- **v1**: Two-step process (create experiment → create job)
- **v2**: Single-step process (create job with experiment metadata)
- **Impact**: Reduced API calls and simplified workflow

### 4. **Enhanced Schema System**
- **v1**: Separate schemas for different job types
- **v2**: Polymorphic schemas using OneOfSchema pattern
- **Impact**: Type-safe job handling with automatic schema selection

### 5. **Improved Status Codes**
- **v1**: Inconsistent use of HTTP status codes
- **v2**: Proper HTTP status codes (201 for creation, 200 for retrieval/updates)
- **Impact**: Better REST compliance and clearer API semantics

## Migration Mapping

### Core Job Operations

#### Creating Jobs

**v1 Experiment Jobs:**
```http
# Step 1: Create experiment
POST /api/v1/orgs/{org_name}/experiments
{
  "name": "my-experiment",
  "network_arch": "classification_pyt",
  "description": "My experiment"
}

# Step 2: Create job
POST /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs
{
  "action": "train",
  "specs": {...}
}
```

**v2 Unified Jobs:**
```http
# Single step: Create job with experiment metadata
POST /api/v2/orgs/{org_name}/jobs
{
  "kind": "experiment",
  "name": "my-experiment",
  "network_arch": "classification_pyt",
  "description": "My experiment",
  "action": "train",
  "specs": {...}
}
```

**v1 Dataset Jobs:**
```http
POST /api/v1/orgs/{org_name}/datasets/{dataset_id}/jobs
{
  "action": "convert",
  "specs": {...}
}
```

**v2 Dataset Jobs:**
```http
POST /api/v2/orgs/{org_name}/jobs
{
  "kind": "dataset",
  "dataset_id": "{dataset_id}",
  "action": "convert",
  "specs": {...}
}
```

#### Retrieving Jobs

**v1:**
```http
GET /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}
GET /api/v1/orgs/{org_name}/datasets/{dataset_id}/jobs/{job_id}
```

**v2:**
```http
GET /api/v2/orgs/{org_name}/jobs/{job_id}
```

#### Listing Jobs

**v1:**
```http
GET /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs
GET /api/v1/orgs/{org_name}/datasets/{dataset_id}/jobs
```

**v2:**
```http
GET /api/v2/orgs/{org_name}/jobs
# Returns all jobs (experiment and dataset) with filtering support
```

#### Job Control Operations

**v1:**
```http
POST /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}:retry
POST /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}:cancel
POST /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}:pause
POST /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}:resume
```

**v2:**
```http
POST /api/v2/orgs/{org_name}/jobs/{job_id}:retry
POST /api/v2/orgs/{org_name}/jobs/{job_id}:cancel
POST /api/v2/orgs/{org_name}/jobs/{job_id}:pause
POST /api/v2/orgs/{org_name}/jobs/{job_id}:resume
```

### Schema Management

#### Retrieving Specs Schema

**v1 Multiple Endpoints:**
```http
# For experiments with job_id
GET /api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs/{job_id}/schema

# For experiments with base_experiment_id
GET /api/v1/orgs/{org_name}/experiments/{experiment_id}/specs/{action}/schema:base?base_experiment_id={id}

# For unknown experiment
GET /api/v1/orgs/{org_name}/experiments/{experiment_id}/specs/{action}/schema
```

**v2 Unified Endpoint:**
```http
# Single endpoint with query parameters
GET /api/v2/orgs/{org_name}/jobs:schema?job_id={job_id}
GET /api/v2/orgs/{org_name}/jobs:schema?base_experiment_id={id}&action={action}
GET /api/v2/orgs/{org_name}/jobs:schema?network_arch={network_arch}&action={action}
```

### Inference Microservices

**v1:** Not available as first-class endpoints

**v2 New Endpoints:**
```http
POST /api/v2/orgs/{org_name}/inference_microservices:start
POST /api/v2/orgs/{org_name}/inference_microservices/{job_id}:inference
GET  /api/v2/orgs/{org_name}/inference_microservices/{job_id}:status
POST /api/v2/orgs/{org_name}/inference_microservices/{job_id}:stop
```

### Base Experiments and Transfer Learning

**v1:**
```http
GET /api/v1/orgs/{org_name}/experiments:base
POST /api/v1/orgs/{org_name}/experiments:load_airgapped
```

**v2:**
```http
GET /api/v2/orgs/{org_name}/jobs:list_base_experiments
POST /api/v2/orgs/{org_name}/jobs:load_airgapped
```

## Schema Changes

### Request Schemas

#### v1 Experiment Creation
```json
{
  "name": "string",
  "description": "string",
  "network_arch": "classification_pyt",
  "base_experiment": "uuid",
  "tags": ["tag1", "tag2"]
}
```

#### v2 Experiment Job Creation
```json
{
  "kind": "experiment",
  "name": "string",
  "description": "string",
  "network_arch": "classification_pyt",
  "base_experiment_ids": ["uuid1", "uuid2"],
  "tags": ["tag1", "tag2"],
  "action": "train",
  "specs": {...}
}
```

#### v2 Dataset Job Creation
```json
{
  "kind": "dataset",
  "dataset_id": "uuid",
  "action": "convert",
  "specs": {...}
}
```

### Response Schemas

#### v2 Polymorphic Job Response
The v2 API uses polymorphic schemas that automatically adapt based on job type:

```json
{
  "id": "uuid",
  "kind": "experiment|dataset",
  "status": "pending|running|done|error|paused|canceled",
  "action": "train|evaluate|export|...",
  "created_on": "2024-01-01T00:00:00Z",
  "last_modified": "2024-01-01T00:00:00Z",
  
  // Experiment-specific fields (when kind="experiment")
  "name": "string",
  "description": "string",
  "network_arch": "classification_pyt",
  "base_experiment_ids": ["uuid"],
  "tags": ["tag1", "tag2"],
  
  // Dataset-specific fields (when kind="dataset")
  "dataset_id": "uuid",
  "tags": ["tag1", "tag2"],
  
  // Common fields
  "specs": {...},
  "job_details": {...},
  "num_gpu": 1,
  "platform_id": "uuid"
}
```

## Removed/Deprecated Endpoints

### Experiment Management (Obsolete in v2)
- `GET /api/v1/orgs/{org_name}/experiments` → Use job listing with filtering
- `POST /api/v1/orgs/{org_name}/experiments` → Integrated into job creation
- `GET /api/v1/orgs/{org_name}/experiments/{id}` → Use job retrieval
- `DELETE /api/v1/orgs/{org_name}/experiments/{id}` → Use job deletion
- `PUT/PATCH /api/v1/orgs/{org_name}/experiments/{id}` → Use job updates

### Bulk Operations (Replaced with Individual Operations)
- `DELETE /api/v1/orgs/{org_name}/experiments/{id}/jobs` → List + delete individual jobs
- `POST /api/v1/orgs/{org_name}/experiments:cancel_all_jobs` → List + cancel individual jobs
- `POST /api/v1/orgs/{org_name}/datasets:cancel_all_jobs` → List + cancel individual jobs

### Internal Endpoints (Not for Client Use)
- `POST /api/v1/orgs/{org_name}/*/jobs/{job_id}:log_update` → Internal only
- `POST /api/v1/orgs/{org_name}/*/jobs/{job_id}:status_update` → Internal only

## Migration Strategy

### 1. **Immediate Changes Required**

#### Update Base URLs
- Change all `/api/v1/` to `/api/v2/`
- Update job-related endpoints to use `/jobs/` instead of `/experiments/{id}/jobs/` or `/datasets/{id}/jobs/`

#### Update Job Creation Flow
```python
# v1 approach
def create_experiment_job_v1(org_name, experiment_data, job_data):
    # Step 1: Create experiment
    exp_response = post(f"/api/v1/orgs/{org_name}/experiments", experiment_data)
    experiment_id = exp_response.json()["id"]
    
    # Step 2: Create job
    job_response = post(f"/api/v1/orgs/{org_name}/experiments/{experiment_id}/jobs", job_data)
    return job_response.json()

# v2 approach
def create_experiment_job_v2(org_name, combined_data):
    # Single step: Create job with experiment metadata
    combined_data["kind"] = "experiment"
    job_response = post(f"/api/v2/orgs/{org_name}/jobs", combined_data)
    return job_response.json()
```

#### Update Schema Retrieval
```python
# v1 approach
def get_schema_v1(org_name, experiment_id, action):
    return get(f"/api/v1/orgs/{org_name}/experiments/{experiment_id}/specs/{action}/schema")

# v2 approach
def get_schema_v2(org_name, **kwargs):
    params = {k: v for k, v in kwargs.items() if v is not None}
    return get(f"/api/v2/orgs/{org_name}/jobs:schema", params=params)
```

### 2. **Gradual Migration Steps**

#### Phase 1: Update Core Job Operations
1. Migrate job creation, retrieval, and control operations
2. Update job listing and filtering logic
3. Test basic job workflows

#### Phase 2: Update Schema Management
1. Consolidate schema retrieval calls
2. Update spec validation logic
3. Test schema-dependent operations

#### Phase 3: Implement New Features
1. Add inference microservice support
2. Implement enhanced filtering and pagination
3. Utilize new polymorphic response handling

#### Phase 4: Cleanup and Optimization
1. Remove v1-specific code paths
2. Optimize for v2 patterns (single job creation, unified endpoints)
3. Update error handling for new status codes

### 3. **Testing Strategy**

#### Compatibility Testing
```python
def test_job_creation_compatibility():
    # Test v1-style data works with v2 endpoint
    v1_experiment_data = {
        "name": "test-experiment",
        "network_arch": "classification_pyt",
        "description": "Test experiment"
    }
    
    v1_job_data = {
        "action": "train",
        "specs": {...}
    }
    
    # Convert to v2 format
    v2_data = {
        "kind": "experiment",
        **v1_experiment_data,
        **v1_job_data
    }
    
    response = post(f"/api/v2/orgs/{org_name}/jobs", v2_data)
    assert response.status_code == 201
```

#### Response Validation
```python
def test_polymorphic_response():
    # Test that response adapts to job type
    experiment_job = create_job(kind="experiment", ...)
    dataset_job = create_job(kind="dataset", ...)
    
    # Experiment job should have experiment-specific fields
    assert "network_arch" in experiment_job
    assert "name" in experiment_job
    
    # Dataset job should have dataset-specific fields
    assert "dataset_id" in dataset_job
    assert "network_arch" not in dataset_job
```

## Error Handling Changes

### Status Code Updates
- **Job Creation**: v1 returned `200`, v2 returns `201`
- **Resource Creation**: All creation operations now return `201`
- **Validation Errors**: More detailed error responses with structured validation details

### Error Response Format
```json
{
  "error_desc": "Detailed error description",
  "error_code": 1,
  "validation_details": {
    "field_name": ["error message 1", "error message 2"]
  }
}
```

## Performance Improvements

### Reduced API Calls
- **v1**: 2 calls for experiment job creation (experiment + job)
- **v2**: 1 call for job creation
- **Impact**: ~50% reduction in API calls for job creation workflows

### Enhanced Filtering
- **v1**: Basic filtering on individual resource endpoints
- **v2**: Advanced filtering across all job types with consistent query parameters
- **Impact**: More efficient data retrieval and reduced client-side filtering

### Pagination Improvements
- **v1**: Inconsistent pagination across endpoints
- **v2**: Consistent pagination with `skip`, `size`, and `sort` parameters
- **Impact**: Better performance for large datasets and consistent UX

## Best Practices for v2

### 1. **Use Polymorphic Schemas**
```python
# Handle different job types gracefully
def process_job_response(job_data):
    if job_data["kind"] == "experiment":
        return process_experiment_job(job_data)
    elif job_data["kind"] == "dataset":
        return process_dataset_job(job_data)
```

### 2. **Leverage Unified Endpoints**
```python
# Single function for all job operations
def get_job(org_name, job_id):
    return get(f"/api/v2/orgs/{org_name}/jobs/{job_id}")

def list_jobs(org_name, **filters):
    return get(f"/api/v2/orgs/{org_name}/jobs", params=filters)
```

### 3. **Use Enhanced Filtering**
```python
# Filter jobs by multiple criteria
jobs = list_jobs(
    org_name="my-org",
    network_arch="classification_pyt",
    status="running",
    user_only=True,
    sort="date-descending"
)
```

### 4. **Handle New Status Codes**
```python
def create_job(org_name, job_data):
    response = post(f"/api/v2/orgs/{org_name}/jobs", job_data)
    if response.status_code == 201:  # v2 uses 201 for creation
        return response.json()
    else:
        handle_error(response)
```

## Troubleshooting Common Migration Issues

### 1. **404 Errors on Job Endpoints**
**Problem**: Using v1 endpoint patterns with v2
**Solution**: Update to unified job endpoints
```python
# Wrong
get(f"/api/v2/orgs/{org}/experiments/{exp_id}/jobs/{job_id}")

# Correct
get(f"/api/v2/orgs/{org}/jobs/{job_id}")
```

### 2. **Schema Validation Errors**
**Problem**: Using v1 schema structure with v2 endpoints
**Solution**: Add `kind` field and update schema structure
```python
# Wrong
job_data = {"action": "train", "specs": {...}}

# Correct
job_data = {"kind": "experiment", "action": "train", "specs": {...}}
```

### 3. **Missing Experiment Metadata**
**Problem**: Expecting separate experiment creation step
**Solution**: Include experiment metadata in job creation
```python
# v2 requires all metadata in single call
job_data = {
    "kind": "experiment",
    "name": "experiment-name",
    "network_arch": "classification_pyt",
    "action": "train",
    "specs": {...}
}
```

### 4. **Status Code Mismatches**
**Problem**: Expecting v1 status codes
**Solution**: Update status code handling
```python
# Update success checks
if response.status_code == 201:  # v2 creation
    handle_success(response)
elif response.status_code == 200:  # v2 retrieval/update
    handle_success(response)
```

## Conclusion

The migration from TAO API v1 to v2 represents a significant improvement in API design, consistency, and functionality. While it requires updates to existing code, the benefits include:

- **Simplified Architecture**: Unified job management reduces complexity
- **Better Performance**: Fewer API calls and enhanced filtering
- **Enhanced Features**: First-class inference microservice support
- **Future-Proof Design**: Extensible polymorphic schemas and consistent patterns

Following this migration guide will ensure a smooth transition to v2 while taking advantage of its enhanced capabilities.

## Support and Resources

- **API Documentation**: Available at `/api/v2/docs`
- **Schema Validation**: Use `/api/v2/orgs/{org_name}/jobs:schema` for dynamic schema retrieval
- **Testing**: Comprehensive test suites available for validation
- **Migration Tools**: Consider creating wrapper functions to ease the transition

For additional support or questions about specific migration scenarios, please refer to the API documentation or contact the development team.
