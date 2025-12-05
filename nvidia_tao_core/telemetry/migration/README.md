# TAO Telemetry Migration Guide

This guide explains how to migrate historical metrics from the old telemetry server to the new server.

## Overview

**Old Server:**
- Legacy metrics (e.g., `total_action_train_pass`)
- Comprehensive metrics (e.g., `network_resnet50_action_train_version_5_3_0_status_pass_gpu_1_NVIDIA_A100_1`)
- Time metrics (e.g., `time_lapsed_today`)

**New Server:**
- All old metrics (preserved for backward compatibility)
- **NEW** Labeled metrics (generated automatically by new jobs via `LabeledMetricsBuilder`)

## Migration Steps

### Step 1: Export Metrics from Old Server

On the **OLD server**, export existing metrics to a backup file:

```bash
# SSH to old server
ssh old-telemetry-server

# Export metrics from MongoDB
cd /path/to/tao-pytorch/tao-core
python nvidia_tao_core/telemetry/migration/export_metrics.py \
    --output /tmp/old_server_metrics.json \
    --pretty
```

**Output:**
- Creates `/tmp/old_server_metrics.json` with all current metrics
- Includes comprehensive, legacy, and time metrics
- Pretty-printed for inspection

### Step 2: Transfer Backup to New Server

Transfer the backup file from old server to new server:

```bash
# From your local machine or new server
scp old-telemetry-server:/tmp/old_server_metrics.json /tmp/
```

### Step 3: Import Metrics to New Server (Dry Run)

On the **NEW server**, preview the import first:

```bash
# SSH to new server
ssh new-telemetry-server

cd /path/to/tao-pytorch/tao-core
python nvidia_tao_core/telemetry/migration/import_metrics.py \
    --input /tmp/old_server_metrics.json \
    --dry-run
```

**Output:**
```
TAO Telemetry: Import Metrics to MongoDB
================================================================================

[1/4] Loading metrics from /tmp/old_server_metrics.json...
   ✓ Loaded 1847 metric entries
   ✓ File size: 1,234,567 bytes (1.18 MB)

[2/4] Loading current metrics from MongoDB...
   ℹ  No existing metrics in MongoDB (will create new)

[3/4] Planning import operation...
   Mode: MERGE (imported metrics added to existing)

   Current metrics: 0
   Imported metrics: 1847
   Final metrics: 1847

[4/4] DRY RUN - Previewing...

DRY RUN MODE - No changes made to MongoDB
================================================================================

What will happen:
  ✓ 1847 total metrics after merge
  ✓ 1847 new metrics will be added
  ✓ 0 existing metrics will be updated

To execute this import, run:
  python import_metrics.py --input /tmp/old_server_metrics.json
```

### Step 4: Import Metrics to New Server (Execute)

If dry run looks good, execute the import:

```bash
python nvidia_tao_core/telemetry/migration/import_metrics.py \
    --input /tmp/old_server_metrics.json
```

**Output:**
```
✅ Import Successful!
================================================================================

Merged metrics:
  Previous: 0
  Imported: 1847
  Final: 1847
```

### Step 5: Deploy Updated Application

Deploy the updated TAO application with `LabeledMetricsBuilder` enabled.

From now on, **new jobs** will automatically generate labeled metrics while old metrics remain for history.

### Step 6: Deploy Updated Exporter

Update the Prometheus exporter to handle labeled metrics:

```bash
# Deploy updated exporter.py to telemetry gateway pod
kubectl apply -f telemetry-gateway-deployment.yaml
```

The updated exporter automatically:
- Detects flat metrics (old format) → creates Gauges
- Detects labeled metrics (new format) → creates Counters/Gauges based on naming convention
- Exposes both formats at `/metrics` endpoint

### Step 7: Verify Prometheus Scraping

Check that Prometheus is scraping the new labeled metrics:

```bash
# Port-forward to Prometheus
kubectl port-forward -n monitoring svc/kube-prometheus-stack-prometheus 9090:9090

# Open browser to http://localhost:9090
# Query: tao_job_total
# Should see labeled metrics with labels
```

### Step 8: Import Grafana Dashboards

Import the new dashboards that use labeled metrics:

1. Open Grafana UI
2. Go to **Dashboards → Import**
3. Upload JSON files from `nvidia_tao_core/telemetry/grafana_dashboards/`:
   - `tao_executive_dashboard.json` - Executive overview
   - `tao_network_analysis.json` - Network-specific analysis
   - `tao_gpu_utilization.json` - GPU usage tracking
   - `tao_version_migration.json` - Version adoption tracking

## Scripts Reference

### export_metrics.py

Export metrics from MongoDB to JSON file.

```bash
# Export with auto-generated filename
python export_metrics.py

# Export to specific file
python export_metrics.py --output /backup/metrics.json

# Pretty print for inspection
python export_metrics.py --output metrics.json --pretty
```

### import_metrics.py

Import metrics from JSON file to MongoDB.

```bash
# Preview import (safe)
python import_metrics.py --input metrics.json --dry-run

# Execute import (merge with existing)
python import_metrics.py --input metrics.json

# Replace all metrics (dangerous!)
python import_metrics.py --input metrics.json --replace
```

**Modes:**
- `--dry-run`: Preview only (default if no --execute)
- No flag: Execute merge (add to existing metrics)
- `--replace`: Delete all existing metrics and replace

**Important:** The script automatically removes `_id` field to avoid MongoDB immutable field errors.

## Migration Workflow Summary

```
┌─────────────────────┐
│   Old Server        │
│  (Legacy + Comp)    │
└──────────┬──────────┘
           │
           │ 1. export_metrics.py
           ↓
     ┌─────────────┐
     │ Backup JSON │
     └──────┬──────┘
            │
            │ 2. Transfer file
            ↓
┌─────────────────────┐
│   New Server        │
│   (Empty MongoDB)   │
└──────────┬──────────┘
           │
           │ 3. import_metrics.py
           ↓
┌─────────────────────┐
│   New Server        │
│  (Legacy + Comp)    │
│                     │
│  New jobs generate  │
│  labeled metrics    │
│  automatically      │
└─────────────────────┘
```

**Note:** Labeled metrics are generated automatically by new jobs via `LabeledMetricsBuilder`. No conversion of old data is needed.

## Rollback Procedure

If something goes wrong during migration:

### Rollback Step 1: Restore from Backup

```bash
# Restore from the backup created before conversion
python import_metrics.py --input /tmp/pre_conversion_backup.json --replace
```

**Warning:** Use `--replace` with caution as it deletes all current metrics!

### Rollback Step 2: Verify Restoration

```bash
# Analyze restored metrics
python convert_to_labeled.py --analyze
```

## Troubleshooting

### Error: "_id field immutable"

**Problem:** MongoDB doesn't allow updating `_id` field.

**Solution:** Fixed in v1.1 of `import_metrics.py` - automatically removes `_id` before update.

### Error: "MongoDB handlers not available"

**Problem:** Can't import from `nvidia_tao_core.microservices`.

**Solution:** Export from MongoDB manually:
```bash
# Export from MongoDB manually
mongo tao --eval 'printjson(db.metrics.findOne())' > /tmp/metrics.json

# Import on new server
python import_metrics.py --input /tmp/metrics.json
```

### No labeled metrics appearing

**Problem:** Labeled metrics not showing up for new jobs.

**Solution:** 
1. Verify `LabeledMetricsBuilder` is enabled in `processor.py`
2. Verify exporter is updated and restarted
3. Check Prometheus `/targets` to ensure scraping is active
4. Run a test job and check MongoDB for labeled metrics (keys with `{}`)

### Old dashboards broken

**Problem:** Old Grafana dashboards showing no data.

**Solution:** Old dashboards should still work! Check:
1. Old metrics are preserved (verify with `--analyze`)
2. Exporter is handling flat metrics (check `/metrics` endpoint)
3. Prometheus datasource is configured correctly

## Best Practices

1. **Always use dry-run first:**
   ```bash
   python import_metrics.py --input backup.json --dry-run  # Preview
   python import_metrics.py --input backup.json            # Execute
   ```

2. **Keep backups:**
   ```bash
   # Regular backups from new server
   python export_metrics.py --output /backup/metrics_$(date +%Y%m%d).json
   ```

3. **Monitor storage:**
   - New jobs generate labeled metrics automatically
   - Monitor MongoDB disk usage
   - Old metrics preserved for history
   - Consider retention policies for very old metrics

4. **Gradual transition:**
   - Import old metrics → Old dashboards work ✓
   - New jobs generate labeled metrics → New dashboards populate ✓
   - PMs get new insights from labeled metrics ✓
   - Old metrics remain for historical comparison ✓

## File Locations

```
tao-core/nvidia_tao_core/telemetry/
├── migration/
│   ├── export_metrics.py        # Export from MongoDB to JSON
│   ├── import_metrics.py        # Import from JSON to MongoDB
│   └── README.md                # This file
├── grafana_dashboards/
│   ├── tao_executive_dashboard.json
│   ├── tao_network_analysis.json
│   ├── tao_gpu_utilization.json
│   ├── tao_version_migration.json
│   └── tao_legacy_metrics.json  # For old comprehensive metrics
└── builders/
    └── labeled.py               # Generates labeled metrics for new jobs
```

## Timeline Estimate

| Step | Time | Risk |
|------|------|------|
| Export from old server | 1 min | Low |
| Transfer file | 1 min | Low |
| Import to new server (dry-run) | 1 min | None |
| Import to new server (execute) | 2 min | Low |
| Deploy application | 5 min | Medium |
| Deploy exporter | 5 min | Medium |
| Verify Prometheus | 2 min | Low |
| Import Grafana dashboards | 5 min | Low |
| **Total** | **~15 min** | **Low** |

## Support

If issues occur during migration:
1. Check rollback procedure above
2. Verify all backups are created
3. Contact TAO infrastructure team
4. Share backup files and error logs

