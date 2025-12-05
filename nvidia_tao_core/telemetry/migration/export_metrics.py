#!/usr/bin/env python3

# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""Export TAO telemetry metrics from MongoDB to JSON file.

This script exports metrics from MongoDB to a JSON file for backup,
analysis, or migration purposes.

Usage:
    # Export to file with timestamp
    python export_metrics.py

    # Export to specific file
    python export_metrics.py --output /path/to/metrics_backup.json

    # Pretty print
    python export_metrics.py --output metrics.json --pretty
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

try:
    from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_metrics
    HAS_MONGO_ACCESS = True
except ImportError:
    HAS_MONGO_ACCESS = False
    print("Error: MongoDB handlers not available.")
    print("Make sure nvidia_tao_core.microservices is installed.")
    sys.exit(1)


def export_metrics(output_file: str, pretty: bool = False) -> bool:
    """Export metrics from MongoDB to JSON file.

    Args:
        output_file: Path to output JSON file
        pretty: If True, format JSON with indentation

    Returns:
        True if successful
    """
    print("=" * 80)
    print("TAO Telemetry: Export Metrics from MongoDB")
    print("=" * 80)

    # Load metrics from MongoDB
    print("\n[1/3] Loading metrics from MongoDB...")
    metrics = get_metrics()

    if not metrics:
        print("   ✗ No metrics found in MongoDB!")
        print("   Database might be empty or connection failed.")
        return False

    # Count metrics
    metric_count = len([k for k in metrics.keys() if k not in ['_id', 'last_updated', 'name']])
    print(f"   ✓ Loaded {metric_count} metric entries")

    # Show some stats
    print("\n[2/3] Analyzing metrics...")

    comprehensive_count = sum(
        1 for k in metrics.keys()
        if k.startswith('network_') and '_version_' in k and '_status_' in k
    )
    legacy_count = sum(
        1 for k in metrics.keys()
        if k.startswith(('total_', 'version_', 'gpu_')) and '_version_' not in k
    )
    labeled_count = sum(
        1 for k in metrics.keys() if '{' in k and '}' in k
    )

    print(f"   Comprehensive metrics: {comprehensive_count}")
    print(f"   Legacy metrics: {legacy_count}")
    print(f"   Labeled metrics: {labeled_count}")
    print(f"   Other metrics: {metric_count - comprehensive_count - legacy_count - labeled_count}")

    # Export to file
    print(f"\n[3/3] Exporting to {output_file}...")

    try:
        # Ensure parent directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(metrics, f, indent=2, default=str, sort_keys=True)
            else:
                json.dump(metrics, f, default=str)

        # Get file size
        file_size = Path(output_file).stat().st_size

        print(f"   ✓ Exported {metric_count} metrics")
        print(f"   ✓ File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
        print(f"   ✓ Saved to: {output_file}")

        print("\n" + "=" * 80)
        print("✅ Export Successful!")
        print("=" * 80)
        print(f"\nMetrics exported to: {output_file}")
        print("\nTo import later, use:")
        print(f"  python import_metrics.py --input {output_file}")

        return True

    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Export TAO telemetry metrics from MongoDB to JSON file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with auto-generated filename
  python export_metrics.py

  # Export to specific file
  python export_metrics.py --output /tmp/metrics_backup.json

  # Export with pretty formatting
  python export_metrics.py --output metrics.json --pretty
        """
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        metavar='FILE',
        help='Output JSON file path (default: metrics_backup_YYYYMMDD_HHMMSS.json)'
    )

    parser.add_argument(
        '--pretty', '-p',
        action='store_true',
        help='Format JSON with indentation for readability'
    )

    args = parser.parse_args()

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Auto-generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'metrics_backup_{timestamp}.json'

    # Export
    success = export_metrics(output_file, args.pretty)

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
