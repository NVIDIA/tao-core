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

"""Import TAO telemetry metrics from JSON file to MongoDB.

This script imports metrics from a JSON file (created by export_metrics.py)
into MongoDB. Can be used for restore, migration, or testing.

Usage:
    # Preview import (dry run)
    python import_metrics.py --input metrics_backup.json --dry-run

    # Execute import (merge with existing)
    python import_metrics.py --input metrics_backup.json

    # Replace all metrics (dangerous!)
    python import_metrics.py --input metrics_backup.json --replace
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from nvidia_tao_core.microservices.utils.stateless_handler_utils import get_metrics, set_metrics
    HAS_MONGO_ACCESS = True
except ImportError:
    HAS_MONGO_ACCESS = False
    print("Error: MongoDB handlers not available.")
    print("Make sure nvidia_tao_core.microservices is installed.")
    sys.exit(1)


def import_metrics(input_file: str, dry_run: bool = True, replace: bool = False) -> bool:
    """Import metrics from JSON file to MongoDB.

    Args:
        input_file: Path to input JSON file
        dry_run: If True, only preview changes
        replace: If True, replace all existing metrics; if False, merge

    Returns:
        True if successful
    """
    print("=" * 80)
    print("TAO Telemetry: Import Metrics to MongoDB")
    print("=" * 80)

    # Load JSON file
    print(f"\n[1/4] Loading metrics from {input_file}...")

    if not Path(input_file).exists():
        print(f"   ✗ File not found: {input_file}")
        return False

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            imported_metrics = json.load(f)

        imported_count = len([k for k in imported_metrics.keys()
                             if k not in ['_id', 'last_updated', 'name']])
        file_size = Path(input_file).stat().st_size

        print(f"   ✓ Loaded {imported_count} metric entries")
        print(f"   ✓ File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    except Exception as e:
        print(f"   ✗ Failed to load file: {e}")
        return False

    # Load current MongoDB metrics
    print("\n[2/4] Loading current metrics from MongoDB...")

    current_metrics = get_metrics()

    if current_metrics:
        current_count = len([k for k in current_metrics.keys()
                            if k not in ['_id', 'last_updated', 'name']])
        print(f"   ✓ Current metrics in MongoDB: {current_count}")
    else:
        current_count = 0
        print("   ℹ  No existing metrics in MongoDB (will create new)")

    # Determine operation
    print("\n[3/4] Planning import operation...")

    if replace:
        print("   Mode: REPLACE (all existing metrics will be deleted!)")
        final_metrics = imported_metrics
        operation = "replace"
    else:
        print("   Mode: MERGE (imported metrics added to existing)")
        # Merge: imported metrics override existing for same keys
        final_metrics = current_metrics.copy() if current_metrics else {}
        final_metrics.update(imported_metrics)
        operation = "merge"

    final_count = len([k for k in final_metrics.keys()
                      if k not in ['_id', 'last_updated', 'name']])

    print(f"\n   Current metrics: {current_count}")
    print(f"   Imported metrics: {imported_count}")
    print(f"   Final metrics: {final_count}")

    if operation == "merge":
        new_count = imported_count - sum(1 for k in imported_metrics.keys() if k in (current_metrics or {}))
        updated_count = sum(1 for k in imported_metrics.keys() if k in (current_metrics or {}))
        print(f"   New metrics to add: {new_count}")
        print(f"   Existing metrics to update: {updated_count}")

    # Execute or dry run
    print(f"\n[4/4] {'DRY RUN - Previewing...' if dry_run else 'Executing import...'}")

    if dry_run:
        print("\n" + "=" * 80)
        print("DRY RUN MODE - No changes made to MongoDB")
        print("=" * 80)

        print("\nWhat will happen:")
        if replace:
            print(f"  ⚠️  REPLACE mode: {current_count} existing metrics will be DELETED")
            print(f"  ✓ {imported_count} metrics will be imported")
        else:
            print(f"  ✓ {final_count} total metrics after merge")
            if operation == "merge":
                print(f"  ✓ {new_count} new metrics will be added")
                print(f"  ✓ {updated_count} existing metrics will be updated")

        print("\nTo execute this import, run:")
        print(f"  python import_metrics.py --input {input_file}")

        if replace:
            print("\n⚠️  WARNING: Using --replace will delete all existing metrics!")

    else:
        # Execute import
        if replace and current_count > 0:
            # Confirm before replacing
            print(f"\n⚠️  WARNING: You are about to REPLACE {current_count} existing metrics!")
            print("   This will DELETE all current metrics and import new ones.")
            response = input("\n   Type 'YES' to confirm: ")
            if response != 'YES':
                print("   Import cancelled.")
                return False

        print("   Saving metrics to MongoDB...")

        try:
            # Remove _id field before update (MongoDB doesn't allow modifying _id)
            metrics_to_save = final_metrics.copy()
            if '_id' in metrics_to_save:
                del metrics_to_save['_id']

            set_metrics(metrics_to_save)

            print("   ✓ Import complete!")

            print("\n" + "=" * 80)
            print("✅ Import Successful!")
            print("=" * 80)

            if replace:
                print(f"\nReplaced all metrics with {imported_count} entries from {input_file}")
            else:
                print("\nMerged metrics:")
                print(f"  Previous: {current_count}")
                print(f"  Imported: {imported_count}")
                print(f"  Final: {final_count}")

            print("\nVerify in MongoDB:")
            print("  mongo tao --eval 'db.metrics.findOne()'")

            return True

        except Exception as e:
            print(f"   ✗ Import failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Import TAO telemetry metrics from JSON file to MongoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview import (safe)
  python import_metrics.py --input metrics_backup.json --dry-run

  # Execute import (merge with existing)
  python import_metrics.py --input metrics_backup.json

  # Replace all metrics (dangerous!)
  python import_metrics.py --input metrics_backup.json --replace

Notes:
  - Default mode is MERGE (adds to existing metrics)
  - REPLACE mode deletes all existing metrics first
  - Always run --dry-run first to preview changes
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        metavar='FILE',
        help='Input JSON file with metrics to import'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview import without making changes (recommended first step)'
    )

    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace all existing metrics (WARNING: deletes current metrics!)'
    )

    args = parser.parse_args()

    # Check file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    # Import
    success = import_metrics(
        input_file=args.input,
        dry_run=args.dry_run,
        replace=args.replace
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
