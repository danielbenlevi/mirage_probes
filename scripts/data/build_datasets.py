#!/usr/bin/env python3
"""Build local parquet datasets from repo-managed raw_data downloads."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.data_helpers.datasets.base import DatasetRegistry
from scripts.data.data_helpers.datasets.converters.medxpertqa_mm import MedXpertQAMMConverter  # noqa: F401
from scripts.data.data_helpers.datasets.converters.microvqa import MicroVQAConverter  # noqa: F401
from scripts.data.data_helpers.datasets.converters.mmmu_pro import MMMUProConverter  # noqa: F401
from scripts.data.data_helpers.datasets.converters.vqa_rad import VQARadConverter  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build local dataset parquet files from raw_data/ into data/."
    )
    parser.add_argument("datasets", nargs="*", help="Datasets to build (default: all)")
    parser.add_argument("--force", "-f", action="store_true", help="Force rebuild even if parquet exists")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name in DatasetRegistry.list_datasets():
            print(f"  - {name}")
        return 0

    datasets_to_build = args.datasets or DatasetRegistry.list_datasets()
    print("=" * 60)
    print("Building Dataset Parquet Files")
    print("=" * 60)
    print(f"Datasets: {datasets_to_build}")
    print(f"Force rebuild: {args.force}")

    success = 0
    failed = 0
    for dataset_name in datasets_to_build:
        print(f"\n{'=' * 40}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 40}")
        try:
            dataset_cls = DatasetRegistry.get(dataset_name)
            dataset_cls().build(force=args.force)
            success += 1
        except Exception as exc:
            print(f"  x Error: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Complete: {success} success, {failed} failed")
    print(f"{'=' * 60}")
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
