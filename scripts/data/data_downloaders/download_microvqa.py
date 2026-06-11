#!/usr/bin/env python3
"""
Download MicroVQA from Hugging Face and place files in this repo's raw_data layout.

By default, this script writes to:
    raw_data/MicroVQA

The MicroVQA converter expects test parquet files matching:
    test-*.parquet

Usage:
    python scripts/data/data_downloaders/download_microvqa.py
    python scripts/data/data_downloaders/download_microvqa.py --force
    python scripts/data/data_downloaders/download_microvqa.py --target-dir /custom/path/MicroVQA
"""

import argparse
import shutil
import sys
from pathlib import Path

# Make sure the local package is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download jmhb/microvqa dataset and prepare raw files."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="jmhb/microvqa",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=ORIGINAL_DATA_SOURCES["microvqa"]["data_dir"],
        help="Directory where test parquet files should be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing test-*.parquet files in target directory.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token (for gated/private datasets).",
    )
    return parser.parse_args()


def require_hf_hub():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from exc
    return snapshot_download


def find_test_parquets(snapshot_dir: Path):
    all_parquets = sorted(snapshot_dir.rglob("*.parquet"))
    if not all_parquets:
        return []

    test_parquets = [p for p in all_parquets if p.name.startswith("test-")]
    if test_parquets:
        return test_parquets

    # Fallback: accept any parquet if test-specific naming isn't present.
    return all_parquets


def copy_parquets(parquets, target_dir: Path, force: bool):
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in parquets:
        dst = target_dir / src.name
        if dst.exists() and not force:
            raise FileExistsError(
                f"File already exists: {dst}. Use --force to overwrite."
            )
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir).expanduser().resolve()

    snapshot_download = require_hf_hub()

    print(f"Downloading dataset: {args.repo_id}")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=["*.parquet"],
            token=args.hf_token,
        )
    )
    print(f"Snapshot cached at: {snapshot_dir}")

    parquets = find_test_parquets(snapshot_dir)
    if not parquets:
        print("No parquet files found in downloaded snapshot.", file=sys.stderr)
        return 1

    copied = copy_parquets(parquets, target_dir, force=args.force)

    print(f"Copied {len(copied)} parquet file(s) to: {target_dir}")
    for p in copied:
        print(f"  - {p.name}")

    print("\nNext step:")
    print("  python scripts/data/build_datasets.py microvqa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
