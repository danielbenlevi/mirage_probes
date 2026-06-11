#!/usr/bin/env python3
"""
Download MMMU-Pro from Hugging Face and place files in this repo's raw_data layout.

Expected converter input location:
    raw_data/MMMU-Pro/standard-4-options/test-*.parquet

Usage:
    python scripts/data/data_downloaders/download_mmmu_pro.py
    python scripts/data/data_downloaders/download_mmmu_pro.py --force
    python scripts/data/data_downloaders/download_mmmu_pro.py --repo-id MMMU/MMMU_Pro
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List

# Make sure the local package is importable when running directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MMMU-Pro raw parquet files for the dataset builder."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="MMMU/MMMU_Pro",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=ORIGINAL_DATA_SOURCES["mmmu_pro"]["data_dir"],
        help="Root target directory (MMMU-Pro) under DATA_ROOT.",
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


def find_standard4_test_parquets(snapshot_dir: Path) -> List[Path]:
    all_parquets = sorted(snapshot_dir.rglob("*.parquet"))
    if not all_parquets:
        return []

    # Common HF parquet conversion layout:
    # refs/convert/parquet/standard (4 options)/test/0000.parquet
    selected = []
    for p in all_parquets:
        parts_lower = [x.lower() for x in p.parts]
        has_standard4 = any(
            key in part
            for part in parts_lower
            for key in ["standard (4 options)", "standard-4-options", "standard_4_options"]
        )
        has_test = "test" in parts_lower
        if has_standard4 and has_test:
            selected.append(p)

    if selected:
        return sorted(selected)

    # Fallback: any parquet under folders with "standard" and "4 options"
    loose = []
    for p in all_parquets:
        joined = "/".join(x.lower() for x in p.parts)
        if "standard" in joined and ("4 options" in joined or "4-options" in joined):
            loose.append(p)
    if loose:
        return sorted(loose)

    return []


def copy_to_converter_layout(parquets: List[Path], target_root: Path, force: bool) -> List[Path]:
    target_dir = target_root / "standard-4-options"
    target_dir.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []
    for src in parquets:
        dst_name = src.name
        if not dst_name.startswith("test-"):
            dst_name = f"test-{dst_name}"
        dst = target_dir / dst_name
        if dst.exists() and not force:
            raise FileExistsError(f"File already exists: {dst}. Use --force to overwrite.")
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> int:
    args = parse_args()
    target_root = Path(args.target_dir).expanduser().resolve()

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

    parquets = find_standard4_test_parquets(snapshot_dir)
    if not parquets:
        print(
            "No standard-4-options test parquet files found in snapshot. "
            "Check repo-id or dataset layout.",
            file=sys.stderr,
        )
        return 1

    copied = copy_to_converter_layout(parquets, target_root, force=args.force)
    print(f"Copied {len(copied)} parquet file(s) to: {target_root / 'standard-4-options'}")
    for p in copied[:20]:
        print(f"  - {p.name}")
    if len(copied) > 20:
        print(f"  ... and {len(copied) - 20} more")

    print("\nNext step:")
    print("  python scripts/data/build_datasets.py mmmu_pro")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
