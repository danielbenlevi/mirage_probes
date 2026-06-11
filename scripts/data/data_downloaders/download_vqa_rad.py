#!/usr/bin/env python3
"""
Download VQA-RAD from Hugging Face and materialize raw files.

Expected converter input layout:
    raw_data/vqa_rad/VQA_RAD Dataset Public.json
    raw_data/vqa_rad/vqa_rad_balanced_split_and_human_eval_inclusions.tsv
    raw_data/vqa_rad/images/<image files>

This script supports parquet-style Hugging Face datasets (default: flaviagiammarino/vqa-rad)
and reconstructs the files expected by the VQA-RAD converter.

Usage:
    python scripts/data/data_downloaders/download_vqa_rad.py
    python scripts/data/data_downloaders/download_vqa_rad.py --force
    python scripts/data/data_downloaders/download_vqa_rad.py --repo-id flaviagiammarino/vqa-rad
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure local package import works when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VQA-RAD and convert to the raw_data format used by this repo."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="flaviagiammarino/vqa-rad",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=ORIGINAL_DATA_SOURCES["vqa_rad"]["data_dir"],
        help="Root target directory (vqa_rad) under DATA_ROOT.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
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


def require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required. Install with: pip install pyarrow") from exc
    return pq


def find_parquet_files(snapshot_dir: Path) -> List[Path]:
    return sorted(snapshot_dir.rglob("*.parquet"))


def infer_split_from_path(path: Path) -> Optional[str]:
    joined = "/".join(p.lower() for p in path.parts)
    if "/train/" in joined:
        return "train"
    if "/test/" in joined:
        return "test"
    return None


def normalize_yes_no(answer: str) -> Optional[str]:
    a = (answer or "").strip().lower().rstrip(".")
    if a in {"yes", "y", "true"}:
        return "yes"
    if a in {"no", "n", "false"}:
        return "no"
    return None


def extract_image_bytes(image_obj, snapshot_dir: Path) -> Tuple[Optional[bytes], Optional[str]]:
    if image_obj is None:
        return None, None

    if isinstance(image_obj, dict):
        b = image_obj.get("bytes")
        p = image_obj.get("path")
        if b:
            return b, p
        if p:
            pth = Path(p)
            if not pth.is_absolute():
                pth = snapshot_dir / pth
            if pth.exists() and pth.is_file():
                return pth.read_bytes(), str(p)
        return None, p

    if isinstance(image_obj, (bytes, bytearray)):
        return bytes(image_obj), None

    if isinstance(image_obj, str):
        pth = Path(image_obj)
        if not pth.is_absolute():
            pth = snapshot_dir / pth
        if pth.exists() and pth.is_file():
            return pth.read_bytes(), image_obj

    return None, None


def choose_image_name(path_hint: Optional[str], fallback_idx: int) -> str:
    if path_hint:
        name = Path(path_hint).name
        if name:
            return name
    return f"vqa_rad_{fallback_idx:06d}.jpg"


def write_if_needed(dst: Path, data: bytes, force: bool) -> bool:
    if dst.exists() and not force:
        return False
    dst.write_bytes(data)
    return True


def materialize_vqa_rad_from_parquet(snapshot_dir: Path, target_dir: Path, force: bool) -> Tuple[int, int, int]:
    pq = require_pyarrow()

    images_dir = target_dir / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / "VQA_RAD Dataset Public.json"
    tsv_path = target_dir / "vqa_rad_balanced_split_and_human_eval_inclusions.tsv"

    if (json_path.exists() or tsv_path.exists()) and not force:
        raise FileExistsError(
            f"{json_path} or {tsv_path} already exists. Use --force to overwrite."
        )

    parquets = find_parquet_files(snapshot_dir)
    if not parquets:
        raise FileNotFoundError("No parquet files found in downloaded VQA-RAD snapshot.")

    rows_with_split: List[Tuple[Dict, str]] = []
    for p in parquets:
        split_guess = infer_split_from_path(p)
        table = pq.read_table(p)
        for row in table.to_pylist():
            split = (row.get("split") or split_guess or "test").lower()
            if split not in {"train", "test"}:
                split = "test"
            rows_with_split.append((row, split))

    data_out: List[Dict] = []
    tsv_rows: List[Tuple[int, str]] = []
    written_images = 0

    for idx, (row, split) in enumerate(rows_with_split, start=1):
        qid = idx

        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question:
            continue

        image_obj = row.get("image") or row.get("images")
        image_bytes, path_hint = extract_image_bytes(image_obj, snapshot_dir)

        image_name = choose_image_name(path_hint=path_hint, fallback_idx=qid)
        if image_bytes:
            if write_if_needed(images_dir / image_name, image_bytes, force=force):
                written_images += 1
        else:
            # Try to copy by name from snapshot if image wasn't embedded.
            src_candidates = list(snapshot_dir.rglob(image_name))
            if src_candidates:
                dst = images_dir / image_name
                if not dst.exists() or force:
                    shutil.copy2(src_candidates[0], dst)
                    written_images += 1

        closed = normalize_yes_no(answer)
        answer_type = "CLOSED" if closed is not None else "OPEN"
        if closed is not None:
            answer = "yes" if closed == "yes" else "no"

        data_out.append(
            {
                "qid": qid,
                "image_name": image_name,
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "question_type": "binary" if answer_type == "CLOSED" else "open",
                "phrase_type": "test_para" if split == "test" else "para",
            }
        )
        tsv_rows.append((qid, split))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data_out, f, indent=2, ensure_ascii=False)

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("QID_unique\tSPLIT_BALANCED\n")
        for qid, split in tsv_rows:
            f.write(f"{qid}\t{split}\n")

    n_test = sum(1 for _, split in tsv_rows if split == "test")
    return len(data_out), n_test, written_images


def main() -> int:
    args = parse_args()
    target_dir = Path(args.target_dir).expanduser().resolve()

    snapshot_download = require_hf_hub()

    print(f"Downloading dataset: {args.repo_id}")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            allow_patterns=[
                "*.parquet",
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.webp",
                "*.gif",
            ],
            token=args.hf_token,
        )
    )
    print(f"Snapshot cached at: {snapshot_dir}")

    n_rows, n_test, n_images = materialize_vqa_rad_from_parquet(
        snapshot_dir=snapshot_dir,
        target_dir=target_dir,
        force=args.force,
    )

    print(f"Wrote {n_rows} rows to: {target_dir / 'VQA_RAD Dataset Public.json'}")
    print(f"Wrote {n_test} split entries marked as test to: {target_dir / 'vqa_rad_balanced_split_and_human_eval_inclusions.tsv'}")
    print(f"Wrote/updated {n_images} image files in: {target_dir / 'images'}")

    print("\nNext step:")
    print("  python scripts/data/build_datasets.py vqa_rad")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
