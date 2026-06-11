#!/usr/bin/env python3
"""
Download MedXpertQA-MM from Hugging Face and materialize raw files.

Expected converter input layout:
    raw_data/MedXpertQA-MM/test.jsonl
    raw_data/MedXpertQA-MM/images/<image files>

Supported source layouts:
1) Canonical MedXpertQA repo style:
   - MM/test.jsonl
   - images.zip (or image files under MM/images)
2) Parquet-based datasets with embedded images/options (fallback)

Usage:
    python scripts/data/data_downloaders/download_medxpertqa_mm.py
    python scripts/data/data_downloaders/download_medxpertqa_mm.py --force
    python scripts/data/data_downloaders/download_medxpertqa_mm.py --repo-id TsinghuaC3I/MedXpertQA
"""

import argparse
import ast
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure local package import works when run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MedXpertQA-MM and convert to the raw_data format used by this repo."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="TsinghuaC3I/MedXpertQA",
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=ORIGINAL_DATA_SOURCES["medxpertqa_mm"]["data_dir"],
        help="Root target directory (MedXpertQA-MM) under DATA_ROOT.",
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


def find_mm_test_jsonl(snapshot_dir: Path) -> Optional[Path]:
    preferred = snapshot_dir / "MM" / "test.jsonl"
    if preferred.exists():
        return preferred

    # Allow alternative folder casing/layout.
    candidates = sorted(snapshot_dir.rglob("test.jsonl"))
    if not candidates:
        return None

    mm_candidates = [p for p in candidates if any(part.lower() == "mm" for part in p.parts)]
    return mm_candidates[0] if mm_candidates else candidates[0]


def find_zip_candidates(snapshot_dir: Path) -> List[Path]:
    zips = sorted(snapshot_dir.rglob("*.zip"))
    if not zips:
        return []
    preferred = [p for p in zips if "image" in p.name.lower()]
    return preferred if preferred else zips


def parse_option_text(option: str) -> Tuple[Optional[str], str]:
    if option is None:
        return None, ""
    s = str(option).strip()
    if len(s) >= 2 and s[0].upper() in "ABCDEFGHIJ" and s[1] in [":", ".", ")", " "]:
        letter = s[0].upper()
        text = s[2:].strip() if s[1] != " " else s[1:].strip()
        return letter, text
    return None, s


def _coerce_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
        return [x]
    try:
        return list(x)
    except Exception:
        return [x]


def extract_option_map(options_raw) -> Dict[str, str]:
    option_map: Dict[str, str] = {}
    if options_raw is None:
        return option_map

    if isinstance(options_raw, str):
        s = options_raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                options_raw = json.loads(s)
            except Exception:
                try:
                    options_raw = ast.literal_eval(s)
                except Exception:
                    pass
        elif s.startswith("[") and s.endswith("]"):
            try:
                options_raw = json.loads(s)
            except Exception:
                try:
                    options_raw = ast.literal_eval(s)
                except Exception:
                    pass

    if isinstance(options_raw, dict):
        for k, v in options_raw.items():
            letter = str(k).strip().upper()[:1]
            if letter in "ABCDEFGHIJ":
                option_map[letter] = str(v).strip()
        return option_map

    options_list = _coerce_list(options_raw)
    next_letters = iter("ABCDEFGHIJ")
    for opt in options_list:
        letter, text = parse_option_text(str(opt))
        if letter is None:
            try:
                letter = next(next_letters)
            except StopIteration:
                break
        option_map[letter] = text
    return option_map


def infer_label(answer_raw, option_map: Dict[str, str]) -> str:
    if answer_raw is None:
        return ""
    ans = str(answer_raw).strip()
    if len(ans) == 1 and ans.upper() in "ABCDEFGHIJ":
        return ans.upper()

    letter, _ = parse_option_text(ans)
    if letter is not None:
        return letter

    ans_norm = ans.lower().strip()
    for k, v in option_map.items():
        if ans_norm == str(v).lower().strip():
            return k
    return ""


def _extract_zip_to_images(images_zip: Path, images_dir: Path, force: bool) -> int:
    extracted = 0
    with zipfile.ZipFile(images_zip, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            name = Path(member.filename)
            if not name.name:
                continue

            # Flatten nested archive paths to plain filenames for converter compatibility.
            dst = images_dir / name.name
            if dst.exists() and not force:
                continue

            data = zf.read(member)
            dst.write_bytes(data)
            extracted += 1
    return extracted


def _normalize_image_refs(row: Dict) -> List[str]:
    refs = row.get("images")
    if refs is None:
        refs = row.get("image")
    if refs is None:
        refs = row.get("image_name")
    if refs is None:
        refs = row.get("image_path")

    if isinstance(refs, str):
        refs = [refs]
    elif isinstance(refs, dict):
        refs = [refs.get("path") or refs.get("image_name") or refs.get("name") or ""]

    refs_list = _coerce_list(refs)
    out = []
    for r in refs_list:
        if isinstance(r, dict):
            path_val = r.get("path") or r.get("image_name") or r.get("name")
            if path_val:
                out.append(str(path_val))
            continue
        s = str(r).strip()
        if s:
            out.append(s)
    return out


def _copy_image_if_found(image_ref: str, snapshot_dir: Path, images_dir: Path, force: bool) -> Optional[str]:
    ref_path = Path(image_ref)
    candidates = [
        snapshot_dir / ref_path,
        snapshot_dir / "MM" / ref_path,
        snapshot_dir / "images" / ref_path,
        snapshot_dir / "MM" / "images" / ref_path,
        snapshot_dir / ref_path.name,
    ]

    src = next((p for p in candidates if p.exists() and p.is_file()), None)
    if src is None:
        # Broad fallback search by filename.
        matches = list(snapshot_dir.rglob(ref_path.name))
        src = matches[0] if matches else None
    if src is None:
        return None

    dst_name = ref_path.name
    dst = images_dir / dst_name
    if not dst.exists() or force:
        shutil.copy2(src, dst)
    return dst_name


def _write_test_jsonl(records: Sequence[Dict], out_path: Path, force: bool) -> None:
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} exists. Use --force to overwrite.")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def materialize_from_mm_jsonl(snapshot_dir: Path, target_dir: Path, force: bool) -> Tuple[int, int]:
    mm_test = find_mm_test_jsonl(snapshot_dir)
    if mm_test is None:
        return 0, 0

    images_dir = target_dir / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    extracted_images = 0
    for z in find_zip_candidates(snapshot_dir):
        extracted_images += _extract_zip_to_images(z, images_dir, force=force)

    records_out: List[Dict] = []
    copied_images = 0

    with open(mm_test, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)

            option_map = extract_option_map(row.get("options") or row.get("choices"))
            label = infer_label(
                row.get("label") if row.get("label") is not None else row.get("answer"),
                option_map,
            )

            image_refs = _normalize_image_refs(row)
            image_names: List[str] = []
            for ref in image_refs:
                name = _copy_image_if_found(ref, snapshot_dir, images_dir, force=force)
                if name:
                    image_names.append(name)
                    copied_images += 1
                elif (images_dir / Path(ref).name).exists():
                    image_names.append(Path(ref).name)

            record = {
                "id": str(row.get("id") or row.get("question_id") or f"MM-{idx}"),
                "question": str(row.get("question", "")).strip(),
                "options": option_map,
                "images": image_names,
                "medical_task": row.get("medical_task", ""),
                "body_system": row.get("body_system", ""),
                "question_type": row.get("question_type", ""),
                "label": label,
            }
            records_out.append(record)

    _write_test_jsonl(records_out, target_dir / "test.jsonl", force=force)
    return len(records_out), max(extracted_images, copied_images)


def find_test_parquets(snapshot_dir: Path) -> List[Path]:
    all_parquets = sorted(snapshot_dir.rglob("*.parquet"))
    if not all_parquets:
        return []

    test_files = [p for p in all_parquets if p.name.startswith("test-")]
    if test_files:
        return test_files

    joined = ["/".join(x.lower() for x in p.parts) for p in all_parquets]
    inferred = [p for p, j in zip(all_parquets, joined) if "/test/" in j or " split=test" in j]
    if inferred:
        return inferred

    # Fallback to all parquet files.
    return all_parquets


def _extract_image_bytes(img_obj, snapshot_dir: Path) -> Optional[bytes]:
    if img_obj is None:
        return None
    if isinstance(img_obj, (bytes, bytearray)):
        return bytes(img_obj)
    if isinstance(img_obj, dict):
        if img_obj.get("bytes"):
            return img_obj["bytes"]
        path_val = img_obj.get("path")
        if path_val:
            p = Path(path_val)
            if not p.is_absolute():
                p = snapshot_dir / p
            if p.exists():
                return p.read_bytes()
    if isinstance(img_obj, str):
        p = Path(img_obj)
        if not p.is_absolute():
            p = snapshot_dir / p
        if p.exists():
            return p.read_bytes()
    return None


def write_medxpert_raw_from_parquet(
    snapshot_dir: Path,
    test_parquets: List[Path],
    target_dir: Path,
    force: bool,
) -> Tuple[int, int]:
    images_dir = target_dir / "images"
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    test_jsonl = target_dir / "test.jsonl"

    if test_jsonl.exists() and not force:
        raise FileExistsError(f"{test_jsonl} exists. Use --force to overwrite.")

    pq = require_pyarrow()
    rows = []
    for p in test_parquets:
        table = pq.read_table(p)
        rows.extend(table.to_pylist())

    written_images = 0
    with open(test_jsonl, "w", encoding="utf-8") as out_f:
        for idx, row in enumerate(rows):
            option_map = extract_option_map(row.get("options"))
            label = infer_label(row.get("answer"), option_map)

            image_entries = row.get("images") or row.get("image") or row.get("images_list") or []
            if not isinstance(image_entries, list):
                try:
                    image_entries = list(image_entries)
                except Exception:
                    image_entries = []

            image_hashes = row.get("image_hash") or []
            if not isinstance(image_hashes, list):
                image_hashes = []

            image_names: List[str] = []
            for j, img_obj in enumerate(image_entries):
                img_bytes = _extract_image_bytes(img_obj, snapshot_dir)
                if not img_bytes:
                    continue
                base = None
                if j < len(image_hashes):
                    base = str(image_hashes[j]).strip()
                if not base:
                    base = f"{row.get('id', idx)}_{j}"
                img_name = f"{base}.jpg"
                img_path = images_dir / img_name
                if img_path.exists() and not force:
                    image_names.append(img_name)
                    continue
                img_path.write_bytes(img_bytes)
                written_images += 1
                image_names.append(img_name)

            record = {
                "id": str(row.get("id", f"MM-{idx}")),
                "question": str(row.get("question", "")).strip(),
                "options": option_map,
                "images": image_names,
                "medical_task": row.get("medical_task", ""),
                "body_system": row.get("body_system", ""),
                "question_type": row.get("question_type", ""),
                "label": label,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(rows), written_images


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
                "*.jsonl",
                "*.zip",
                "*.parquet",
                "*.jpg",
                "*.jpeg",
                "*.png",
                "*.bmp",
                "*.webp",
            ],
            token=args.hf_token,
        )
    )
    print(f"Snapshot cached at: {snapshot_dir}")

    n_rows, n_images = materialize_from_mm_jsonl(snapshot_dir, target_dir, args.force)
    if n_rows == 0:
        test_parquets = find_test_parquets(snapshot_dir)
        if not test_parquets:
            print(
                "No MM/test.jsonl or suitable parquet files found in snapshot.",
                file=sys.stderr,
            )
            return 1
        n_rows, n_images = write_medxpert_raw_from_parquet(
            snapshot_dir=snapshot_dir,
            test_parquets=test_parquets,
            target_dir=target_dir,
            force=args.force,
        )

    print(f"Wrote {n_rows} records to: {target_dir / 'test.jsonl'}")
    print(f"Wrote/updated image files in: {target_dir / 'images'}")
    print(f"Image write activity count: {n_images}")

    print("\nNext step:")
    print("  python scripts/data/build_datasets.py medxpertqa_mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
