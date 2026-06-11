#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.data.gen_mutations_get_responses as core
from vlm_model_registry import (
    DEFAULT_VLM_KEY,
    VLM_CHOICES,
    resolve_vlm_config,
    resolve_vllm_model_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build contrastive conversation pairs from mirage flip questions. "
            "Each pair contains one non-mirage-like variant and one mirage-like variant, "
            "including both with-image and without-image conversations."
        )
    )
    parser.add_argument(
        "--mirage_flip_path",
        type=str,
        default="",
        help=(
            "Input mirage flip artifact path. If omitted, defaults to "
            "./tmp_artifacts/<vlm_key>/mirage_flip_questions.json."
        ),
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default=DEFAULT_VLM_KEY,
        choices=VLM_CHOICES,
        help="Canonical VLM alias used for model-scoped artifact defaults.",
    )
    parser.add_argument(
        "--vllm_model_override",
        type=str,
        default="",
        help="Optional explicit served model name to record in metadata.",
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="",
        help="Deprecated alias for --vllm_model_override.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help=(
            "Output JSON path. If omitted, uses the standard artifact path, or a separate "
            "neutral-inclusive path when --neutral_as_non_mirage is set."
        ),
    )
    parser.add_argument(
        "--include_model_key_in_name",
        action="store_true",
        help="Append _<vlm_key> to default output filename.",
    )
    parser.add_argument(
        "--neutral_as_non_mirage",
        action="store_true",
        help="Treat variants with mirage_label='neutral' as eligible non-mirage variants.",
    )
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=-1,
        help="Optional cap on number of output pairs. -1 means no cap.",
    )
    return parser.parse_args()


def _variant_mirage_state(variant: Dict) -> str:
    wo = (variant.get("without_image", {}) or {})
    mirage_like = wo.get("mirage_like")
    if mirage_like is True:
        return "mirage"
    if mirage_like is False:
        return "non_mirage"
    label = str(wo.get("mirage_label", "")).strip().lower()
    if label.startswith("neutral"):
        return "neutral"
    return "unknown"


def _load_flip_groups(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list in {path}, got: {type(payload)}")
    return payload


def _resolve_default_flip_path(vlm_key: str) -> Path:
    data_root = Path("./tmp_artifacts").resolve()
    direct = (data_root / str(vlm_key) / "mirage_flip_questions.json").resolve()
    if direct.exists():
        return direct

    # Fallback for generation outputs saved under run-specific folders such as:
    # tmp_artifacts/gen_<vlm_key>*/mirage_flip_questions.json
    gen_candidates = sorted(
        data_root.glob(f"gen_{str(vlm_key)}*/mirage_flip_questions.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if gen_candidates:
        return gen_candidates[0].resolve()

    legacy = (data_root / "mirage_flip_questions.json").resolve()
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "Could not find mirage_flip_questions.json for "
        f"vlm_key='{vlm_key}'. Tried:\n"
        f"- {direct}\n"
        f"- {data_root}/gen_{vlm_key}*/mirage_flip_questions.json\n"
        f"- {legacy}\n"
        "Pass --mirage_flip_path explicitly if your artifact is elsewhere."
    )


def _select_variants(group: Dict, neutral_as_non_mirage: bool) -> Optional[Tuple[Dict, Dict]]:
    variants = group.get("variants", []) or []
    mirage = [v for v in variants if _variant_mirage_state(v) == "mirage"]
    if neutral_as_non_mirage:
        non_mirage = [v for v in variants if _variant_mirage_state(v) in {"non_mirage", "neutral"}]
    else:
        non_mirage = [v for v in variants if _variant_mirage_state(v) == "non_mirage"]
    if not non_mirage or not mirage:
        return None

    # Deterministic selection to keep artifact stable.
    non_mirage_sorted = sorted(non_mirage, key=lambda x: str(x.get("variant_id", "")))
    mirage_sorted = sorted(mirage, key=lambda x: str(x.get("variant_id", "")))
    return non_mirage_sorted[0], mirage_sorted[0]


def _build_with_image_conversation(variant_record: Dict, image_bytes_list: List[bytes]) -> List[Dict]:
    prompt_text = variant_record.get("prompt_text", "")
    system_prompt = variant_record.get("system_prompt", "")
    assistant_response = (variant_record.get("with_image", {}) or {}).get("response", "")

    messages = core._make_vllm_messages(
        prompt_text=prompt_text,
        image_bytes_list=image_bytes_list,
        system_prompt=system_prompt,
    )
    messages.append({"role": "assistant", "content": assistant_response})
    return messages


def _build_without_image_conversation(variant_record: Dict) -> List[Dict]:
    prompt_text = variant_record.get("prompt_text", "")
    system_prompt = variant_record.get("system_prompt", "")
    assistant_response = (variant_record.get("without_image", {}) or {}).get("response", "")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": assistant_response},
    ]


def _load_vqa_rad_items_raw_no_pandas(mirage_root: Path) -> List[Dict]:
    data_dir = mirage_root / "raw_data" / "vqa_rad"
    image_dir = data_dir / "images"
    splits_file = data_dir / "vqa_rad_balanced_split_and_human_eval_inclusions.tsv"
    data_file = data_dir / "VQA_RAD Dataset Public.json"

    if not splits_file.exists() or not data_file.exists():
        raise FileNotFoundError(
            "VQA-RAD raw files missing. Expected both: "
            f"{splits_file} and {data_file}"
        )

    test_ids = set()
    with open(splits_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if str(row.get("SPLIT_BALANCED", "")).strip().lower() == "test":
                test_ids.add(str(row.get("QID_unique", "")).strip())

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    output: List[Dict] = []
    idx = 0
    for row in raw_data:
        qid = str(row.get("qid", "")).strip()
        if qid not in test_ids:
            continue
        if str(row.get("answer_type", "")).upper() != "CLOSED":
            continue

        images: List[bytes] = []
        image_name = row.get("image_name", "")
        if image_name:
            image_path = image_dir / image_name
            if image_path.exists():
                images.append(image_path.read_bytes())

        output.append(
            {
                "unique_id": f"vqa_rad_raw_{idx}",
                "images": images,
            }
        )
        idx += 1

    return output


def _load_dataset_items_for_images(mirage_root: Path, dataset_name: str) -> List[Dict]:
    try:
        return core._load_dataset_items(mirage_root=mirage_root, dataset_name=dataset_name)
    except Exception:
        if dataset_name == "vqa_rad":
            return _load_vqa_rad_items_raw_no_pandas(mirage_root)
        raise


def main() -> None:
    args = parse_args()
    vlm_cfg = resolve_vlm_config(args.vlm)
    legacy_override = str(args.vllm_model).strip()
    explicit_override = str(args.vllm_model_override).strip()
    if explicit_override and legacy_override:
        raise ValueError("Specify only one of --vllm_model_override or --vllm_model (deprecated).")
    resolved_vllm_model = resolve_vllm_model_name(
        vlm_cfg.key,
        override_model_name=(explicit_override or legacy_override),
    )
    mirage_root = REPO_ROOT.resolve()
    if str(args.mirage_flip_path).strip():
        flip_path = Path(args.mirage_flip_path).expanduser().resolve()
    else:
        flip_path = _resolve_default_flip_path(vlm_cfg.key)
    if args.output_path.strip():
        out_path = Path(args.output_path).expanduser().resolve()
    else:
        data_root = Path("./tmp_artifacts")
        if vlm_cfg.key == DEFAULT_VLM_KEY:
            # Keep legacy default location for the default VLM.
            out_root = data_root
        else:
            out_root = data_root / vlm_cfg.key
        if args.neutral_as_non_mirage:
            base_name = "contrastive_conversation_pairs_neutral_as_non_mirage"
        else:
            base_name = "contrastive_conversation_pairs"
        if args.include_model_key_in_name:
            base_name = f"{base_name}_{vlm_cfg.key}"
        out_path = (out_root / f"{base_name}.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    groups = _load_flip_groups(flip_path)
    image_lookup_by_dataset_uid: Dict[str, Dict[str, List[bytes]]] = {}
    image_lookup_by_dataset_qid: Dict[str, Dict[str, List[bytes]]] = {}

    pairs: List[Dict] = []
    skipped_missing_variants = 0
    skipped_missing_images = 0

    for group in groups:
        if not group.get("mirage_flip_flag", False):
            continue

        selected = _select_variants(group, neutral_as_non_mirage=bool(args.neutral_as_non_mirage))
        if selected is None:
            skipped_missing_variants += 1
            continue

        non_mirage_variant, mirage_variant = selected
        dataset_name = str(group.get("dataset", "")).strip()
        unique_id = str(group.get("unique_id", "")).strip()
        question_id = str(group.get("question_id", "")).strip()

        if dataset_name not in image_lookup_by_dataset_uid:
            items = _load_dataset_items_for_images(mirage_root=mirage_root, dataset_name=dataset_name)
            image_lookup_by_dataset_uid[dataset_name] = {
                str(item["unique_id"]): item.get("images", []) or [] for item in items
            }
            image_lookup_by_dataset_qid[dataset_name] = {
                str(item.get("question_id", "")).strip(): item.get("images", []) or []
                for item in items
                if str(item.get("question_id", "")).strip()
            }

        image_bytes_list = image_lookup_by_dataset_uid[dataset_name].get(unique_id)
        if image_bytes_list is None and question_id:
            image_bytes_list = image_lookup_by_dataset_qid[dataset_name].get(question_id)
        if image_bytes_list is None:
            skipped_missing_images += 1
            continue

        pair = {
            "vlm_key": vlm_cfg.key,
            "vllm_model": resolved_vllm_model,
            "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
            "source_dataset": dataset_name,
            "source_unique_id": unique_id,
            "source_question_id": question_id,
            "non_mirage_variant_id": str(non_mirage_variant.get("variant_id", "")),
            "mirage_variant_id": str(mirage_variant.get("variant_id", "")),
            "non_mirage_conversation": _build_with_image_conversation(non_mirage_variant, image_bytes_list),
            "mirage_conversation": _build_with_image_conversation(mirage_variant, image_bytes_list),
            "non_mirage_without_image_conversation": _build_without_image_conversation(non_mirage_variant),
            "mirage_without_image_conversation": _build_without_image_conversation(mirage_variant),
        }
        pairs.append(pair)

        if args.max_pairs > 0 and len(pairs) >= args.max_pairs:
            break

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    metadata = {
        "vlm_key": vlm_cfg.key,
        "vllm_model": resolved_vllm_model,
        "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
        "mirage_flip_path": str(flip_path),
        "output_path": str(out_path),
        "neutral_as_non_mirage": bool(args.neutral_as_non_mirage),
        "include_model_key_in_name": bool(args.include_model_key_in_name),
        "input_groups": len(groups),
        "output_pairs": len(pairs),
        "skipped_missing_variants": skipped_missing_variants,
        "skipped_missing_images": skipped_missing_images,
    }
    with open(out_path.with_suffix(".metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
