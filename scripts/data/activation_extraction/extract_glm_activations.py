#!/usr/bin/env python3
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.data.gen_mutations_get_responses as gen_core
import scripts.training.train_log_reg_all_examples as all_core
import scripts.training.train_log_reg_contrastive as pair_core


GLM_VLM_KEY = "glm_4_6v_flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-extract GLM-4.6V activations once for contrastive and all-examples trainers. "
            "The resulting caches are load-only inputs for glm trainer runs."
        )
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional explicit local model path override for GLM-4.6V-Flash.",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="./tmp_artifacts/contrastive_conversation_pairs.json",
    )
    parser.add_argument(
        "--neutral_pairs_path",
        type=str,
        default="./tmp_artifacts/contrastive_conversation_pairs_neutral_as_non_mirage.json",
    )
    parser.add_argument(
        "--include_neutral_pairs_source",
        action="store_true",
        help="Also include conversations from neutral-as-non-mirage contrastive pairs artifact if present.",
    )
    parser.add_argument(
        "--responses_path",
        type=str,
        default="./tmp_artifacts/responses.json",
    )
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--max_responses", type=int, default=-1)
    parser.add_argument(
        "--all_examples_max_samples",
        type=int,
        default=2000,
        help=(
            "Maximum number of all-examples rows to extract (best-effort balanced by "
            "benchmark, class, and class-within-benchmark). Set <=0 for no cap."
        ),
    )
    parser.add_argument(
        "--all_examples_sampling_seed",
        type=int,
        default=42,
        help="Random seed used by all-examples capped sampling.",
    )
    parser.add_argument("--contrastive_output_path", type=str, default="")
    parser.add_argument("--all_examples_output_path", type=str, default="")
    parser.add_argument("--skip_contrastive", action="store_true")
    parser.add_argument("--skip_all_examples", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--extract_additional_feature_caches",
        action="store_true",
        help=(
            "Write dedicated GLM pre-extracted caches for additional probe families "
            "(attention-head/post-attention/MLP) without residual families."
        ),
    )
    parser.add_argument("--include_attention_probes", action="store_true", default=False)
    parser.add_argument("--include_mlp_probes", action="store_true", default=False)
    parser.add_argument(
        "--include_additional_attention_mlp_probes",
        action="store_true",
        default=False,
        help="Alias to enable both --include_attention_probes and --include_mlp_probes.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="HF device_map passed to model loader. Keep 'auto' to shard across visible GPUs.",
    )
    parser.add_argument("--max_memory_per_gpu_gib", type=float, default=0.0)
    parser.add_argument("--max_memory_cpu_gib", type=float, default=0.0)
    parser.add_argument(
        "--min_visible_gpus",
        type=int,
        default=2,
        help="Fail if fewer than this many CUDA devices are visible.",
    )
    return parser.parse_args()


def _resolve_model_path(model_path: str) -> str:
    override = str(model_path or "").strip()
    if override:
        return override
    return pair_core._default_model_path_for_vlm(GLM_VLM_KEY)


def _resolve_scoped_input_path(base_path: Path, vlm_key: str) -> Path:
    resolver = getattr(pair_core, "_resolve_model_scoped_artifact_path", None)
    if callable(resolver):
        return Path(
            resolver(
                base_path=Path(base_path),
                vlm_key=str(vlm_key),
                include_gen_prefix_dirs=True,
            )
        )
    candidate = base_path.parent / vlm_key / base_path.name
    if candidate.exists():
        return candidate
    return base_path


def _default_contrastive_output_path() -> Path:
    return pair_core._glm_preextracted_contrastive_path()


def _default_all_examples_output_path() -> Path:
    return pair_core._glm_preextracted_all_examples_path()


def _default_contrastive_additional_output_path() -> Path:
    return pair_core._glm_preextracted_contrastive_additional_path()


def _default_all_examples_additional_output_path() -> Path:
    return pair_core._glm_preextracted_all_examples_additional_path()


def _ensure_writable_output(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not overwrite):
        raise FileExistsError(
            f"Output already exists: {path}. Re-run with --overwrite to replace it."
        )


def _extract_feature_rows(
    model,
    items: Sequence[Tuple[str, List[Dict]]],
    include_attention_probes: bool,
    include_mlp_probes: bool,
    include_residual_probes: bool,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    layer_features: Dict[str, List[np.ndarray]] = defaultdict(list)
    layer_order: List[str] = []
    seen = set()

    for signature_key, conversation in tqdm(items, desc="Extracting activations", unit="sample", dynamic_ncols=True):
        del signature_key
        model_messages = pair_core._to_model_messages(conversation=conversation, model_key=GLM_VLM_KEY)
        sample_feats = pair_core._extract_sample_features_only(
            model=model,
            messages=model_messages,
            include_additional_attention_mlp_probes=False,
            include_attention_probes=bool(include_attention_probes),
            include_mlp_probes=bool(include_mlp_probes),
            include_residual_probes=bool(include_residual_probes),
            model_key=GLM_VLM_KEY,
        )
        for key, value in sample_feats.items():
            if key not in seen:
                seen.add(key)
                layer_order.append(key)
            layer_features[key].append(value.to(torch.float32).cpu().numpy())

    layer_order = sorted(layer_order, key=pair_core._layer_sort_key)
    layer_features_out = {
        key: np.asarray(layer_features[key], dtype=np.float32)
        for key in layer_order
    }
    return layer_features_out, layer_order


def _build_contrastive_items(
    pairs_paths: Sequence[Path],
    max_pairs: int,
) -> Tuple[List[Tuple[str, List[Dict]]], List[Tuple[str, List[Dict]]], Dict]:
    with_sig_to_conv: Dict[str, List[Dict]] = {}
    with_sig_to_meta: Dict[str, Dict] = {}
    without_sig_to_conv: Dict[str, List[Dict]] = {}
    without_sig_to_meta: Dict[str, Dict] = {}

    total_pairs_loaded = 0
    for path in pairs_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        if not isinstance(pairs, list):
            continue
        if int(max_pairs) > 0:
            pairs = pairs[: int(max_pairs)]
        total_pairs_loaded += int(len(pairs))

        for pair_idx, pair in enumerate(pairs):
            benchmark = pair_core._infer_benchmark_from_pair(pair)
            for label, key, wo_key in (
                (0, "non_mirage_conversation", "non_mirage_without_image_conversation"),
                (1, "mirage_conversation", "mirage_without_image_conversation"),
            ):
                conv = pair.get(key, [])
                if not isinstance(conv, list):
                    continue
                sig_key = pair_core._signature_key_from_conv(conv)
                if sig_key not in with_sig_to_conv:
                    with_sig_to_conv[sig_key] = conv
                    with_sig_to_meta[sig_key] = {
                        "label": int(label),
                        "benchmark": str(benchmark),
                        "source_pairs_path": str(path),
                        "source_pair_index": int(pair_idx),
                        "source_dataset": str(pair.get("source_dataset", "")),
                        "source_unique_id": str(pair.get("source_unique_id", "")),
                        "source_question_id": str(pair.get("source_question_id", "")),
                    }

                wo_conv = pair.get(wo_key, [])
                if isinstance(wo_conv, list) and wo_conv:
                    wo_sig = pair_core._signature_key_from_conv(wo_conv)
                    if wo_sig not in without_sig_to_conv:
                        without_sig_to_conv[wo_sig] = wo_conv
                        without_sig_to_meta[wo_sig] = {
                            "label": int(label),
                            "benchmark": str(benchmark),
                            "source_pairs_path": str(path),
                            "source_pair_index": int(pair_idx),
                            "source_dataset": str(pair.get("source_dataset", "")),
                            "source_unique_id": str(pair.get("source_unique_id", "")),
                            "source_question_id": str(pair.get("source_question_id", "")),
                        }

    with_keys = list(with_sig_to_conv.keys())
    without_keys = list(without_sig_to_conv.keys())
    with_items = [(k, with_sig_to_conv[k]) for k in with_keys]
    without_items = [(k, without_sig_to_conv[k]) for k in without_keys]
    metadata = {
        "num_source_pairs_total": int(total_pairs_loaded),
        "num_unique_with_image_conversations": int(len(with_items)),
        "num_unique_without_image_conversations": int(len(without_items)),
        "source_paths": [str(p) for p in pairs_paths],
        "with_signature_metadata": [with_sig_to_meta[k] for k in with_keys],
        "without_signature_metadata": [without_sig_to_meta[k] for k in without_keys],
    }
    return with_items, without_items, metadata


def _build_all_examples_items(
    responses_path: Path,
    mirage_root: Path,
    max_responses: int,
    all_examples_max_samples: int,
    sampling_seed: int,
) -> Tuple[List[Tuple[str, List[Dict]]], List[Tuple[str, List[Dict]]], Dict]:
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    if not isinstance(responses, list):
        raise ValueError(f"Expected list payload in responses path: {responses_path}")
    responses = [r for r in responses if str(r.get("dataset", "")) in set(all_core.ALL_BENCHMARK_DATASETS)]
    if int(max_responses) > 0:
        responses = responses[: int(max_responses)]
    num_responses_considered_pre_sampling = int(len(responses))

    sampled_rows, sampling_meta = _sample_balanced_all_examples_rows(
        rows=responses,
        max_samples=int(all_examples_max_samples),
        seed=int(sampling_seed),
    )

    image_lookup_uid, image_lookup_qid = all_core._build_image_lookup(
        mirage_root=mirage_root,
        responses=sampled_rows,
    )
    with_sig_to_conv: Dict[str, List[Dict]] = {}
    with_sig_meta: Dict[str, Dict] = {}
    without_sig_to_conv: Dict[str, List[Dict]] = {}
    without_sig_meta: Dict[str, Dict] = {}
    skipped_missing_images = 0

    for row in sampled_rows:
        ds = str(row.get("dataset", ""))
        uid = str(row.get("unique_id", ""))
        qid = str(row.get("question_id", ""))
        imgs = image_lookup_uid.get((ds, uid)) if uid else None
        if imgs is None:
            imgs = image_lookup_qid.get((ds, qid))
        if not imgs:
            skipped_missing_images += 1
            continue

        system_prompt = str(row.get("system_prompt", ""))
        prompt_text = str(row.get("prompt_text", ""))
        with_resp = str((row.get("with_image", {}) or {}).get("response", ""))
        without_resp = str((row.get("without_image", {}) or {}).get("response", ""))

        with_conv = gen_core._make_vllm_messages(
            prompt_text=prompt_text,
            image_bytes_list=imgs,
            system_prompt=system_prompt,
        )
        with_conv.append({"role": "assistant", "content": with_resp})

        without_conv = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": without_resp},
        ]

        meta = {
            "dataset": ds,
            "unique_id": uid,
            "question_id": qid,
            "variant_id": row.get("variant_id", ""),
        }

        with_sig = pair_core._signature_key_from_conv(with_conv)
        if with_sig not in with_sig_to_conv:
            with_sig_to_conv[with_sig] = with_conv
            with_sig_meta[with_sig] = dict(meta)

        without_sig = pair_core._signature_key_from_conv(without_conv)
        if without_sig not in without_sig_to_conv:
            without_sig_to_conv[without_sig] = without_conv
            without_sig_meta[without_sig] = dict(meta)

    with_keys = list(with_sig_to_conv.keys())
    without_keys = list(without_sig_to_conv.keys())
    with_items = [(k, with_sig_to_conv[k]) for k in with_keys]
    without_items = [(k, without_sig_to_conv[k]) for k in without_keys]
    metadata = {
        "responses_path": str(responses_path),
        "num_responses_considered_pre_sampling": int(num_responses_considered_pre_sampling),
        "num_responses_considered": int(len(sampled_rows)),
        "sampling": sampling_meta,
        "num_with_image_unique_conversations": int(len(with_items)),
        "num_without_image_unique_conversations": int(len(without_items)),
        "num_rows_skipped_missing_images": int(skipped_missing_images),
        "with_signature_metadata": [with_sig_meta[k] for k in with_keys],
        "without_signature_metadata": [without_sig_meta[k] for k in without_keys],
    }
    return with_items, without_items, metadata


def _row_binary_label(row: Dict) -> Optional[int]:
    wo = row.get("without_image", {}) or {}
    mirage_like = wo.get("mirage_like")
    if mirage_like is True:
        return 1
    if mirage_like is False:
        return 0
    return None


def _sample_balanced_all_examples_rows(
    rows: Sequence[Dict],
    max_samples: int,
    seed: int,
) -> Tuple[List[Dict], Dict]:
    rows_list = list(rows)
    rng = np.random.default_rng(int(seed))

    bucket_to_rows: Dict[Tuple[str, int], List[Dict]] = defaultdict(list)
    skipped_non_binary_or_missing_dataset = 0
    for row in rows_list:
        ds = str(row.get("dataset", "")).strip()
        label = _row_binary_label(row)
        if (not ds) or (label is None):
            skipped_non_binary_or_missing_dataset += 1
            continue
        bucket_to_rows[(ds, int(label))].append(row)

    eligible_total = int(sum(len(v) for v in bucket_to_rows.values()))
    if eligible_total <= 0:
        return [], {
            "seed": int(seed),
            "max_samples": int(max_samples),
            "num_rows_input": int(len(rows_list)),
            "num_rows_eligible_binary": 0,
            "num_rows_selected_for_extraction": 0,
            "num_rows_skipped_non_binary_or_missing_dataset": int(skipped_non_binary_or_missing_dataset),
            "bucket_counts_available": {},
            "bucket_counts_selected": {},
            "dataset_counts_available": {},
            "dataset_counts_selected": {},
            "class_counts_available": {"0": 0, "1": 0},
            "class_counts_selected": {"0": 0, "1": 0},
        }

    target = eligible_total if int(max_samples) <= 0 else min(int(max_samples), eligible_total)
    buckets = sorted(bucket_to_rows.keys())
    for key in buckets:
        rng.shuffle(bucket_to_rows[key])

    alloc: Dict[Tuple[str, int], int] = {key: 0 for key in buckets}
    if buckets:
        base = target // len(buckets)
        for key in buckets:
            alloc[key] = min(int(base), len(bucket_to_rows[key]))
    assigned = int(sum(alloc.values()))
    remaining = int(target - assigned)

    while remaining > 0:
        candidates = [key for key in buckets if alloc[key] < len(bucket_to_rows[key])]
        if not candidates:
            break
        rng.shuffle(candidates)
        candidates = sorted(
            candidates,
            key=lambda key: (len(bucket_to_rows[key]) - alloc[key]),
            reverse=True,
        )
        progressed = False
        for key in candidates:
            if remaining <= 0:
                break
            if alloc[key] >= len(bucket_to_rows[key]):
                continue
            alloc[key] += 1
            remaining -= 1
            progressed = True
        if not progressed:
            break

    selected_rows: List[Dict] = []
    bucket_counts_available: Dict[str, int] = {}
    bucket_counts_selected: Dict[str, int] = {}
    dataset_counts_available: Dict[str, int] = defaultdict(int)
    dataset_counts_selected: Dict[str, int] = defaultdict(int)
    class_counts_available: Dict[str, int] = {"0": 0, "1": 0}
    class_counts_selected: Dict[str, int] = {"0": 0, "1": 0}

    for (ds, label) in buckets:
        available = int(len(bucket_to_rows[(ds, label)]))
        chosen = int(alloc[(ds, label)])
        key_str = f"{ds}::class_{label}"
        bucket_counts_available[key_str] = available
        bucket_counts_selected[key_str] = chosen
        dataset_counts_available[ds] += available
        dataset_counts_selected[ds] += chosen
        class_counts_available[str(label)] += available
        class_counts_selected[str(label)] += chosen
        if chosen > 0:
            selected_rows.extend(bucket_to_rows[(ds, label)][:chosen])

    rng.shuffle(selected_rows)
    sampling_meta = {
        "seed": int(seed),
        "max_samples": int(max_samples),
        "num_rows_input": int(len(rows_list)),
        "num_rows_eligible_binary": int(eligible_total),
        "num_rows_selected_for_extraction": int(len(selected_rows)),
        "num_rows_skipped_non_binary_or_missing_dataset": int(skipped_non_binary_or_missing_dataset),
        "bucket_counts_available": bucket_counts_available,
        "bucket_counts_selected": bucket_counts_selected,
        "dataset_counts_available": dict(sorted(dataset_counts_available.items())),
        "dataset_counts_selected": dict(sorted(dataset_counts_selected.items())),
        "class_counts_available": class_counts_available,
        "class_counts_selected": class_counts_selected,
    }
    return selected_rows, sampling_meta


def main() -> None:
    args = parse_args()
    extract_additional_mode = bool(args.extract_additional_feature_caches)
    include_attention = bool(
        args.include_attention_probes
        or args.include_additional_attention_mlp_probes
        or extract_additional_mode
    )
    include_mlp = bool(
        args.include_mlp_probes
        or args.include_additional_attention_mlp_probes
        or extract_additional_mode
    )
    include_residual = not bool(extract_additional_mode)
    if bool(args.skip_contrastive) and bool(args.skip_all_examples):
        raise ValueError("Nothing to do: both --skip_contrastive and --skip_all_examples were set.")

    visible_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if visible_gpus < int(args.min_visible_gpus):
        raise RuntimeError(
            f"Expected at least {int(args.min_visible_gpus)} visible GPUs for GLM extraction, "
            f"but found {visible_gpus}. Set CUDA_VISIBLE_DEVICES accordingly."
        )

    model_path = _resolve_model_path(args.model_path)
    model = pair_core.load_vlm_for_extraction(
        model_path=model_path,
        attn_implementation=str(args.attn_implementation),
        device_map_raw=str(args.device_map),
        max_memory_per_gpu_gib=float(args.max_memory_per_gpu_gib),
        max_memory_cpu_gib=float(args.max_memory_cpu_gib),
    )
    pair_core._force_attention_backend(model, str(args.attn_implementation))

    if not bool(args.skip_contrastive):
        contrastive_output = (
            Path(args.contrastive_output_path).expanduser().resolve()
            if str(args.contrastive_output_path).strip()
            else (
                _default_contrastive_additional_output_path().resolve()
                if bool(extract_additional_mode)
                else _default_contrastive_output_path().resolve()
            )
        )
        _ensure_writable_output(path=contrastive_output, overwrite=bool(args.overwrite))

        pairs_paths: List[Path] = []
        main_pairs_path = _resolve_scoped_input_path(Path(args.pairs_path).expanduser().resolve(), GLM_VLM_KEY)
        pairs_paths.append(main_pairs_path)
        if bool(args.include_neutral_pairs_source):
            neutral_pairs_path = _resolve_scoped_input_path(
                Path(args.neutral_pairs_path).expanduser().resolve(),
                GLM_VLM_KEY,
            )
            if neutral_pairs_path.exists():
                pairs_paths.append(neutral_pairs_path)

        contrastive_items, contrastive_without_items, contrastive_meta = _build_contrastive_items(
            pairs_paths=pairs_paths,
            max_pairs=int(args.max_pairs),
        )
        if not contrastive_items:
            raise RuntimeError("No contrastive conversations found for GLM pre-extraction.")

        contrastive_features, contrastive_layer_order = _extract_feature_rows(
            model=model,
            items=contrastive_items,
            include_attention_probes=bool(include_attention),
            include_mlp_probes=bool(include_mlp),
            include_residual_probes=bool(include_residual),
        )
        contrastive_without_features: Dict[str, np.ndarray] = {}
        contrastive_without_layer_order: List[str] = []
        if contrastive_without_items:
            contrastive_without_features, contrastive_without_layer_order = _extract_feature_rows(
                model=model,
                items=contrastive_without_items,
                include_attention_probes=bool(include_attention),
                include_mlp_probes=bool(include_mlp),
                include_residual_probes=bool(include_residual),
            )
            if contrastive_without_layer_order != contrastive_layer_order:
                raise RuntimeError(
                    "With-image and without-image layer orders differ in GLM contrastive extraction."
                )
        contrastive_payload = {
            "cache_type": "glm_preextracted_contrastive_features",
            "vlm": GLM_VLM_KEY,
            "model_path": str(model_path),
            "feature_extraction_version": int(pair_core.FEATURE_EXTRACTION_VERSION),
            "include_attention_probes": bool(include_attention),
            "include_mlp_probes": bool(include_mlp),
            "include_residual_probes": bool(include_residual),
            "include_additional_attention_mlp_probes": bool(include_attention or include_mlp),
            "extract_additional_feature_caches": bool(extract_additional_mode),
            "layer_order": contrastive_layer_order,
            "layer_features": contrastive_features,
            "signature_keys": [k for k, _ in contrastive_items],
            "without_image_layer_features": contrastive_without_features,
            "without_image_signature_keys": [k for k, _ in contrastive_without_items],
            "metadata": contrastive_meta,
        }
        torch.save(contrastive_payload, contrastive_output)
        print(
            json.dumps(
                {
                    "contrastive_output_path": str(contrastive_output),
                    "num_with_image_conversations": int(len(contrastive_items)),
                    "num_without_image_conversations": int(len(contrastive_without_items)),
                    "num_features": int(len(contrastive_layer_order)),
                },
                indent=2,
            )
        )

    if not bool(args.skip_all_examples):
        all_examples_output = (
            Path(args.all_examples_output_path).expanduser().resolve()
            if str(args.all_examples_output_path).strip()
            else (
                _default_all_examples_additional_output_path().resolve()
                if bool(extract_additional_mode)
                else _default_all_examples_output_path().resolve()
            )
        )
        _ensure_writable_output(path=all_examples_output, overwrite=bool(args.overwrite))

        responses_path = _resolve_scoped_input_path(Path(args.responses_path).expanduser().resolve(), GLM_VLM_KEY)
        with_items, without_items, all_examples_meta = _build_all_examples_items(
            responses_path=responses_path,
            mirage_root=REPO_ROOT.resolve(),
            max_responses=int(args.max_responses),
            all_examples_max_samples=int(args.all_examples_max_samples),
            sampling_seed=int(args.all_examples_sampling_seed),
        )
        if not with_items:
            raise RuntimeError("No with-image all-examples conversations found for GLM pre-extraction.")
        if not without_items:
            raise RuntimeError("No without-image all-examples conversations found for GLM pre-extraction.")

        with_features, with_layer_order = _extract_feature_rows(
            model=model,
            items=with_items,
            include_attention_probes=bool(include_attention),
            include_mlp_probes=bool(include_mlp),
            include_residual_probes=bool(include_residual),
        )
        without_features, without_layer_order = _extract_feature_rows(
            model=model,
            items=without_items,
            include_attention_probes=bool(include_attention),
            include_mlp_probes=bool(include_mlp),
            include_residual_probes=bool(include_residual),
        )
        if with_layer_order != without_layer_order:
            raise RuntimeError(
                "With-image and without-image layer orders differ in GLM all-examples extraction."
            )

        all_examples_payload = {
            "cache_type": "glm_preextracted_all_examples_features",
            "vlm": GLM_VLM_KEY,
            "model_path": str(model_path),
            "feature_extraction_version": int(pair_core.FEATURE_EXTRACTION_VERSION),
            "include_attention_probes": bool(include_attention),
            "include_mlp_probes": bool(include_mlp),
            "include_residual_probes": bool(include_residual),
            "include_additional_attention_mlp_probes": bool(include_attention or include_mlp),
            "extract_additional_feature_caches": bool(extract_additional_mode),
            "layer_order": with_layer_order,
            "with_image": {
                "signature_keys": [k for k, _ in with_items],
                "layer_features": with_features,
            },
            "without_image": {
                "signature_keys": [k for k, _ in without_items],
                "layer_features": without_features,
            },
            "metadata": all_examples_meta,
        }
        torch.save(all_examples_payload, all_examples_output)
        print(
            json.dumps(
                {
                    "all_examples_output_path": str(all_examples_output),
                    "num_with_image_conversations": int(len(with_items)),
                    "num_without_image_conversations": int(len(without_items)),
                    "num_features": int(len(with_layer_order)),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
