#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.data.gen_mutations_get_responses as core
import scripts.training.train_log_reg_contrastive as pair_core

MIN_RESPONSE_TOKENS = 10
DEFAULT_VLM = "ovis"
VLM_MODEL_PATHS = {
    "ovis": "AIDC-AI/Ovis2.5-2B",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "glm_4_6v_flash": "zai-org/GLM-4.6V-Flash",
}
ALL_BENCHMARK_DATASETS = ["vqa_rad", "mmmu_pro", "medxpertqa_mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train VLM probes from with-image examples using BClean labels: "
            "class1 from current mirage-like labels, class0 from "
            "(without_image incorrect AND with_image correct), "
            "with benchmark-balanced class sampling and benchmark-stratified 70/10/20 "
            "train/valid/test splits across multiple random seeds."
        )
    )
    parser.add_argument(
        "--responses_path",
        type=str,
        default="./tmp_artifacts/responses.json",
        help=(
            "All-examples responses artifact. If left at the default value, OVIS/QWEN use "
            "data/final_data/*_all_responses.json and GLM uses tmp_artifacts/responses.json."
        ),
    )
    parser.add_argument(
        "--vlm",
        type=str,
        choices=sorted(VLM_MODEL_PATHS.keys()),
        default=DEFAULT_VLM,
        help="VLM family used for activation extraction.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional explicit model path override. Defaults to --vlm canonical path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude_short_responses_in_training_examples",
        dest="exclude_short_responses_in_training_examples",
        action="store_true",
        help=(
            "Exclude training-candidate rows where with-image response has "
            f"fewer than {MIN_RESPONSE_TOKENS} whitespace-separated tokens."
        ),
    )
    parser.add_argument(
        "--no_exclude_short_responses_in_training_examples",
        dest="exclude_short_responses_in_training_examples",
        action="store_false",
        help="Keep short-response rows in training-candidate selection.",
    )
    parser.set_defaults(exclude_short_responses_in_training_examples=True)
    parser.add_argument(
        "--neutral_as_non_mirage",
        dest="neutral_as_non_mirage",
        action="store_true",
        help="Treat rows with mirage_label='neutral*' as class 0 (non-mirage-like).",
    )
    parser.add_argument(
        "--no_neutral_as_non_mirage",
        dest="neutral_as_non_mirage",
        action="store_false",
        help="Exclude neutral rows from class 0.",
    )
    parser.set_defaults(neutral_as_non_mirage=False)
    parser.add_argument(
        "--max_questions",
        type=int,
        default=-1,
        help="Optional cap on number of unique questions used per class after per-question sampling.",
    )
    parser.add_argument(
        "--target_examples_per_class",
        type=int,
        default=500,
        help=(
            "Target number of examples per class before split. For each benchmark, selected count "
            "is identical across class 0 and class 1, but can differ across benchmarks. "
            "Class 0 still prioritizes non-neutral rows first, "
            "then fills with random neutral rows when enabled."
        ),
    )
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--vqa_only_examples",
        action="store_true",
        help="Train/eval on VQA-RAD rows only.",
    )
    benchmark_group.add_argument(
        "--mmmu_only_examples",
        action="store_true",
        help="Train/eval on MMMU-Pro rows only.",
    )
    benchmark_group.add_argument(
        "--medxpert_only_examples",
        action="store_true",
        help="Train/eval on MedXpertQA-MM rows only.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument(
        "--regularization_values",
        type=str,
        default="0,0.01,0.1,1.0",
        help="Comma-separated C values for sweep (selection by validation accuracy).",
    )
    parser.add_argument(
        "--num_split_seeds",
        type=int,
        default=5,
        help="Number of random split seeds for reporting averaged test accuracy.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=200,
        help="Validation-loss early stopping patience (high by default).",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement to reset early-stopping patience.",
    )
    parser.add_argument("--probe_epochs", type=int, default=800)
    parser.add_argument("--probe_lr", type=float, default=0.03)
    parser.add_argument(
        "--multi_init_probe_selection",
        dest="multi_init_probe_selection",
        action="store_true",
        default=True,
        help="Try multiple random initializations per (feature, C, split) and keep best by validation accuracy.",
    )
    parser.add_argument(
        "--no_multi_init_probe_selection",
        dest="multi_init_probe_selection",
        action="store_false",
        help="Disable multi-initialization probe selection (single init per C).",
    )
    parser.add_argument(
        "--probe_num_initializations",
        type=int,
        default=3,
        help="Number of random probe initializations per C when multi-init selection is enabled.",
    )
    parser.add_argument(
        "--include_additional_attention_mlp_probes",
        action="store_true",
        default=False,
        help=(
            "If set, also train/evaluate additional LLM probe families: per-head attention "
            "activations, post-attention activations, and MLP activations. Default: off."
        ),
    )
    parser.add_argument(
        "--normalize_features",
        dest="normalize_features",
        action="store_true",
        default=True,
        help="Normalize features using train-set mean/std before probe training.",
    )
    parser.add_argument(
        "--no_normalize_features",
        dest="normalize_features",
        action="store_false",
        help="Disable train-set feature normalization.",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=0,
        help="If >0, apply train-fitted PCA to this many components before training.",
    )
    parser.add_argument(
        "--features_cache_path",
        type=str,
        default="./tmp_artifacts/bclean_all_examples_layer_features.pt",
    )
    parser.add_argument(
        "--force_reextract",
        dest="force_reextract",
        action="store_true",
        default=True,
        help="Force re-extraction of activations instead of loading cached features (default: on).",
    )
    parser.add_argument(
        "--no_force_reextract",
        dest="force_reextract",
        action="store_false",
        help="Allow reuse of --features_cache_path when cache metadata matches.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./tmp_artifacts/bclean_all_examples_probe_results",
    )
    pair_core.add_model_loading_args(parser)
    return parser.parse_args()


def _build_image_lookup(mirage_root: Path, responses: List[Dict]) -> Tuple[Dict[Tuple[str, str], List[bytes]], Dict[Tuple[str, str], List[bytes]]]:
    datasets = sorted({str(r.get("dataset", "")) for r in responses if str(r.get("dataset", ""))})
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]] = {}
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]] = {}
    for ds in datasets:
        items = core._load_dataset_items(mirage_root=mirage_root, dataset_name=ds)
        for item in items:
            uid = str(item.get("unique_id", ""))
            qid = str(item.get("question_id", ""))
            imgs = item.get("images", []) or []
            image_lookup_uid[(ds, uid)] = imgs
            if qid:
                image_lookup_qid[(ds, qid)] = imgs
    return image_lookup_uid, image_lookup_qid


def _resolve_selected_benchmark(args: argparse.Namespace) -> Optional[str]:
    if bool(args.vqa_only_examples):
        return "vqa_rad"
    if bool(args.mmmu_only_examples):
        return "mmmu_pro"
    if bool(args.medxpert_only_examples):
        return "medxpertqa_mm"
    return None


def _resolve_model_path(args: argparse.Namespace) -> str:
    if str(args.model_path).strip():
        return str(args.model_path).strip()
    resolver = getattr(pair_core, "_default_model_path_for_vlm", None)
    if callable(resolver):
        try:
            return str(resolver(str(args.vlm)))
        except Exception:
            pass
    return VLM_MODEL_PATHS[str(args.vlm)]


def _scope_default_path(path: Path, vlm: str, marker: str) -> Path:
    p = Path(path)
    if marker not in str(p):
        return p
    if vlm == DEFAULT_VLM:
        return p
    if p.suffix:
        return p.with_name(f"{p.stem}_{vlm}{p.suffix}")
    return Path(f"{p}_{vlm}")


def _select_balanced_examples(
    responses: List[Dict],
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
    seed: int,
    max_questions: int,
    include_short_response_filter: bool,
    neutral_as_non_mirage: bool,
    target_examples_per_class: int,
    selected_benchmark: Optional[str],
    excluded_datasets: List[str],
) -> Tuple[List[Dict], Dict]:
    rng = np.random.default_rng(seed)

    grouped: Dict[Tuple[Tuple[str, str], int], List[Dict]] = defaultdict(list)
    skipped_short_response_count = 0
    for row in responses:
        with_image_response = str((row.get("with_image", {}) or {}).get("response", ""))
        if include_short_response_filter:
            if core._count_tokens(with_image_response) < MIN_RESPONSE_TOKENS:
                skipped_short_response_count += 1
                continue

        label = _row_binary_label(row, neutral_as_non_mirage=neutral_as_non_mirage)
        if label is None:
            continue

        ds = str(row.get("dataset", ""))
        uid = str(row.get("unique_id", ""))
        qid = str(row.get("question_id", ""))
        question_key = (ds, uid if uid else qid)
        if not question_key[1]:
            continue

        imgs = image_lookup_uid.get((ds, uid)) if uid else None
        if imgs is None:
            imgs = image_lookup_qid.get((ds, qid))
        if not imgs:
            continue

        if not str(with_image_response).strip():
            continue

        grouped[(question_key, label)].append(row)

    selected_class0 = []
    selected_class1 = []
    for (_question_key, label), rows in grouped.items():
        pick = rows[int(rng.integers(0, len(rows)))]
        if label == 0:
            selected_class0.append(pick)
        else:
            selected_class1.append(pick)

    if max_questions > 0:
        if len(selected_class0) > max_questions:
            idx = rng.choice(len(selected_class0), size=max_questions, replace=False)
            selected_class0 = [selected_class0[int(i)] for i in idx]
        if len(selected_class1) > max_questions:
            idx = rng.choice(len(selected_class1), size=max_questions, replace=False)
            selected_class1 = [selected_class1[int(i)] for i in idx]

    # BClean labels define class0 by correctness transition (without-image wrong,
    # with-image right). Keep all selected class0 rows in the primary bucket while
    # preserving the same balancing/allocation procedure as all-examples.
    class0_non_neutral = list(selected_class0)
    class0_neutral = []

    class0_non_by_ds: Dict[str, List[Dict]] = defaultdict(list)
    class0_neu_by_ds: Dict[str, List[Dict]] = defaultdict(list)
    class1_by_ds: Dict[str, List[Dict]] = defaultdict(list)

    for r in class0_non_neutral:
        class0_non_by_ds[str(r.get("dataset", ""))].append(r)
    for r in class0_neutral:
        class0_neu_by_ds[str(r.get("dataset", ""))].append(r)
    for r in selected_class1:
        class1_by_ds[str(r.get("dataset", ""))].append(r)

    represented_datasets = sorted(
        ds
        for ds in set(class1_by_ds.keys()) | set(class0_non_by_ds.keys()) | set(class0_neu_by_ds.keys())
        if (len(class1_by_ds.get(ds, [])) > 0)
        and ((len(class0_non_by_ds.get(ds, [])) + len(class0_neu_by_ds.get(ds, []))) > 0)
    )
    if not represented_datasets:
        raise ValueError(
            "No represented datasets remain after applying dataset and label filters."
        )

    pair_caps_by_ds = {
        ds: len(class0_non_by_ds.get(ds, [])) + len(class0_neu_by_ds.get(ds, []))
        for ds in represented_datasets
    }
    pair_caps_by_ds = {
        ds: min(
            int(pair_caps_by_ds.get(ds, 0)),
            int(len(class1_by_ds.get(ds, []))),
        )
        for ds in represented_datasets
    }
    if min(pair_caps_by_ds.values()) < 3:
        raise ValueError(
            "Not enough examples to build benchmark-parity classes with >=3 examples per dataset. "
            f"pair_caps_by_ds={pair_caps_by_ds}"
        )

    desired_per_class = int(target_examples_per_class)
    total_pair_cap = int(sum(pair_caps_by_ds.values()))
    if desired_per_class <= 0:
        desired_per_class_effective = total_pair_cap
    else:
        desired_per_class_effective = min(desired_per_class, total_pair_cap)

    # Allocate per-benchmark pair counts that are identical across classes.
    # We keep at least 3 per benchmark so each (class,dataset) bucket can still be split
    # into train/valid/test with at least one item in each split.
    base_by_ds = {ds: 3 for ds in represented_datasets}
    remaining = desired_per_class_effective - 3 * len(represented_datasets)
    if remaining < 0:
        raise ValueError(
            "Target examples per class too small for benchmark-aware split with >=3 per benchmark. "
            f"target_examples_per_class={desired_per_class_effective}, "
            f"num_benchmarks={len(represented_datasets)}"
        )

    extra_capacity_by_ds = {ds: int(pair_caps_by_ds[ds] - 3) for ds in represented_datasets}
    extra_total_capacity = int(sum(extra_capacity_by_ds.values()))
    extra_target = min(remaining, extra_total_capacity)
    selected_pairs_by_ds = dict(base_by_ds)
    if extra_target > 0:
        # Proportional extra allocation with largest-remainder rounding.
        raw_extra = {
            ds: (extra_target * float(extra_capacity_by_ds[ds]) / float(extra_total_capacity))
            if extra_total_capacity > 0
            else 0.0
            for ds in represented_datasets
        }
        floor_extra = {ds: int(np.floor(raw_extra[ds])) for ds in represented_datasets}
        for ds in represented_datasets:
            selected_pairs_by_ds[ds] += floor_extra[ds]
        assigned_extra = int(sum(floor_extra.values()))
        leftover = int(extra_target - assigned_extra)
        if leftover > 0:
            remainder_order = sorted(
                represented_datasets,
                key=lambda ds: (raw_extra[ds] - floor_extra[ds]),
                reverse=True,
            )
            for ds in remainder_order:
                if leftover <= 0:
                    break
                if selected_pairs_by_ds[ds] < pair_caps_by_ds[ds]:
                    selected_pairs_by_ds[ds] += 1
                    leftover -= 1

    picked_class0: List[Dict] = []
    picked_class1: List[Dict] = []
    used_non_neutral = 0
    used_neutral = 0
    selected_per_dataset: Dict[str, Dict] = {}
    for ds in represented_datasets:
        ds_non = class0_non_by_ds.get(ds, [])
        ds_neu = class0_neu_by_ds.get(ds, [])
        ds_cls1 = class1_by_ds.get(ds, [])
        per_ds_target = int(selected_pairs_by_ds[ds])
        idx1 = rng.choice(len(ds_cls1), size=per_ds_target, replace=False)
        ds_pick1 = [ds_cls1[int(i)] for i in idx1]
        picked_class1.extend(ds_pick1)

        if len(ds_non) >= per_ds_target:
            idx0_non = rng.choice(len(ds_non), size=per_ds_target, replace=False)
            ds_pick0 = [ds_non[int(i)] for i in idx0_non]
            ds_non_ct = per_ds_target
            ds_neu_ct = 0
        else:
            ds_pick0 = list(ds_non)
            remain = per_ds_target - len(ds_pick0)
            if len(ds_neu) < remain:
                raise ValueError(
                    f"Dataset {ds} cannot fill class0 quota. "
                    f"non_neutral={len(ds_non)}, neutral={len(ds_neu)}, need={per_ds_target}"
                )
            idx0_neu = rng.choice(len(ds_neu), size=remain, replace=False) if remain > 0 else []
            ds_pick0.extend(ds_neu[int(i)] for i in idx0_neu)
            ds_non_ct = len(ds_non)
            ds_neu_ct = remain
        picked_class0.extend(ds_pick0)
        used_non_neutral += ds_non_ct
        used_neutral += ds_neu_ct

        selected_per_dataset[ds] = {
            "class0_selected": int(len(ds_pick0)),
            "class1_selected": int(len(ds_pick1)),
            "class0_non_neutral_selected": int(ds_non_ct),
            "class0_neutral_selected": int(ds_neu_ct),
            "class0_available_total": int(len(ds_non) + len(ds_neu)),
            "class1_available_total": int(len(ds_cls1)),
        }

    balanced_rows = picked_class0 + picked_class1
    rng.shuffle(balanced_rows)
    stats = {
        "selected_benchmark": selected_benchmark,
        "excluded_datasets": [str(x) for x in excluded_datasets],
        "benchmark_parity_across_classes_enabled": True,
        "represented_datasets": represented_datasets,
        "desired_examples_per_class_pre_split": int(desired_per_class_effective),
        "selected_examples_per_dataset_pre_split": {
            ds: int(selected_pairs_by_ds[ds]) for ds in represented_datasets
        },
        "selected_examples_per_class_pre_split": {
            "class0": int(sum(selected_pairs_by_ds.values())),
            "class1": int(sum(selected_pairs_by_ds.values())),
        },
        "selected_per_dataset_per_class": selected_per_dataset,
        "pair_caps_by_ds": {k: int(v) for k, v in pair_caps_by_ds.items()},
        "class0_non_neutral_available": int(len(class0_non_neutral)),
        "class0_neutral_available": int(len(class0_neutral)),
        "class1_available": int(len(selected_class1)),
        "class0_non_neutral_selected": int(used_non_neutral),
        "class0_neutral_selected": int(used_neutral),
        "exclude_short_responses_in_training_examples": bool(include_short_response_filter),
        "min_response_tokens_required": (
            int(MIN_RESPONSE_TOKENS) if include_short_response_filter else None
        ),
        "short_response_examples_excluded_pre_sampling": int(skipped_short_response_count),
    }
    return balanced_rows, stats


def _build_conversation_from_row(
    row: Dict,
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
    neutral_as_non_mirage: bool,
) -> Dict:
    ds = str(row.get("dataset", ""))
    uid = str(row.get("unique_id", ""))
    qid = str(row.get("question_id", ""))
    imgs = image_lookup_uid.get((ds, uid)) if uid else None
    if imgs is None:
        imgs = image_lookup_qid.get((ds, qid))
    if not imgs:
        raise RuntimeError(f"No image bytes found for row dataset={ds}, unique_id={uid}, question_id={qid}")

    prompt_text = str(row.get("prompt_text", ""))
    system_prompt = str(row.get("system_prompt", ""))
    with_resp = str((row.get("with_image", {}) or {}).get("response", ""))
    label = _row_binary_label(row, neutral_as_non_mirage=neutral_as_non_mirage)
    if label is None:
        raise ValueError("Row does not map to a binary label under current neutral mode.")

    conv = core._make_vllm_messages(
        prompt_text=prompt_text,
        image_bytes_list=imgs,
        system_prompt=system_prompt,
    )
    conv.append({"role": "assistant", "content": with_resp})
    return {
        "conversation": conv,
        "label": label,
        "sample_name": f"{ds}::{uid}::{row.get('variant_id', '')}::class_{label}",
        "meta": {
            "dataset": ds,
            "unique_id": uid,
            "question_id": qid,
            "variant_id": row.get("variant_id", ""),
            "label": label,
        },
    }


def _row_binary_label(row: Dict, neutral_as_non_mirage: bool) -> int | None:
    wo = row.get("without_image", {}) or {}
    wi = row.get("with_image", {}) or {}

    # Positive class keeps current mirage-like labels.
    if wo.get("mirage_like") is True:
        return 1

    # BClean negative class: without-image is wrong while with-image is right.
    if (wo.get("correct") is False) and (wi.get("correct") is True):
        return 0

    # Keep signature parity with all-examples script; neutral flag is not used
    # under BClean labeling.
    _ = neutral_as_non_mirage
    return None


def _is_neutral_class0_row(row: Dict) -> bool:
    return False


def _is_non_neutral_class0_row(row: Dict) -> bool:
    return _row_binary_label(row, neutral_as_non_mirage=False) == 0


def _split_balanced_train_val_test(
    labels: np.ndarray,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_fraction <= 0.0 or test_fraction <= 0.0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError(
            "Expected 0 < val_fraction < 1, 0 < test_fraction < 1, and "
            f"val_fraction + test_fraction < 1. Got val={val_fraction}, test={test_fraction}."
        )

    y = np.asarray(labels, dtype=np.int64)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) < 3 or len(idx1) < 3:
        raise ValueError(
            "Need >=3 examples per class for train/val/test split, "
            f"got class0={len(idx0)}, class1={len(idx1)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def _split_class_indices(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(idx)
        n_val = max(1, int(round(n * val_fraction)))
        n_test = max(1, int(round(n * test_fraction)))
        while n_val + n_test > n - 1:
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break
        val_idx_cls = idx[:n_val]
        test_idx_cls = idx[n_val : n_val + n_test]
        train_idx_cls = idx[n_val + n_test :]
        return train_idx_cls, val_idx_cls, test_idx_cls

    train0, val0, test0 = _split_class_indices(idx0)
    train1, val1, test1 = _split_class_indices(idx1)

    train_idx = np.concatenate([train0, train1])
    val_idx = np.concatenate([val0, val1])
    test_idx = np.concatenate([test0, test1])

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _split_balanced_train_test(
    labels: np.ndarray,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if test_fraction <= 0.0 or test_fraction >= 1.0:
        raise ValueError(f"Expected 0 < test_fraction < 1. Got {test_fraction}.")

    y = np.asarray(labels, dtype=np.int64)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) < 2 or len(idx1) < 2:
        raise ValueError(
            "Need >=2 examples per class for train/test split, "
            f"got class0={len(idx0)}, class1={len(idx1)}"
        )

    rng = np.random.default_rng(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def _split_class_indices(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = len(idx)
        n_test = max(1, int(round(n * test_fraction)))
        if n_test >= n:
            n_test = n - 1
        test_idx_cls = idx[:n_test]
        train_idx_cls = idx[n_test:]
        return train_idx_cls, test_idx_cls

    train0, test0 = _split_class_indices(idx0)
    train1, test1 = _split_class_indices(idx1)

    train_idx = np.concatenate([train0, train1])
    test_idx = np.concatenate([test0, test1])

    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    return train_mask, test_mask


def _split_balanced_train_val_test_by_class_dataset(
    labels: np.ndarray,
    sample_meta: List[Dict],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError(f"Expected 0 < val_fraction < 1. Got {val_fraction}.")
    if test_fraction <= 0.0 or test_fraction >= 1.0:
        raise ValueError(f"Expected 0 < test_fraction < 1. Got {test_fraction}.")
    if (val_fraction + test_fraction) >= 1.0:
        raise ValueError(
            "Expected val_fraction + test_fraction < 1. "
            f"Got val_fraction={val_fraction}, test_fraction={test_fraction}."
        )

    y = np.asarray(labels, dtype=np.int64)
    if len(sample_meta) != len(y):
        raise ValueError("sample_meta length must match labels length.")

    bucket_to_indices: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for i, meta in enumerate(sample_meta):
        ds = str(meta.get("dataset", ""))
        bucket_to_indices[(int(y[i]), ds)].append(i)

    represented = sorted(set(ds for (_label, ds) in bucket_to_indices.keys()))
    if not represented:
        raise ValueError("No dataset buckets found for train/valid/test split.")

    rng = np.random.default_rng(seed)
    train_idx_all: List[int] = []
    val_idx_all: List[int] = []
    test_idx_all: List[int] = []
    bucket_summary = {}

    for label in [0, 1]:
        for ds in represented:
            idx = np.asarray(bucket_to_indices.get((label, ds), []), dtype=np.int64)
            if len(idx) < 3:
                raise ValueError(
                    "Need >=3 samples in each (class,dataset) bucket for train/valid/test split. "
                    f"Found label={label}, dataset={ds}, n={len(idx)}"
                )
            rng.shuffle(idx)
            n = len(idx)
            n_val = max(1, int(round(n * val_fraction)))
            n_test = max(1, int(round(n * test_fraction)))
            while (n_val + n_test) > (n - 1):
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
            val_idx = idx[:n_val]
            test_idx = idx[n_val : n_val + n_test]
            train_idx = idx[n_val + n_test :]
            if len(train_idx) < 1:
                raise ValueError(
                    "Unable to allocate at least one train sample in (class,dataset) bucket. "
                    f"label={label}, dataset={ds}, n={n}, n_val={n_val}, n_test={n_test}"
                )
            train_idx_all.extend(train_idx.tolist())
            val_idx_all.extend(val_idx.tolist())
            test_idx_all.extend(test_idx.tolist())
            bucket_summary[f"class_{label}::{ds}"] = {
                "total": int(n),
                "train": int(len(train_idx)),
                "val": int(len(val_idx)),
                "test": int(len(test_idx)),
            }

    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[np.asarray(train_idx_all, dtype=np.int64)] = True
    val_mask[np.asarray(val_idx_all, dtype=np.int64)] = True
    test_mask[np.asarray(test_idx_all, dtype=np.int64)] = True
    split_stats = {
        "represented_datasets": represented,
        "bucket_summary": bucket_summary,
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
    }
    return train_mask, val_mask, test_mask, split_stats


def _is_additional_attention_mlp_feature(feature_name: str) -> bool:
    base = str(feature_name).split("__")[0]
    return ("/attention_head_" in base) or ("/post_attention" in base) or ("/mlp" in base)


def _fit_probe_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    c_value: float,
    normalize_features: bool,
    pca_components: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> Dict:
    preproc, X_train_s = pair_core._fit_feature_preprocessor(
        X_train=X_train,
        normalize_features=normalize_features,
        pca_components=pca_components,
    )
    X_val_s = pair_core._apply_feature_preprocessor(X_val, preproc)
    X_eval_s = pair_core._apply_feature_preprocessor(X_eval, preproc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = torch.nn.Linear(X_train_s.shape[1], 1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    weight_decay = 0.0 if float(c_value) == 0.0 else (1.0 / float(c_value))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_train_t = torch.from_numpy(X_train_s).to(device=device, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32).unsqueeze(-1)
    x_val_t = torch.from_numpy(X_val_s).to(device=device, dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32).unsqueeze(-1)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    model.train()
    for epoch in range(int(epochs)):
        logits = model(x_train_t)
        loss = loss_fn(logits, y_train_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            model.eval()
            val_logits = model(x_val_t)
            val_loss = float(loss_fn(val_logits, y_val_t).item())
            model.train()

        if val_loss < (best_val_loss - float(early_stopping_min_delta)):
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            # Patience is intentionally large by default to make stopping tolerant to noisy val loss.
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(early_stopping_patience):
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device=device) for k, v in best_state.items()})

    def _predict(arr: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            logits = model(x_t).squeeze(-1)
            return (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu().numpy()

    model.eval()
    train_pred = _predict(X_train_s)
    val_pred = _predict(X_val_s)
    eval_pred = _predict(X_eval_s)
    return {
        "train_pred": train_pred,
        "val_pred": val_pred,
        "eval_pred": eval_pred,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }


def _sweep_probe_with_validation(
    X: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    split_seed: int,
    reg_values: List[float],
    args: argparse.Namespace,
    benchmark_labels: Optional[np.ndarray] = None,
) -> Dict:
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    test_benchmark_labels: Optional[List[str]] = None
    if benchmark_labels is not None:
        test_benchmark_labels = [str(x) for x in benchmark_labels[test_mask].tolist()]

    best = None
    sweep = []
    num_inits = int(args.probe_num_initializations) if bool(args.multi_init_probe_selection) else 1
    if num_inits < 1:
        raise ValueError("--probe_num_initializations must be >= 1.")

    for c in reg_values:
        init_runs = []
        best_init = None
        for init_idx in range(num_inits):
            init_seed = int(split_seed) + (init_idx * 100_003) + int(round(float(c) * 10_000))
            out = _fit_probe_with_early_stopping(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_eval=X_test,
                seed=init_seed,
                epochs=int(args.probe_epochs),
                lr=float(args.probe_lr),
                c_value=float(c),
                normalize_features=bool(args.normalize_features),
                pca_components=int(args.pca_components),
                early_stopping_patience=int(args.early_stopping_patience),
                early_stopping_min_delta=float(args.early_stopping_min_delta),
            )
            train_pred = out["train_pred"]
            val_pred = out["val_pred"]
            test_pred = out["eval_pred"]

            train_acc = float((train_pred == y_train).mean())
            val_acc = float((val_pred == y_val).mean())
            test_acc = float((test_pred == y_test).mean())
            class0_mask = y_test == 0
            class1_mask = y_test == 1
            class0_test_acc = float((test_pred[class0_mask] == y_test[class0_mask]).mean()) if class0_mask.any() else None
            class1_test_acc = float((test_pred[class1_mask] == y_test[class1_mask]).mean()) if class1_mask.any() else None
            benchmark_test_acc, benchmark_class0_test_acc, benchmark_class1_test_acc = pair_core._compute_benchmark_test_metrics(
                y_true=y_test,
                y_pred=test_pred,
                benchmark_labels=test_benchmark_labels,
            )

            init_row = {
                "init_index": int(init_idx),
                "init_seed": int(init_seed),
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc,
                "class0_test_accuracy": class0_test_acc,
                "class1_test_accuracy": class1_test_acc,
                "benchmark_test_accuracy": benchmark_test_acc,
                "benchmark_class0_test_accuracy": benchmark_class0_test_acc,
                "benchmark_class1_test_accuracy": benchmark_class1_test_acc,
                "early_stopped_best_epoch": int(out["best_epoch"]),
                "early_stopped_best_val_loss": float(out["best_val_loss"]),
            }
            init_runs.append(init_row)
            if (
                best_init is None
                or (val_acc > best_init["val_accuracy"])
                or (
                    val_acc == best_init["val_accuracy"]
                    and test_acc > best_init["test_accuracy"]
                )
                or (
                    val_acc == best_init["val_accuracy"]
                    and test_acc == best_init["test_accuracy"]
                    and int(init_idx) < int(best_init["init_index"])
                )
            ):
                best_init = init_row

        row = {
            "c_value": float(c),
            "num_initializations_tried": int(num_inits),
            "best_init_index": int(best_init["init_index"]),
            "best_init_seed": int(best_init["init_seed"]),
            "train_accuracy": float(best_init["train_accuracy"]),
            "val_accuracy": float(best_init["val_accuracy"]),
            "test_accuracy": float(best_init["test_accuracy"]),
            "class0_test_accuracy": best_init["class0_test_accuracy"],
            "class1_test_accuracy": best_init["class1_test_accuracy"],
            "benchmark_test_accuracy": best_init["benchmark_test_accuracy"],
            "benchmark_class0_test_accuracy": best_init["benchmark_class0_test_accuracy"],
            "benchmark_class1_test_accuracy": best_init["benchmark_class1_test_accuracy"],
            "early_stopped_best_epoch": int(best_init["early_stopped_best_epoch"]),
            "early_stopped_best_val_loss": float(best_init["early_stopped_best_val_loss"]),
            "initialization_runs": init_runs,
        }
        sweep.append(row)

        if (
            best is None
            or (float(best_init["val_accuracy"]) > best["val_accuracy_at_best_c"])
            or (
                float(best_init["val_accuracy"]) == best["val_accuracy_at_best_c"]
                and float(c) < float(best["best_c"])
            )
        ):
            best = {
                "best_c": float(c),
                "best_train_accuracy": float(best_init["train_accuracy"]),
                "val_accuracy_at_best_c": float(best_init["val_accuracy"]),
                "test_accuracy_at_best_c": float(best_init["test_accuracy"]),
                "class0_test_accuracy_at_best_c": best_init["class0_test_accuracy"],
                "class1_test_accuracy_at_best_c": best_init["class1_test_accuracy"],
                "benchmark_test_accuracy_at_best_c": best_init["benchmark_test_accuracy"],
                "benchmark_class0_test_accuracy_at_best_c": best_init["benchmark_class0_test_accuracy"],
                "benchmark_class1_test_accuracy_at_best_c": best_init["benchmark_class1_test_accuracy"],
                "best_epoch_at_best_c": int(best_init["early_stopped_best_epoch"]),
                "best_val_loss_at_best_c": float(best_init["early_stopped_best_val_loss"]),
                "best_init_index_at_best_c": int(best_init["init_index"]),
                "best_init_seed_at_best_c": int(best_init["init_seed"]),
            }

    return {
        "split_seed": int(split_seed),
        "selection_metric": "val_accuracy",
        **best,
        "sweep": sweep,
    }


def main() -> None:
    args = parse_args()
    args.responses_path = str(
        pair_core._canonicalize_input_arg_for_vlm(
            path_value=args.responses_path,
            vlm_key=str(args.vlm),
            artifact_name="responses.json",
        )
    )
    selected_benchmark = _resolve_selected_benchmark(args)
    selected_dataset_set: Optional[Set[str]] = None
    if selected_benchmark is not None:
        selected_dataset_set = {selected_benchmark}

    model_path = _resolve_model_path(args)
    save_dir_path = _scope_default_path(
        Path(args.save_dir),
        vlm=str(args.vlm),
        marker="/bclean_all_examples_probe_results",
    )
    cache_path = _scope_default_path(
        Path(args.features_cache_path),
        vlm=str(args.vlm),
        marker="bclean_all_examples_layer_features.pt",
    )

    save_dir = Path(save_dir_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    mirage_root = REPO_ROOT.resolve()

    responses_path = pair_core._resolve_responses_path_for_vlm(
        base_responses_path=Path(args.responses_path),
        vlm_key=str(args.vlm),
    )

    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    responses = [r for r in responses if str(r.get("dataset", "")) in set(ALL_BENCHMARK_DATASETS)]
    if selected_dataset_set is not None:
        responses = [r for r in responses if str(r.get("dataset", "")) in selected_dataset_set]
    if not responses:
        raise ValueError("No responses remain after applying benchmark filter.")
    image_lookup_uid, image_lookup_qid = _build_image_lookup(mirage_root=mirage_root, responses=responses)
    balanced_rows, selection_stats = _select_balanced_examples(
        responses=responses,
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
        seed=int(args.seed),
        max_questions=int(args.max_questions),
        include_short_response_filter=bool(args.exclude_short_responses_in_training_examples),
        neutral_as_non_mirage=bool(args.neutral_as_non_mirage),
        target_examples_per_class=int(args.target_examples_per_class),
        selected_benchmark=selected_benchmark,
        excluded_datasets=[] if selected_benchmark is not None else ["microvqa"],
    )

    selected = [
        _build_conversation_from_row(
            row,
            image_lookup_uid=image_lookup_uid,
            image_lookup_qid=image_lookup_qid,
            neutral_as_non_mirage=bool(args.neutral_as_non_mirage),
        )
        for row in balanced_rows
    ]
    labels = [int(x["label"]) for x in selected]
    sample_names = [str(x["sample_name"]) for x in selected]
    conversations = [x["conversation"] for x in selected]
    sample_meta = [x["meta"] for x in selected]

    if pair_core._uses_preextracted_activation_store(str(args.vlm)):
        family = pair_core._preextract_family_for_vlm(str(args.vlm)).upper()
        if bool(args.force_reextract):
            print(f"Ignoring --force_reextract for {family}; loading pre-extracted activations cache.")
        layer_features, _unused_without, layer_order, cache_path = pair_core._load_preextracted_all_examples_subset(
            vlm_key=str(args.vlm),
            with_conversations=conversations,
            without_conversations=None,
            require_without=False,
            include_attention_probes=bool(args.include_additional_attention_mlp_probes),
            include_mlp_probes=bool(args.include_additional_attention_mlp_probes),
            requested_model_path=str(model_path),
        )
    elif cache_path.exists() and not args.force_reextract:
        print(f"Loading cached activations from: {cache_path}")
        payload = torch.load(cache_path)
        expected_glm_image_normalization = str(args.vlm) == "glm_4_6v_flash"
        cached_glm_image_normalization = bool(payload.get("glm_image_normalization_applied", False))
        cached_model_path = str(payload.get("model_path", ""))
        if cached_model_path and cached_model_path != str(model_path):
            raise RuntimeError(
                "Cached activations were extracted with a different model_path. "
                "Re-run with --force_reextract or use a model-scoped cache path."
            )
        if expected_glm_image_normalization and (not cached_glm_image_normalization):
            raise RuntimeError(
                "Cached GLM activations were extracted before GLM image normalization was applied. "
                "Re-run with --force_reextract."
            )
        cached_names = [str(x) for x in payload["sample_names"]]
        if cached_names != sample_names:
            raise RuntimeError(
                "Cached sample ordering does not match current selection/split settings. "
                "Re-run with --force_reextract."
            )
        layer_features = payload["layer_features"]
        labels = payload["labels"]
        sample_names = payload["sample_names"]
        sample_meta = payload["sample_meta"]
        layer_order = payload["layer_order"]

        cached_include_additional = bool(payload.get("include_additional_attention_mlp_probes", False))
        requested_include_additional = bool(args.include_additional_attention_mlp_probes)
        if requested_include_additional and not cached_include_additional:
            raise RuntimeError(
                "Cache was extracted without additional attention/MLP probes, but "
                "--include_additional_attention_mlp_probes was requested. "
                "Re-run with --force_reextract."
            )
        if (not requested_include_additional) and cached_include_additional:
            layer_order = [k for k in layer_order if not _is_additional_attention_mlp_feature(k)]
            layer_features = {
                k: v for k, v in layer_features.items() if not _is_additional_attention_mlp_feature(k)
            }
    else:
        print(f"Extracting activations and caching to: {cache_path}")
        model = pair_core.load_vlm_for_extraction(
            model_path=model_path,
            attn_implementation=None,
            device_map_raw=str(getattr(args, "device_map", "")),
            max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
            max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
        )

        layer_features: Dict[str, List[np.ndarray]] = defaultdict(list)
        layer_order: List[str] = []
        seen_layers = set()

        for i, conv in enumerate(
            tqdm(
                conversations,
                total=len(conversations),
                desc="Extracting activations",
                unit="sample",
                dynamic_ncols=True,
            )
        ):
            ovis_messages = pair_core._to_ovis_messages(conv)
            sample_feats = pair_core._extract_sample_features_only(
                model=model,
                messages=ovis_messages,
                include_additional_attention_mlp_probes=bool(args.include_additional_attention_mlp_probes),
                model_key=str(args.vlm),
            )

            for key, value in sample_feats.items():
                if key not in seen_layers:
                    seen_layers.add(key)
                    layer_order.append(key)
                layer_features[key].append(value.to(torch.float32).cpu().numpy())

        layer_order = sorted(layer_order, key=pair_core._layer_sort_key)
        torch.save(
            {
                "layer_features": layer_features,
                "labels": labels,
                "sample_names": sample_names,
                "sample_meta": sample_meta,
                "layer_order": layer_order,
                "vlm": str(args.vlm),
                "model_path": model_path,
                "responses_path": str(responses_path),
                "seed": int(args.seed),
                "neutral_as_non_mirage": bool(args.neutral_as_non_mirage),
                "include_additional_attention_mlp_probes": bool(args.include_additional_attention_mlp_probes),
                "glm_image_normalization_applied": bool(str(args.vlm) == "glm_4_6v_flash"),
            },
            cache_path,
        )

    y = np.asarray(labels, dtype=np.int64)
    benchmark_labels = np.asarray([str(m.get("dataset", "unknown")) for m in sample_meta], dtype=object)
    val_fraction = float(args.val_fraction)
    test_fraction = float(args.test_fraction)
    if val_fraction <= 0.0 or test_fraction <= 0.0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError(
            "Require 0 < val_fraction < 1, 0 < test_fraction < 1, and val_fraction + test_fraction < 1. "
            f"Got val_fraction={val_fraction}, test_fraction={test_fraction}."
        )

    reg_values = [float(x.strip()) for x in args.regularization_values.split(",") if x.strip()]
    if not reg_values:
        raise ValueError("--regularization_values must contain at least one value.")
    if int(args.num_split_seeds) <= 0:
        raise ValueError("--num_split_seeds must be >= 1.")

    split_seed_rng = np.random.default_rng(int(args.seed))
    split_seeds = split_seed_rng.choice(
        1_000_000_000,
        size=int(args.num_split_seeds),
        replace=False,
    ).astype(np.int64).tolist()

    split_payloads = []
    print("Split sizes by seed:")
    for split_seed in split_seeds:
        train_mask, val_mask, test_mask, split_stats = _split_balanced_train_val_test_by_class_dataset(
            labels=y,
            sample_meta=sample_meta,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=int(split_seed),
        )
        train_y = y[train_mask]
        val_y = y[val_mask]
        test_y = y[test_mask]
        split_sizes = {
            "train_total": int(train_mask.sum()),
            "train_class0": int((train_y == 0).sum()),
            "train_class1": int((train_y == 1).sum()),
            "val_total": int(val_mask.sum()),
            "val_class0": int((val_y == 0).sum()),
            "val_class1": int((val_y == 1).sum()),
            "test_total": int(test_mask.sum()),
            "test_class0": int((test_y == 0).sum()),
            "test_class1": int((test_y == 1).sum()),
        }
        print(
            f"  seed={split_seed}: "
            f"train={split_sizes['train_total']} (c0={split_sizes['train_class0']}, c1={split_sizes['train_class1']}), "
            f"val={split_sizes['val_total']} (c0={split_sizes['val_class0']}, c1={split_sizes['val_class1']}), "
            f"test={split_sizes['test_total']} (c0={split_sizes['test_class0']}, c1={split_sizes['test_class1']})"
        )
        split_payloads.append(
            {
                "split_seed": int(split_seed),
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
                "split_stats": split_stats,
                "split_sizes": split_sizes,
            }
        )

    per_feature_results: Dict[str, Dict] = {}
    for feature_name in tqdm(
        layer_order,
        desc=f"Training probes ({int(args.num_split_seeds)} split seeds, C sweep + early stopping on validation)",
    ):
        X = np.asarray(layer_features[feature_name], dtype=np.float32)
        seed_runs = []
        for split_payload in split_payloads:
            run = _sweep_probe_with_validation(
                X=X,
                y=y,
                train_mask=split_payload["train_mask"],
                val_mask=split_payload["val_mask"],
                test_mask=split_payload["test_mask"],
                split_seed=int(split_payload["split_seed"]),
                reg_values=reg_values,
                args=args,
                benchmark_labels=benchmark_labels,
            )
            run["split_sizes"] = split_payload["split_sizes"]
            seed_runs.append(run)

        test_scores = [float(r["test_accuracy_at_best_c"]) for r in seed_runs]
        val_scores = [float(r["val_accuracy_at_best_c"]) for r in seed_runs]
        class0_scores = [float(r["class0_test_accuracy_at_best_c"]) for r in seed_runs if r["class0_test_accuracy_at_best_c"] is not None]
        class1_scores = [float(r["class1_test_accuracy_at_best_c"]) for r in seed_runs if r["class1_test_accuracy_at_best_c"] is not None]
        benchmark_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=seed_runs,
            key="benchmark_test_accuracy_at_best_c",
        )
        benchmark_class0_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=seed_runs,
            key="benchmark_class0_test_accuracy_at_best_c",
        )
        benchmark_class1_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=seed_runs,
            key="benchmark_class1_test_accuracy_at_best_c",
        )
        per_feature_results[feature_name] = {
            "num_split_seeds": int(len(seed_runs)),
            "split_seeds": [int(r["split_seed"]) for r in seed_runs],
            "seed_runs": seed_runs,
            "selection_metric": "val_accuracy",
            "mean_test_accuracy_at_best_c": float(np.mean(test_scores)),
            "std_test_accuracy_at_best_c": float(np.std(test_scores)),
            "mean_val_accuracy_at_best_c": float(np.mean(val_scores)),
            "std_val_accuracy_at_best_c": float(np.std(val_scores)),
            "mean_class0_test_accuracy_at_best_c": float(np.mean(class0_scores)) if class0_scores else None,
            "mean_class1_test_accuracy_at_best_c": float(np.mean(class1_scores)) if class1_scores else None,
            "mean_benchmark_test_accuracy_at_best_c": benchmark_scores,
            "mean_benchmark_class0_test_accuracy_at_best_c": benchmark_class0_scores,
            "mean_benchmark_class1_test_accuracy_at_best_c": benchmark_class1_scores,
            # Back-compat aliases expected by downstream analysis.
            "test_accuracy_at_best_c": float(np.mean(test_scores)),
            "val_accuracy_at_best_c": float(np.mean(val_scores)),
        }

    llm_layers = sorted(
        {
            name.split("__")[0]
            for name in per_feature_results
            if re.fullmatch(r"language_model/layer_\d+__[^/]+", name) is not None
        },
        key=lambda s: int(re.search(r"layer_(\d+)", s).group(1)),
    )
    llm_strategy_results: Dict[str, Dict] = {}
    for llm_layer in llm_layers:
        strategy_acc = {}
        strategy_class_acc = {}
        for strategy in pair_core.LLM_STRATEGIES:
            k = f"{llm_layer}__{strategy}"
            if k in per_feature_results:
                strategy_acc[strategy] = float(per_feature_results[k]["mean_test_accuracy_at_best_c"])
                strategy_class_acc[strategy] = {
                    "class0_test_accuracy": per_feature_results[k]["mean_class0_test_accuracy_at_best_c"],
                    "class1_test_accuracy": per_feature_results[k]["mean_class1_test_accuracy_at_best_c"],
                }
        best_strategy = None
        best_acc = -1.0
        for k, v in strategy_acc.items():
            if v > best_acc:
                best_acc = v
                best_strategy = k
        llm_strategy_results[llm_layer] = {
            "strategy_test_accuracies": strategy_acc,
            "strategy_per_class_test_accuracies": strategy_class_acc,
            "best_strategy": best_strategy,
            "best_test_accuracy": best_acc,
            "best_class0_test_accuracy": strategy_class_acc.get(best_strategy, {}).get("class0_test_accuracy"),
            "best_class1_test_accuracy": strategy_class_acc.get(best_strategy, {}).get("class1_test_accuracy"),
            "best_strategy_selection_metric": "test_accuracy",
        }

    all_feature_results = [
        {
            "feature": k,
            "num_split_seeds": int(v["num_split_seeds"]),
            "split_seeds": v["split_seeds"],
            "selection_metric": v["selection_metric"],
            "mean_test_accuracy_at_best_c": float(v["mean_test_accuracy_at_best_c"]),
            "std_test_accuracy_at_best_c": float(v["std_test_accuracy_at_best_c"]),
            "mean_val_accuracy_at_best_c": float(v["mean_val_accuracy_at_best_c"]),
            "std_val_accuracy_at_best_c": float(v["std_val_accuracy_at_best_c"]),
            "mean_class0_test_accuracy_at_best_c": v["mean_class0_test_accuracy_at_best_c"],
            "mean_class1_test_accuracy_at_best_c": v["mean_class1_test_accuracy_at_best_c"],
            # Back-compat aliases expected by downstream analysis.
            "test_accuracy_at_best_c": float(v["test_accuracy_at_best_c"]),
            "val_accuracy_at_best_c": float(v["val_accuracy_at_best_c"]),
            "class0_test_accuracy_at_best_c": v["mean_class0_test_accuracy_at_best_c"],
            "class1_test_accuracy_at_best_c": v["mean_class1_test_accuracy_at_best_c"],
            "seed_runs": v["seed_runs"],
        }
        for k, v in sorted(per_feature_results.items(), key=lambda kv: pair_core._layer_sort_key(kv[0]))
    ]

    first_split = split_payloads[0]["split_sizes"] if split_payloads else None
    summary_payload = {
        "num_samples": int(len(y)),
        "num_train_examples": int(first_split["train_total"]) if first_split else None,
        "num_val_examples": int(first_split["val_total"]) if first_split else None,
        "num_test_examples": int(first_split["test_total"]) if first_split else None,
        "num_class0_total": int((y == 0).sum()),
        "num_class1_total": int((y == 1).sum()),
        "training_mode": "bclean_all_examples_benchmark_balanced_train_val_test_multiseed",
        "num_split_seeds": int(args.num_split_seeds),
        "split_seeds": [int(s) for s in split_seeds],
        "val_fraction": float(val_fraction),
        "test_fraction": float(test_fraction),
        "regularization_sweep_c_values": reg_values,
        "hyperparam_selection_metric": "val_accuracy",
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "probe_epochs": int(args.probe_epochs),
        "probe_lr": float(args.probe_lr),
        "multi_init_probe_selection": bool(args.multi_init_probe_selection),
        "probe_num_initializations": int(args.probe_num_initializations),
        "normalize_features": bool(args.normalize_features),
        "pca_components": int(args.pca_components),
        "include_additional_attention_mlp_probes": bool(args.include_additional_attention_mlp_probes),
        "neutral_as_non_mirage": bool(args.neutral_as_non_mirage),
        "exclude_short_responses_in_training_examples": bool(
            args.exclude_short_responses_in_training_examples
        ),
        "label_definition": {
            "class1": "without_image.mirage_like == True",
            "class0": "without_image.correct == False AND with_image.correct == True",
            "excluded": "all other rows",
        },
        "selected_benchmark": selected_benchmark,
        "included_datasets": sorted(list(selected_dataset_set)) if selected_dataset_set is not None else list(ALL_BENCHMARK_DATASETS),
        "excluded_datasets": [] if selected_dataset_set is not None else ["microvqa"],
        "target_examples_per_class": int(args.target_examples_per_class),
        "selection_stats": selection_stats,
        "split_stats_by_seed": [
            {
                "split_seed": int(sp["split_seed"]),
                "split_sizes": sp["split_sizes"],
                "split_stats": sp["split_stats"],
            }
            for sp in split_payloads
        ],
        "llm_strategies": pair_core.LLM_STRATEGIES,
    }

    vlm_tag = str(args.vlm)
    all_features_path = save_dir / f"{vlm_tag}_bclean_all_feature_probe_accuracies.json"
    llm_strategy_path = save_dir / f"{vlm_tag}_bclean_llm_layer_best_strategy_results.json"
    sample_meta_path = save_dir / f"{vlm_tag}_bclean_sample_metadata.json"
    config_path = save_dir / f"{vlm_tag}_bclean_run_config.json"

    with open(all_features_path, "w", encoding="utf-8") as f:
        json.dump(all_feature_results, f, indent=2, ensure_ascii=False)
    with open(llm_strategy_path, "w", encoding="utf-8") as f:
        json.dump(llm_strategy_results, f, indent=2, ensure_ascii=False)
    with open(sample_meta_path, "w", encoding="utf-8") as f:
        json.dump(sample_meta, f, indent=2, ensure_ascii=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary_payload,
                "responses_path": str(responses_path),
                "repo_root": str(mirage_root),
                "vlm": str(args.vlm),
                "model_path": model_path,
                "device_map": str(getattr(args, "device_map", "")),
                "max_memory_per_gpu_gib": float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
                "max_memory_cpu_gib": float(getattr(args, "max_memory_cpu_gib", 0.0)),
                "features_cache_path": str(cache_path),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved feature accuracies: {all_features_path}")
    print(f"Saved LLM strategy best-per-layer: {llm_strategy_path}")
    print(f"Saved sample metadata: {sample_meta_path}")
    print(f"Saved run config: {config_path}")
    print("\nBest strategy per LLM layer (mean test accuracy across split seeds):")
    for layer in llm_layers:
        info = llm_strategy_results[layer]
        score = info.get("best_test_accuracy")
        c0 = info.get("best_class0_test_accuracy")
        c1 = info.get("best_class1_test_accuracy")
        c0_txt = f"{c0:.4f}" if c0 is not None else "n/a"
        c1_txt = f"{c1:.4f}" if c1 is not None else "n/a"
        print(
            f"{layer}: {info['best_strategy']} "
            f"(test_accuracy={score:.4f}, class0={c0_txt}, class1={c1_txt})"
        )


if __name__ == "__main__":
    main()
