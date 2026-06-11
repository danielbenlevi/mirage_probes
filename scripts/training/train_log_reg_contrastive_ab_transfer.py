#!/usr/bin/env python3
"""Train/evaluate contrastive logreg probes split by A/B question class.

This script runs one combined experiment:
1) Train A_probe on contrastive pairs where both variants are GPT-labeled A.
2) Train B_probe on contrastive pairs where both variants are GPT-labeled B.
3) Evaluate each probe on held-out all-examples pools labeled A and labeled B.

Outputs include per-feature/layer training summaries and held-out multiseed
evaluation summaries in the same style as train_log_reg_contrastive.py.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.training.train_log_reg_contrastive as core


DEFAULT_BENCHMARKS = ("mmmu_pro", "medxpertqa_mm")
TARGET_LABELS = ("A", "B")
SHORT_Q_EVAL_TARGET = "SHORT_Q"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train two contrastive logreg probes (A_probe, B_probe) and evaluate "
            "each on A-labeled and B-labeled held-out pools from all-examples cache."
        )
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="./data/final_data/qwen_contrastive.json",
    )
    parser.add_argument(
        "--responses_path",
        type=str,
        default="./data/final_data/qwen_all_responses.json",
    )
    parser.add_argument(
        "--contrastive_labels_path",
        type=str,
        default="./handoff/handoff/labels/predictions_contrastive.json",
    )
    parser.add_argument(
        "--all_examples_labels_path",
        type=str,
        default="./handoff/handoff/labels/predictions_all_responses_with_false.json",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./tmp_artifacts/contrastive_ab_transfer_probe_results",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="qwen3_vl_32b_instruct",
        choices=["qwen3_vl_32b_instruct"],
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regularization_values",
        type=str,
        default="0,0.01,0.1,1.0",
        help="Comma-separated C values for sweep; must include 0.",
    )
    parser.add_argument("--num_split_seeds", type=int, default=5)
    parser.add_argument("--num_eval_seeds", type=int, default=5)
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.20,
        help="Fixed train/valid split fraction for this experiment (defaults to 80/20).",
    )
    parser.add_argument("--probe_epochs", type=int, default=800)
    parser.add_argument("--probe_lr", type=float, default=0.03)
    parser.add_argument("--multi_init_probe_selection", action="store_true", default=True)
    parser.add_argument("--no_multi_init_probe_selection", dest="multi_init_probe_selection", action="store_false")
    parser.add_argument("--probe_num_initializations", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=200)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--normalize_features", action="store_true", default=True)
    parser.add_argument("--no_normalize_features", dest="normalize_features", action="store_false")
    parser.add_argument("--pca_components", type=int, default=0)
    parser.add_argument(
        "--num_holdout_mirage_true",
        type=int,
        default=50,
        help="Requested per-seed held-out positives for each eval target (A/B).",
    )
    parser.add_argument(
        "--num_holdout_mirage_false",
        type=int,
        default=50,
        help="Requested per-seed held-out negatives for each eval target (A/B).",
    )
    parser.add_argument(
        "--exclude_short_responses_in_training_pairs",
        dest="exclude_short_responses_in_training_pairs",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_exclude_short_responses_in_training_pairs",
        dest="exclude_short_responses_in_training_pairs",
        action="store_false",
    )
    parser.add_argument(
        "--exclude_short_responses_in_holdout",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--features_cache_path",
        type=str,
        default="",
        help="Optional explicit contrastive pre-extracted cache path override.",
    )
    parser.add_argument(
        "--all_examples_features_cache_path",
        type=str,
        default="",
        help="Optional explicit all-examples pre-extracted cache path override.",
    )
    parser.add_argument(
        "--b_only_per_benchmark_mode",
        action="store_true",
        default=False,
        help=(
            "If set, run only B->B probes, split into two independent runs: "
            "one for mmmu_pro-only and one for medxpertqa_mm-only."
        ),
    )
    parser.add_argument(
        "--b_only_per_benchmark_mode_filter_mirage_without_image_correct",
        action="store_true",
        default=False,
        help=(
            "If set, run the same B-only-per-benchmark mode as --b_only_per_benchmark_mode, "
            "but additionally filter mirage/class1 examples to without_image.correct==True "
            "for both training and held-out pools before balancing."
        ),
    )
    parser.add_argument(
        "--parallel_benchmark_gpus",
        type=str,
        default="0,1",
        help=(
            "Comma-separated GPU ids used to run per-benchmark modes in parallel. "
            "Default: '0,1'."
        ),
    )
    parser.add_argument(
        "--no_parallel_benchmark_runs",
        action="store_true",
        default=False,
        help="Disable parallel execution for per-benchmark modes.",
    )
    parser.add_argument(
        "--mmmu_pro_short_question_mode",
        action="store_true",
        default=False,
        help=(
            "If set, run one mmmu_pro-only contrastive probe where train/eval candidates are "
            "filtered by question length (< --short_question_word_limit) instead of A/B labels."
        ),
    )
    parser.add_argument(
        "--short_question_word_limit",
        type=int,
        default=15,
        help="Strict upper bound for short-question filtering; examples must have fewer words.",
    )
    return parser.parse_args()


def _resolve_model_path(args: argparse.Namespace) -> str:
    if str(args.model_path).strip():
        return str(args.model_path).strip()
    return core._default_model_path_for_vlm(str(args.vlm))


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _load_contrastive_label_lookup(
    labels_payload: Dict[str, Any],
    allowed_benchmarks: Set[str],
) -> Dict[Tuple[int, str], str]:
    out: Dict[Tuple[int, str], str] = {}
    for row in labels_payload.get("predictions", []):
        ds = str(row.get("source_dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        kind = str(row.get("variant_kind", "")).strip()
        if kind not in {"non_mirage", "mirage"}:
            continue
        label = str(row.get("predicted_label", "")).strip().upper()
        if label not in TARGET_LABELS:
            continue
        pair_idx = int(row.get("pair_idx"))
        out[(pair_idx, kind)] = label
    return out


def _load_all_examples_label_lookup(
    labels_payload: Dict[str, Any],
    allowed_benchmarks: Set[str],
) -> Dict[Tuple[str, str, str], str]:
    out: Dict[Tuple[str, str, str], str] = {}
    for row in labels_payload.get("predictions", []):
        ds = str(row.get("source_dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        uid = str(row.get("unique_id", "")).strip()
        vid = str(row.get("variant_id", "")).strip()
        label = str(row.get("predicted_label", "")).strip().upper()
        if label not in TARGET_LABELS:
            continue
        out[(ds, uid, vid)] = label
    return out


def _build_without_image_correct_key_set(
    responses: Sequence[Dict[str, Any]],
    allowed_benchmarks: Set[str],
) -> Set[Tuple[str, str, str]]:
    out: Set[Tuple[str, str, str]] = set()
    for row in responses:
        ds = str(row.get("dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        wo = row.get("without_image", {}) or {}
        if wo.get("correct") is not True:
            continue
        uid = str(row.get("unique_id", "")).strip()
        vid = str(row.get("variant_id", "")).strip()
        out.add((ds, uid, vid))
    return out


def _count_words_in_question_text(text: str) -> int:
    cleaned = str(text or "").replace("\n", " ").strip()
    if not cleaned:
        return 0
    return int(len(re.findall(r"\S+", cleaned)))


def _extract_user_question_text_from_conversation(conversation: Sequence[Dict[str, Any]]) -> str:
    for turn in conversation:
        if str(turn.get("role", "")).strip() != "user":
            continue
        content = turn.get("content")
        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, dict) and str(item.get("type", "")).strip() == "text":
                    chunks.append(str(item.get("text", "")))
            text = " ".join(chunks).strip()
        else:
            text = str(content or "").strip()
        if not text:
            continue
        head = re.split(r"\n\s*\n", text, maxsplit=1)[0].strip()
        return head if head else text
    return ""


def _build_response_question_lookup(
    responses: Sequence[Dict[str, Any]],
    allowed_benchmarks: Set[str],
) -> Dict[Tuple[str, str, str], str]:
    out: Dict[Tuple[str, str, str], str] = {}
    for row in responses:
        ds = str(row.get("dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        uid = str(row.get("unique_id", "")).strip()
        vid = str(row.get("variant_id", "")).strip()
        q_text = str(row.get("variant_question_text", "")).strip()
        if not q_text:
            prompt = str(row.get("prompt_text", "")).strip()
            q_text = re.split(r"\n\s*\n", prompt, maxsplit=1)[0].strip() if prompt else ""
        out[(ds, uid, vid)] = q_text
    return out


def _select_pairs_for_probe(
    pairs: Sequence[Dict[str, Any]],
    pair_label_lookup: Dict[Tuple[int, str], str],
    allowed_benchmarks: Set[str],
    target_label: str,
    require_mirage_without_image_correct: bool,
    without_image_correct_keys: Set[Tuple[str, str, str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts = {
        "pairs_seen_in_allowed_benchmarks": 0,
        "pairs_selected": 0,
        "pairs_missing_any_label": 0,
        "pairs_mixed_or_other_label": 0,
        "pairs_filtered_mirage_without_image_incorrect": 0,
    }
    selected_by_benchmark: Dict[str, int] = defaultdict(int)

    for pair_idx, pair in enumerate(pairs):
        ds = str(pair.get("source_dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        counts["pairs_seen_in_allowed_benchmarks"] += 1
        non_label = pair_label_lookup.get((int(pair_idx), "non_mirage"))
        mir_label = pair_label_lookup.get((int(pair_idx), "mirage"))
        if non_label is None or mir_label is None:
            counts["pairs_missing_any_label"] += 1
            continue
        if non_label == target_label and mir_label == target_label:
            if require_mirage_without_image_correct:
                mir_key = (
                    ds,
                    str(pair.get("source_unique_id", "")).strip(),
                    str(pair.get("mirage_variant_id", "")).strip(),
                )
                if mir_key not in without_image_correct_keys:
                    counts["pairs_filtered_mirage_without_image_incorrect"] += 1
                    continue
            selected.append(pair)
            counts["pairs_selected"] += 1
            selected_by_benchmark[ds] += 1
        else:
            counts["pairs_mixed_or_other_label"] += 1

    if not selected:
        raise RuntimeError(
            f"No contrastive pairs selected for target label '{target_label}'. "
            f"Counts={counts}"
        )
    stats = {
        **counts,
        "selected_pairs_by_benchmark": {k: int(v) for k, v in sorted(selected_by_benchmark.items())},
        "target_label": target_label,
    }
    return selected, stats


def _select_pairs_for_short_questions(
    pairs: Sequence[Dict[str, Any]],
    allowed_benchmarks: Set[str],
    question_word_limit: int,
    response_question_lookup: Dict[Tuple[str, str, str], str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts = {
        "pairs_seen_in_allowed_benchmarks": 0,
        "pairs_selected": 0,
        "pairs_missing_question_text": 0,
        "pairs_filtered_long_question": 0,
    }
    selected_by_benchmark: Dict[str, int] = defaultdict(int)

    for pair in pairs:
        ds = str(pair.get("source_dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        counts["pairs_seen_in_allowed_benchmarks"] += 1

        uid = str(pair.get("source_unique_id", "")).strip()
        non_vid = str(pair.get("non_mirage_variant_id", "")).strip()
        mir_vid = str(pair.get("mirage_variant_id", "")).strip()
        non_text = response_question_lookup.get((ds, uid, non_vid), "").strip()
        mir_text = response_question_lookup.get((ds, uid, mir_vid), "").strip()
        if not non_text:
            non_text = _extract_user_question_text_from_conversation(pair.get("non_mirage_conversation", []) or [])
        if not mir_text:
            mir_text = _extract_user_question_text_from_conversation(pair.get("mirage_conversation", []) or [])

        if not non_text or not mir_text:
            counts["pairs_missing_question_text"] += 1
            continue

        non_wc = _count_words_in_question_text(non_text)
        mir_wc = _count_words_in_question_text(mir_text)
        if non_wc < int(question_word_limit) and mir_wc < int(question_word_limit):
            selected.append(pair)
            counts["pairs_selected"] += 1
            selected_by_benchmark[ds] += 1
        else:
            counts["pairs_filtered_long_question"] += 1

    if not selected:
        raise RuntimeError(
            f"No contrastive pairs selected for short-question filter (<{question_word_limit} words). "
            f"Counts={counts}"
        )
    stats = {
        **counts,
        "selected_pairs_by_benchmark": {k: int(v) for k, v in sorted(selected_by_benchmark.items())},
        "question_word_limit_exclusive": int(question_word_limit),
        "question_length_filter_mode": "both_pair_variants_under_limit",
    }
    return selected, stats


def _build_training_examples_from_pairs(
    pairs: Sequence[Dict[str, Any]],
    exclude_short_responses: bool,
) -> Tuple[List[List[Dict]], List[int], List[int], List[str], List[str], int]:
    conversations: List[List[Dict]] = []
    labels: List[int] = []
    pair_ids: List[int] = []
    pair_benchmarks: List[str] = []
    sample_names: List[str] = []
    skipped_short = 0
    kept_pair_id = 0

    for pair in pairs:
        non_conv = pair["non_mirage_conversation"]
        mir_conv = pair["mirage_conversation"]

        if exclude_short_responses:
            too_short = False
            if core._conversation_has_image_input(non_conv):
                _, _, non_assistant = core._conversation_signature_from_conv(non_conv)
                if core.core._count_tokens(non_assistant) < core.MIN_RESPONSE_TOKENS:
                    too_short = True
            if core._conversation_has_image_input(mir_conv):
                _, _, mir_assistant = core._conversation_signature_from_conv(mir_conv)
                if core.core._count_tokens(mir_assistant) < core.MIN_RESPONSE_TOKENS:
                    too_short = True
            if too_short:
                skipped_short += 1
                continue

        ds = str(pair.get("source_dataset", "unknown"))
        pair_benchmarks.append(ds)

        conversations.append(non_conv)
        labels.append(0)
        pair_ids.append(kept_pair_id)
        sample_names.append(f"pair_{kept_pair_id}_non_mirage")

        conversations.append(mir_conv)
        labels.append(1)
        pair_ids.append(kept_pair_id)
        sample_names.append(f"pair_{kept_pair_id}_mirage")

        kept_pair_id += 1

    if not conversations:
        raise RuntimeError("No training conversations remained after filtering.")
    return conversations, labels, pair_ids, pair_benchmarks, sample_names, int(skipped_short)


def _make_split_payloads(
    y: np.ndarray,
    pair_ids_arr: np.ndarray,
    pair_benchmarks: List[str],
    seed: int,
    num_split_seeds: int,
    val_fraction: float,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    split_seed_rng = np.random.default_rng(int(seed))
    split_seeds = split_seed_rng.choice(
        1_000_000_000,
        size=int(num_split_seeds),
        replace=False,
    ).astype(np.int64).tolist()

    benchmark_labels = np.asarray([str(pair_benchmarks[int(pid)]) for pid in pair_ids_arr.tolist()], dtype=object)
    split_payloads: List[Dict[str, Any]] = []
    for split_seed in split_seeds:
        train_mask, val_mask, val_pairs, val_pairs_by_benchmark = core._split_pair_benchmark_fraction_validation(
            pair_ids=pair_ids_arr,
            pair_benchmarks=pair_benchmarks,
            val_fraction=float(val_fraction),
            seed=int(split_seed),
        )
        split_payloads.append(
            {
                "split_seed": int(split_seed),
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": np.zeros(len(val_mask), dtype=bool),
                "validation_pairs": [int(x) for x in val_pairs],
                "validation_pairs_by_benchmark": {
                    k: [int(x) for x in v] for k, v in val_pairs_by_benchmark.items()
                },
                "split_sizes": {
                    "train_total": int(train_mask.sum()),
                    "val_total": int(val_mask.sum()),
                },
                "split_label_counts": {
                    "train_class0": int((y[train_mask] == 0).sum()),
                    "train_class1": int((y[train_mask] == 1).sum()),
                    "val_class0": int((y[val_mask] == 0).sum()),
                    "val_class1": int((y[val_mask] == 1).sum()),
                },
                "split_benchmark_class_counts": {
                    "train": core._label_counts_by_benchmark_and_class(
                        y=y,
                        benchmark_labels=benchmark_labels.tolist(),
                        mask=train_mask,
                    ),
                    "val": core._label_counts_by_benchmark_and_class(
                        y=y,
                        benchmark_labels=benchmark_labels.tolist(),
                        mask=val_mask,
                    ),
                },
            }
        )
    return split_seeds, split_payloads


def _train_per_feature(
    layer_features: Dict[str, np.ndarray],
    layer_order: Sequence[str],
    y: np.ndarray,
    benchmark_labels: np.ndarray,
    split_payloads: Sequence[Dict[str, Any]],
    reg_values: Sequence[float],
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    per_feature_results: Dict[str, Dict[str, Any]] = {}
    for feature_name in tqdm(
        layer_order,
        desc=f"Training probes ({int(args.num_split_seeds)} split seeds, C sweep + early stopping on validation)",
    ):
        X = np.asarray(layer_features[feature_name], dtype=np.float32)
        seed_runs = []
        for split_payload in split_payloads:
            run = core._sweep_probe_with_validation(
                X=X,
                y=y,
                train_mask=split_payload["train_mask"],
                val_mask=split_payload["val_mask"],
                test_mask=split_payload["test_mask"],
                split_seed=int(split_payload["split_seed"]),
                reg_values=list(reg_values),
                args=args,
                benchmark_labels=benchmark_labels,
            )
            run["validation_pairs"] = split_payload["validation_pairs"]
            run["validation_pairs_by_benchmark"] = split_payload["validation_pairs_by_benchmark"]
            run["split_sizes"] = split_payload["split_sizes"]
            seed_runs.append(run)

        train_scores = [float(r["best_train_accuracy"]) for r in seed_runs]
        val_scores = [float(r["validation_accuracy_at_best_c"]) for r in seed_runs]
        test_scores = [float(r["test_accuracy_at_best_c"]) for r in seed_runs if r.get("test_accuracy_at_best_c") is not None]
        class0_scores = [float(r["class0_test_accuracy_at_best_c"]) for r in seed_runs if r.get("class0_test_accuracy_at_best_c") is not None]
        class1_scores = [float(r["class1_test_accuracy_at_best_c"]) for r in seed_runs if r.get("class1_test_accuracy_at_best_c") is not None]
        benchmark_scores = core._mean_dict_metrics_from_seed_runs(seed_runs=seed_runs, key="benchmark_test_accuracy_at_best_c")
        benchmark_class0_scores = core._mean_dict_metrics_from_seed_runs(seed_runs=seed_runs, key="benchmark_class0_test_accuracy_at_best_c")
        benchmark_class1_scores = core._mean_dict_metrics_from_seed_runs(seed_runs=seed_runs, key="benchmark_class1_test_accuracy_at_best_c")
        best_c_values = [float(r["best_c"]) for r in seed_runs]
        best_c_mode = sorted(best_c_values)[0] if not best_c_values else max(
            sorted(set(best_c_values)),
            key=lambda c: (best_c_values.count(c), -c),
        )

        per_feature_results[str(feature_name)] = {
            "num_split_seeds": int(len(seed_runs)),
            "split_seeds": [int(r["split_seed"]) for r in seed_runs],
            "seed_runs": seed_runs,
            "selection_metric": "validation_accuracy",
            "best_c_values_by_split": best_c_values,
            "best_c": float(best_c_mode),
            "mean_train_accuracy_at_best_c": float(np.mean(train_scores)),
            "std_train_accuracy_at_best_c": float(np.std(train_scores)),
            "mean_validation_accuracy_at_best_c": float(np.mean(val_scores)),
            "std_validation_accuracy_at_best_c": float(np.std(val_scores)),
            "mean_test_accuracy_at_best_c": (float(np.mean(test_scores)) if test_scores else None),
            "std_test_accuracy_at_best_c": (float(np.std(test_scores)) if test_scores else None),
            "mean_class0_test_accuracy_at_best_c": float(np.mean(class0_scores)) if class0_scores else None,
            "mean_class1_test_accuracy_at_best_c": float(np.mean(class1_scores)) if class1_scores else None,
            "mean_benchmark_test_accuracy_at_best_c": benchmark_scores,
            "mean_benchmark_class0_test_accuracy_at_best_c": benchmark_class0_scores,
            "mean_benchmark_class1_test_accuracy_at_best_c": benchmark_class1_scores,
            "validation_accuracy_at_best_c": float(np.mean(val_scores)),
            "test_accuracy_at_best_c": (float(np.mean(test_scores)) if test_scores else None),
        }
    return per_feature_results


def _build_llm_strategy_results(per_feature_results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    llm_layers = sorted(
        {
            name.split("__")[0]
            for name in per_feature_results
            if re.fullmatch(r"language_model/layer_\d+__[^/]+", name) is not None
        },
        key=lambda s: int(re.search(r"layer_(\d+)", s).group(1)),
    )
    out: Dict[str, Dict[str, Any]] = {}
    for llm_layer in llm_layers:
        strategy_acc = {}
        for strategy in core.LLM_STRATEGIES:
            key = f"{llm_layer}__{strategy}"
            if key in per_feature_results:
                strategy_acc[strategy] = float(per_feature_results[key]["mean_validation_accuracy_at_best_c"])
        best_strategy = None
        best_acc = -1.0
        for strategy in core.LLM_STRATEGIES:
            if strategy not in strategy_acc:
                continue
            score = float(strategy_acc[strategy])
            if score > best_acc:
                best_acc = score
                best_strategy = strategy
        best_feature = f"{llm_layer}__{best_strategy}" if best_strategy is not None else None
        out[str(llm_layer)] = {
            "strategy_validation_accuracies": strategy_acc,
            "best_strategy": best_strategy,
            "best_feature": best_feature,
            "best_validation_accuracy": best_acc,
        }
    return out


def _load_preextracted_all_examples_store(
    args: argparse.Namespace,
    model_path: str,
) -> Tuple[Dict[str, Any], Dict[str, int], Path]:
    cache_path = (
        Path(args.all_examples_features_cache_path)
        if str(args.all_examples_features_cache_path).strip()
        else core._preextracted_all_examples_path_for_vlm(vlm_key=str(args.vlm), use_additional_feature_cache=False)
    )
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing pre-extracted all-examples cache: {cache_path}")
    payload = torch.load(cache_path)
    expected_cache_type = "qwen_preextracted_all_examples_features"
    if str(payload.get("cache_type", "")) != expected_cache_type:
        raise RuntimeError(
            f"Invalid all-examples cache_type: {payload.get('cache_type')}, expected {expected_cache_type}"
        )
    core._maybe_warn_preextracted_cache_model_mismatch(
        payload=payload,
        requested_model_path=str(model_path),
        cache_path=cache_path,
        family_label="QWEN",
    )
    with_payload = payload.get("with_image", {}) or {}
    sig_keys = [str(x) for x in with_payload.get("signature_keys", [])]
    sig_to_idx = {sig: i for i, sig in enumerate(sig_keys)}
    feature_store = with_payload.get("layer_features", {})
    return feature_store, sig_to_idx, cache_path


def _build_labeled_holdout_pool(
    responses: Sequence[Dict[str, Any]],
    seen_signatures: Set[Tuple[str, str, str]],
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
    label_lookup: Dict[Tuple[str, str, str], str],
    target_label: str,
    allowed_benchmarks: Set[str],
    exclude_short_responses_in_holdout: bool,
    require_mirage_without_image_correct: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    pool_true: List[Dict[str, Any]] = []
    pool_false: List[Dict[str, Any]] = []
    seen_pool_signatures: Set[Tuple[str, str, str]] = set()
    skipped_short = 0

    for row in responses:
        ds = str(row.get("dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        uid = str(row.get("unique_id", "")).strip()
        vid = str(row.get("variant_id", "")).strip()
        label = label_lookup.get((ds, uid, vid))
        if label != target_label:
            continue

        wo = row.get("without_image", {}) or {}
        mirage_like = wo.get("mirage_like")
        if mirage_like not in (True, False):
            continue
        if require_mirage_without_image_correct and bool(mirage_like):
            if wo.get("correct") is not True:
                continue

        sig = core._conversation_signature_from_response_row(row)
        if sig in seen_signatures or sig in seen_pool_signatures:
            continue

        qid = str(row.get("question_id", "")).strip()
        imgs = image_lookup_uid.get((ds, uid))
        if imgs is None:
            imgs = image_lookup_qid.get((ds, qid))
        if not imgs:
            continue

        with_image_response = str((row.get("with_image", {}) or {}).get("response", ""))
        if not core._norm_text(with_image_response):
            continue
        if exclude_short_responses_in_holdout:
            if core.core._count_tokens(with_image_response) < core.MIN_RESPONSE_TOKENS:
                skipped_short += 1
                continue

        conv = core.core._make_vllm_messages(
            prompt_text=row.get("prompt_text", ""),
            image_bytes_list=imgs,
            system_prompt=row.get("system_prompt", ""),
        )
        conv.append({"role": "assistant", "content": with_image_response})
        item = {
            "dataset": ds,
            "unique_id": uid,
            "variant_id": vid,
            "conversation": conv,
            "mirage_like": bool(mirage_like),
            "ab_label": str(target_label),
        }
        if mirage_like is True:
            pool_true.append(item)
        else:
            pool_false.append(item)
        seen_pool_signatures.add(sig)

    return pool_true, pool_false, int(skipped_short)


def _build_short_question_holdout_pool(
    responses: Sequence[Dict[str, Any]],
    seen_signatures: Set[Tuple[str, str, str]],
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
    allowed_benchmarks: Set[str],
    question_word_limit: int,
    exclude_short_responses_in_holdout: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    pool_true: List[Dict[str, Any]] = []
    pool_false: List[Dict[str, Any]] = []
    seen_pool_signatures: Set[Tuple[str, str, str]] = set()
    stats = {
        "rows_considered": 0,
        "rows_filtered_missing_question_text": 0,
        "rows_filtered_long_question": 0,
        "rows_filtered_missing_image": 0,
        "rows_filtered_empty_response": 0,
        "rows_filtered_short_response": 0,
        "rows_filtered_seen_signature": 0,
    }

    for row in responses:
        ds = str(row.get("dataset", "")).strip()
        if ds not in allowed_benchmarks:
            continue
        stats["rows_considered"] += 1
        q_text = str(row.get("variant_question_text", "")).strip()
        if not q_text:
            prompt = str(row.get("prompt_text", "")).strip()
            q_text = re.split(r"\n\s*\n", prompt, maxsplit=1)[0].strip() if prompt else ""
        if not q_text:
            stats["rows_filtered_missing_question_text"] += 1
            continue
        if _count_words_in_question_text(q_text) >= int(question_word_limit):
            stats["rows_filtered_long_question"] += 1
            continue

        uid = str(row.get("unique_id", "")).strip()
        vid = str(row.get("variant_id", "")).strip()
        wo = row.get("without_image", {}) or {}
        mirage_like = wo.get("mirage_like")
        if mirage_like not in (True, False):
            continue

        sig = core._conversation_signature_from_response_row(row)
        if sig in seen_signatures or sig in seen_pool_signatures:
            stats["rows_filtered_seen_signature"] += 1
            continue

        qid = str(row.get("question_id", "")).strip()
        imgs = image_lookup_uid.get((ds, uid))
        if imgs is None:
            imgs = image_lookup_qid.get((ds, qid))
        if not imgs:
            stats["rows_filtered_missing_image"] += 1
            continue

        with_image_response = str((row.get("with_image", {}) or {}).get("response", ""))
        if not core._norm_text(with_image_response):
            stats["rows_filtered_empty_response"] += 1
            continue
        if exclude_short_responses_in_holdout:
            if core.core._count_tokens(with_image_response) < core.MIN_RESPONSE_TOKENS:
                stats["rows_filtered_short_response"] += 1
                continue

        conv = core.core._make_vllm_messages(
            prompt_text=row.get("prompt_text", ""),
            image_bytes_list=imgs,
            system_prompt=row.get("system_prompt", ""),
        )
        conv.append({"role": "assistant", "content": with_image_response})
        item = {
            "dataset": ds,
            "unique_id": uid,
            "variant_id": vid,
            "conversation": conv,
            "mirage_like": bool(mirage_like),
            "question_word_count": int(_count_words_in_question_text(q_text)),
            "question_word_limit_exclusive": int(question_word_limit),
        }
        if bool(mirage_like):
            pool_true.append(item)
        else:
            pool_false.append(item)
        seen_pool_signatures.add(sig)

    return pool_true, pool_false, stats


def _prepare_holdout_payloads_short_questions(
    responses: Sequence[Dict[str, Any]],
    seen_signatures: Set[Tuple[str, str, str]],
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
    allowed_benchmarks: Set[str],
    args: argparse.Namespace,
    sig_to_idx: Dict[str, int],
) -> Dict[str, Any]:
    pool_true, pool_false, short_question_filter_stats = _build_short_question_holdout_pool(
        responses=responses,
        seen_signatures=seen_signatures,
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
        allowed_benchmarks=allowed_benchmarks,
        question_word_limit=int(args.short_question_word_limit),
        exclude_short_responses_in_holdout=bool(args.exclude_short_responses_in_holdout),
    )
    if not pool_true or not pool_false:
        raise RuntimeError(
            "Short-question hold-out pool has insufficient class coverage: "
            f"true={len(pool_true)}, false={len(pool_false)}"
        )

    raw_holdout_pool_sizes = {"mirage_true": int(len(pool_true)), "mirage_false": int(len(pool_false))}
    raw_counts_by_benchmark = core._holdout_pool_counts_by_benchmark(pool_true=pool_true, pool_false=pool_false)
    raw_selection_plan = core._plan_balanced_holdout_selection(
        pool_true=pool_true,
        pool_false=pool_false,
        requested_num_true=int(args.num_holdout_mirage_true),
        requested_num_false=int(args.num_holdout_mirage_false),
    )

    pool_true_f, pool_false_f, filter_stats = core._filter_holdout_pool_to_available_signatures(
        pool_true=pool_true,
        pool_false=pool_false,
        available_signature_keys=set(sig_to_idx.keys()),
    )
    if not pool_true_f or not pool_false_f:
        raise RuntimeError(
            "After filtering to pre-extracted signatures, short-question hold-out pool is empty in one class: "
            f"true={len(pool_true_f)}, false={len(pool_false_f)}"
        )

    filtered_counts_by_benchmark = core._holdout_pool_counts_by_benchmark(pool_true=pool_true_f, pool_false=pool_false_f)
    filtered_selection_plan = core._plan_balanced_holdout_selection(
        pool_true=pool_true_f,
        pool_false=pool_false_f,
        requested_num_true=int(args.num_holdout_mirage_true),
        requested_num_false=int(args.num_holdout_mirage_false),
    )

    eval_seeds = [int(args.seed) + i for i in range(int(args.num_eval_seeds))]
    holdout_payloads_by_seed: Dict[int, Dict[str, Any]] = {}
    holdout_selected_sizes_by_seed: Dict[str, Any] = {}
    for eval_seed in eval_seeds:
        selected_examples, selected_counts_by_benchmark = core._select_holdout_examples_balanced_by_benchmark(
            pool_true=pool_true_f,
            pool_false=pool_false_f,
            selected_pairs_by_benchmark={
                str(k): int(v)
                for k, v in dict(filtered_selection_plan["selected_pairs_by_benchmark"]).items()
            },
            seed=int(eval_seed),
        )
        y_holdout = np.asarray([1 if bool(x["mirage_like"]) else 0 for x in selected_examples], dtype=np.int64)
        benchmark_labels_holdout = [str(x.get("dataset", "unknown")) for x in selected_examples]
        holdout_indices: List[int] = []
        for item in selected_examples:
            sig = core._signature_key_from_holdout_item(item)
            if sig is None:
                raise RuntimeError("Held-out item missing conversation signature fields.")
            idx = sig_to_idx.get(sig)
            if idx is None:
                raise RuntimeError("Held-out item signature missing after pre-filtering.")
            holdout_indices.append(int(idx))
        holdout_idx_arr = np.asarray(holdout_indices, dtype=np.int64)
        holdout_payloads_by_seed[int(eval_seed)] = {
            "y_holdout": y_holdout,
            "benchmark_labels_holdout": benchmark_labels_holdout,
            "holdout_indices": holdout_idx_arr,
            "num_examples": int(len(y_holdout)),
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
            "num_examples_per_class": {
                "mirage_true": int((y_holdout == 1).sum()),
                "mirage_false": int((y_holdout == 0).sum()),
            },
            "eval_target": SHORT_Q_EVAL_TARGET,
        }
        holdout_selected_sizes_by_seed[str(int(eval_seed))] = {
            "num_examples_total": int(len(y_holdout)),
            "num_examples_per_class": {
                "mirage_true": int((y_holdout == 1).sum()),
                "mirage_false": int((y_holdout == 0).sum()),
            },
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
        }

    return {
        "eval_target": SHORT_Q_EVAL_TARGET,
        "question_word_limit_exclusive": int(args.short_question_word_limit),
        "pool_sizes_before_preextract_filter": raw_holdout_pool_sizes,
        "pool_sizes_by_benchmark_before_preextract_filter": raw_counts_by_benchmark,
        "selection_plan_before_preextract_filter": raw_selection_plan,
        "pool_sizes_after_preextract_filter": {
            "mirage_true": int(len(pool_true_f)),
            "mirage_false": int(len(pool_false_f)),
        },
        "pool_sizes_by_benchmark_after_preextract_filter": filtered_counts_by_benchmark,
        "selection_plan_after_preextract_filter": filtered_selection_plan,
        "pool_filter_to_preextract_cache": filter_stats,
        "short_question_filter_stats": short_question_filter_stats,
        "holdout_payloads_by_seed": holdout_payloads_by_seed,
        "holdout_selected_sizes_by_seed": holdout_selected_sizes_by_seed,
    }


def _prepare_holdout_payloads_for_label(
    target_label: str,
    responses: Sequence[Dict[str, Any]],
    seen_signatures: Set[Tuple[str, str, str]],
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
    all_examples_label_lookup: Dict[Tuple[str, str, str], str],
    allowed_benchmarks: Set[str],
    args: argparse.Namespace,
    sig_to_idx: Dict[str, int],
    require_mirage_without_image_correct: bool,
) -> Dict[str, Any]:
    pool_true, pool_false, skipped_short = _build_labeled_holdout_pool(
        responses=responses,
        seen_signatures=seen_signatures,
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
        label_lookup=all_examples_label_lookup,
        target_label=str(target_label),
        allowed_benchmarks=allowed_benchmarks,
        exclude_short_responses_in_holdout=bool(args.exclude_short_responses_in_holdout),
        require_mirage_without_image_correct=bool(require_mirage_without_image_correct),
    )
    if not pool_true or not pool_false:
        raise RuntimeError(
            f"Target '{target_label}' hold-out pool has insufficient class coverage: "
            f"true={len(pool_true)}, false={len(pool_false)}"
        )

    raw_holdout_pool_sizes = {"mirage_true": int(len(pool_true)), "mirage_false": int(len(pool_false))}
    raw_counts_by_benchmark = core._holdout_pool_counts_by_benchmark(pool_true=pool_true, pool_false=pool_false)
    raw_selection_plan = core._plan_balanced_holdout_selection(
        pool_true=pool_true,
        pool_false=pool_false,
        requested_num_true=int(args.num_holdout_mirage_true),
        requested_num_false=int(args.num_holdout_mirage_false),
    )

    pool_true_f, pool_false_f, filter_stats = core._filter_holdout_pool_to_available_signatures(
        pool_true=pool_true,
        pool_false=pool_false,
        available_signature_keys=set(sig_to_idx.keys()),
    )
    if not pool_true_f or not pool_false_f:
        raise RuntimeError(
            f"After filtering to pre-extracted signatures, target '{target_label}' hold-out "
            f"pool is empty in one class: true={len(pool_true_f)}, false={len(pool_false_f)}"
        )

    filtered_counts_by_benchmark = core._holdout_pool_counts_by_benchmark(pool_true=pool_true_f, pool_false=pool_false_f)
    filtered_selection_plan = core._plan_balanced_holdout_selection(
        pool_true=pool_true_f,
        pool_false=pool_false_f,
        requested_num_true=int(args.num_holdout_mirage_true),
        requested_num_false=int(args.num_holdout_mirage_false),
    )

    eval_seeds = [int(args.seed) + i for i in range(int(args.num_eval_seeds))]
    holdout_payloads_by_seed: Dict[int, Dict[str, Any]] = {}
    holdout_selected_sizes_by_seed: Dict[str, Any] = {}
    for eval_seed in eval_seeds:
        selected_examples, selected_counts_by_benchmark = core._select_holdout_examples_balanced_by_benchmark(
            pool_true=pool_true_f,
            pool_false=pool_false_f,
            selected_pairs_by_benchmark={
                str(k): int(v)
                for k, v in dict(filtered_selection_plan["selected_pairs_by_benchmark"]).items()
            },
            seed=int(eval_seed),
        )
        y_holdout = np.asarray([1 if bool(x["mirage_like"]) else 0 for x in selected_examples], dtype=np.int64)
        benchmark_labels_holdout = [str(x.get("dataset", "unknown")) for x in selected_examples]
        holdout_indices: List[int] = []
        for item in selected_examples:
            sig = core._signature_key_from_holdout_item(item)
            if sig is None:
                raise RuntimeError("Held-out item missing conversation signature fields.")
            idx = sig_to_idx.get(sig)
            if idx is None:
                raise RuntimeError("Held-out item signature missing after pre-filtering.")
            holdout_indices.append(int(idx))
        holdout_idx_arr = np.asarray(holdout_indices, dtype=np.int64)
        holdout_payloads_by_seed[int(eval_seed)] = {
            "y_holdout": y_holdout,
            "benchmark_labels_holdout": benchmark_labels_holdout,
            "holdout_indices": holdout_idx_arr,
            "num_examples": int(len(y_holdout)),
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
            "num_examples_per_class": {
                "mirage_true": int((y_holdout == 1).sum()),
                "mirage_false": int((y_holdout == 0).sum()),
            },
            "target_label": str(target_label),
        }
        holdout_selected_sizes_by_seed[str(int(eval_seed))] = {
            "num_examples_total": int(len(y_holdout)),
            "num_examples_per_class": {
                "mirage_true": int((y_holdout == 1).sum()),
                "mirage_false": int((y_holdout == 0).sum()),
            },
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
        }

    return {
        "target_label": str(target_label),
        "pool_sizes_before_preextract_filter": raw_holdout_pool_sizes,
        "pool_sizes_by_benchmark_before_preextract_filter": raw_counts_by_benchmark,
        "selection_plan_before_preextract_filter": raw_selection_plan,
        "pool_sizes_after_preextract_filter": {
            "mirage_true": int(len(pool_true_f)),
            "mirage_false": int(len(pool_false_f)),
        },
        "pool_sizes_by_benchmark_after_preextract_filter": filtered_counts_by_benchmark,
        "selection_plan_after_preextract_filter": filtered_selection_plan,
        "pool_filter_to_preextract_cache": filter_stats,
        "num_candidates_skipped_short_responses": int(skipped_short),
        "holdout_payloads_by_seed": holdout_payloads_by_seed,
        "holdout_selected_sizes_by_seed": holdout_selected_sizes_by_seed,
    }


def _evaluate_feature_runs_on_holdout_target(
    per_feature_results: Dict[str, Dict[str, Any]],
    split_payloads: Sequence[Dict[str, Any]],
    layer_features: Dict[str, np.ndarray],
    y: np.ndarray,
    holdout_payloads_by_seed: Dict[int, Dict[str, Any]],
    preextracted_holdout_feature_store: Dict[str, Any],
    args: argparse.Namespace,
    eval_target_label: str,
) -> Dict[str, Dict[str, Any]]:
    eval_seeds = [int(args.seed) + i for i in range(int(args.num_eval_seeds))]
    results: Dict[str, Dict[str, Any]] = {}
    for feature_name in sorted(per_feature_results.keys(), key=core._layer_sort_key):
        layer_runs: List[Dict[str, Any]] = []
        X_all = np.asarray(layer_features[feature_name], dtype=np.float32)
        for eval_idx, eval_seed in enumerate(eval_seeds):
            split_payload = split_payloads[eval_idx % len(split_payloads)]
            split_seed = int(split_payload["split_seed"])
            run_by_seed = {
                int(r["split_seed"]): r for r in per_feature_results[feature_name]["seed_runs"]
            }
            if split_seed not in run_by_seed:
                raise RuntimeError(f"Missing split run for split_seed={split_seed}, feature={feature_name}.")
            split_run = run_by_seed[split_seed]
            best_c = float(split_run["best_c"])

            X_train = X_all[split_payload["train_mask"]]
            y_train = y[split_payload["train_mask"]]
            X_val = X_all[split_payload["val_mask"]]
            y_val = y[split_payload["val_mask"]]

            holdout_payload = holdout_payloads_by_seed[int(eval_seed)]
            holdout_indices = holdout_payload["holdout_indices"]
            if feature_name not in preextracted_holdout_feature_store:
                raise RuntimeError(f"Held-out feature '{feature_name}' missing from all-examples cache store.")
            X_holdout = core._gather_feature_rows_from_store(
                feature_store=preextracted_holdout_feature_store[feature_name],
                row_indices=np.asarray(holdout_indices, dtype=np.int64),
            )
            y_holdout = holdout_payload["y_holdout"]
            benchmark_labels_holdout = holdout_payload.get("benchmark_labels_holdout", [])

            fit = core._fit_fixed_c_with_multi_init(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_eval=X_holdout,
                split_seed=split_seed,
                c_value=best_c,
                args=args,
            )
            holdout_pred = fit["eval_pred"]
            holdout_acc = float((holdout_pred == y_holdout).mean())
            class1_mask = y_holdout == 1
            class0_mask = y_holdout == 0
            benchmark_test_acc, benchmark_class0_test_acc, benchmark_class1_test_acc = core._compute_benchmark_test_metrics(
                y_true=y_holdout,
                y_pred=holdout_pred,
                benchmark_labels=benchmark_labels_holdout,
            )
            layer_runs.append(
                {
                    "eval_target_label": str(eval_target_label),
                    "eval_seed": int(eval_seed),
                    "split_seed": int(split_seed),
                    "feature": str(feature_name),
                    "best_c": float(best_c),
                    "train_accuracy": float(fit["train_accuracy"]),
                    "validation_accuracy": float(fit["validation_accuracy"]),
                    "test_accuracy": float(holdout_acc),
                    "test_accuracy_mirage_true": float((holdout_pred[class1_mask] == y_holdout[class1_mask]).mean()) if class1_mask.any() else None,
                    "test_accuracy_mirage_false": float((holdout_pred[class0_mask] == y_holdout[class0_mask]).mean()) if class0_mask.any() else None,
                    "benchmark_test_accuracy": benchmark_test_acc,
                    "benchmark_class0_test_accuracy": benchmark_class0_test_acc,
                    "benchmark_class1_test_accuracy": benchmark_class1_test_acc,
                    "macro_benchmark_test_accuracy": core._macro_average_metric_dict(benchmark_test_acc),
                    "macro_benchmark_class0_test_accuracy": core._macro_average_metric_dict(benchmark_class0_test_acc),
                    "macro_benchmark_class1_test_accuracy": core._macro_average_metric_dict(benchmark_class1_test_acc),
                    "num_holdout_examples": int(len(y_holdout)),
                    "num_holdout_examples_per_class": holdout_payload.get("num_examples_per_class"),
                    "num_holdout_examples_by_benchmark": holdout_payload.get("selected_counts_by_benchmark"),
                    "best_init_index": int(fit["init_index"]),
                    "best_init_seed": int(fit["init_seed"]),
                }
            )

        results[str(feature_name)] = {
            "best_feature": str(feature_name),
            "selection_scope": "all_features",
            "best_strategy": (
                str(feature_name).split("__", 1)[1] if "__" in str(feature_name) else None
            ),
            "best_validation_accuracy": float(
                per_feature_results[feature_name].get("mean_validation_accuracy_at_best_c", float("nan"))
            ),
            "num_eval_seeds": int(len(layer_runs)),
            "eval_seeds": [int(r["eval_seed"]) for r in layer_runs],
            "train_accuracy_mean": float(np.mean([r["train_accuracy"] for r in layer_runs])),
            "validation_accuracy_mean": float(np.mean([r["validation_accuracy"] for r in layer_runs])),
            "test_accuracy_mean": float(np.mean([r["test_accuracy"] for r in layer_runs])),
            "macro_benchmark_test_accuracy_mean": float(
                np.mean(
                    [
                        float(r["macro_benchmark_test_accuracy"])
                        for r in layer_runs
                        if r.get("macro_benchmark_test_accuracy") is not None
                    ]
                )
            ) if any(r.get("macro_benchmark_test_accuracy") is not None for r in layer_runs) else None,
            "train_accuracy_std": float(np.std([r["train_accuracy"] for r in layer_runs])),
            "validation_accuracy_std": float(np.std([r["validation_accuracy"] for r in layer_runs])),
            "test_accuracy_std": float(np.std([r["test_accuracy"] for r in layer_runs])),
            "macro_benchmark_test_accuracy_std": float(
                np.std(
                    [
                        float(r["macro_benchmark_test_accuracy"])
                        for r in layer_runs
                        if r.get("macro_benchmark_test_accuracy") is not None
                    ]
                )
            ) if any(r.get("macro_benchmark_test_accuracy") is not None for r in layer_runs) else None,
            "seed_runs": layer_runs,
        }
    return results


def _to_all_feature_results(per_feature_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "feature": k,
            "num_split_seeds": int(v["num_split_seeds"]),
            "split_seeds": v["split_seeds"],
            "selection_metric": v["selection_metric"],
            "mean_train_accuracy_at_best_c": float(v["mean_train_accuracy_at_best_c"]),
            "std_train_accuracy_at_best_c": float(v["std_train_accuracy_at_best_c"]),
            "mean_validation_accuracy_at_best_c": float(v["mean_validation_accuracy_at_best_c"]),
            "std_validation_accuracy_at_best_c": float(v["std_validation_accuracy_at_best_c"]),
            "mean_test_accuracy_at_best_c": (
                float(v["mean_test_accuracy_at_best_c"])
                if v.get("mean_test_accuracy_at_best_c") is not None
                else None
            ),
            "std_test_accuracy_at_best_c": (
                float(v["std_test_accuracy_at_best_c"])
                if v.get("std_test_accuracy_at_best_c") is not None
                else None
            ),
            "mean_class0_test_accuracy_at_best_c": v["mean_class0_test_accuracy_at_best_c"],
            "mean_class1_test_accuracy_at_best_c": v["mean_class1_test_accuracy_at_best_c"],
            "validation_accuracy_at_best_c": float(v["validation_accuracy_at_best_c"]),
            "test_accuracy_at_best_c": (
                float(v["test_accuracy_at_best_c"])
                if v.get("test_accuracy_at_best_c") is not None
                else None
            ),
            "class0_test_accuracy_at_best_c": v["mean_class0_test_accuracy_at_best_c"],
            "class1_test_accuracy_at_best_c": v["mean_class1_test_accuracy_at_best_c"],
            "seed_runs": v["seed_runs"],
        }
        for k, v in sorted(per_feature_results.items(), key=lambda kv: core._layer_sort_key(kv[0]))
    ]


def _run_single_probe(
    probe_label: str,
    pairs: Sequence[Dict[str, Any]],
    pair_label_lookup: Dict[Tuple[int, str], str],
    responses: Sequence[Dict[str, Any]],
    all_examples_label_lookup: Dict[Tuple[str, str, str], str],
    model_path: str,
    args: argparse.Namespace,
    save_dir: Path,
    allowed_benchmarks: Set[str],
    eval_targets: Sequence[str] = TARGET_LABELS,
    require_mirage_without_image_correct: bool = False,
    without_image_correct_keys: Set[Tuple[str, str, str]] | None = None,
) -> Dict[str, Any]:
    wo_correct_keys = without_image_correct_keys if without_image_correct_keys is not None else set()
    selected_pairs, pair_selection_stats = _select_pairs_for_probe(
        pairs=pairs,
        pair_label_lookup=pair_label_lookup,
        allowed_benchmarks=allowed_benchmarks,
        target_label=str(probe_label),
        require_mirage_without_image_correct=bool(require_mirage_without_image_correct),
        without_image_correct_keys=wo_correct_keys,
    )
    conversations, labels, pair_ids, pair_benchmarks, sample_names, skipped_short_training_pairs = _build_training_examples_from_pairs(
        pairs=selected_pairs,
        exclude_short_responses=bool(args.exclude_short_responses_in_training_pairs),
    )

    requested_strategies = list(core.LLM_STRATEGIES)
    contrastive_cache_path = (
        Path(args.features_cache_path)
        if str(args.features_cache_path).strip()
        else None
    )
    layer_features, layer_order, resolved_contrastive_cache_path = core._load_preextracted_contrastive_subset(
        vlm_key=str(args.vlm),
        conversations=conversations,
        include_attention_probes=False,
        include_mlp_probes=False,
        include_residual_probes=True,
        llm_feature_strategies=requested_strategies,
        use_additional_feature_preextract_cache=False,
        requested_model_path=str(model_path),
        cache_path=contrastive_cache_path,
    )

    y = np.asarray(labels, dtype=np.int64)
    pair_ids_arr = np.asarray(pair_ids, dtype=np.int64)
    benchmark_labels = np.asarray([str(pair_benchmarks[int(pid)]) for pid in pair_ids_arr.tolist()], dtype=object)

    reg_values = [float(x.strip()) for x in str(args.regularization_values).split(",") if x.strip()]
    if not reg_values:
        raise ValueError("--regularization_values must contain at least one value.")
    if 0.0 not in reg_values:
        raise ValueError("--regularization_values must include 0.")

    split_seeds, split_payloads = _make_split_payloads(
        y=y,
        pair_ids_arr=pair_ids_arr,
        pair_benchmarks=pair_benchmarks,
        seed=int(args.seed),
        num_split_seeds=int(args.num_split_seeds),
        val_fraction=float(args.val_fraction),
    )

    per_feature_results = _train_per_feature(
        layer_features=layer_features,
        layer_order=layer_order,
        y=y,
        benchmark_labels=benchmark_labels,
        split_payloads=split_payloads,
        reg_values=reg_values,
        args=args,
    )
    llm_strategy_results = _build_llm_strategy_results(per_feature_results)
    all_feature_results = _to_all_feature_results(per_feature_results)

    image_lookup_uid, image_lookup_qid = core._build_image_lookup_from_responses(
        mirage_root=REPO_ROOT.resolve(),
        responses=list(responses),
    )
    seen_signatures = {core._conversation_signature_from_conv(c) for c in conversations}
    preextracted_holdout_feature_store, sig_to_idx, resolved_all_examples_cache = _load_preextracted_all_examples_store(
        args=args,
        model_path=str(model_path),
    )

    holdout_by_target: Dict[str, Any] = {}
    holdout_eval_results_by_target: Dict[str, Any] = {}
    for eval_target in eval_targets:
        holdout_meta = _prepare_holdout_payloads_for_label(
            target_label=str(eval_target),
            responses=responses,
            seen_signatures=seen_signatures,
            image_lookup_uid=image_lookup_uid,
            image_lookup_qid=image_lookup_qid,
            all_examples_label_lookup=all_examples_label_lookup,
            allowed_benchmarks=allowed_benchmarks,
            args=args,
            sig_to_idx=sig_to_idx,
            require_mirage_without_image_correct=bool(require_mirage_without_image_correct),
        )
        holdout_by_target[str(eval_target)] = holdout_meta
        holdout_eval_results_by_target[str(eval_target)] = _evaluate_feature_runs_on_holdout_target(
            per_feature_results=per_feature_results,
            split_payloads=split_payloads,
            layer_features=layer_features,
            y=y,
            holdout_payloads_by_seed=holdout_meta["holdout_payloads_by_seed"],
            preextracted_holdout_feature_store=preextracted_holdout_feature_store,
            args=args,
            eval_target_label=str(eval_target),
        )

    training_pair_counts_by_benchmark: Dict[str, int] = defaultdict(int)
    for bench in pair_benchmarks:
        training_pair_counts_by_benchmark[str(bench)] += 1
    training_sample_counts_by_benchmark_and_class = core._label_counts_by_benchmark_and_class(
        y=y,
        benchmark_labels=benchmark_labels.tolist(),
        mask=None,
    )

    split_details_summary = [
        {
            "split_seed": int(sp["split_seed"]),
            "validation_pairs": [int(x) for x in sp["validation_pairs"]],
            "validation_pairs_by_benchmark": {
                k: [int(x) for x in v] for k, v in sp["validation_pairs_by_benchmark"].items()
            },
            "split_sizes": sp["split_sizes"],
            "split_label_counts": sp["split_label_counts"],
            "split_benchmark_class_counts": sp["split_benchmark_class_counts"],
        }
        for sp in split_payloads
    ]

    summary_payload = {
        "probe_label": str(probe_label),
        "allowed_benchmarks": sorted(list(allowed_benchmarks)),
        "pair_selection_stats": pair_selection_stats,
        "num_pairs": int(len(set(pair_ids))),
        "num_samples": int(len(y)),
        "num_class0_samples": int((y == 0).sum()),
        "num_class1_samples": int((y == 1).sum()),
        "training_pair_counts_by_benchmark": {str(k): int(v) for k, v in sorted(training_pair_counts_by_benchmark.items())},
        "training_sample_counts_by_benchmark_and_class": training_sample_counts_by_benchmark_and_class,
        "training_mode": "pair_ab_filtered_benchmark_fraction_validation_multiseed",
        "num_split_seeds": int(args.num_split_seeds),
        "split_seeds": [int(s) for s in split_seeds],
        "validation_fraction": float(args.val_fraction),
        "regularization_sweep_c_values": reg_values,
        "hyperparam_selection_metric": "validation_accuracy",
        "probe_epochs": int(args.probe_epochs),
        "probe_lr": float(args.probe_lr),
        "multi_init_probe_selection": bool(args.multi_init_probe_selection),
        "probe_num_initializations": int(args.probe_num_initializations),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "include_attention_probes": False,
        "include_mlp_probes": False,
        "include_residual_probes": True,
        "requested_llm_feature_strategies": requested_strategies,
        "feature_extraction_version": int(core.FEATURE_EXTRACTION_VERSION),
        "heldout_eval_all_features": True,
        "num_eval_seeds": int(args.num_eval_seeds),
        "eval_seeds": [int(args.seed) + i for i in range(int(args.num_eval_seeds))],
        "num_holdout_mirage_true_requested": int(args.num_holdout_mirage_true),
        "num_holdout_mirage_false_requested": int(args.num_holdout_mirage_false),
        "exclude_short_responses_in_training_pairs": bool(args.exclude_short_responses_in_training_pairs),
        "num_training_pairs_skipped_short_responses": int(skipped_short_training_pairs),
        "exclude_short_responses_in_holdout": bool(args.exclude_short_responses_in_holdout),
        "require_mirage_without_image_correct_filter": bool(require_mirage_without_image_correct),
        "holdout_by_target": {
            k: {
                "pool_sizes_before_preextract_filter": v["pool_sizes_before_preextract_filter"],
                "pool_sizes_by_benchmark_before_preextract_filter": v["pool_sizes_by_benchmark_before_preextract_filter"],
                "selection_plan_before_preextract_filter": v["selection_plan_before_preextract_filter"],
                "pool_sizes_after_preextract_filter": v["pool_sizes_after_preextract_filter"],
                "pool_sizes_by_benchmark_after_preextract_filter": v["pool_sizes_by_benchmark_after_preextract_filter"],
                "selection_plan_after_preextract_filter": v["selection_plan_after_preextract_filter"],
                "pool_filter_to_preextract_cache": v["pool_filter_to_preextract_cache"],
                "num_candidates_skipped_short_responses": v["num_candidates_skipped_short_responses"],
                "holdout_selected_sizes_by_seed": v["holdout_selected_sizes_by_seed"],
            }
            for k, v in holdout_by_target.items()
        },
        "pairs_path": str(Path(args.pairs_path).resolve()),
        "responses_path": str(Path(args.responses_path).resolve()),
        "contrastive_labels_path": str(Path(args.contrastive_labels_path).resolve()),
        "all_examples_labels_path": str(Path(args.all_examples_labels_path).resolve()),
        "vlm": str(args.vlm),
        "model_path": str(model_path),
        "contrastive_preextract_path": str(resolved_contrastive_cache_path),
        "all_examples_preextract_path": str(resolved_all_examples_cache),
        "normalize_features": bool(args.normalize_features),
        "pca_components": int(args.pca_components),
        "split_details": split_details_summary,
    }

    vlm_tag = str(args.vlm)
    all_features_path = save_dir / f"{vlm_tag}_all_feature_probe_accuracies.json"
    llm_strategy_path = save_dir / f"{vlm_tag}_llm_layer_best_strategy_results.json"
    config_path = save_dir / f"{vlm_tag}_run_config.json"
    sample_names_path = save_dir / f"{vlm_tag}_sample_names.json"
    heldout_paths: Dict[str, str] = {}

    _write_json(all_features_path, all_feature_results)
    _write_json(llm_strategy_path, llm_strategy_results)
    _write_json(config_path, summary_payload)
    _write_json(sample_names_path, sample_names)
    for tgt in eval_targets:
        tgt_up = str(tgt).upper()
        tgt_path = save_dir / f"{vlm_tag}_llm_residual_layer_heldout_eval_on_{tgt_up}.json"
        _write_json(tgt_path, holdout_eval_results_by_target[str(tgt_up)])
        heldout_paths[str(tgt_up)] = str(tgt_path)

    print(f"[{probe_label}] Saved feature accuracies: {all_features_path}")
    for tgt in sorted(heldout_paths.keys()):
        print(f"[{probe_label}] Saved held-out eval on {tgt}: {heldout_paths[tgt]}")
    print(f"[{probe_label}] Saved run config: {config_path}")

    return {
        "probe_label": str(probe_label),
        "save_dir": str(save_dir),
        "all_feature_results_path": str(all_features_path),
        "llm_strategy_results_path": str(llm_strategy_path),
        "heldout_eval_paths": heldout_paths,
        "heldout_eval_on_A_path": heldout_paths.get("A"),
        "heldout_eval_on_B_path": heldout_paths.get("B"),
        "run_config_path": str(config_path),
        "heldout_eval_results_by_target": holdout_eval_results_by_target,
    }


def _run_single_probe_short_question_mode(
    pairs: Sequence[Dict[str, Any]],
    responses: Sequence[Dict[str, Any]],
    model_path: str,
    args: argparse.Namespace,
    save_dir: Path,
    allowed_benchmarks: Set[str],
) -> Dict[str, Any]:
    question_lookup = _build_response_question_lookup(
        responses=responses,
        allowed_benchmarks=allowed_benchmarks,
    )
    selected_pairs, pair_selection_stats = _select_pairs_for_short_questions(
        pairs=pairs,
        allowed_benchmarks=allowed_benchmarks,
        question_word_limit=int(args.short_question_word_limit),
        response_question_lookup=question_lookup,
    )
    conversations, labels, pair_ids, pair_benchmarks, sample_names, skipped_short_training_pairs = _build_training_examples_from_pairs(
        pairs=selected_pairs,
        exclude_short_responses=bool(args.exclude_short_responses_in_training_pairs),
    )

    requested_strategies = list(core.LLM_STRATEGIES)
    contrastive_cache_path = (
        Path(args.features_cache_path)
        if str(args.features_cache_path).strip()
        else None
    )
    layer_features, layer_order, resolved_contrastive_cache_path = core._load_preextracted_contrastive_subset(
        vlm_key=str(args.vlm),
        conversations=conversations,
        include_attention_probes=False,
        include_mlp_probes=False,
        include_residual_probes=True,
        llm_feature_strategies=requested_strategies,
        use_additional_feature_preextract_cache=False,
        requested_model_path=str(model_path),
        cache_path=contrastive_cache_path,
    )

    y = np.asarray(labels, dtype=np.int64)
    pair_ids_arr = np.asarray(pair_ids, dtype=np.int64)
    benchmark_labels = np.asarray([str(pair_benchmarks[int(pid)]) for pid in pair_ids_arr.tolist()], dtype=object)

    reg_values = [float(x.strip()) for x in str(args.regularization_values).split(",") if x.strip()]
    if not reg_values:
        raise ValueError("--regularization_values must contain at least one value.")
    if 0.0 not in reg_values:
        raise ValueError("--regularization_values must include 0.")

    split_seeds, split_payloads = _make_split_payloads(
        y=y,
        pair_ids_arr=pair_ids_arr,
        pair_benchmarks=pair_benchmarks,
        seed=int(args.seed),
        num_split_seeds=int(args.num_split_seeds),
        val_fraction=float(args.val_fraction),
    )

    per_feature_results = _train_per_feature(
        layer_features=layer_features,
        layer_order=layer_order,
        y=y,
        benchmark_labels=benchmark_labels,
        split_payloads=split_payloads,
        reg_values=reg_values,
        args=args,
    )
    llm_strategy_results = _build_llm_strategy_results(per_feature_results)
    all_feature_results = _to_all_feature_results(per_feature_results)

    image_lookup_uid, image_lookup_qid = core._build_image_lookup_from_responses(
        mirage_root=REPO_ROOT.resolve(),
        responses=list(responses),
    )
    seen_signatures = {core._conversation_signature_from_conv(c) for c in conversations}
    preextracted_holdout_feature_store, sig_to_idx, resolved_all_examples_cache = _load_preextracted_all_examples_store(
        args=args,
        model_path=str(model_path),
    )

    holdout_meta = _prepare_holdout_payloads_short_questions(
        responses=responses,
        seen_signatures=seen_signatures,
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
        allowed_benchmarks=allowed_benchmarks,
        args=args,
        sig_to_idx=sig_to_idx,
    )
    holdout_eval_results = _evaluate_feature_runs_on_holdout_target(
        per_feature_results=per_feature_results,
        split_payloads=split_payloads,
        layer_features=layer_features,
        y=y,
        holdout_payloads_by_seed=holdout_meta["holdout_payloads_by_seed"],
        preextracted_holdout_feature_store=preextracted_holdout_feature_store,
        args=args,
        eval_target_label=SHORT_Q_EVAL_TARGET,
    )

    training_pair_counts_by_benchmark: Dict[str, int] = defaultdict(int)
    for bench in pair_benchmarks:
        training_pair_counts_by_benchmark[str(bench)] += 1
    training_sample_counts_by_benchmark_and_class = core._label_counts_by_benchmark_and_class(
        y=y,
        benchmark_labels=benchmark_labels.tolist(),
        mask=None,
    )

    split_details_summary = [
        {
            "split_seed": int(sp["split_seed"]),
            "validation_pairs": [int(x) for x in sp["validation_pairs"]],
            "validation_pairs_by_benchmark": {
                k: [int(x) for x in v] for k, v in sp["validation_pairs_by_benchmark"].items()
            },
            "split_sizes": sp["split_sizes"],
            "split_label_counts": sp["split_label_counts"],
            "split_benchmark_class_counts": sp["split_benchmark_class_counts"],
        }
        for sp in split_payloads
    ]

    summary_payload = {
        "probe_label": "short_question",
        "allowed_benchmarks": sorted(list(allowed_benchmarks)),
        "pair_selection_stats": pair_selection_stats,
        "num_pairs": int(len(set(pair_ids))),
        "num_samples": int(len(y)),
        "num_class0_samples": int((y == 0).sum()),
        "num_class1_samples": int((y == 1).sum()),
        "training_pair_counts_by_benchmark": {str(k): int(v) for k, v in sorted(training_pair_counts_by_benchmark.items())},
        "training_sample_counts_by_benchmark_and_class": training_sample_counts_by_benchmark_and_class,
        "training_mode": "pair_short_question_filtered_benchmark_fraction_validation_multiseed",
        "question_word_limit_exclusive": int(args.short_question_word_limit),
        "num_split_seeds": int(args.num_split_seeds),
        "split_seeds": [int(s) for s in split_seeds],
        "validation_fraction": float(args.val_fraction),
        "regularization_sweep_c_values": reg_values,
        "hyperparam_selection_metric": "validation_accuracy",
        "probe_epochs": int(args.probe_epochs),
        "probe_lr": float(args.probe_lr),
        "multi_init_probe_selection": bool(args.multi_init_probe_selection),
        "probe_num_initializations": int(args.probe_num_initializations),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "include_attention_probes": False,
        "include_mlp_probes": False,
        "include_residual_probes": True,
        "requested_llm_feature_strategies": requested_strategies,
        "feature_extraction_version": int(core.FEATURE_EXTRACTION_VERSION),
        "heldout_eval_all_features": True,
        "heldout_eval_target": SHORT_Q_EVAL_TARGET,
        "num_eval_seeds": int(args.num_eval_seeds),
        "eval_seeds": [int(args.seed) + i for i in range(int(args.num_eval_seeds))],
        "num_holdout_mirage_true_requested": int(args.num_holdout_mirage_true),
        "num_holdout_mirage_false_requested": int(args.num_holdout_mirage_false),
        "exclude_short_responses_in_training_pairs": bool(args.exclude_short_responses_in_training_pairs),
        "num_training_pairs_skipped_short_responses": int(skipped_short_training_pairs),
        "exclude_short_responses_in_holdout": bool(args.exclude_short_responses_in_holdout),
        "holdout_short_question": {
            "question_word_limit_exclusive": holdout_meta["question_word_limit_exclusive"],
            "pool_sizes_before_preextract_filter": holdout_meta["pool_sizes_before_preextract_filter"],
            "pool_sizes_by_benchmark_before_preextract_filter": holdout_meta["pool_sizes_by_benchmark_before_preextract_filter"],
            "selection_plan_before_preextract_filter": holdout_meta["selection_plan_before_preextract_filter"],
            "pool_sizes_after_preextract_filter": holdout_meta["pool_sizes_after_preextract_filter"],
            "pool_sizes_by_benchmark_after_preextract_filter": holdout_meta["pool_sizes_by_benchmark_after_preextract_filter"],
            "selection_plan_after_preextract_filter": holdout_meta["selection_plan_after_preextract_filter"],
            "pool_filter_to_preextract_cache": holdout_meta["pool_filter_to_preextract_cache"],
            "short_question_filter_stats": holdout_meta["short_question_filter_stats"],
            "holdout_selected_sizes_by_seed": holdout_meta["holdout_selected_sizes_by_seed"],
        },
        "pairs_path": str(Path(args.pairs_path).resolve()),
        "responses_path": str(Path(args.responses_path).resolve()),
        "vlm": str(args.vlm),
        "model_path": str(model_path),
        "contrastive_preextract_path": str(resolved_contrastive_cache_path),
        "all_examples_preextract_path": str(resolved_all_examples_cache),
        "normalize_features": bool(args.normalize_features),
        "pca_components": int(args.pca_components),
        "split_details": split_details_summary,
    }

    vlm_tag = str(args.vlm)
    all_features_path = save_dir / f"{vlm_tag}_all_feature_probe_accuracies.json"
    llm_strategy_path = save_dir / f"{vlm_tag}_llm_layer_best_strategy_results.json"
    config_path = save_dir / f"{vlm_tag}_run_config.json"
    sample_names_path = save_dir / f"{vlm_tag}_sample_names.json"
    heldout_path = save_dir / f"{vlm_tag}_llm_residual_layer_heldout_eval_on_{SHORT_Q_EVAL_TARGET}.json"

    _write_json(all_features_path, all_feature_results)
    _write_json(llm_strategy_path, llm_strategy_results)
    _write_json(config_path, summary_payload)
    _write_json(sample_names_path, sample_names)
    _write_json(heldout_path, holdout_eval_results)

    print(f"[short_question] Saved feature accuracies: {all_features_path}")
    print(f"[short_question] Saved held-out eval on {SHORT_Q_EVAL_TARGET}: {heldout_path}")
    print(f"[short_question] Saved run config: {config_path}")

    return {
        "probe_label": "short_question",
        "save_dir": str(save_dir),
        "all_feature_results_path": str(all_features_path),
        "llm_strategy_results_path": str(llm_strategy_path),
        "heldout_eval_path": str(heldout_path),
        "run_config_path": str(config_path),
    }


def _summarize_transfer_grid(
    probe_a: Dict[str, Any],
    probe_b: Dict[str, Any],
) -> Dict[str, Any]:
    def _best_feature_score(payload: Dict[str, Any], eval_target: str) -> Dict[str, Any]:
        results = payload["heldout_eval_results_by_target"][eval_target]
        best_feature = None
        best_mean = float("-inf")
        best_std = None
        for feature, info in results.items():
            score = float(info.get("test_accuracy_mean", float("-inf")))
            if score > best_mean:
                best_mean = score
                best_feature = feature
                best_std = float(info.get("test_accuracy_std")) if info.get("test_accuracy_std") is not None else None
        return {
            "best_feature_by_mean_test_accuracy": best_feature,
            "mean_test_accuracy": (float(best_mean) if best_feature is not None else None),
            "std_test_accuracy": best_std,
        }

    aa = _best_feature_score(probe_a, "A")
    ab = _best_feature_score(probe_a, "B")
    ba = _best_feature_score(probe_b, "A")
    bb = _best_feature_score(probe_b, "B")
    return {
        "A_probe_on_A": aa,
        "A_probe_on_B": ab,
        "B_probe_on_A": ba,
        "B_probe_on_B": bb,
    }


def _run_b_only_single_benchmark(
    benchmark: str,
    *,
    require_mirage_without_image_correct: bool,
    args_dict: Dict[str, Any],
    gpu_id: int | None,
) -> Dict[str, Any]:
    parent_visible_raw = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    if torch.cuda.is_available():
        current_visible_raw = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
        parent_visible_ids: List[int] = []
        current_visible_ids: List[int] = []
        if parent_visible_raw:
            try:
                parent_visible_ids = [int(x.strip()) for x in parent_visible_raw.split(",") if x.strip()]
            except Exception:
                parent_visible_ids = []
        if current_visible_raw:
            try:
                current_visible_ids = [int(x.strip()) for x in current_visible_raw.split(",") if x.strip()]
            except Exception:
                current_visible_ids = []
        local_device_index = 0
        if gpu_id is not None:
            if parent_visible_ids:
                if int(gpu_id) in parent_visible_ids:
                    local_device_index = int(parent_visible_ids.index(int(gpu_id)))
                elif 0 <= int(gpu_id) < len(parent_visible_ids):
                    local_device_index = int(gpu_id)
                else:
                    local_device_index = 0
            elif current_visible_ids:
                if int(gpu_id) in current_visible_ids:
                    local_device_index = int(current_visible_ids.index(int(gpu_id)))
                elif 0 <= int(gpu_id) < len(current_visible_ids):
                    local_device_index = int(gpu_id)
                else:
                    local_device_index = 0
            else:
                local_device_index = int(gpu_id)
        torch.cuda.set_device(int(local_device_index))
        print(
            f"[worker:{benchmark}] CUDA pinning -> requested_gpu_id={gpu_id}, "
            f"parent_CUDA_VISIBLE_DEVICES='{parent_visible_raw}', "
            f"current_CUDA_VISIBLE_DEVICES='{current_visible_raw}', "
            f"local_device_index={local_device_index}, "
            f"current_device={torch.cuda.current_device()}"
        )
    args = argparse.Namespace(**copy.deepcopy(args_dict))
    save_root = Path(args.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    model_path = _resolve_model_path(args)

    allowed_benchmarks = {str(benchmark)}
    pairs = _read_json(Path(args.pairs_path))
    responses = _read_json(Path(args.responses_path))
    contrastive_labels_payload = _read_json(Path(args.contrastive_labels_path))
    all_examples_labels_payload = _read_json(Path(args.all_examples_labels_path))

    pair_label_lookup = _load_contrastive_label_lookup(
        labels_payload=contrastive_labels_payload,
        allowed_benchmarks=allowed_benchmarks,
    )
    all_examples_label_lookup = _load_all_examples_label_lookup(
        labels_payload=all_examples_labels_payload,
        allowed_benchmarks=allowed_benchmarks,
    )
    wo_correct_keys = (
        _build_without_image_correct_key_set(responses=responses, allowed_benchmarks=allowed_benchmarks)
        if bool(require_mirage_without_image_correct)
        else set()
    )

    suffix = "_mirage_wo_correct" if bool(require_mirage_without_image_correct) else ""
    run_dir = save_root / f"probe_B_{benchmark}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    out = _run_single_probe(
        probe_label="B",
        pairs=pairs,
        pair_label_lookup=pair_label_lookup,
        responses=responses,
        all_examples_label_lookup=all_examples_label_lookup,
        model_path=model_path,
        args=args,
        save_dir=run_dir,
        allowed_benchmarks=allowed_benchmarks,
        eval_targets=("A", "B"),
        require_mirage_without_image_correct=bool(require_mirage_without_image_correct),
        without_image_correct_keys=wo_correct_keys,
    )
    return {
        "benchmark": str(benchmark),
        "save_dir": out["save_dir"],
        "heldout_eval_on_A_path": out["heldout_eval_on_A_path"],
        "heldout_eval_on_B_path": out["heldout_eval_on_B_path"],
        "run_config_path": out["run_config_path"],
        "gpu_id": (int(gpu_id) if gpu_id is not None else None),
    }


def _parse_parallel_gpu_ids(gpu_csv: str) -> List[int]:
    raw = str(gpu_csv).strip()
    # Allow shorthand like "01" or "23" (interpreted as [0,1] or [2,3]).
    if "," not in raw and raw.isdigit() and len(raw) > 1:
        return [int(ch) for ch in raw]
    ids = [x.strip() for x in raw.split(",") if x.strip()]
    out: List[int] = []
    for x in ids:
        out.append(int(x))
    return out


def _run_b_only_per_benchmark_mode(
    args: argparse.Namespace,
    *,
    require_mirage_without_image_correct: bool,
) -> Dict[str, Any]:
    gpu_ids = _parse_parallel_gpu_ids(args.parallel_benchmark_gpus)
    print(f"[orchestrator] Parsed benchmark GPU ids: {gpu_ids}")
    args_dict = dict(vars(args))
    per_benchmark_summary: Dict[str, Any] = {}

    run_parallel = (not bool(args.no_parallel_benchmark_runs)) and len(gpu_ids) >= len(DEFAULT_BENCHMARKS)
    if run_parallel:
        with ProcessPoolExecutor(
            max_workers=len(DEFAULT_BENCHMARKS),
            mp_context=get_context("spawn"),
        ) as ex:
            futures = {
                ex.submit(
                    _run_b_only_single_benchmark,
                    benchmark,
                    require_mirage_without_image_correct=bool(require_mirage_without_image_correct),
                    args_dict=args_dict,
                    gpu_id=gpu_ids[idx],
                ): benchmark
                for idx, benchmark in enumerate(DEFAULT_BENCHMARKS)
            }
            for fut in as_completed(futures):
                bench = futures[fut]
                out = fut.result()
                per_benchmark_summary[str(bench)] = out
                print(
                    f"[parallel] Completed benchmark={bench} on gpu={out.get('gpu_id')} "
                    f"save_dir={out.get('save_dir')}"
                )
    else:
        if bool(args.no_parallel_benchmark_runs):
            print("[orchestrator] Parallel benchmark runs disabled via --no_parallel_benchmark_runs.")
        elif len(gpu_ids) < len(DEFAULT_BENCHMARKS):
            print(
                f"[orchestrator] Falling back to sequential benchmark runs: "
                f"need >= {len(DEFAULT_BENCHMARKS)} gpu ids, got {len(gpu_ids)}."
            )
        for idx, benchmark in enumerate(DEFAULT_BENCHMARKS):
            gpu_id = gpu_ids[idx] if idx < len(gpu_ids) else None
            out = _run_b_only_single_benchmark(
                benchmark=benchmark,
                require_mirage_without_image_correct=bool(require_mirage_without_image_correct),
                args_dict=args_dict,
                gpu_id=gpu_id,
            )
            per_benchmark_summary[str(benchmark)] = out

    return {
        "runs": dict(sorted(per_benchmark_summary.items(), key=lambda kv: kv[0])),
        "used_parallel_execution": bool(run_parallel),
        "configured_parallel_benchmark_gpus": str(args.parallel_benchmark_gpus),
    }


def main() -> None:
    args = parse_args()
    save_root = Path(args.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)

    if int(args.short_question_word_limit) <= 0:
        raise ValueError("--short_question_word_limit must be positive.")

    model_path = _resolve_model_path(args)
    pairs = _read_json(Path(args.pairs_path))
    responses = _read_json(Path(args.responses_path))

    if bool(args.mmmu_pro_short_question_mode):
        allowed_benchmarks = {"mmmu_pro"}
        run_dir = save_root / "probe_mmmu_pro_short_question"
        run_dir.mkdir(parents=True, exist_ok=True)
        out = _run_single_probe_short_question_mode(
            pairs=pairs,
            responses=responses,
            model_path=model_path,
            args=args,
            save_dir=run_dir,
            allowed_benchmarks=allowed_benchmarks,
        )
        summary_path = save_root / "mmmu_pro_short_question_summary.json"
        _write_json(
            summary_path,
            {
                "experiment": "contrastive_mmmu_pro_short_question",
                "mode_flag": "mmmu_pro_short_question_mode",
                "question_word_limit_exclusive": int(args.short_question_word_limit),
                "run": {
                    "save_dir": out["save_dir"],
                    "heldout_eval_path": out["heldout_eval_path"],
                    "run_config_path": out["run_config_path"],
                },
            },
        )
        print(f"Saved mmmu_pro short-question summary: {summary_path}")
        return

    allowed_benchmarks = set(DEFAULT_BENCHMARKS)
    contrastive_labels_payload = _read_json(Path(args.contrastive_labels_path))
    all_examples_labels_payload = _read_json(Path(args.all_examples_labels_path))

    pair_label_lookup = _load_contrastive_label_lookup(
        labels_payload=contrastive_labels_payload,
        allowed_benchmarks=allowed_benchmarks,
    )
    all_examples_label_lookup = _load_all_examples_label_lookup(
        labels_payload=all_examples_labels_payload,
        allowed_benchmarks=allowed_benchmarks,
    )

    if bool(args.b_only_per_benchmark_mode_filter_mirage_without_image_correct):
        mode_out = _run_b_only_per_benchmark_mode(
            args=args,
            require_mirage_without_image_correct=True,
        )
        summary_path = save_root / "b_only_per_benchmark_mirage_wo_correct_summary.json"
        _write_json(
            summary_path,
            {
                "experiment": "contrastive_b_only_per_benchmark_mirage_wo_correct",
                "mode_flag": "b_only_per_benchmark_mode_filter_mirage_without_image_correct",
                "parallel_execution": bool(mode_out["used_parallel_execution"]),
                "parallel_benchmark_gpus": str(mode_out["configured_parallel_benchmark_gpus"]),
                "runs": mode_out["runs"],
            },
        )
        print(f"Saved B-only per-benchmark + mirage wo-correct summary: {summary_path}")
        return

    if bool(args.b_only_per_benchmark_mode):
        mode_out = _run_b_only_per_benchmark_mode(
            args=args,
            require_mirage_without_image_correct=False,
        )
        summary_path = save_root / "b_only_per_benchmark_summary.json"
        _write_json(
            summary_path,
            {
                "experiment": "contrastive_b_only_per_benchmark",
                "mode_flag": "b_only_per_benchmark_mode",
                "parallel_execution": bool(mode_out["used_parallel_execution"]),
                "parallel_benchmark_gpus": str(mode_out["configured_parallel_benchmark_gpus"]),
                "runs": mode_out["runs"],
            },
        )
        print(f"Saved B-only per-benchmark summary: {summary_path}")
        return

    probe_a_dir = save_root / "probe_A"
    probe_b_dir = save_root / "probe_B"
    probe_a_dir.mkdir(parents=True, exist_ok=True)
    probe_b_dir.mkdir(parents=True, exist_ok=True)

    probe_a_results = _run_single_probe(
        probe_label="A",
        pairs=pairs,
        pair_label_lookup=pair_label_lookup,
        responses=responses,
        all_examples_label_lookup=all_examples_label_lookup,
        model_path=model_path,
        args=args,
        save_dir=probe_a_dir,
        allowed_benchmarks=allowed_benchmarks,
    )
    probe_b_results = _run_single_probe(
        probe_label="B",
        pairs=pairs,
        pair_label_lookup=pair_label_lookup,
        responses=responses,
        all_examples_label_lookup=all_examples_label_lookup,
        model_path=model_path,
        args=args,
        save_dir=probe_b_dir,
        allowed_benchmarks=allowed_benchmarks,
    )

    transfer_grid = _summarize_transfer_grid(probe_a=probe_a_results, probe_b=probe_b_results)
    transfer_payload = {
        "experiment": "contrastive_ab_transfer",
        "probe_a": {
            "save_dir": probe_a_results["save_dir"],
            "heldout_eval_on_A_path": probe_a_results["heldout_eval_on_A_path"],
            "heldout_eval_on_B_path": probe_a_results["heldout_eval_on_B_path"],
        },
        "probe_b": {
            "save_dir": probe_b_results["save_dir"],
            "heldout_eval_on_A_path": probe_b_results["heldout_eval_on_A_path"],
            "heldout_eval_on_B_path": probe_b_results["heldout_eval_on_B_path"],
        },
        "transfer_grid_best_feature_summaries": transfer_grid,
    }
    transfer_path = save_root / "transfer_grid_summary.json"
    _write_json(transfer_path, transfer_payload)
    print(f"Saved transfer grid summary: {transfer_path}")


if __name__ == "__main__":
    main()
