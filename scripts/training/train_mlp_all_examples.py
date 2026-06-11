#!/usr/bin/env python3
"""Train probe models on all examples.

This script is the all-examples MLP entrypoint and exposes reusable helpers for
concat-residual and activation-difference all-examples probe scripts.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.training.train_log_reg_all_examples as all_core
import scripts.training.train_log_reg_contrastive as pair_core
import scripts.training.train_mlp_contrastive as core


FEATURE_EXTRACTION_VERSION = 1


def build_arg_parser(description: str, default_save_dir: str, default_cache_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
        "--contrastive_pairs_path",
        type=str,
        default="./tmp_artifacts/contrastive_conversation_pairs.json",
        help=(
            "Contrastive pairs artifact used for optional all-examples training augmentation. "
            "If left at the default value, OVIS/QWEN use data/final_data/*_contrastive.json "
            "and GLM uses tmp_artifacts/contrastive_conversation_pairs.json."
        ),
    )
    parser.add_argument(
        "--contrastive_neutral_pairs_path",
        type=str,
        default="./tmp_artifacts/contrastive_conversation_pairs_neutral_as_non_mirage.json",
        help=(
            "Neutral-inclusive contrastive pairs artifact for optional augmentation. "
            "Defaults to tmp_artifacts/contrastive_conversation_pairs_neutral_as_non_mirage.json."
        ),
    )
    parser.add_argument(
        "--include_contrastive_pairs_in_training_examples",
        dest="include_contrastive_pairs_in_training_examples",
        action="store_true",
        help="Augment all-examples data with contrastive pair examples (eligible for train/val/test splits).",
    )
    parser.add_argument(
        "--no_include_contrastive_pairs_in_training_examples",
        dest="include_contrastive_pairs_in_training_examples",
        action="store_false",
        help="Disable contrastive-pair augmentation for all-examples trainers.",
    )
    parser.set_defaults(include_contrastive_pairs_in_training_examples=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude_short_responses_in_training_examples",
        dest="exclude_short_responses_in_training_examples",
        action="store_true",
        help=(
            "Exclude training-candidate rows where with-image response has "
            f"fewer than {all_core.MIN_RESPONSE_TOKENS} whitespace-separated tokens."
        ),
    )
    parser.add_argument(
        "--no_exclude_short_responses_in_training_examples",
        dest="exclude_short_responses_in_training_examples",
        action="store_false",
        help="Keep short-response rows in training-candidate selection.",
    )
    parser.set_defaults(exclude_short_responses_in_training_examples=True)
    parser.add_argument("--num_split_seeds", type=int, default=5)
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="all",
        choices=core.BENCHMARK_CHOICES,
        help="Restrict training/eval to one benchmark or use all represented benchmarks.",
    )
    parser.add_argument(
        "--neutral_as_non_mirage",
        dest="neutral_as_non_mirage",
        action="store_true",
        help="Treat rows with mirage_label='neutral*' as class 0.",
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
        help="Optional cap on unique questions per class after per-question sampling.",
    )
    parser.add_argument(
        "--target_examples_per_class",
        type=int,
        default=500,
        help="Target examples per class before split.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=0.20)
    parser.add_argument(
        "--vlm",
        type=str,
        default="ovis",
        choices=sorted(core.MODEL_REGISTRY.keys()),
        help="Model family key used for artifact naming and default model resolution.",
    )
    parser.add_argument("--model_path_override", type=str, default="")
    parser.add_argument(
        "--regularization_values",
        type=str,
        default="0,0.01,0.1,1.0",
        help="Comma-separated C values (weight_decay = 0 if C=0 else 1/C).",
    )
    parser.add_argument("--probe_epochs", type=int, default=800)
    parser.add_argument("--probe_lr", type=float, default=0.03)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--num_probe_inits", type=int, default=3)
    parser.add_argument("--disable_multi_init", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=200)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--include_attention_probes",
        action="store_true",
        default=False,
        help=(
            "If set, include additional attention probe families: per-head attention "
            "activations and post-attention activations. Default: off."
        ),
    )
    parser.add_argument(
        "--include_mlp_probes",
        action="store_true",
        default=False,
        help=(
            "If set, include additional MLP probe family: per-layer MLP activations. "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--include_additional_attention_mlp_probes",
        action="store_true",
        default=False,
        help=(
            "Backward-compatible alias for enabling both --include_attention_probes and "
            "--include_mlp_probes."
        ),
    )
    parser.add_argument(
        "--include_residual_probes",
        dest="include_residual_probes",
        action="store_true",
        default=True,
        help=(
            "Include baseline residual probe families (vision encoder, projector, residual stream). "
            "Default: on."
        ),
    )
    parser.add_argument(
        "--no_include_residual_probes",
        dest="include_residual_probes",
        action="store_false",
        help="Exclude baseline residual probe families.",
    )
    parser.add_argument(
        "--llm_feature_strategies",
        type=str,
        default=",".join(pair_core.LLM_STRATEGIES),
        help=(
            "Comma-separated feature strategies to include. Choices: "
            + ",".join(pair_core.LLM_STRATEGIES)
            + "."
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
    parser.add_argument("--pca_components", type=int, default=0)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
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
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--features_cache_path", type=str, default=default_cache_path)
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    pair_core.add_model_loading_args(parser)
    pair_core.add_preextract_cache_selection_args(parser)
    return parser


def _filter_responses_by_benchmark(responses: List[Dict], benchmark_mode: str) -> List[Dict]:
    if benchmark_mode == "all":
        allowed = {"vqa_rad", "mmmu_pro", "medxpertqa_mm"}
        return [row for row in responses if str(row.get("dataset", "")) in allowed]
    return [row for row in responses if str(row.get("dataset", "")) == benchmark_mode]


def _build_without_image_conversation(row: Dict) -> List[Dict]:
    system_prompt = str(row.get("system_prompt", ""))
    prompt_text = str(row.get("prompt_text", ""))
    without_resp = str((row.get("without_image", {}) or {}).get("response", ""))
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
        {"role": "assistant", "content": without_resp},
    ]


def _load_or_extract_features(
    args: argparse.Namespace,
    model_path: str,
    with_conversations: List[List[Dict]],
    without_conversations: List[List[Dict]],
    labels: List[int],
    sample_names: List[str],
    sample_meta: List[Dict],
    require_without_image: bool,
    cache_path: Path,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], List[str], Path]:
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    requested_include_attention = bool(
        bool(getattr(args, "include_attention_probes", False))
        or bool(getattr(args, "include_additional_attention_mlp_probes", False))
    )
    requested_include_mlp = bool(
        bool(getattr(args, "include_mlp_probes", False))
        or bool(getattr(args, "include_additional_attention_mlp_probes", False))
    )
    requested_include_residual = bool(getattr(args, "include_residual_probes", True))
    requested_llm_feature_strategies = pair_core._parse_requested_llm_feature_strategies(
        getattr(args, "llm_feature_strategies", ",".join(pair_core.LLM_STRATEGIES))
    )

    if pair_core._uses_preextracted_activation_store(str(args.vlm)):
        family = pair_core._preextract_family_for_vlm(str(args.vlm)).upper()
        if bool(getattr(args, "force_reextract", False)):
            print(f"Ignoring --force_reextract for {family}; loading pre-extracted activations cache.")
        with_map, without_map, layer_order, resolved_cache = pair_core._load_preextracted_all_examples_subset(
            vlm_key=str(args.vlm),
            with_conversations=with_conversations,
            without_conversations=without_conversations if bool(require_without_image) else None,
            require_without=bool(require_without_image),
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            llm_feature_strategies=requested_llm_feature_strategies,
            use_additional_feature_preextract_cache=getattr(
                args, "use_additional_feature_preextract_cache", None
            ),
            requested_model_path=str(model_path),
        )
        return with_map, without_map, layer_order, resolved_cache

    use_cache = cache_path.exists() and (not bool(args.force_reextract))
    if use_cache:
        payload = torch.load(cache_path)
        cached_names = [str(x) for x in payload.get("sample_names", [])]
        expected_glm_image_normalization = str(args.vlm) == "glm_4_6v_flash"
        cached_glm_image_normalization = bool(payload.get("glm_image_normalization_applied", False))
        if cached_names != [str(x) for x in sample_names]:
            use_cache = False
        elif expected_glm_image_normalization and (not cached_glm_image_normalization):
            use_cache = False
        elif require_without_image and ("layer_features_without" not in payload):
            use_cache = False

    if use_cache:
        payload = torch.load(cache_path)
        cached_with = payload.get("layer_features_with")
        if cached_with is None:
            # Backward compatibility with simple cache payloads.
            cached_with = payload.get("layer_features")
        if cached_with is None:
            raise RuntimeError(
                "Cache payload missing with-image feature map. Re-run with --force_reextract."
            )
        requested_keys = pair_core._resolve_requested_feature_keys(
            layer_order=payload["layer_order"],
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            llm_feature_strategies=requested_llm_feature_strategies,
        )
        filtered_with, filtered_order = pair_core._filter_layer_feature_map(
            layer_features=cached_with,
            layer_order=payload["layer_order"],
            requested_keys=requested_keys,
        )
        filtered_without, _ = pair_core._filter_layer_feature_map(
            layer_features=payload.get("layer_features_without", {}),
            layer_order=filtered_order,
            requested_keys=filtered_order,
        )
        return (filtered_with, filtered_without, filtered_order, cache_path)

    model = pair_core.load_vlm_for_extraction(
        model_path=model_path,
        attn_implementation=str(args.attn_implementation),
        device_map_raw=str(getattr(args, "device_map", "")),
        max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
        max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
    )
    pair_core._force_attention_backend(model, str(args.attn_implementation))

    layer_features_with: Dict[str, List[np.ndarray]] = defaultdict(list)
    layer_features_without: Dict[str, List[np.ndarray]] = defaultdict(list)
    seen = set()
    layer_order: List[str] = []

    for idx in tqdm(range(len(with_conversations)), desc="Extracting activations", unit="sample", dynamic_ncols=True):
        with_messages = pair_core._to_ovis_messages(with_conversations[idx])
        with_feats = pair_core._extract_sample_features_only(
            model=model,
            messages=with_messages,
            include_additional_attention_mlp_probes=False,
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            model_key=str(args.vlm),
        )
        for key, value in with_feats.items():
            if key not in seen:
                seen.add(key)
                layer_order.append(key)
            layer_features_with[key].append(value.to(torch.float32).cpu().numpy())

        if require_without_image:
            wo_messages = pair_core._to_ovis_messages(without_conversations[idx])
            wo_feats = pair_core._extract_sample_features_only(
                model=model,
                messages=wo_messages,
                include_additional_attention_mlp_probes=False,
                include_attention_probes=bool(requested_include_attention),
                include_mlp_probes=bool(requested_include_mlp),
                include_residual_probes=bool(requested_include_residual),
                model_key=str(args.vlm),
            )
            for key, value in wo_feats.items():
                layer_features_without[key].append(value.to(torch.float32).cpu().numpy())

    layer_order = sorted(layer_order, key=pair_core._layer_sort_key)
    requested_keys = pair_core._resolve_requested_feature_keys(
        layer_order=layer_order,
        include_attention_probes=bool(requested_include_attention),
        include_mlp_probes=bool(requested_include_mlp),
        include_residual_probes=bool(requested_include_residual),
        llm_feature_strategies=requested_llm_feature_strategies,
    )
    layer_features_with, layer_order = pair_core._filter_layer_feature_map(
        layer_features=layer_features_with,
        layer_order=layer_order,
        requested_keys=requested_keys,
    )
    layer_features_without, _ = pair_core._filter_layer_feature_map(
        layer_features=layer_features_without,
        layer_order=layer_order,
        requested_keys=layer_order,
    )
    torch.save(
        {
            "layer_features_with": layer_features_with,
            "layer_features_without": layer_features_without,
            "labels": labels,
            "sample_names": sample_names,
            "sample_meta": sample_meta,
            "layer_order": layer_order,
            "vlm": str(args.vlm),
            "model_path": model_path,
            "glm_image_normalization_applied": bool(str(args.vlm) == "glm_4_6v_flash"),
            "feature_extraction_version": int(FEATURE_EXTRACTION_VERSION),
            "requires_without_image": bool(require_without_image),
            "include_attention_probes": bool(requested_include_attention),
            "include_mlp_probes": bool(requested_include_mlp),
            "include_residual_probes": bool(requested_include_residual),
            "include_additional_attention_mlp_probes": bool(
                getattr(args, "include_additional_attention_mlp_probes", False)
            ),
            "llm_feature_strategies": list(requested_llm_feature_strategies),
        },
        cache_path,
    )
    return layer_features_with, layer_features_without, layer_order, cache_path


def _build_concat_llm_residual_features(layer_features_with: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    strategy_features: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    post_attention_strategy_features: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    mlp_strategy_features: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    attention_head_strategy_features: Dict[Tuple[str, int], List[Tuple[int, np.ndarray]]] = defaultdict(list)

    for key, value in layer_features_with.items():
        key_s = str(key)
        arr = np.asarray(value, dtype=np.float32)

        m_res = re.fullmatch(r"language_model/layer_(\d+)__([^/]+)", key_s)
        if m_res is not None:
            layer_num = int(m_res.group(1))
            strategy = m_res.group(2)
            strategy_features[strategy].append((layer_num, arr))
            continue

        m_post = re.fullmatch(r"language_model/layer_(\d+)/post_attention__([^/]+)", key_s)
        if m_post is not None:
            layer_num = int(m_post.group(1))
            strategy = m_post.group(2)
            post_attention_strategy_features[strategy].append((layer_num, arr))
            continue

        m_mlp = re.fullmatch(r"language_model/layer_(\d+)/mlp__([^/]+)", key_s)
        if m_mlp is not None:
            layer_num = int(m_mlp.group(1))
            strategy = m_mlp.group(2)
            mlp_strategy_features[strategy].append((layer_num, arr))
            continue

        m_head = re.fullmatch(r"language_model/layer_(\d+)/attention_head_(\d+)__([^/]+)", key_s)
        if m_head is not None:
            layer_num = int(m_head.group(1))
            head_idx = int(m_head.group(2))
            strategy = m_head.group(3)
            attention_head_strategy_features[(strategy, head_idx)].append((layer_num, arr))
            continue

    out: Dict[str, np.ndarray] = {}
    # Preserve legacy residual concat outputs when residual stream features are present.
    for strategy, items in strategy_features.items():
        ordered = [arr for _layer, arr in sorted(items, key=lambda t: t[0])]
        if ordered:
            out[f"language_model/all_layers_concat__{strategy}"] = np.concatenate(ordered, axis=1)

    # Additional concat outputs: one per attention head index (across layers), plus post-attn and MLP families.
    for (strategy, head_idx), items in attention_head_strategy_features.items():
        ordered = [arr for _layer, arr in sorted(items, key=lambda t: t[0])]
        if ordered:
            out[f"language_model/all_layers_concat_attention_head_{head_idx}__{strategy}"] = np.concatenate(ordered, axis=1)

    for strategy, items in post_attention_strategy_features.items():
        ordered = [arr for _layer, arr in sorted(items, key=lambda t: t[0])]
        if ordered:
            out[f"language_model/all_layers_concat_post_attention__{strategy}"] = np.concatenate(ordered, axis=1)

    for strategy, items in mlp_strategy_features.items():
        ordered = [arr for _layer, arr in sorted(items, key=lambda t: t[0])]
        if ordered:
            out[f"language_model/all_layers_concat_mlp__{strategy}"] = np.concatenate(ordered, axis=1)

    if not out:
        raise RuntimeError("No language-model features found for concat mode.")
    return out


def _materialize_feature_matrices(
    feature_variant: str,
    layer_features_with: Dict[str, List[np.ndarray]],
    layer_features_without: Dict[str, List[np.ndarray]],
    layer_order: List[str],
    labels: List[int],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, List[str]]:
    y = np.asarray(labels, dtype=np.int64)
    if feature_variant == "raw":
        return ({k: np.asarray(layer_features_with[k], dtype=np.float32) for k in layer_order}, y, list(layer_order))

    if feature_variant == "concat_llm_residual":
        concat_map = _build_concat_llm_residual_features(layer_features_with)
        order = sorted(concat_map.keys())
        return concat_map, y, order

    if feature_variant == "activation_diff":
        common = [k for k in layer_order if k in layer_features_without]
        if not common:
            raise RuntimeError("No overlapping feature keys found for activation-difference mode.")
        diff_map = {
            k: np.asarray(layer_features_with[k], dtype=np.float32) - np.asarray(layer_features_without[k], dtype=np.float32)
            for k in common
        }
        return diff_map, y, common

    raise ValueError(f"Unsupported feature_variant '{feature_variant}'.")


def _split_payloads_all_examples(
    args: argparse.Namespace,
    y: np.ndarray,
    sample_meta: List[Dict],
) -> Tuple[List[Dict], List[int]]:
    if int(args.num_split_seeds) <= 0:
        raise ValueError("--num_split_seeds must be >= 1.")

    val_fraction = float(args.val_fraction)
    test_fraction = float(args.test_fraction)
    if val_fraction <= 0.0 or test_fraction <= 0.0 or (val_fraction + test_fraction) >= 1.0:
        raise ValueError(
            "Require 0 < val_fraction < 1, 0 < test_fraction < 1, and val_fraction + test_fraction < 1. "
            f"Got val_fraction={val_fraction}, test_fraction={test_fraction}."
        )

    split_seed_rng = np.random.default_rng(int(args.seed))
    split_seeds = split_seed_rng.choice(
        1_000_000_000,
        size=int(args.num_split_seeds),
        replace=False,
    ).astype(np.int64).tolist()

    payloads: List[Dict] = []
    for split_seed in split_seeds:
        train_mask, val_mask, test_mask, split_stats = all_core._split_balanced_train_val_test_by_class_dataset(
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
        payloads.append(
            {
                "split_seed": int(split_seed),
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
                "split_stats": split_stats,
                "split_sizes": split_sizes,
            }
        )

    return payloads, [int(s) for s in split_seeds]


def _summarize_feature_runs(seed_runs: List[Dict]) -> Dict:
    train_scores = [float(r["best_train_accuracy"]) for r in seed_runs]
    val_scores = [float(r["val_accuracy_at_best_c"]) for r in seed_runs]
    test_scores = [float(r["test_accuracy_at_best_c"]) for r in seed_runs]
    class0_scores = [
        float(r["class0_test_accuracy_at_best_c"])
        for r in seed_runs
        if r.get("class0_test_accuracy_at_best_c") is not None
    ]
    class1_scores = [
        float(r["class1_test_accuracy_at_best_c"])
        for r in seed_runs
        if r.get("class1_test_accuracy_at_best_c") is not None
    ]
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

    return {
        "num_split_seeds": int(len(seed_runs)),
        "split_seeds": [int(r["split_seed"]) for r in seed_runs],
        "seed_runs": seed_runs,
        "selection_metric": "val_accuracy",
        "mean_train_accuracy_at_best_c": float(np.mean(train_scores)),
        "std_train_accuracy_at_best_c": float(np.std(train_scores)),
        "mean_val_accuracy_at_best_c": float(np.mean(val_scores)),
        "std_val_accuracy_at_best_c": float(np.std(val_scores)),
        "mean_test_accuracy_at_best_c": float(np.mean(test_scores)),
        "std_test_accuracy_at_best_c": float(np.std(test_scores)),
        "mean_class0_test_accuracy_at_best_c": float(np.mean(class0_scores)) if class0_scores else None,
        "mean_class1_test_accuracy_at_best_c": float(np.mean(class1_scores)) if class1_scores else None,
        "mean_benchmark_test_accuracy_at_best_c": benchmark_scores,
        "mean_benchmark_class0_test_accuracy_at_best_c": benchmark_class0_scores,
        "mean_benchmark_class1_test_accuracy_at_best_c": benchmark_class1_scores,
    }


def _summarize_llm_layers(per_feature_results: Dict[str, Dict]) -> Dict[str, Dict]:
    llm_layers = sorted(
        {
            name.split("__")[0]
            for name in per_feature_results
            if re.fullmatch(r"language_model/layer_\d+__[^/]+", name) is not None
        },
        key=lambda s: int(re.search(r"layer_(\d+)", s).group(1)),
    )

    out: Dict[str, Dict] = {}
    for llm_layer in llm_layers:
        strategy_to_val: Dict[str, float] = {}
        strategy_to_test: Dict[str, float] = {}
        for strategy in pair_core.LLM_STRATEGIES:
            key = f"{llm_layer}__{strategy}"
            info = per_feature_results.get(key)
            if info is None:
                continue
            strategy_to_val[strategy] = float(info["mean_val_accuracy_at_best_c"])
            strategy_to_test[strategy] = float(info["mean_test_accuracy_at_best_c"])

        best_strategy = None
        best_val = -1.0
        for strategy, score in strategy_to_val.items():
            if score > best_val:
                best_val = score
                best_strategy = strategy

        out[llm_layer] = {
            "strategy_val_accuracies": strategy_to_val,
            "strategy_test_accuracies": strategy_to_test,
            "best_strategy": best_strategy,
            "best_val_accuracy": float(best_val) if best_strategy is not None else None,
            "best_test_accuracy": strategy_to_test.get(best_strategy) if best_strategy is not None else None,
            "best_strategy_selection_metric": "val_accuracy",
        }

    return out


def run_all_examples_experiment(
    args: argparse.Namespace,
    script_name: str,
    probe_type: str,
    feature_variant: str,
) -> None:
    vlm_key = str(args.vlm)
    requested_include_attention = bool(
        bool(getattr(args, "include_attention_probes", False))
        or bool(getattr(args, "include_additional_attention_mlp_probes", False))
    )
    requested_include_mlp = bool(
        bool(getattr(args, "include_mlp_probes", False))
        or bool(getattr(args, "include_additional_attention_mlp_probes", False))
    )
    requested_include_residual = bool(getattr(args, "include_residual_probes", True))
    additional_feature_experiment_mode = pair_core._is_additional_feature_experiment_mode(
        include_attention_probes=bool(requested_include_attention),
        include_mlp_probes=bool(requested_include_mlp),
        include_residual_probes=bool(requested_include_residual),
    )
    requested_llm_feature_strategies = pair_core._parse_requested_llm_feature_strategies(
        getattr(args, "llm_feature_strategies", ",".join(pair_core.LLM_STRATEGIES))
    )
    if additional_feature_experiment_mode:
        print(
            "Additional-feature experiment mode detected; applying runtime policy: "
            "num_split_seeds=3, single-init probes."
        )
        args.num_split_seeds = 3
        args.disable_multi_init = True
        args.num_probe_inits = 1
    model_path = core.resolve_model_path(vlm_key, str(args.model_path_override))
    run_id = core.make_run_id(seed=int(args.seed), run_id=str(args.run_id))
    cache_path = core._scope_default_path(
        Path(args.features_cache_path),
        vlm=vlm_key,
        marker="all_examples_layer_features",
    )
    save_root = core._scope_default_path(
        Path(args.save_dir),
        vlm=vlm_key,
        marker="_probe_results",
    )

    mirage_root = REPO_ROOT.resolve()
    responses_path = pair_core._resolve_responses_path_for_vlm(
        base_responses_path=Path(args.responses_path),
        vlm_key=vlm_key,
    )
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)

    responses = _filter_responses_by_benchmark(responses, str(args.benchmark_mode))
    require_without = feature_variant == "activation_diff"
    preextract_response_filter_stats = None
    if pair_core._uses_preextracted_activation_store(str(args.vlm)):
        preextract_path = pair_core._preextracted_all_examples_path_for_vlm(str(args.vlm))
        if preextract_path.exists():
            preextract_payload = torch.load(preextract_path)
            with_payload = preextract_payload.get("with_image", {}) or {}
            available_with_sig_keys = {
                str(x) for x in with_payload.get("signature_keys", [])
            }
            before_ct = int(len(responses))
            filtered_responses = []
            for row in responses:
                sig = pair_core._signature_key_from_tuple(
                    pair_core._conversation_signature_from_response_row(row)
                )
                if sig in available_with_sig_keys:
                    filtered_responses.append(row)
            responses = filtered_responses
            after_ct = int(len(responses))
            preextract_response_filter_stats = {
                "responses_before_signature_filter": before_ct,
                "responses_after_signature_filter": after_ct,
                "responses_dropped_by_signature_filter": int(before_ct - after_ct),
                "preextract_path": str(preextract_path),
            }
            if before_ct != after_ct:
                print(
                    "Pre-filtered responses to signatures present in pre-extracted all-examples cache: "
                    f"dropped={before_ct - after_ct}, kept={after_ct}"
                )
    short_without_rows_excluded_pre_selection = 0
    if bool(require_without) and bool(args.exclude_short_responses_in_training_examples):
        filtered_responses: List[Dict] = []
        for row in responses:
            without_image_response = str((row.get("without_image", {}) or {}).get("response", ""))
            if core._count_tokens(without_image_response) < int(all_core.MIN_RESPONSE_TOKENS):
                short_without_rows_excluded_pre_selection += 1
                continue
            filtered_responses.append(row)
        if short_without_rows_excluded_pre_selection > 0:
            print(
                "Activation-diff short-response filter (without-image) applied before sampling: "
                f"dropped={short_without_rows_excluded_pre_selection}, kept={len(filtered_responses)}"
            )
        responses = filtered_responses
    if not responses:
        raise ValueError("No responses remain after benchmark filtering.")

    image_lookup_uid, image_lookup_qid = all_core._build_image_lookup(mirage_root=mirage_root, responses=responses)
    balanced_rows, selection_stats = all_core._select_balanced_examples(
        responses=responses,
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
        seed=int(args.seed),
        max_questions=int(args.max_questions),
        include_short_response_filter=bool(args.exclude_short_responses_in_training_examples),
        neutral_as_non_mirage=bool(args.neutral_as_non_mirage),
        target_examples_per_class=int(args.target_examples_per_class),
        selected_benchmark=None if str(args.benchmark_mode) == "all" else str(args.benchmark_mode),
        excluded_datasets=[] if str(args.benchmark_mode) != "all" else ["microvqa"],
    )

    selected_with = [
        all_core._build_conversation_from_row(
            row,
            image_lookup_uid=image_lookup_uid,
            image_lookup_qid=image_lookup_qid,
            neutral_as_non_mirage=bool(args.neutral_as_non_mirage),
        )
        for row in balanced_rows
    ]
    contrastive_aug_added = 0
    short_without_aug_examples_excluded_pre_selection = 0
    dropped_not_in_preextracted_cache = 0
    extra_without_conversations: List[List[Dict]] = []
    if bool(args.include_contrastive_pairs_in_training_examples):
        base_pairs_path = Path(
            args.contrastive_neutral_pairs_path
            if bool(args.neutral_as_non_mirage)
            else args.contrastive_pairs_path
        )
        pairs_path = all_core._resolve_scoped_path_for_vlm(base_pairs_path, str(args.vlm))
        selected_dataset_set = (
            {str(args.benchmark_mode)} if str(args.benchmark_mode) != "all" else set(all_core.ALL_BENCHMARK_DATASETS)
        )
        contrastive_aug_examples = all_core._load_contrastive_pair_augmentation_examples(
            pairs_path=pairs_path,
            selected_dataset_set=selected_dataset_set,
        )
        seen_sig_keys = {
            pair_core._signature_key_from_conv(x["conversation"])
            for x in selected_with
            if isinstance(x.get("conversation"), list)
        }
        for ex in contrastive_aug_examples:
            ex_without_conv = (
                ex.get("without_conversation", []) if isinstance(ex.get("without_conversation", []), list) else []
            )
            if bool(require_without) and bool(args.exclude_short_responses_in_training_examples):
                _, _, without_assistant = pair_core._conversation_signature_from_conv(ex_without_conv)
                if core._count_tokens(without_assistant) < int(all_core.MIN_RESPONSE_TOKENS):
                    short_without_aug_examples_excluded_pre_selection += 1
                    continue
            sig_key = pair_core._signature_key_from_conv(ex["conversation"])
            if sig_key in seen_sig_keys:
                continue
            selected_with.append(ex)
            seen_sig_keys.add(sig_key)
            contrastive_aug_added += 1
            extra_without_conversations.append(ex_without_conv)
        if short_without_aug_examples_excluded_pre_selection > 0:
            print(
                "Activation-diff short-response filter (without-image) applied to contrastive augmentation "
                f"examples: dropped={short_without_aug_examples_excluded_pre_selection}"
            )

    labels = [int(x["label"]) for x in selected_with]
    sample_names = [str(x["sample_name"]) for x in selected_with]
    sample_meta = [x["meta"] for x in selected_with]
    with_conversations = [x["conversation"] for x in selected_with]
    without_conversations = [_build_without_image_conversation(row) for row in balanced_rows] + list(extra_without_conversations)

    if pair_core._uses_preextracted_activation_store(str(args.vlm)):
        preextract_path = pair_core._preextracted_all_examples_path_for_vlm(str(args.vlm))
        if preextract_path.exists():
            preextract_payload = torch.load(preextract_path)
            with_payload = preextract_payload.get("with_image", {}) or {}
            without_payload = preextract_payload.get("without_image", {}) or {}
            available_with_sig_keys = {
                str(x) for x in with_payload.get("signature_keys", [])
            }
            available_without_sig_keys = {
                str(x) for x in without_payload.get("signature_keys", [])
            }

            filtered_selected_with: List[Dict] = []
            filtered_without_conversations: List[List[Dict]] = []
            for item, wo_conv in zip(selected_with, without_conversations):
                with_sig = pair_core._signature_key_from_conv(item["conversation"])
                if with_sig not in available_with_sig_keys:
                    continue
                if bool(require_without):
                    wo_sig = pair_core._signature_key_from_conv(wo_conv)
                    if wo_sig not in available_without_sig_keys:
                        continue
                filtered_selected_with.append(item)
                filtered_without_conversations.append(wo_conv)

            dropped_not_in_preextracted_cache = int(len(selected_with) - len(filtered_selected_with))
            if dropped_not_in_preextracted_cache > 0:
                print(
                    "Filtered selected all-examples rows to those present in pre-extracted cache: "
                    f"dropped={dropped_not_in_preextracted_cache}, kept={len(filtered_selected_with)}"
                )

            selected_with = filtered_selected_with
            without_conversations = filtered_without_conversations
            labels = [int(x["label"]) for x in selected_with]
            sample_names = [str(x["sample_name"]) for x in selected_with]
            sample_meta = [x["meta"] for x in selected_with]
            with_conversations = [x["conversation"] for x in selected_with]

    selection_stats = dict(selection_stats)
    selection_stats["activation_diff_without_short_filter_enabled"] = bool(
        require_without and bool(args.exclude_short_responses_in_training_examples)
    )
    selection_stats["short_without_response_rows_excluded_pre_sampling"] = int(
        short_without_rows_excluded_pre_selection
    )
    selection_stats["short_without_response_aug_examples_excluded_pre_sampling"] = int(
        short_without_aug_examples_excluded_pre_selection
    )
    selection_stats["short_without_response_examples_excluded_post_sampling"] = int(
        0
    )
    selection_stats["short_without_response_examples_excluded_total"] = int(
        short_without_rows_excluded_pre_selection + short_without_aug_examples_excluded_pre_selection
    )

    if bool(require_without):
        for conv in without_conversations:
            if not isinstance(conv, list) or (not conv):
                raise RuntimeError(
                    "Activation-diff all-examples mode requires without-image conversations for all "
                    "selected samples. Rebuild contrastive pairs artifact with without-image fields."
                )
    layer_features_with, layer_features_without, layer_order, cache_path = _load_or_extract_features(
        args=args,
        model_path=model_path,
        with_conversations=with_conversations,
        without_conversations=without_conversations,
        labels=labels,
        sample_names=sample_names,
        sample_meta=sample_meta,
        require_without_image=require_without,
        cache_path=cache_path,
    )
    if additional_feature_experiment_mode:
        original_feature_count = len(layer_order)
        filtered_layer_order = pair_core._keep_every_other_layer_feature_names(layer_order)
        layer_features_with, layer_order = pair_core._filter_layer_feature_map(
            layer_features=layer_features_with,
            layer_order=layer_order,
            requested_keys=filtered_layer_order,
        )
        layer_features_without, _ = pair_core._filter_layer_feature_map(
            layer_features=layer_features_without,
            layer_order=layer_order,
            requested_keys=filtered_layer_order,
        )
        print(
            f"Additional-feature runtime policy: every-other-layer filter kept "
            f"{len(layer_order)}/{original_feature_count} features."
        )
        if not layer_order:
            raise RuntimeError("Every-other-layer filtering removed all features.")

    X_by_feature, y, feature_order = _materialize_feature_matrices(
        feature_variant=feature_variant,
        layer_features_with=layer_features_with,
        layer_features_without=layer_features_without,
        layer_order=layer_order,
        labels=labels,
    )
    if not feature_order:
        raise RuntimeError(
            "No features remain after applying probe-family/strategy filters. "
            "Check --include_*_probes and --llm_feature_strategies."
        )
    benchmark_labels = np.asarray([str(m.get("dataset", "unknown")) for m in sample_meta], dtype=object)

    split_payloads, split_seeds = _split_payloads_all_examples(
        args=args,
        y=y,
        sample_meta=sample_meta,
    )
    reg_values = core.parse_regularization_values(str(args.regularization_values))

    per_feature_results: Dict[str, Dict] = {}
    for feature_name in tqdm(feature_order, desc=f"Training {probe_type} probes", unit="feature", dynamic_ncols=True):
        X = np.asarray(X_by_feature[feature_name], dtype=np.float32)
        seed_runs: List[Dict] = []
        for sp in split_payloads:
            run = core.sweep_probe_with_validation(
                probe_type=probe_type,
                X=X,
                y=y,
                train_mask=sp["train_mask"],
                val_mask=sp["val_mask"],
                test_mask=sp["test_mask"],
                split_seed=int(sp["split_seed"]),
                reg_values=reg_values,
                args=args,
                benchmark_labels=benchmark_labels,
            )
            run["split_sizes"] = sp["split_sizes"]
            run["split_stats"] = sp["split_stats"]
            seed_runs.append(run)
        per_feature_results[feature_name] = _summarize_feature_runs(seed_runs)

    llm_strategy_summary = _summarize_llm_layers(per_feature_results)

    run_dir = save_root / f"{script_name}_{args.vlm}_{args.benchmark_mode}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_feature_rows = [
        {
            "feature": key,
            **info,
        }
        for key, info in sorted(
            per_feature_results.items(),
            key=lambda kv: pair_core._layer_sort_key(kv[0]) if "layer_" in kv[0] else (99, 0, kv[0]),
        )
    ]

    vlm_tag = str(args.vlm)
    all_feature_path = run_dir / f"{vlm_tag}_all_feature_probe_accuracies_{run_id}.json"
    llm_path = run_dir / f"{vlm_tag}_llm_layer_best_strategy_results_{run_id}.json"
    sample_meta_path = run_dir / f"{vlm_tag}_sample_metadata_{run_id}.json"
    config_path = run_dir / f"{vlm_tag}_run_config_{run_id}.json"

    with open(all_feature_path, "w", encoding="utf-8") as f:
        json.dump(all_feature_rows, f, indent=2, ensure_ascii=False)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_strategy_summary, f, indent=2, ensure_ascii=False)
    with open(sample_meta_path, "w", encoding="utf-8") as f:
        json.dump(sample_meta, f, indent=2, ensure_ascii=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "script_name": script_name,
                "run_id": run_id,
                "probe_type": probe_type,
                "feature_variant": feature_variant,
                "vlm": str(args.vlm),
                "model_path": model_path,
                "device_map": str(getattr(args, "device_map", "")),
                "max_memory_per_gpu_gib": float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
                "max_memory_cpu_gib": float(getattr(args, "max_memory_cpu_gib", 0.0)),
                "responses_path": str(responses_path),
                "repo_root": str(mirage_root),
                "benchmark_mode": str(args.benchmark_mode),
                "neutral_as_non_mirage": bool(args.neutral_as_non_mirage),
                "include_contrastive_pairs_in_training_examples": bool(
                    args.include_contrastive_pairs_in_training_examples
                ),
                "num_contrastive_pair_examples_added": int(contrastive_aug_added),
                "num_examples_dropped_not_in_preextracted_cache": int(dropped_not_in_preextracted_cache),
                "exclude_short_responses_in_training_examples": bool(
                    args.exclude_short_responses_in_training_examples
                ),
                "num_samples": int(len(y)),
                "num_class0_total": int((y == 0).sum()),
                "num_class1_total": int((y == 1).sum()),
                "num_features": int(len(feature_order)),
                "val_fraction": float(args.val_fraction),
                "test_fraction": float(args.test_fraction),
                "regularization_sweep_c_values": reg_values,
                "num_split_seeds": int(args.num_split_seeds),
                "split_seeds": split_seeds,
                "multi_init_enabled": (not bool(args.disable_multi_init)),
                "num_probe_inits": int(max(1, args.num_probe_inits)),
                "probe_epochs": int(args.probe_epochs),
                "probe_lr": float(args.probe_lr),
                "mlp_hidden_dim": int(args.mlp_hidden_dim),
                "normalize_features": bool(args.normalize_features),
                "pca_components": int(args.pca_components),
                "early_stopping_patience": int(args.early_stopping_patience),
                "early_stopping_min_delta": float(args.early_stopping_min_delta),
                "include_attention_probes": bool(requested_include_attention),
                "include_mlp_probes": bool(requested_include_mlp),
                "include_residual_probes": bool(requested_include_residual),
                "include_additional_attention_mlp_probes": bool(
                    getattr(args, "include_additional_attention_mlp_probes", False)
                ),
                "use_additional_feature_preextract_cache": getattr(
                    args, "use_additional_feature_preextract_cache", None
                ),
                "requested_llm_feature_strategies": list(requested_llm_feature_strategies),
                "features_cache_path": str(cache_path),
                "feature_extraction_version": int(FEATURE_EXTRACTION_VERSION),
                "selection_stats": selection_stats,
                "preextract_response_filter_stats": preextract_response_filter_stats,
                "run_dir": str(run_dir),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved feature accuracies: {all_feature_path}")
    print(f"Saved LLM strategy summary: {llm_path}")
    print(f"Saved sample metadata: {sample_meta_path}")
    print(f"Saved run config: {config_path}")


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(
        description=(
            "Train two-layer MLP probes on all examples with benchmark-aware split seeds, "
            "validation-selected C sweep, and 3-init selection per C by default."
        ),
        default_save_dir="./tmp_artifacts/mlp_all_examples_probe_results",
        default_cache_path="./tmp_artifacts/all_examples_layer_features_mlp.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.responses_path = str(
        pair_core._canonicalize_input_arg_for_vlm(
            path_value=args.responses_path,
            vlm_key=str(args.vlm),
            artifact_name="responses.json",
        )
    )
    args.contrastive_pairs_path = str(
        pair_core._canonicalize_input_arg_for_vlm(
            path_value=args.contrastive_pairs_path,
            vlm_key=str(args.vlm),
            artifact_name="contrastive_conversation_pairs.json",
        )
    )
    args.contrastive_neutral_pairs_path = str(
        pair_core._canonicalize_input_arg_for_vlm(
            path_value=args.contrastive_neutral_pairs_path,
            vlm_key=str(args.vlm),
            artifact_name="contrastive_conversation_pairs_neutral_as_non_mirage.json",
        )
    )
    run_all_examples_experiment(
        args=args,
        script_name="train_mlp_all_examples",
        probe_type="mlp",
        feature_variant="raw",
    )


if __name__ == "__main__":
    main()
