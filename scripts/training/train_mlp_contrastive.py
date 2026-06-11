#!/usr/bin/env python3
"""Train probe models on contrastive examples.

This script is the contrastive MLP entrypoint and also exposes reusable helpers
for concat-residual and activation-difference contrastive probe scripts.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.training.train_log_reg_contrastive as pair_core


MODEL_REGISTRY: Dict[str, str] = {
    "ovis": "AIDC-AI/Ovis2.5-2B",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "glm_4_6v_flash": "zai-org/GLM-4.6V-Flash",
}
DEFAULT_VLM = "ovis"
BENCHMARK_CHOICES = ["all", "vqa_rad", "medxpertqa_mm", "mmmu_pro"]
FEATURE_EXTRACTION_VERSION = 1
MIN_RESPONSE_TOKENS = 10
DEFAULT_VALIDATION_PAIRS_SINGLE_BENCHMARK = 5


def resolve_model_path(vlm: str, model_path_override: str) -> str:
    override = str(model_path_override or "").strip()
    if override:
        return override
    if vlm not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported --vlm '{vlm}'. Choices: {sorted(MODEL_REGISTRY.keys())}")
    resolver = getattr(pair_core, "_default_model_path_for_vlm", None)
    if callable(resolver):
        try:
            return str(resolver(vlm))
        except Exception:
            pass
    return MODEL_REGISTRY[vlm]


def make_run_id(seed: int, run_id: str) -> str:
    if run_id:
        return str(run_id)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_seed{int(seed)}"


def parse_regularization_values(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if str(x).strip()]
    if not vals:
        raise ValueError("--regularization_values must contain at least one value.")
    return vals


def _scope_default_path(path: Path, vlm: str, marker: str) -> Path:
    p = Path(path)
    if marker not in str(p):
        return p
    if vlm == DEFAULT_VLM:
        return p
    if p.suffix:
        return p.with_name(f"{p.stem}_{vlm}{p.suffix}")
    return Path(f"{p}_{vlm}")


def build_arg_parser(description: str, default_save_dir: str, default_cache_path: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--pairs_path",
        type=str,
        default="./tmp_artifacts/contrastive_conversation_pairs.json",
        help=(
            "Contrastive pairs artifact. If left at the default value, OVIS/QWEN use "
            "data/final_data/*_contrastive.json and GLM uses tmp_artifacts/contrastive_conversation_pairs.json."
        ),
    )
    parser.add_argument(
        "--neutral_pairs_path",
        type=str,
        default="./tmp_artifacts/contrastive_conversation_pairs_neutral_as_non_mirage.json",
        help=(
            "Alternative pairs artifact where neutral is treated as non-mirage. "
            "Defaults to tmp_artifacts/contrastive_conversation_pairs_neutral_as_non_mirage.json."
        ),
    )
    parser.add_argument(
        "--neutral_as_non_mirage_pairs",
        action="store_true",
        help="Use neutral-inclusive contrastive pairs artifact for training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_split_seeds", type=int, default=5)
    parser.add_argument(
        "--exclude_short_responses_in_training_pairs",
        dest="exclude_short_responses_in_training_pairs",
        action="store_true",
        help=(
            "Exclude pair candidates where either class conversation's assistant response has "
            f"fewer than {MIN_RESPONSE_TOKENS} whitespace-separated tokens."
        ),
    )
    parser.add_argument(
        "--no_exclude_short_responses_in_training_pairs",
        dest="exclude_short_responses_in_training_pairs",
        action="store_false",
        help="Keep short-response pairs in training-candidate selection.",
    )
    parser.set_defaults(exclude_short_responses_in_training_pairs=True)
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument(
        "--benchmark_mode",
        type=str,
        default="all",
        choices=BENCHMARK_CHOICES,
        help="Restrict training/eval to one benchmark or use all represented benchmarks.",
    )
    parser.add_argument(
        "--single_benchmark_validation_pairs",
        type=int,
        default=DEFAULT_VALIDATION_PAIRS_SINGLE_BENCHMARK,
        help="Validation pair count when --benchmark_mode is not 'all' (defaults to 5; GLM defaults to 2).",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="ovis",
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model family key used for artifact naming and default model resolution.",
    )
    parser.add_argument(
        "--model_path_override",
        type=str,
        default="",
        help="Optional explicit model path/id override.",
    )
    parser.add_argument(
        "--regularization_values",
        type=str,
        default="0,0.01,0.1,1.0",
        help="Comma-separated C values (weight_decay = 0 if C=0 else 1/C).",
    )
    parser.add_argument("--probe_epochs", type=int, default=800)
    parser.add_argument("--probe_lr", type=float, default=0.03)
    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument(
        "--num_probe_inits",
        type=int,
        default=3,
        help="Random initialization count per hyperparameter candidate.",
    )
    parser.add_argument(
        "--disable_multi_init",
        action="store_true",
        help="If set, train one initialization per hyperparameter candidate.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=200,
        help="Validation-loss early stopping patience.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="Minimum validation-loss improvement to reset patience.",
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
        help="If >0, apply train-fitted PCA to this many components.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention backend when model supports Ovis-style config fields.",
    )
    parser.add_argument("--force_reextract", action="store_true")
    parser.add_argument("--run_id", type=str, default="")
    parser.add_argument("--features_cache_path", type=str, default=default_cache_path)
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
        "--responses_path",
        type=str,
        default="./tmp_artifacts/responses.json",
        help=(
            "responses.json used for unseen holdout evaluation. If left at the default value, "
            "OVIS/QWEN use data/final_data/*_all_responses.json and GLM uses tmp_artifacts/responses.json."
        ),
    )
    parser.add_argument(
        "--num_holdout_mirage_true",
        type=int,
        default=50,
        help="Per-seed unseen holdout size for mirage_like=true.",
    )
    parser.add_argument(
        "--num_holdout_mirage_false",
        type=int,
        default=50,
        help="Per-seed unseen holdout size for mirage_like=false.",
    )
    parser.add_argument(
        "--num_eval_seeds",
        type=int,
        default=5,
        help="Number of unseen holdout evaluation seeds.",
    )
    parser.add_argument(
        "--exclude_short_responses_in_holdout",
        action="store_true",
        help=(
            "Exclude holdout candidates where with-image response has fewer than "
            f"{MIN_RESPONSE_TOKENS} whitespace-separated tokens."
        ),
    )
    parser.add_argument("--save_dir", type=str, default=default_save_dir)
    pair_core.add_model_loading_args(parser)
    pair_core.add_preextract_cache_selection_args(parser)
    return parser


def _filter_pairs_by_benchmark(pairs: List[Dict], benchmark_mode: str, vlm_key: str) -> List[Dict]:
    supported = set(pair_core._supported_contrastive_benchmarks_for_vlm(vlm_key))
    if benchmark_mode == "all":
        return [p for p in pairs if pair_core._infer_benchmark_from_pair(p) in supported]
    return [p for p in pairs if pair_core._infer_benchmark_from_pair(p) == benchmark_mode]


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _build_contrastive_samples(
    pairs: List[Dict],
    include_short_response_filter: bool,
    require_without: bool = False,
    min_response_tokens: int = MIN_RESPONSE_TOKENS,
) -> Tuple[List[List[Dict]], List[List[Dict]], List[int], List[int], List[str], List[str], int]:
    conversations: List[List[Dict]] = []
    without_conversations: List[List[Dict]] = []
    labels: List[int] = []
    pair_ids: List[int] = []
    sample_names: List[str] = []
    pair_benchmarks: List[str] = []
    skipped_short_pairs = 0
    kept_pair_id = 0

    for pair in pairs:
        non_conv = pair["non_mirage_conversation"]
        mirage_conv = pair["mirage_conversation"]
        non_without_conv = pair.get("non_mirage_without_image_conversation", [])
        mirage_without_conv = pair.get("mirage_without_image_conversation", [])
        if include_short_response_filter:
            too_short_with_image_response = False
            if pair_core._conversation_has_image_input(non_conv):
                _, _, non_assistant = pair_core._conversation_signature_from_conv(non_conv)
                if _count_tokens(non_assistant) < int(min_response_tokens):
                    too_short_with_image_response = True
            if pair_core._conversation_has_image_input(mirage_conv):
                _, _, mirage_assistant = pair_core._conversation_signature_from_conv(mirage_conv)
                if _count_tokens(mirage_assistant) < int(min_response_tokens):
                    too_short_with_image_response = True
            if bool(require_without):
                _, _, non_without_assistant = pair_core._conversation_signature_from_conv(
                    non_without_conv if isinstance(non_without_conv, list) else []
                )
                if _count_tokens(non_without_assistant) < int(min_response_tokens):
                    too_short_with_image_response = True
                _, _, mirage_without_assistant = pair_core._conversation_signature_from_conv(
                    mirage_without_conv if isinstance(mirage_without_conv, list) else []
                )
                if _count_tokens(mirage_without_assistant) < int(min_response_tokens):
                    too_short_with_image_response = True
            if too_short_with_image_response:
                skipped_short_pairs += 1
                continue

        pair_benchmarks.append(pair_core._infer_benchmark_from_pair(pair))

        conversations.append(non_conv)
        without_conversations.append(non_without_conv if isinstance(non_without_conv, list) else [])
        labels.append(0)
        pair_ids.append(kept_pair_id)
        sample_names.append(f"pair_{kept_pair_id}_non_mirage")

        conversations.append(mirage_conv)
        without_conversations.append(mirage_without_conv if isinstance(mirage_without_conv, list) else [])
        labels.append(1)
        pair_ids.append(kept_pair_id)
        sample_names.append(f"pair_{kept_pair_id}_mirage")
        kept_pair_id += 1

    return conversations, without_conversations, labels, pair_ids, sample_names, pair_benchmarks, int(skipped_short_pairs)


def _extract_features(
    conversations: List[List[Dict]],
    model_path: str,
    attn_implementation: str,
    model_key: str,
    include_attention_probes: bool = False,
    include_mlp_probes: bool = False,
    include_residual_probes: bool = True,
    device_map_raw: str = "",
    max_memory_per_gpu_gib: float = 0.0,
    max_memory_cpu_gib: float = 0.0,
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    model = pair_core.load_vlm_for_extraction(
        model_path=model_path,
        attn_implementation=attn_implementation,
        device_map_raw=device_map_raw,
        max_memory_per_gpu_gib=max_memory_per_gpu_gib,
        max_memory_cpu_gib=max_memory_cpu_gib,
    )
    pair_core._force_attention_backend(model, attn_implementation)

    layer_features: Dict[str, List[np.ndarray]] = defaultdict(list)
    seen = set()
    layer_order: List[str] = []

    for conv in tqdm(conversations, desc="Extracting activations", unit="sample", dynamic_ncols=True):
        messages = pair_core._to_ovis_messages(conv)
        sample_feats = pair_core._extract_sample_features_only(
            model=model,
            messages=messages,
            include_additional_attention_mlp_probes=False,
            include_attention_probes=bool(include_attention_probes),
            include_mlp_probes=bool(include_mlp_probes),
            include_residual_probes=bool(include_residual_probes),
            model_key=model_key,
        )
        for key, value in sample_feats.items():
            if key not in seen:
                seen.add(key)
                layer_order.append(key)
            layer_features[key].append(value.to(torch.float32).cpu().numpy())

    layer_order = sorted(layer_order, key=pair_core._layer_sort_key)
    return layer_features, layer_order


def _align_with_without_feature_maps(
    layer_features_with: Dict[str, List[np.ndarray]],
    layer_features_without: Dict[str, List[np.ndarray]],
    with_layer_order: List[str],
    without_layer_order: Optional[List[str]] = None,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], List[str]]:
    if not layer_features_without:
        return layer_features_with, layer_features_without, list(with_layer_order)

    with_keys = set(str(k) for k in layer_features_with.keys())
    without_keys = set(str(k) for k in layer_features_without.keys())
    common_keys = sorted(with_keys.intersection(without_keys), key=pair_core._layer_sort_key)
    if not common_keys:
        raise RuntimeError(
            "With-image and without-image extraction produced disjoint feature keys; "
            "cannot compute activation differences."
        )

    dropped_with = sorted(with_keys.difference(without_keys), key=pair_core._layer_sort_key)
    dropped_without = sorted(without_keys.difference(with_keys), key=pair_core._layer_sort_key)
    if dropped_with or dropped_without:
        with_count = len(with_layer_order) if with_layer_order else len(with_keys)
        without_count = len(without_layer_order) if without_layer_order else len(without_keys)
        print(
            "Warning: contrastive with/without feature sets differ; aligning to common keys. "
            f"with_keys={with_count} without_keys={without_count} common_keys={len(common_keys)} "
            f"dropped_with_only={len(dropped_with)} dropped_without_only={len(dropped_without)}"
        )

    aligned_with = {k: layer_features_with[k] for k in common_keys}
    aligned_without = {k: layer_features_without[k] for k in common_keys}
    return aligned_with, aligned_without, common_keys


def _load_or_extract_contrastive_features(
    args: argparse.Namespace,
    model_path: str,
    conversations: List[List[Dict]],
    without_conversations: List[List[Dict]],
    require_without_image: bool,
    labels: List[int],
    pair_ids: List[int],
    sample_names: List[str],
    cache_path: Path,
    model_key: str,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], List[str], Path]:
    if not cache_path.is_absolute():
        cache_path = Path.cwd() / cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    requested_include_attention = bool(
        getattr(args, "include_attention_probes", False) or getattr(args, "include_additional_attention_mlp_probes", False)
    )
    requested_include_mlp = bool(
        getattr(args, "include_mlp_probes", False) or getattr(args, "include_additional_attention_mlp_probes", False)
    )
    requested_include_residual = bool(getattr(args, "include_residual_probes", True))
    requested_llm_feature_strategies = pair_core._parse_requested_llm_feature_strategies(
        getattr(args, "llm_feature_strategies", ",".join(pair_core.LLM_STRATEGIES))
    )

    if pair_core._uses_preextracted_activation_store(str(args.vlm)):
        family = pair_core._preextract_family_for_vlm(str(args.vlm)).upper()
        if bool(getattr(args, "force_reextract", False)):
            print(f"Ignoring --force_reextract for {family}; loading pre-extracted activations cache.")
        if bool(require_without_image):
            try:
                with_map, without_map, layer_order, resolved_cache = pair_core._load_preextracted_contrastive_with_without_subset(
                    vlm_key=str(args.vlm),
                    with_conversations=conversations,
                    without_conversations=without_conversations,
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
            except RuntimeError as exc:
                # Backward compatibility: older contrastive pre-extract caches may not include
                # without-image tensors/signatures. Fall back to all-examples lookup in that case.
                if "does not include without-image tensors/signatures" not in str(exc):
                    raise
                with_map, without_map, layer_order, resolved_cache = pair_core._load_preextracted_all_examples_subset(
                    vlm_key=str(args.vlm),
                    with_conversations=conversations,
                    without_conversations=without_conversations,
                    require_without=True,
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
        layer_features, layer_order, resolved_cache = pair_core._load_preextracted_contrastive_subset(
            vlm_key=str(args.vlm),
            conversations=conversations,
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            llm_feature_strategies=requested_llm_feature_strategies,
            use_additional_feature_preextract_cache=getattr(
                args, "use_additional_feature_preextract_cache", None
            ),
            requested_model_path=str(model_path),
        )
        return layer_features, {}, layer_order, resolved_cache

    use_cache = cache_path.exists() and (not bool(args.force_reextract))
    if use_cache:
        payload = torch.load(cache_path)
        cached_names = [str(x) for x in payload.get("sample_names", [])]
        expected_glm_image_normalization = bool(pair_core._is_glm_vlm(str(args.vlm)))
        cached_glm_image_normalization = bool(payload.get("glm_image_normalization_applied", False))
        if cached_names != [str(x) for x in sample_names]:
            use_cache = False
        elif expected_glm_image_normalization and (not cached_glm_image_normalization):
            use_cache = False
        elif bool(require_without_image) and ("layer_features_without" not in payload):
            use_cache = False

    if use_cache:
        payload = torch.load(cache_path)
        layer_features = payload.get("layer_features_with", payload.get("layer_features", {}))
        layer_features_without = payload.get("layer_features_without", {})
        layer_order = payload["layer_order"]
        if bool(require_without_image):
            layer_features, layer_features_without, layer_order = _align_with_without_feature_maps(
                layer_features_with=layer_features,
                layer_features_without=layer_features_without,
                with_layer_order=[str(x) for x in layer_order],
                without_layer_order=None,
            )
        requested_keys = pair_core._resolve_requested_feature_keys(
            layer_order=layer_order,
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            llm_feature_strategies=requested_llm_feature_strategies,
        )
        layer_features, layer_order = pair_core._filter_layer_feature_map(
            layer_features=layer_features,
            layer_order=layer_order,
            requested_keys=requested_keys,
        )
        layer_features_without, _ = pair_core._filter_layer_feature_map(
            layer_features=layer_features_without,
            layer_order=layer_order,
            requested_keys=layer_order,
        )
        return layer_features, layer_features_without, layer_order, cache_path

    layer_features_with, layer_order = _extract_features(
        conversations=conversations,
        model_path=model_path,
        attn_implementation=str(args.attn_implementation),
        model_key=model_key,
        include_attention_probes=bool(requested_include_attention),
        include_mlp_probes=bool(requested_include_mlp),
        include_residual_probes=bool(requested_include_residual),
        device_map_raw=str(getattr(args, "device_map", "")),
        max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
        max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
    )
    layer_features_without: Dict[str, List[np.ndarray]] = {}
    if bool(require_without_image):
        for idx, conv in enumerate(without_conversations):
            if not isinstance(conv, list) or (not conv):
                raise RuntimeError(
                    "Activation-diff contrastive mode requires without-image conversations in "
                    "contrastive pair artifacts. Rebuild via build_contrastive_pairs.py."
                )
        layer_features_without, without_layer_order = _extract_features(
            conversations=without_conversations,
            model_path=model_path,
            attn_implementation=str(args.attn_implementation),
            model_key=model_key,
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            device_map_raw=str(getattr(args, "device_map", "")),
            max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
            max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
        )
        layer_features_with, layer_features_without, layer_order = _align_with_without_feature_maps(
            layer_features_with=layer_features_with,
            layer_features_without=layer_features_without,
            with_layer_order=layer_order,
            without_layer_order=without_layer_order,
        )
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
            "layer_features": layer_features_with,
            "labels": labels,
            "pair_ids": pair_ids,
            "sample_names": sample_names,
            "layer_order": layer_order,
            "model_path": model_path,
            "vlm": str(args.vlm),
            "glm_image_normalization_applied": bool(pair_core._is_glm_vlm(str(args.vlm))),
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


def _split_payloads_contrastive(
    args: argparse.Namespace,
    y: np.ndarray,
    pair_ids_arr: np.ndarray,
    pair_benchmarks: List[str],
) -> Tuple[List[Dict], List[int]]:
    if int(args.num_split_seeds) <= 0:
        raise ValueError("--num_split_seeds must be >= 1.")

    split_seed_rng = np.random.default_rng(int(args.seed))
    split_seeds = split_seed_rng.choice(
        1_000_000_000,
        size=int(args.num_split_seeds),
        replace=False,
    ).astype(np.int64).tolist()

    payloads: List[Dict] = []
    single_validation_pairs = int(args.single_benchmark_validation_pairs)
    if (
        pair_core._is_glm_vlm(str(args.vlm))
        and single_validation_pairs == int(DEFAULT_VALIDATION_PAIRS_SINGLE_BENCHMARK)
    ):
        single_validation_pairs = int(pair_core._single_benchmark_validation_pairs_for_vlm(str(args.vlm)))
    qwen_val_fraction = pair_core._contrastive_validation_fraction_for_vlm(str(args.vlm))

    for split_seed in split_seeds:
        if qwen_val_fraction is not None:
            train_mask, val_mask, heldout_pairs, heldout_by_bench = pair_core._split_pair_benchmark_fraction_validation(
                pair_ids=pair_ids_arr,
                pair_benchmarks=pair_benchmarks,
                val_fraction=float(qwen_val_fraction),
                seed=int(split_seed),
            )
        elif str(args.benchmark_mode) == "all":
            if pair_core._is_glm_vlm(str(args.vlm)):
                train_mask, val_mask, heldout_pairs = pair_core._split_pair_fixed_validation_count(
                    pair_ids=pair_ids_arr,
                    num_validation_pairs=int(pair_core._all_mode_validation_pairs_for_vlm(str(args.vlm))),
                    seed=int(split_seed),
                )
                heldout_by_bench = pair_core._group_pair_ids_by_benchmark(
                    pair_ids=[int(x) for x in heldout_pairs],
                    pair_benchmarks=pair_benchmarks,
                )
            else:
                train_mask, val_mask, heldout_pairs, heldout_by_bench = pair_core._split_pair_benchmark_stratified_validation(
                    pair_ids=pair_ids_arr,
                    pair_benchmarks=pair_benchmarks,
                    seed=int(split_seed),
                )
        else:
            train_mask, val_mask, heldout_pairs = pair_core._split_pair_fixed_validation_count(
                pair_ids=pair_ids_arr,
                num_validation_pairs=int(single_validation_pairs),
                seed=int(split_seed),
            )
            heldout_by_bench = {str(args.benchmark_mode): [int(x) for x in heldout_pairs]}

        test_mask = np.zeros(len(val_mask), dtype=bool)
        train_y = y[train_mask]
        val_y = y[val_mask]
        split_sizes = {
            "train_total": int(train_mask.sum()),
            "train_class0": int((train_y == 0).sum()),
            "train_class1": int((train_y == 1).sum()),
            "val_total": int(val_mask.sum()),
            "val_class0": int((val_y == 0).sum()),
            "val_class1": int((val_y == 1).sum()),
            "test_total": 0,
            "test_class0": 0,
            "test_class1": 0,
        }
        payloads.append(
            {
                "split_seed": int(split_seed),
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": test_mask,
                "split_sizes": split_sizes,
                "heldout_pairs": [int(x) for x in heldout_pairs],
                "heldout_pairs_by_benchmark": heldout_by_bench,
            }
        )

    return payloads, [int(s) for s in split_seeds]


def _fit_linear_with_early_stopping(
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
    return {
        "train_pred": _predict(X_train_s),
        "val_pred": _predict(X_val_s),
        "eval_pred": _predict(X_eval_s),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }


def _fit_mlp_with_early_stopping(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    c_value: float,
    hidden_dim: int,
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

    model = torch.nn.Sequential(
        torch.nn.Linear(X_train_s.shape[1], int(hidden_dim)),
        torch.nn.GELU(),
        torch.nn.Linear(int(hidden_dim), 1),
    ).to(device)
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
    return {
        "train_pred": _predict(X_train_s),
        "val_pred": _predict(X_val_s),
        "eval_pred": _predict(X_eval_s),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }


def _run_single_probe(
    probe_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    split_seed: int,
    init_seed: int,
    c_value: float,
    args: argparse.Namespace,
    test_benchmark_labels: Optional[List[str]] = None,
) -> Dict:
    if probe_type == "mlp":
        out = _fit_mlp_with_early_stopping(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_eval=X_test,
            seed=int(init_seed),
            epochs=int(args.probe_epochs),
            lr=float(args.probe_lr),
            c_value=float(c_value),
            hidden_dim=int(args.mlp_hidden_dim),
            normalize_features=bool(args.normalize_features),
            pca_components=int(args.pca_components),
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
        )
    elif probe_type in {"linear", "logreg"}:
        out = _fit_linear_with_early_stopping(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_eval=X_test,
            seed=int(init_seed),
            epochs=int(args.probe_epochs),
            lr=float(args.probe_lr),
            c_value=float(c_value),
            normalize_features=bool(args.normalize_features),
            pca_components=int(args.pca_components),
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
        )
    else:
        raise ValueError(f"Unsupported probe_type '{probe_type}'.")

    train_pred = out["train_pred"]
    val_pred = out["val_pred"]
    test_pred = out["eval_pred"]

    train_acc = float((train_pred == y_train).mean())
    val_acc = float((val_pred == y_val).mean())
    if y_test.size > 0:
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
    else:
        test_acc = None
        class0_test_acc = None
        class1_test_acc = None
        benchmark_test_acc = {}
        benchmark_class0_test_acc = {}
        benchmark_class1_test_acc = {}

    return {
        "split_seed": int(split_seed),
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


def sweep_probe_with_validation(
    probe_type: str,
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

    n_inits = 1 if bool(args.disable_multi_init) else int(max(1, args.num_probe_inits))

    best = None
    sweep_rows: List[Dict] = []
    for c_val in reg_values:
        init_runs: List[Dict] = []
        for init_idx in range(n_inits):
            init_seed = pair_core._probe_init_seed(
                split_seed=int(split_seed),
                c_value=float(c_val),
                init_idx=int(init_idx),
            )
            init_result = _run_single_probe(
                probe_type=probe_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                split_seed=int(split_seed),
                init_seed=int(init_seed),
                c_value=float(c_val),
                args=args,
                test_benchmark_labels=test_benchmark_labels,
            )
            init_result["init_index"] = int(init_idx)
            init_runs.append(init_result)

        best_init = sorted(
            init_runs,
            key=lambda r: (
                -float(r["val_accuracy"]),
                -float(r["train_accuracy"]),
                int(r["init_index"]),
            ),
        )[0]

        row = {
            "c_value": float(c_val),
            "selected_init_index": int(best_init["init_index"]),
            "selected_init_seed": int(best_init["init_seed"]),
            "train_accuracy": float(best_init["train_accuracy"]),
            "val_accuracy": float(best_init["val_accuracy"]),
            "test_accuracy": (
                float(best_init["test_accuracy"])
                if best_init.get("test_accuracy") is not None
                else None
            ),
            "class0_test_accuracy": best_init["class0_test_accuracy"],
            "class1_test_accuracy": best_init["class1_test_accuracy"],
            "benchmark_test_accuracy": best_init["benchmark_test_accuracy"],
            "benchmark_class0_test_accuracy": best_init["benchmark_class0_test_accuracy"],
            "benchmark_class1_test_accuracy": best_init["benchmark_class1_test_accuracy"],
            "early_stopped_best_epoch": int(best_init["early_stopped_best_epoch"]),
            "early_stopped_best_val_loss": float(best_init["early_stopped_best_val_loss"]),
            "init_runs": init_runs,
        }
        sweep_rows.append(row)

        if (
            (best is None)
            or (float(row["val_accuracy"]) > float(best["val_accuracy_at_best_c"]))
            or (
                float(row["val_accuracy"]) == float(best["val_accuracy_at_best_c"])
                and float(c_val) < float(best["best_c"])
            )
        ):
            best = {
                "best_c": float(c_val),
                "best_train_accuracy": float(row["train_accuracy"]),
                "val_accuracy_at_best_c": float(row["val_accuracy"]),
                "test_accuracy_at_best_c": (float(row["test_accuracy"]) if row["test_accuracy"] is not None else None),
                "class0_test_accuracy_at_best_c": row["class0_test_accuracy"],
                "class1_test_accuracy_at_best_c": row["class1_test_accuracy"],
                "benchmark_test_accuracy_at_best_c": row["benchmark_test_accuracy"],
                "benchmark_class0_test_accuracy_at_best_c": row["benchmark_class0_test_accuracy"],
                "benchmark_class1_test_accuracy_at_best_c": row["benchmark_class1_test_accuracy"],
                "best_epoch_at_best_c": int(row["early_stopped_best_epoch"]),
                "best_val_loss_at_best_c": float(row["early_stopped_best_val_loss"]),
                "best_init_index_at_best_c": int(row["selected_init_index"]),
                "best_init_seed_at_best_c": int(row["selected_init_seed"]),
            }

    return {
        "split_seed": int(split_seed),
        "selection_metric": "val_accuracy",
        "probe_type": str(probe_type),
        **best,
        "sweep": sweep_rows,
    }


def _summarize_feature_runs(seed_runs: List[Dict]) -> Dict:
    train_scores = [float(r["best_train_accuracy"]) for r in seed_runs]
    val_scores = [float(r["val_accuracy_at_best_c"]) for r in seed_runs]
    test_scores = [
        float(r["test_accuracy_at_best_c"])
        for r in seed_runs
        if r.get("test_accuracy_at_best_c") is not None
    ]
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
        "mean_test_accuracy_at_best_c": (float(np.mean(test_scores)) if test_scores else None),
        "std_test_accuracy_at_best_c": (float(np.std(test_scores)) if test_scores else None),
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
            if info.get("mean_test_accuracy_at_best_c") is not None:
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


def _fit_fixed_c_with_multi_init(
    probe_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    split_seed: int,
    c_value: float,
    args: argparse.Namespace,
    eval_benchmark_labels: Optional[List[str]] = None,
) -> Dict:
    n_inits = 1 if bool(args.disable_multi_init) else int(max(1, args.num_probe_inits))
    best = None
    for init_idx in range(n_inits):
        init_seed = pair_core._probe_init_seed(
            split_seed=int(split_seed),
            c_value=float(c_value),
            init_idx=int(init_idx),
        )
        candidate = _run_single_probe(
            probe_type=probe_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_eval,
            y_test=y_eval,
            split_seed=int(split_seed),
            init_seed=int(init_seed),
            c_value=float(c_value),
            args=args,
            test_benchmark_labels=eval_benchmark_labels,
        )
        candidate["init_index"] = int(init_idx)
        if (
            best is None
            or (float(candidate["val_accuracy"]) > float(best["val_accuracy"]))
            or (
                float(candidate["val_accuracy"]) == float(best["val_accuracy"])
                and float(candidate["train_accuracy"]) > float(best["train_accuracy"])
            )
            or (
                float(candidate["val_accuracy"]) == float(best["val_accuracy"])
                and float(candidate["train_accuracy"]) == float(best["train_accuracy"])
                and int(init_idx) < int(best["init_index"])
            )
        ):
            best = candidate
    return best


def _build_holdout_pool_with_without(
    responses: List[Dict],
    seen_signatures: set,
    selected_benchmark: Optional[str],
    allowed_benchmarks: Sequence[str],
    include_short_response_filter: bool,
    require_without: bool,
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
) -> Tuple[List[Dict], List[Dict], int]:
    pool_true: List[Dict] = []
    pool_false: List[Dict] = []
    skipped_short_response_count = 0
    seen_pool_signatures = set()
    allowed_benchmarks_set = {str(x) for x in allowed_benchmarks}

    for row in responses:
        ds = str(row.get("dataset", ""))
        if selected_benchmark is not None:
            if ds != selected_benchmark:
                continue
        elif ds not in allowed_benchmarks_set:
            continue

        wo = row.get("without_image", {}) or {}
        mirage_like = wo.get("mirage_like")
        if mirage_like not in (True, False):
            continue

        sig = pair_core._conversation_signature_from_response_row(row)
        if sig in seen_signatures or sig in seen_pool_signatures:
            continue

        uid = str(row.get("unique_id", ""))
        qid = str(row.get("question_id", ""))
        imgs = image_lookup_uid.get((ds, uid))
        if imgs is None:
            imgs = image_lookup_qid.get((ds, qid))
        if not imgs:
            continue

        with_image_response = str((row.get("with_image", {}) or {}).get("response", ""))
        without_image_response = str((row.get("without_image", {}) or {}).get("response", ""))
        if not pair_core._norm_text(with_image_response):
            continue
        if bool(require_without) and (not pair_core._norm_text(without_image_response)):
            continue
        if include_short_response_filter:
            with_short = _count_tokens(with_image_response) < int(MIN_RESPONSE_TOKENS)
            without_short = bool(require_without) and (_count_tokens(without_image_response) < int(MIN_RESPONSE_TOKENS))
            if with_short or without_short:
                skipped_short_response_count += 1
                continue

        with_conv = pair_core.core._make_vllm_messages(
            prompt_text=row.get("prompt_text", ""),
            image_bytes_list=imgs,
            system_prompt=row.get("system_prompt", ""),
        )
        with_conv.append({"role": "assistant", "content": with_image_response})
        without_conv = pair_core.core._make_vllm_messages(
            prompt_text=row.get("prompt_text", ""),
            image_bytes_list=None,
            system_prompt=row.get("system_prompt", ""),
        )
        without_conv.append({"role": "assistant", "content": without_image_response})

        item = {
            "dataset": ds,
            "mirage_like": bool(mirage_like),
            "with_conversation": with_conv,
            "without_conversation": without_conv,
        }
        if bool(mirage_like):
            pool_true.append(item)
        else:
            pool_false.append(item)
        seen_pool_signatures.add(sig)

    return pool_true, pool_false, int(skipped_short_response_count)


def _extract_holdout_feature_matrices_for_seed(
    args: argparse.Namespace,
    model_path: str,
    vlm_key: str,
    feature_variant: str,
    feature_order: Sequence[str],
    selected_examples: Sequence[Dict],
    expected_feature_dims: Optional[Dict[str, int]] = None,
) -> Dict[str, np.ndarray]:
    require_without = feature_variant == "activation_diff"
    with_conversations = [x["with_conversation"] for x in selected_examples]
    without_conversations = [x["without_conversation"] for x in selected_examples]
    labels = [1 if bool(x["mirage_like"]) else 0 for x in selected_examples]
    pair_ids = list(range(len(selected_examples)))
    sample_names = [f"holdout_{i}" for i in range(len(selected_examples))]

    requested_include_attention = bool(
        getattr(args, "include_attention_probes", False) or getattr(args, "include_additional_attention_mlp_probes", False)
    )
    requested_include_mlp = bool(
        getattr(args, "include_mlp_probes", False) or getattr(args, "include_additional_attention_mlp_probes", False)
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
    if pair_core._uses_preextracted_activation_store(str(vlm_key)):
        with_map, without_map, layer_order, _ = pair_core._load_preextracted_all_examples_subset(
            vlm_key=str(vlm_key),
            with_conversations=with_conversations,
            without_conversations=(without_conversations if bool(require_without) else None),
            require_without=bool(require_without),
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            llm_feature_strategies=requested_llm_feature_strategies,
            use_additional_feature_preextract_cache=getattr(
                args, "use_additional_feature_preextract_cache", None
            ),
            requested_model_path=str(model_path),
        )
    else:
        with_map, layer_order = _extract_features(
            conversations=with_conversations,
            model_path=model_path,
            attn_implementation=str(args.attn_implementation),
            model_key=vlm_key,
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            device_map_raw=str(getattr(args, "device_map", "")),
            max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
            max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
        )
        without_map: Dict[str, List[np.ndarray]] = {}
        if bool(require_without):
            without_map, without_layer_order = _extract_features(
                conversations=without_conversations,
                model_path=model_path,
                attn_implementation=str(args.attn_implementation),
                model_key=vlm_key,
                include_attention_probes=bool(requested_include_attention),
                include_mlp_probes=bool(requested_include_mlp),
                include_residual_probes=bool(requested_include_residual),
                device_map_raw=str(getattr(args, "device_map", "")),
                max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
                max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
            )
            with_map, without_map, layer_order = _align_with_without_feature_maps(
                layer_features_with=with_map,
                layer_features_without=without_map,
                with_layer_order=layer_order,
                without_layer_order=without_layer_order,
            )
    requested_keys = pair_core._resolve_requested_feature_keys(
        layer_order=layer_order,
        include_attention_probes=bool(requested_include_attention),
        include_mlp_probes=bool(requested_include_mlp),
        include_residual_probes=bool(requested_include_residual),
        llm_feature_strategies=requested_llm_feature_strategies,
    )
    with_map, layer_order = pair_core._filter_layer_feature_map(
        layer_features=with_map,
        layer_order=layer_order,
        requested_keys=requested_keys,
    )
    without_map, _ = pair_core._filter_layer_feature_map(
        layer_features=without_map,
        layer_order=layer_order,
        requested_keys=layer_order,
    )
    if additional_feature_experiment_mode:
        filtered_layer_order = pair_core._keep_every_other_layer_feature_names(layer_order)
        with_map, layer_order = pair_core._filter_layer_feature_map(
            layer_features=with_map,
            layer_order=layer_order,
            requested_keys=filtered_layer_order,
        )
        without_map, _ = pair_core._filter_layer_feature_map(
            layer_features=without_map,
            layer_order=layer_order,
            requested_keys=filtered_layer_order,
        )

    X_by_feature_holdout, y_holdout_arr, _pids, _sample_names, _order = _materialize_feature_matrices(
        layer_features_with=with_map,
        layer_features_without=without_map,
        layer_order=layer_order,
        feature_variant=feature_variant,
        labels=labels,
        pair_ids=pair_ids,
        sample_names=sample_names,
    )
    if int(len(y_holdout_arr)) != int(len(selected_examples)):
        raise RuntimeError("Holdout matrix materialization produced unexpected label length.")

    missing = [name for name in feature_order if name not in X_by_feature_holdout]
    if missing:
        raise RuntimeError(
            "Holdout extraction is missing feature keys required by training feature_order. "
            f"missing_count={len(missing)}"
        )
    out = {name: np.asarray(X_by_feature_holdout[name], dtype=np.float32) for name in feature_order}
    if expected_feature_dims:
        mismatched: List[str] = []
        for name in feature_order:
            expected_dim = expected_feature_dims.get(str(name))
            if expected_dim is None:
                continue
            arr = out.get(str(name))
            if arr is None:
                continue
            if arr.ndim != 2 or int(arr.shape[1]) != int(expected_dim):
                mismatched.append(
                    f"{name}: holdout_dim={arr.shape[1] if arr.ndim == 2 else 'ndim!=2'}, expected_dim={expected_dim}"
                )
        if mismatched:
            preview = "; ".join(mismatched[:8])
            raise RuntimeError(
                "Holdout feature dimensionality mismatch vs training feature matrices. "
                f"Examples: {preview}"
            )
    return out

def _build_concat_llm_residual_features(layer_features: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    strategy_features: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    post_attention_strategy_features: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    mlp_strategy_features: Dict[str, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    attention_head_strategy_features: Dict[Tuple[str, int], List[Tuple[int, np.ndarray]]] = defaultdict(list)

    for key, value in layer_features.items():
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
        if not ordered:
            continue
        out[f"language_model/all_layers_concat__{strategy}"] = np.concatenate(ordered, axis=1)

    # Additional concat outputs: one per attention head index (across layers), plus post-attn and MLP families.
    for (strategy, head_idx), items in attention_head_strategy_features.items():
        ordered = [arr for _layer, arr in sorted(items, key=lambda t: t[0])]
        if not ordered:
            continue
        out[f"language_model/all_layers_concat_attention_head_{head_idx}__{strategy}"] = np.concatenate(ordered, axis=1)

    for strategy, items in post_attention_strategy_features.items():
        ordered = [arr for _layer, arr in sorted(items, key=lambda t: t[0])]
        if not ordered:
            continue
        out[f"language_model/all_layers_concat_post_attention__{strategy}"] = np.concatenate(ordered, axis=1)

    for strategy, items in mlp_strategy_features.items():
        ordered = [arr for _layer, arr in sorted(items, key=lambda t: t[0])]
        if not ordered:
            continue
        out[f"language_model/all_layers_concat_mlp__{strategy}"] = np.concatenate(ordered, axis=1)

    if not out:
        raise RuntimeError("No language-model features found for concat mode.")
    return out


def _build_activation_diff_features_contrastive(
    layer_features_with: Dict[str, List[np.ndarray]],
    layer_features_without: Dict[str, List[np.ndarray]],
    labels: List[int],
    pair_ids: List[int],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if not layer_features_without:
        raise RuntimeError("Activation-diff mode requires without-image feature tensors.")
    common_keys = [k for k in layer_features_with.keys() if k in layer_features_without]
    if not common_keys:
        raise RuntimeError("No overlapping feature keys for with-image/without-image activation diff.")
    transformed: Dict[str, np.ndarray] = {}
    for key in common_keys:
        with_arr = np.asarray(layer_features_with[key], dtype=np.float32)
        without_arr = np.asarray(layer_features_without[key], dtype=np.float32)
        if with_arr.shape != without_arr.shape:
            raise RuntimeError(
                f"With/without feature shape mismatch for key '{key}': "
                f"{with_arr.shape} vs {without_arr.shape}"
            )
        transformed[key] = with_arr - without_arr

    return (
        transformed,
        np.asarray(labels, dtype=np.int64),
        np.asarray(pair_ids, dtype=np.int64),
    )


def _materialize_feature_matrices(
    layer_features_with: Dict[str, List[np.ndarray]],
    layer_features_without: Dict[str, List[np.ndarray]],
    layer_order: List[str],
    feature_variant: str,
    labels: List[int],
    pair_ids: List[int],
    sample_names: List[str],
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[str], List[str]]:
    if feature_variant == "raw":
        matrix_map = {k: np.asarray(layer_features_with[k], dtype=np.float32) for k in layer_order}
        return (
            matrix_map,
            np.asarray(labels, dtype=np.int64),
            np.asarray(pair_ids, dtype=np.int64),
            [str(x) for x in sample_names],
            list(layer_order),
        )

    if feature_variant == "concat_llm_residual":
        concat_map = _build_concat_llm_residual_features(layer_features_with)
        order = sorted(concat_map.keys())
        return (
            concat_map,
            np.asarray(labels, dtype=np.int64),
            np.asarray(pair_ids, dtype=np.int64),
            [str(x) for x in sample_names],
            order,
        )

    if feature_variant == "activation_diff":
        diff_map, diff_y, diff_pair_ids = _build_activation_diff_features_contrastive(
            layer_features_with=layer_features_with,
            layer_features_without=layer_features_without,
            labels=labels,
            pair_ids=pair_ids,
        )
        order = sorted(diff_map.keys(), key=pair_core._layer_sort_key)
        return diff_map, diff_y, diff_pair_ids, [str(x) for x in sample_names], order

    raise ValueError(f"Unsupported feature_variant '{feature_variant}'.")


def run_contrastive_experiment(
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
            "num_split_seeds=3, single-init probes, num_eval_seeds=3."
        )
        args.num_split_seeds = 3
        args.disable_multi_init = True
        args.num_probe_inits = 1
        args.num_eval_seeds = 3
    supported_benchmarks = pair_core._supported_contrastive_benchmarks_for_vlm(vlm_key)
    if str(args.benchmark_mode) != "all" and str(args.benchmark_mode) not in supported_benchmarks:
        raise ValueError(
            f"benchmark_mode='{args.benchmark_mode}' is not supported for --vlm {vlm_key}. "
            f"Supported: {supported_benchmarks}"
        )
    model_path = resolve_model_path(str(args.vlm), str(args.model_path_override))
    run_id = make_run_id(seed=int(args.seed), run_id=str(args.run_id))
    cache_path = _scope_default_path(
        Path(args.features_cache_path),
        vlm=vlm_key,
        marker="contrastive_pair_layer_features",
    )
    save_root = _scope_default_path(
        Path(args.save_dir),
        vlm=vlm_key,
        marker="_probe_results",
    )

    pairs_path = Path(args.neutral_pairs_path if args.neutral_as_non_mirage_pairs else args.pairs_path)
    # Prefer the latest model-scoped artifact when available (including default VLM),
    # and fall back to legacy unscoped paths for backward compatibility.
    pairs_path = pair_core._resolve_model_scoped_artifact_path(
        base_path=pairs_path,
        vlm_key=vlm_key,
        include_gen_prefix_dirs=True,
    )
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    pairs = _filter_pairs_by_benchmark(
        pairs=pairs,
        benchmark_mode=str(args.benchmark_mode),
        vlm_key=vlm_key,
    )
    if not pairs:
        raise ValueError("No contrastive pairs remain after filtering.")

    (
        conversations,
        without_conversations,
        labels,
        pair_ids,
        sample_names,
        pair_benchmarks,
        skipped_short_training_pairs_count,
    ) = _build_contrastive_samples(
        pairs,
        include_short_response_filter=bool(args.exclude_short_responses_in_training_pairs),
        require_without=(feature_variant == "activation_diff"),
    )
    if not conversations:
        raise ValueError(
            "No contrastive training pairs remain after short-response filtering. "
            "Re-run with --no_exclude_short_responses_in_training_pairs to keep short pairs."
        )
    require_without = feature_variant == "activation_diff"
    if bool(require_without):
        for conv in without_conversations:
            if not isinstance(conv, list) or (not conv):
                raise RuntimeError(
                    "Activation-diff contrastive mode now requires without-image conversations in "
                    "contrastive pair artifacts. Rebuild via build_contrastive_pairs.py."
                )
    layer_features_with, layer_features_without, layer_order, cache_path = _load_or_extract_contrastive_features(
        args=args,
        model_path=model_path,
        conversations=conversations,
        without_conversations=without_conversations,
        require_without_image=bool(require_without),
        labels=labels,
        pair_ids=pair_ids,
        sample_names=sample_names,
        cache_path=cache_path,
        model_key=vlm_key,
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

    X_by_feature, y, pair_ids_arr, used_sample_names, feature_order = _materialize_feature_matrices(
        layer_features_with=layer_features_with,
        layer_features_without=layer_features_without,
        layer_order=layer_order,
        feature_variant=feature_variant,
        labels=labels,
        pair_ids=pair_ids,
        sample_names=sample_names,
    )
    if not feature_order:
        raise RuntimeError(
            "No features remain after applying probe-family/strategy filters. "
            "Check --include_*_probes and --llm_feature_strategies."
        )

    single_validation_pairs = int(args.single_benchmark_validation_pairs)
    if (
        pair_core._is_glm_vlm(vlm_key)
        and single_validation_pairs == int(DEFAULT_VALIDATION_PAIRS_SINGLE_BENCHMARK)
    ):
        single_validation_pairs = int(pair_core._single_benchmark_validation_pairs_for_vlm(vlm_key))
    qwen_val_fraction = pair_core._contrastive_validation_fraction_for_vlm(vlm_key)

    split_payloads, split_seeds = _split_payloads_contrastive(
        args=args,
        y=y,
        pair_ids_arr=pair_ids_arr,
        pair_benchmarks=pair_benchmarks,
    )
    benchmark_labels = np.asarray([str(pair_benchmarks[int(pid)]) for pid in pair_ids_arr.tolist()], dtype=object)
    reg_values = parse_regularization_values(str(args.regularization_values))

    per_feature_results: Dict[str, Dict] = {}
    per_feature_seed_runs: Dict[str, List[Dict]] = {}
    for feature_name in tqdm(feature_order, desc=f"Training {probe_type} probes", unit="feature", dynamic_ncols=True):
        X = np.asarray(X_by_feature[feature_name], dtype=np.float32)
        seed_runs = []
        for sp in split_payloads:
            run = sweep_probe_with_validation(
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
            run["heldout_pairs"] = sp["heldout_pairs"]
            run["heldout_pairs_by_benchmark"] = sp["heldout_pairs_by_benchmark"]
            seed_runs.append(run)
        per_feature_seed_runs[feature_name] = seed_runs
        per_feature_results[feature_name] = _summarize_feature_runs(seed_runs)
    feature_dims_by_name: Dict[str, int] = {
        str(name): int(np.asarray(X_by_feature[name], dtype=np.float32).shape[1])
        for name in feature_order
    }

    selected_benchmark: Optional[str] = None if str(args.benchmark_mode) == "all" else str(args.benchmark_mode)
    responses_path = pair_core._resolve_responses_path_for_vlm(
        base_responses_path=Path(args.responses_path),
        vlm_key=str(args.vlm),
    )
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    mirage_root = REPO_ROOT.resolve()
    image_lookup_uid, image_lookup_qid = pair_core._build_image_lookup_from_responses(
        mirage_root=mirage_root,
        responses=responses,
    )
    seen_signatures = {pair_core._conversation_signature_from_conv(c) for c in conversations}
    require_without = feature_variant == "activation_diff"
    pool_true, pool_false, skipped_short_holdout_candidates = _build_holdout_pool_with_without(
        responses=responses,
        seen_signatures=seen_signatures,
        selected_benchmark=selected_benchmark,
        allowed_benchmarks=supported_benchmarks,
        include_short_response_filter=bool(args.exclude_short_responses_in_holdout),
        require_without=bool(require_without),
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
    )
    raw_holdout_pool_sizes = {
        "mirage_true": int(len(pool_true)),
        "mirage_false": int(len(pool_false)),
    }
    holdout_pool_filter_to_preextract_cache = {
        "mirage_true_before": int(len(pool_true)),
        "mirage_false_before": int(len(pool_false)),
        "mirage_true_after": int(len(pool_true)),
        "mirage_false_after": int(len(pool_false)),
        "mirage_true_dropped": 0,
        "mirage_false_dropped": 0,
    }
    if pair_core._uses_preextracted_activation_store(str(args.vlm)):
        pre_path = pair_core._preextracted_all_examples_path_for_vlm(str(args.vlm))
        if pre_path.exists():
            pre_payload = torch.load(pre_path)
            with_payload = pre_payload.get("with_image", {}) or {}
            available_signature_keys = {str(x) for x in with_payload.get("signature_keys", [])}
            pool_true, pool_false, holdout_pool_filter_to_preextract_cache = pair_core._filter_holdout_pool_to_available_signatures(
                pool_true=pool_true,
                pool_false=pool_false,
                available_signature_keys=available_signature_keys,
            )
    holdout_pool_counts_by_benchmark_before_preextract_filter = pair_core._holdout_pool_counts_by_benchmark(
        pool_true=pool_true,
        pool_false=pool_false,
    )
    holdout_selection_plan = pair_core._plan_balanced_holdout_selection(
        pool_true=pool_true,
        pool_false=pool_false,
        requested_num_true=int(args.num_holdout_mirage_true),
        requested_num_false=int(args.num_holdout_mirage_false),
    )
    eval_seeds = [int(args.seed) + i for i in range(int(args.num_eval_seeds))]
    holdout_payloads_by_seed: Dict[int, Dict[str, Any]] = {}
    holdout_selected_sizes_by_seed: Dict[str, Dict[str, Any]] = {}
    for eval_seed in tqdm(eval_seeds, desc="Building unseen holdout payloads", unit="seed", dynamic_ncols=True):
        selected_examples, selected_counts_by_benchmark = pair_core._select_holdout_examples_balanced_by_benchmark(
            pool_true=pool_true,
            pool_false=pool_false,
            selected_pairs_by_benchmark={
                str(k): int(v)
                for k, v in dict(holdout_selection_plan["selected_pairs_by_benchmark"]).items()
            },
            seed=int(eval_seed),
        )
        y_holdout = np.asarray(
            [1 if bool(item.get("mirage_like")) else 0 for item in selected_examples],
            dtype=np.int64,
        )
        benchmark_labels_holdout = [str(item.get("dataset", "unknown")) for item in selected_examples]
        holdout_features = _extract_holdout_feature_matrices_for_seed(
            args=args,
            model_path=model_path,
            vlm_key=vlm_key,
            feature_variant=feature_variant,
            feature_order=feature_order,
            selected_examples=selected_examples,
            expected_feature_dims=feature_dims_by_name,
        )
        holdout_payloads_by_seed[int(eval_seed)] = {
            "y_holdout": y_holdout,
            "benchmark_labels_holdout": benchmark_labels_holdout,
            "features": holdout_features,
            "num_examples": int(len(y_holdout)),
            "num_examples_per_class": {
                "mirage_true": int((y_holdout == 1).sum()),
                "mirage_false": int((y_holdout == 0).sum()),
            },
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
        }
        holdout_selected_sizes_by_seed[str(int(eval_seed))] = {
            "num_examples_total": int(len(y_holdout)),
            "num_examples_per_class": {
                "mirage_true": int((y_holdout == 1).sum()),
                "mirage_false": int((y_holdout == 0).sum()),
            },
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
        }

    for feature_name in feature_order:
        X = np.asarray(X_by_feature[feature_name], dtype=np.float32)
        run_by_seed = {
            int(r["split_seed"]): r for r in per_feature_seed_runs.get(feature_name, [])
        }
        heldout_runs: List[Dict[str, Any]] = []
        for eval_idx, eval_seed in enumerate(eval_seeds):
            split_payload = split_payloads[eval_idx % len(split_payloads)]
            split_seed = int(split_payload["split_seed"])
            split_run = run_by_seed.get(split_seed)
            if split_run is None:
                raise RuntimeError(
                    f"Missing split run for split_seed={split_seed}, feature={feature_name}."
                )
            best_c = float(split_run["best_c"])
            X_train = X[split_payload["train_mask"]]
            y_train = y[split_payload["train_mask"]]
            X_val = X[split_payload["val_mask"]]
            y_val = y[split_payload["val_mask"]]
            holdout_payload = holdout_payloads_by_seed[int(eval_seed)]
            X_holdout = np.asarray(holdout_payload["features"][feature_name], dtype=np.float32)
            y_holdout = np.asarray(holdout_payload["y_holdout"], dtype=np.int64)
            fit = _fit_fixed_c_with_multi_init(
                probe_type=probe_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_eval=X_holdout,
                y_eval=y_holdout,
                split_seed=int(split_seed),
                c_value=float(best_c),
                args=args,
                eval_benchmark_labels=[str(x) for x in holdout_payload["benchmark_labels_holdout"]],
            )
            heldout_runs.append(
                {
                    "eval_seed": int(eval_seed),
                    "split_seed": int(split_seed),
                    "best_c": float(best_c),
                    "test_accuracy": fit.get("test_accuracy"),
                    "test_accuracy_mirage_true": fit.get("class1_test_accuracy"),
                    "test_accuracy_mirage_false": fit.get("class0_test_accuracy"),
                    "benchmark_test_accuracy": fit.get("benchmark_test_accuracy"),
                    "benchmark_class0_test_accuracy": fit.get("benchmark_class0_test_accuracy"),
                    "benchmark_class1_test_accuracy": fit.get("benchmark_class1_test_accuracy"),
                    "num_holdout_examples": int(len(y_holdout)),
                    "num_holdout_examples_per_class": holdout_payload.get("num_examples_per_class"),
                    "num_holdout_examples_by_benchmark": holdout_payload.get("selected_counts_by_benchmark"),
                    "best_init_index": int(fit["init_index"]),
                    "best_init_seed": int(fit["init_seed"]),
                    "validation_accuracy_for_init_selection": float(fit["val_accuracy"]),
                }
            )
        heldout_test_scores = [
            float(x["test_accuracy"])
            for x in heldout_runs
            if x.get("test_accuracy") is not None
        ]
        heldout_benchmark_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=heldout_runs,
            key="benchmark_test_accuracy",
        )
        heldout_benchmark_class0_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=heldout_runs,
            key="benchmark_class0_test_accuracy",
        )
        heldout_benchmark_class1_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=heldout_runs,
            key="benchmark_class1_test_accuracy",
        )
        class0_scores = [
            float(x["test_accuracy_mirage_false"])
            for x in heldout_runs
            if x.get("test_accuracy_mirage_false") is not None
        ]
        class1_scores = [
            float(x["test_accuracy_mirage_true"])
            for x in heldout_runs
            if x.get("test_accuracy_mirage_true") is not None
        ]
        per_feature_results[feature_name]["mean_test_accuracy_at_best_c"] = (
            float(np.mean(heldout_test_scores)) if heldout_test_scores else None
        )
        per_feature_results[feature_name]["std_test_accuracy_at_best_c"] = (
            float(np.std(heldout_test_scores)) if heldout_test_scores else None
        )
        per_feature_results[feature_name]["mean_class0_test_accuracy_at_best_c"] = (
            float(np.mean(class0_scores)) if class0_scores else None
        )
        per_feature_results[feature_name]["mean_class1_test_accuracy_at_best_c"] = (
            float(np.mean(class1_scores)) if class1_scores else None
        )
        per_feature_results[feature_name]["mean_benchmark_test_accuracy_at_best_c"] = heldout_benchmark_scores
        per_feature_results[feature_name]["mean_benchmark_class0_test_accuracy_at_best_c"] = heldout_benchmark_class0_scores
        per_feature_results[feature_name]["mean_benchmark_class1_test_accuracy_at_best_c"] = heldout_benchmark_class1_scores
        per_feature_results[feature_name]["test_accuracy_at_best_c"] = per_feature_results[feature_name]["mean_test_accuracy_at_best_c"]
        per_feature_results[feature_name]["heldout_seed_runs"] = heldout_runs

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
    config_path = run_dir / f"{vlm_tag}_run_config_{run_id}.json"

    with open(all_feature_path, "w", encoding="utf-8") as f:
        json.dump(all_feature_rows, f, indent=2, ensure_ascii=False)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_strategy_summary, f, indent=2, ensure_ascii=False)
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
                "pairs_path": str(pairs_path),
                "responses_path": str(responses_path),
                "repo_root": str(mirage_root),
                "neutral_as_non_mirage_pairs": bool(args.neutral_as_non_mirage_pairs),
                "benchmark_mode": str(args.benchmark_mode),
                "supported_contrastive_benchmarks": supported_benchmarks,
                "num_pairs": int(len(set(pair_ids_arr.tolist()))),
                "num_samples": int(len(y)),
                "single_benchmark_validation_pairs": int(single_validation_pairs),
                "all_mode_validation_pairs": int(pair_core._all_mode_validation_pairs_for_vlm(str(args.vlm))),
                "qwen_validation_fraction_if_applicable": qwen_val_fraction,
                "exclude_short_responses_in_training_pairs": bool(
                    args.exclude_short_responses_in_training_pairs
                ),
                "exclude_short_responses_in_holdout": bool(args.exclude_short_responses_in_holdout),
                "min_response_tokens_required": (
                    int(MIN_RESPONSE_TOKENS) if bool(args.exclude_short_responses_in_training_pairs) else None
                ),
                "num_pairs_skipped_short_responses": int(skipped_short_training_pairs_count),
                "num_holdout_candidates_skipped_short_responses": int(skipped_short_holdout_candidates),
                "num_holdout_mirage_true": int(args.num_holdout_mirage_true),
                "num_holdout_mirage_false": int(args.num_holdout_mirage_false),
                "num_eval_seeds": int(args.num_eval_seeds),
                "eval_seeds": [int(x) for x in eval_seeds],
                "holdout_pool_sizes_before_preextract_filter": raw_holdout_pool_sizes,
                "holdout_pool_filter_to_preextract_cache": holdout_pool_filter_to_preextract_cache,
                "holdout_pool_sizes_by_benchmark_after_preextract_filter": holdout_pool_counts_by_benchmark_before_preextract_filter,
                "holdout_selection_plan_after_preextract_filter": holdout_selection_plan,
                "holdout_selected_sizes_by_seed": holdout_selected_sizes_by_seed,
                "num_features": int(len(feature_order)),
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
                "run_dir": str(run_dir),
                "sample_names_preview": used_sample_names[:5],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved feature accuracies: {all_feature_path}")
    print(f"Saved LLM strategy summary: {llm_path}")
    print(f"Saved run config: {config_path}")


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(
        description=(
            "Train two-layer MLP probes on contrastive examples with multi-seed pair splits, "
            "validation-selected C sweep, and 3-init selection per C by default."
        ),
        default_save_dir="./tmp_artifacts/mlp_contrastive_probe_results",
        default_cache_path="./tmp_artifacts/contrastive_pair_layer_features_mlp.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.pairs_path = str(
        pair_core._canonicalize_input_arg_for_vlm(
            path_value=args.pairs_path,
            vlm_key=str(args.vlm),
            artifact_name="contrastive_conversation_pairs.json",
        )
    )
    args.neutral_pairs_path = str(
        pair_core._canonicalize_input_arg_for_vlm(
            path_value=args.neutral_pairs_path,
            vlm_key=str(args.vlm),
            artifact_name="contrastive_conversation_pairs_neutral_as_non_mirage.json",
        )
    )
    args.responses_path = str(
        pair_core._canonicalize_input_arg_for_vlm(
            path_value=args.responses_path,
            vlm_key=str(args.vlm),
            artifact_name="responses.json",
        )
    )
    run_contrastive_experiment(
        args=args,
        script_name="train_mlp_contrastive",
        probe_type="mlp",
        feature_variant="raw",
    )


if __name__ == "__main__":
    main()
