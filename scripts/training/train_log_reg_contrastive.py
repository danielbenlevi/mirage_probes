#!/usr/bin/env python3
import argparse
import base64
import inspect
import io
import json
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText  # type: ignore
except Exception:
    AutoModelForImageTextToText = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.data.gen_mutations_get_responses as core

LLM_STRATEGIES = [
    "current_visual_plus_last",
    "text_nonspecial_mean",
    "all_nonspecial_mean",
]
DEFAULT_SINGLE_BENCHMARK_VALIDATION_PAIRS = 5
GLM_SINGLE_BENCHMARK_VALIDATION_PAIRS = 2
GLM_ALL_MODE_VALIDATION_PAIRS = 3
QWEN_CONTRASTIVE_VALIDATION_FRACTION = 0.10
FEATURE_EXTRACTION_VERSION = 2
MIN_RESPONSE_TOKENS = 10
DEFAULT_VLM = "ovis"
VLM_MODEL_PATHS = {
    "ovis": "AIDC-AI/Ovis2.5-2B",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "glm_4_6v_flash": "zai-org/GLM-4.6V-Flash",
}
SUPPORTED_CONTRASTIVE_BENCHMARKS = ["vqa_rad", "mmmu_pro", "medxpertqa_mm"]
GLM_SUPPORTED_CONTRASTIVE_BENCHMARKS = ["vqa_rad", "mmmu_pro"]
GLM_IMAGE_MAX_EDGE = int(getattr(core, "GLM_IMAGE_MAX_EDGE", 1708))
GLM_IMAGE_MIN_EDGE = int(getattr(core, "GLM_IMAGE_MIN_EDGE", 28))
PIL_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)
_PROCESSOR_CACHE: Dict[str, Any] = {}
TMP_ARTIFACTS_ROOT = Path("./tmp_artifacts")
DEFAULT_RESPONSES_INPUT_PATH = TMP_ARTIFACTS_ROOT / "responses.json"
DEFAULT_CONTRASTIVE_PAIRS_INPUT_PATH = TMP_ARTIFACTS_ROOT / "contrastive_conversation_pairs.json"
DEFAULT_NEUTRAL_CONTRASTIVE_PAIRS_INPUT_PATH = (
    TMP_ARTIFACTS_ROOT / "contrastive_conversation_pairs_neutral_as_non_mirage.json"
)
QWEN_PREEXTRACTED_ROOT = TMP_ARTIFACTS_ROOT / "qwen3_vl_32b_instruct"
QWEN_PREEXTRACTED_CONTRASTIVE_PATH = (
    QWEN_PREEXTRACTED_ROOT / "qwen3_vl_32b_instruct_preextracted_contrastive_features.pt"
)
QWEN_PREEXTRACTED_ALL_EXAMPLES_PATH = (
    QWEN_PREEXTRACTED_ROOT / "qwen3_vl_32b_instruct_preextracted_all_examples_features.pt"
)
QWEN_PREEXTRACTED_CONTRASTIVE_ADDITIONAL_PATH = (
    QWEN_PREEXTRACTED_ROOT / "qwen3_vl_32b_instruct_preextracted_contrastive_additional_features.pt"
)
QWEN_PREEXTRACTED_ALL_EXAMPLES_ADDITIONAL_PATH = (
    QWEN_PREEXTRACTED_ROOT / "qwen3_vl_32b_instruct_preextracted_all_examples_additional_features.pt"
)
GLM_PREEXTRACTED_ROOT = TMP_ARTIFACTS_ROOT / "glm_4_6v_flash"
GLM_PREEXTRACTED_CONTRASTIVE_PATH = (
    GLM_PREEXTRACTED_ROOT / "glm_4_6v_flash_preextracted_contrastive_features.pt"
)
GLM_PREEXTRACTED_ALL_EXAMPLES_PATH = (
    GLM_PREEXTRACTED_ROOT / "glm_4_6v_flash_preextracted_all_examples_features.pt"
)
GLM_PREEXTRACTED_CONTRASTIVE_ADDITIONAL_PATH = (
    GLM_PREEXTRACTED_ROOT / "glm_4_6v_flash_preextracted_contrastive_additional_features.pt"
)
GLM_PREEXTRACTED_ALL_EXAMPLES_ADDITIONAL_PATH = (
    GLM_PREEXTRACTED_ROOT / "glm_4_6v_flash_preextracted_all_examples_additional_features.pt"
)
FINAL_DATA_ROOT = REPO_ROOT / "data" / "final_data"
RENAMED_ARTIFACTS_BY_VLM: Dict[str, Dict[str, Path]] = {
    "ovis": {
        "responses.json": FINAL_DATA_ROOT / "ovis_all_responses.json",
        "contrastive_conversation_pairs.json": FINAL_DATA_ROOT / "ovis_contrastive.json",
    },
    "qwen3_vl_32b_instruct": {
        "responses.json": FINAL_DATA_ROOT / "qwen_all_responses.json",
        "contrastive_conversation_pairs.json": FINAL_DATA_ROOT / "qwen_contrastive.json",
    },
}

DEFAULT_ARTIFACT_NAMES = {
    "responses.json",
    "contrastive_conversation_pairs.json",
    "contrastive_conversation_pairs_neutral_as_non_mirage.json",
}


def _is_glm_vlm(vlm_key: str) -> bool:
    return str(vlm_key).strip().lower() == "glm_4_6v_flash"


def _is_qwen_vlm(vlm_key: str) -> bool:
    return str(vlm_key).strip().lower() == "qwen3_vl_32b_instruct"


def _preextract_family_for_vlm(vlm_key: str) -> str:
    key = str(vlm_key).strip().lower()
    if key == "qwen3_vl_32b_instruct":
        return "qwen"
    if key == "glm_4_6v_flash":
        return "glm"
    return ""


def _uses_preextracted_activation_store(vlm_key: str) -> bool:
    return _preextract_family_for_vlm(vlm_key) in {"qwen", "glm"}


def _qwen_preextracted_contrastive_path() -> Path:
    return Path(QWEN_PREEXTRACTED_CONTRASTIVE_PATH)


def _qwen_preextracted_all_examples_path() -> Path:
    return Path(QWEN_PREEXTRACTED_ALL_EXAMPLES_PATH)


def _qwen_preextracted_contrastive_additional_path() -> Path:
    return Path(QWEN_PREEXTRACTED_CONTRASTIVE_ADDITIONAL_PATH)


def _qwen_preextracted_all_examples_additional_path() -> Path:
    return Path(QWEN_PREEXTRACTED_ALL_EXAMPLES_ADDITIONAL_PATH)


def _glm_preextracted_contrastive_path() -> Path:
    return Path(GLM_PREEXTRACTED_CONTRASTIVE_PATH)


def _glm_preextracted_all_examples_path() -> Path:
    return Path(GLM_PREEXTRACTED_ALL_EXAMPLES_PATH)


def _glm_preextracted_contrastive_additional_path() -> Path:
    return Path(GLM_PREEXTRACTED_CONTRASTIVE_ADDITIONAL_PATH)


def _glm_preextracted_all_examples_additional_path() -> Path:
    return Path(GLM_PREEXTRACTED_ALL_EXAMPLES_ADDITIONAL_PATH)


def _preextracted_contrastive_path_for_vlm(
    vlm_key: str,
    use_additional_feature_cache: bool = False,
) -> Path:
    family = _preextract_family_for_vlm(vlm_key)
    if family == "qwen":
        if bool(use_additional_feature_cache):
            return _qwen_preextracted_contrastive_additional_path()
        return _qwen_preextracted_contrastive_path()
    if family == "glm":
        if bool(use_additional_feature_cache):
            return _glm_preextracted_contrastive_additional_path()
        return _glm_preextracted_contrastive_path()
    raise ValueError(f"No pre-extracted contrastive path configured for vlm_key='{vlm_key}'")


def _preextracted_all_examples_path_for_vlm(
    vlm_key: str,
    use_additional_feature_cache: bool = False,
) -> Path:
    family = _preextract_family_for_vlm(vlm_key)
    if family == "qwen":
        if bool(use_additional_feature_cache):
            return _qwen_preextracted_all_examples_additional_path()
        return _qwen_preextracted_all_examples_path()
    if family == "glm":
        if bool(use_additional_feature_cache):
            return _glm_preextracted_all_examples_additional_path()
        return _glm_preextracted_all_examples_path()
    raise ValueError(f"No pre-extracted all-examples path configured for vlm_key='{vlm_key}'")


def _resolve_additional_preextract_cache_selection(
    include_attention_probes: bool,
    include_mlp_probes: bool,
    include_residual_probes: bool,
    explicit_use_additional_feature_cache: Optional[bool],
) -> bool:
    if explicit_use_additional_feature_cache is not None:
        return bool(explicit_use_additional_feature_cache)
    return (not bool(include_residual_probes)) and bool(
        bool(include_attention_probes) or bool(include_mlp_probes)
    )


def _is_additional_feature_experiment_mode(
    include_attention_probes: bool,
    include_mlp_probes: bool,
    include_residual_probes: bool,
) -> bool:
    return (not bool(include_residual_probes)) and bool(
        bool(include_attention_probes) or bool(include_mlp_probes)
    )


def _keep_every_other_layer_feature_names(feature_names: Sequence[str]) -> List[str]:
    kept: List[str] = []
    for name in feature_names:
        key = str(name)
        base = key.split("__")[0]
        m = re.search(r"/layer_(\d+)(?:/|$)", base)
        if m is None:
            kept.append(key)
            continue
        layer_num = int(m.group(1))
        if (layer_num % 2) == 1:
            kept.append(key)
    return kept


def _supported_contrastive_benchmarks_for_vlm(vlm_key: str) -> List[str]:
    if _is_glm_vlm(vlm_key):
        return list(GLM_SUPPORTED_CONTRASTIVE_BENCHMARKS)
    return list(SUPPORTED_CONTRASTIVE_BENCHMARKS)


def _single_benchmark_validation_pairs_for_vlm(vlm_key: str) -> int:
    if _is_glm_vlm(vlm_key):
        return int(GLM_SINGLE_BENCHMARK_VALIDATION_PAIRS)
    return int(DEFAULT_SINGLE_BENCHMARK_VALIDATION_PAIRS)


def _all_mode_validation_pairs_for_vlm(vlm_key: str) -> int:
    if _is_glm_vlm(vlm_key):
        return int(GLM_ALL_MODE_VALIDATION_PAIRS)
    return 9


def _contrastive_validation_fraction_for_vlm(vlm_key: str) -> Optional[float]:
    if _is_qwen_vlm(vlm_key):
        return float(QWEN_CONTRASTIVE_VALIDATION_FRACTION)
    return None


def _probe_init_seed(split_seed: int, c_value: float, init_idx: int) -> int:
    return int(split_seed) + (int(init_idx) * 100_003) + int(round(float(c_value) * 10_000))


def _default_model_path_for_vlm(vlm_key: str) -> str:
    key = str(vlm_key).strip().lower()
    fallback = str(VLM_MODEL_PATHS[key])
    try:
        from vlm_model_registry import resolve_vlm_config

        local_path = resolve_vlm_config(key).default_local_path
        if local_path.exists():
            return str(local_path)
    except Exception:
        pass
    return fallback


def add_model_loading_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device_map",
        type=str,
        default="",
        help=(
            "Optional HF device_map for model loading. Use 'auto' for multi-GPU sharding, "
            "or pass a JSON mapping string."
        ),
    )
    parser.add_argument(
        "--max_memory_per_gpu_gib",
        type=float,
        default=0.0,
        help=(
            "Optional per-GPU memory cap (GiB) used when --device_map is set. "
            "0 disables explicit max_memory."
        ),
    )
    parser.add_argument(
        "--max_memory_cpu_gib",
        type=float,
        default=0.0,
        help="Optional CPU memory cap (GiB) for HF max_memory when --device_map is set.",
    )
    return parser


def add_preextract_cache_selection_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--use_additional_feature_preextract_cache",
        dest="use_additional_feature_preextract_cache",
        action="store_true",
        default=None,
        help=(
            "Use the dedicated Qwen/GLM pre-extracted cache containing additional probe families "
            "(attention-head/post-attention/MLP). Default behavior auto-selects this cache when "
            "running additional-only features (no residual probes)."
        ),
    )
    parser.add_argument(
        "--no_use_additional_feature_preextract_cache",
        dest="use_additional_feature_preextract_cache",
        action="store_false",
        help="Force use of the baseline residual pre-extracted cache.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train VLM probes on contrastive mirage pairs with multiseed train/validation splits, "
            "then run held-out unseen evaluation for selected feature groups."
        )
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default=str(DEFAULT_CONTRASTIVE_PAIRS_INPUT_PATH),
        help=(
            "Contrastive pairs artifact. If left at the default value, OVIS/QWEN use "
            "data/final_data/*_contrastive.json and GLM uses tmp_artifacts/contrastive_conversation_pairs.json."
        ),
    )
    parser.add_argument(
        "--neutral_pairs_path",
        type=str,
        default=str(DEFAULT_NEUTRAL_CONTRASTIVE_PAIRS_INPUT_PATH),
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
        help="Optional explicit model path override. Defaults to the canonical path for --vlm.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_pairs", type=int, default=-1)
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--vqa_only_pairs",
        action="store_true",
        help=(
            "Train/validate using only contrastive pairs inferred as vqa_rad benchmark. "
            "Uses fixed validation holdout (default: 5 pairs; GLM default: 2 pairs)."
        ),
    )
    benchmark_group.add_argument(
        "--mmmu_only_pairs",
        action="store_true",
        help="Train/validate/eval using only contrastive pairs from MMMU-Pro.",
    )
    benchmark_group.add_argument(
        "--medxpert_only_pairs",
        action="store_true",
        help="Train/validate/eval using only contrastive pairs from MedXpertQA-MM.",
    )
    parser.add_argument(
        "--num_test_pairs",
        type=int,
        default=4,
        help=(
            "Deprecated and ignored. Holdout size is fixed by --vlm and mode "
            "(non-GLM all-mode: 9 pairs benchmark-stratified; GLM all-mode: 3 pairs total)."
        ),
    )
    parser.add_argument(
        "--regularization_values",
        type=str,
        default="0,0.01,0.1,1.0",
        help="Comma-separated C values for sweep (include 0 for no regularization).",
    )
    parser.add_argument(
        "--num_split_seeds",
        type=int,
        default=5,
        help="Number of random pair split seeds used for multiseed train/validation training.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention backend for Ovis."
        ),
    )
    parser.add_argument(
        "--probe_epochs",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--probe_lr",
        type=float,
        default=0.03,
    )
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
        help="Disable multi-init selection and train one initialization per (feature, C, split).",
    )
    parser.add_argument(
        "--probe_num_initializations",
        type=int,
        default=3,
        help="Number of initializations tried per (feature, C, split) when multi-init is enabled.",
    )
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
        default=",".join(LLM_STRATEGIES),
        help=(
            "Comma-separated feature strategies to include. Choices: "
            + ",".join(LLM_STRATEGIES)
            + "."
        ),
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
        default=str(TMP_ARTIFACTS_ROOT / "contrastive_pair_layer_features_with_attn.pt"),
    )
    parser.add_argument(
        "--responses_path",
        type=str,
        default=str(DEFAULT_RESPONSES_INPUT_PATH),
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
        help="Number of held-out evaluation seeds for final held-out feature-group reporting.",
    )
    parser.add_argument(
        "--heldout_eval_all_features",
        action="store_true",
        default=False,
        help=(
            "If set, run held-out evaluation for every trained feature instead of only "
            "validation-selected best features per group/layer."
        ),
    )
    parser.add_argument(
        "--exclude_short_responses_in_training_pairs",
        dest="exclude_short_responses_in_training_pairs",
        action="store_true",
        help=(
            "Exclude training pair candidates where either class conversation's assistant response has "
            f"fewer than {MIN_RESPONSE_TOKENS} whitespace-separated tokens."
        ),
    )
    parser.add_argument(
        "--no_exclude_short_responses_in_training_pairs",
        dest="exclude_short_responses_in_training_pairs",
        action="store_false",
        help="Keep short-response training pairs.",
    )
    parser.set_defaults(exclude_short_responses_in_training_pairs=True)
    parser.add_argument(
        "--exclude_short_responses_in_holdout",
        action="store_true",
        help=(
            "Exclude holdout candidates where either with-image or without-image response has "
            f"fewer than {MIN_RESPONSE_TOKENS} whitespace-separated tokens."
        ),
    )
    parser.add_argument("--force_reextract", action="store_true")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(TMP_ARTIFACTS_ROOT / "contrastive_probe_results"),
    )
    add_model_loading_args(parser)
    add_preextract_cache_selection_args(parser)
    return parser.parse_args()


def _resolve_selected_benchmark(args: argparse.Namespace) -> Optional[str]:
    if bool(args.vqa_only_pairs):
        return "vqa_rad"
    if bool(args.mmmu_only_pairs):
        return "mmmu_pro"
    if bool(args.medxpert_only_pairs):
        return "medxpertqa_mm"
    return None


def _resolve_model_path(args: argparse.Namespace) -> str:
    if str(args.model_path).strip():
        return str(args.model_path).strip()
    return _default_model_path_for_vlm(str(args.vlm))


def _resolve_device_map(device_map_raw: str) -> Optional[Any]:
    raw = str(device_map_raw or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if lowered in {"auto", "balanced", "balanced_low_0", "sequential"}:
        return lowered
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "--device_map must be one of: auto, balanced, balanced_low_0, sequential, "
            "or a JSON mapping."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("--device_map JSON value must be an object mapping module names to devices.")
    return parsed


def _format_gib_value(x: float) -> str:
    return f"{float(x):g}GiB"


def _build_max_memory_map(
    max_memory_per_gpu_gib: float,
    max_memory_cpu_gib: float,
) -> Optional[Dict[Any, str]]:
    mm: Dict[Any, str] = {}
    if torch.cuda.is_available() and float(max_memory_per_gpu_gib) > 0:
        for gpu_idx in range(int(torch.cuda.device_count())):
            mm[gpu_idx] = _format_gib_value(float(max_memory_per_gpu_gib))
    if float(max_memory_cpu_gib) > 0:
        mm["cpu"] = _format_gib_value(float(max_memory_cpu_gib))
    return mm or None


def _transformers_has_qwen3_vl_support() -> bool:
    try:
        from transformers import Qwen3VLForConditionalGeneration  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _has_accelerate() -> bool:
    try:
        import accelerate  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def load_vlm_for_extraction(
    model_path: str,
    attn_implementation: Optional[str] = None,
    device_map_raw: str = "",
    max_memory_per_gpu_gib: float = 0.0,
    max_memory_cpu_gib: float = 0.0,
):
    device_map = _resolve_device_map(device_map_raw)
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = str(attn_implementation)
    if device_map is not None:
        if not _has_accelerate():
            raise RuntimeError(
                "Model loading requested --device_map, but accelerate is not installed in this "
                f"environment ({sys.executable}). Install it with "
                f"`{sys.executable} -m pip install accelerate`, or re-run with --device_map '' "
                "(single-GPU load)."
            )
        kwargs["device_map"] = device_map
        max_memory = _build_max_memory_map(
            max_memory_per_gpu_gib=float(max_memory_per_gpu_gib),
            max_memory_cpu_gib=float(max_memory_cpu_gib),
        )
        if max_memory is not None:
            kwargs["max_memory"] = max_memory

    def _loader_name(loader_cls: Any) -> str:
        return getattr(loader_cls, "__name__", str(loader_cls))

    def _try_loader(loader_cls: Any) -> Tuple[Optional[Any], Optional[str]]:
        attempt_kwargs = dict(kwargs)
        try:
            return loader_cls.from_pretrained(model_path, **attempt_kwargs), None
        except TypeError as exc:
            if "attn_implementation" in attempt_kwargs:
                attempt_kwargs.pop("attn_implementation", None)
                try:
                    return loader_cls.from_pretrained(model_path, **attempt_kwargs), None
                except Exception as exc_wo_attn:
                    return None, f"{_loader_name(loader_cls)} failed after dropping attn_implementation: {exc_wo_attn}"
            return None, f"{_loader_name(loader_cls)} type error: {exc}"
        except Exception as exc:
            return None, f"{_loader_name(loader_cls)} failed: {exc}"

    preferred_loaders: List[Any] = [AutoModelForCausalLM]
    try:
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "")).strip().lower()
        cfg_name = str(type(cfg).__name__).lower()
        if model_type == "qwen3_vl" and not _transformers_has_qwen3_vl_support():
            raise RuntimeError(
                "Detected a qwen3_vl checkpoint, but current transformers does not provide "
                "Qwen3VLForConditionalGeneration. Use an environment with transformers "
                "that supports qwen3_vl (e.g., python3 from a compatible environment) "
                "for extraction, then reuse pre-extracted activation caches."
            )
        if model_type == "qwen3_vl" or "qwen3vl" in cfg_name:
            kwargs["config"] = cfg
            if AutoModelForImageTextToText is not None:
                preferred_loaders = [AutoModelForImageTextToText, AutoModel]
            else:
                preferred_loaders = [AutoModel]
            preferred_loaders.append(AutoModelForCausalLM)
        elif model_type == "glm4v" or "glm4v" in cfg_name:
            text_cfg = getattr(cfg, "text_config", None)
            rope_scaling = getattr(text_cfg, "rope_scaling", None) if text_cfg is not None else None
            rope_parameters = getattr(text_cfg, "rope_parameters", None) if text_cfg is not None else None
            if text_cfg is not None and isinstance(rope_parameters, dict):
                needs_rope_scaling = not isinstance(rope_scaling, dict) or "mrope_section" not in rope_scaling
                if needs_rope_scaling:
                    normalized_rope_scaling = dict(rope_parameters)
                    partial_rotary_factor = normalized_rope_scaling.get("partial_rotary_factor", None)
                    mrope_section = normalized_rope_scaling.get("mrope_section", None)
                    if isinstance(mrope_section, list) and mrope_section and partial_rotary_factor not in (None, 0, 1):
                        try:
                            f = float(partial_rotary_factor)
                            if f > 0:
                                normalized_rope_scaling["mrope_section"] = [
                                    int(round(float(x) / f)) for x in mrope_section
                                ]
                        except Exception:
                            pass
                    if "type" not in normalized_rope_scaling and "rope_type" in normalized_rope_scaling:
                        normalized_rope_scaling["type"] = normalized_rope_scaling["rope_type"]
                    text_cfg.rope_scaling = normalized_rope_scaling
                    rope_theta = normalized_rope_scaling.get("rope_theta", None)
                    if rope_theta is not None and hasattr(text_cfg, "rope_theta"):
                        try:
                            text_cfg.rope_theta = float(rope_theta)
                        except Exception:
                            pass
            kwargs["config"] = cfg
            if AutoModelForImageTextToText is not None:
                preferred_loaders = [AutoModelForImageTextToText, AutoModel]
            else:
                preferred_loaders = [AutoModel]
            preferred_loaders.append(AutoModelForCausalLM)
        else:
            if AutoModelForImageTextToText is not None:
                preferred_loaders.append(AutoModelForImageTextToText)
            preferred_loaders.append(AutoModel)
    except Exception:
        if AutoModelForImageTextToText is not None:
            preferred_loaders.append(AutoModelForImageTextToText)
        preferred_loaders.append(AutoModel)

    model = None
    loader_errors: List[str] = []
    for loader_cls in preferred_loaders:
        loaded, err = _try_loader(loader_cls)
        if loaded is not None:
            model = loaded
            break
        if err is not None:
            loader_errors.append(err)

    if model is None:
        raise RuntimeError(
            "Failed to load model for extraction. Tried loaders: "
            + ", ".join(_loader_name(x) for x in preferred_loaders)
            + ". Errors: "
            + " | ".join(loader_errors[-3:])
        )

    if device_map is None and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def _resolve_model_input_device(model) -> torch.device:
    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for loc in hf_device_map.values():
            if isinstance(loc, torch.device):
                if loc.type not in {"cpu", "meta"}:
                    return loc
            elif isinstance(loc, int):
                return torch.device(f"cuda:{int(loc)}")
            elif isinstance(loc, str):
                key = loc.strip().lower()
                if key in {"cpu", "disk", "meta"}:
                    continue
                if key.startswith("cuda"):
                    return torch.device(key)

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    for buf in model.buffers():
        if buf.device.type != "meta":
            return buf.device

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _scope_default_path(path: Path, vlm: str, marker: str) -> Path:
    p = Path(path)
    if marker not in str(p):
        return p
    if vlm == DEFAULT_VLM:
        return p
    if p.suffix:
        return p.with_name(f"{p.stem}_{vlm}{p.suffix}")
    return Path(f"{p}_{vlm}")


def _resolve_model_scoped_artifact_path(
    base_path: Path,
    vlm_key: str,
    include_gen_prefix_dirs: bool = True,
) -> Path:
    p = Path(base_path)
    key = str(vlm_key).strip()
    if not key:
        return p

    candidates: List[Tuple[Path, int]] = []
    base_parent = p.parent.as_posix()
    is_legacy_default_input = (
        p.name in DEFAULT_ARTIFACT_NAMES
        and base_parent in {".", TMP_ARTIFACTS_ROOT.as_posix(), f"./{TMP_ARTIFACTS_ROOT.as_posix()}"}
    )
    scoped_candidate = p.parent / key / p.name
    if scoped_candidate.exists():
        candidates.append((scoped_candidate, 20))

    renamed_artifacts = RENAMED_ARTIFACTS_BY_VLM.get(str(key).strip().lower(), {})
    renamed_candidate = renamed_artifacts.get(p.name)
    if renamed_candidate is not None and Path(renamed_candidate).exists():
        candidates.append((Path(renamed_candidate), 40 if is_legacy_default_input else 30))

    if include_gen_prefix_dirs:
        for gen_dir in sorted(p.parent.glob(f"gen_{key}*")):
            if not gen_dir.is_dir():
                continue
            gen_candidate = gen_dir / p.name
            if gen_candidate.exists():
                candidates.append((gen_candidate, 10))

    if not candidates:
        return p

    def _score(item: Tuple[Path, int]) -> Tuple[int, float, int]:
        path_obj, priority = item
        try:
            return (int(priority), float(path_obj.stat().st_mtime), len(str(path_obj)))
        except Exception:
            return (int(priority), float("-inf"), len(str(path_obj)))

    return max(candidates, key=_score)[0]


def _default_input_artifact_path_for_vlm(vlm_key: str, artifact_name: str) -> Path:
    key = str(vlm_key).strip().lower()
    renamed_artifacts = RENAMED_ARTIFACTS_BY_VLM.get(key, {})
    renamed_candidate = renamed_artifacts.get(str(artifact_name))
    if renamed_candidate is not None:
        return Path(renamed_candidate)
    return TMP_ARTIFACTS_ROOT / str(artifact_name)


def _canonicalize_input_arg_for_vlm(path_value: Optional[str], vlm_key: str, artifact_name: str) -> Path:
    raw = str(path_value or "").strip()
    if not raw:
        return _default_input_artifact_path_for_vlm(vlm_key=vlm_key, artifact_name=artifact_name)

    candidate = Path(raw)
    generic_default = TMP_ARTIFACTS_ROOT / str(artifact_name)
    if candidate == generic_default or candidate.as_posix() in {
        generic_default.as_posix(),
        f"./{generic_default.as_posix()}",
    }:
        return _default_input_artifact_path_for_vlm(vlm_key=vlm_key, artifact_name=artifact_name)
    return candidate


def _resolve_responses_path_for_vlm(base_responses_path: Path, vlm_key: str) -> Path:
    return _resolve_model_scoped_artifact_path(
        base_path=Path(base_responses_path),
        vlm_key=str(vlm_key),
        include_gen_prefix_dirs=True,
    )


def _norm_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def _conversation_signature_from_conv(conv: List[Dict]) -> Tuple[str, str, str]:
    system_text = ""
    user_text = ""
    assistant_text = ""
    for m in conv:
        role = m.get("role")
        if role == "system" and isinstance(m.get("content"), str):
            system_text = m.get("content", "")
        elif role == "user":
            c = m.get("content", [])
            if isinstance(c, list):
                for item in c:
                    if item.get("type") == "text":
                        user_text = item.get("text", "")
                        break
            elif isinstance(c, str):
                user_text = c
        elif role == "assistant" and isinstance(m.get("content"), str):
            assistant_text = m.get("content", "")
    return (_norm_text(system_text), _norm_text(user_text), _norm_text(assistant_text))


def _conversation_has_image_input(conv: List[Dict]) -> bool:
    for m in conv:
        if str(m.get("role", "")) != "user":
            continue
        content = m.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            item_type = str(item.get("type", ""))
            if item_type in {"image_url", "image"}:
                return True
    return False


def _conversation_signature_from_response_row(row: Dict) -> Tuple[str, str, str]:
    return (
        _norm_text(str(row.get("system_prompt", ""))),
        _norm_text(str(row.get("prompt_text", ""))),
        _norm_text(str((row.get("with_image", {}) or {}).get("response", ""))),
    )


def _signature_key_from_tuple(sig: Tuple[str, str, str]) -> str:
    return json.dumps([str(sig[0]), str(sig[1]), str(sig[2])], ensure_ascii=False)


def _signature_key_from_conv(conv: List[Dict]) -> str:
    return _signature_key_from_tuple(_conversation_signature_from_conv(conv))


def _signature_key_from_holdout_item(item: Dict[str, Any]) -> Optional[str]:
    conv = item.get("conversation")
    if not isinstance(conv, list):
        conv = item.get("with_conversation")
    if not isinstance(conv, list):
        return None
    return _signature_key_from_conv(conv)


def _resolve_requested_feature_keys(
    layer_order: Sequence[str],
    include_attention_probes: bool,
    include_mlp_probes: bool,
    include_residual_probes: bool = True,
    llm_feature_strategies: Optional[Sequence[str]] = None,
) -> List[str]:
    requested_strategies = _parse_requested_llm_feature_strategies(llm_feature_strategies)
    requested_set = set(requested_strategies)
    keys = list(layer_order)
    if not include_attention_probes:
        keys = [k for k in keys if not _is_attention_probe_feature(k)]
    if not include_mlp_probes:
        keys = [k for k in keys if not _is_mlp_probe_feature(k)]
    if not include_residual_probes:
        keys = [k for k in keys if _is_additional_attention_mlp_feature(k)]
    keys = [k for k in keys if _feature_strategy_from_name(k) in requested_set]
    return keys


def _parse_requested_llm_feature_strategies(raw_strategies: Optional[Sequence[str] | str]) -> List[str]:
    if raw_strategies is None:
        return list(LLM_STRATEGIES)
    if isinstance(raw_strategies, str):
        tokens = [x.strip() for x in str(raw_strategies).split(",") if x.strip()]
    else:
        tokens = [str(x).strip() for x in raw_strategies if str(x).strip()]
    if not tokens:
        raise ValueError(
            "--llm_feature_strategies must contain at least one strategy. "
            f"Allowed: {LLM_STRATEGIES}"
        )
    unknown = [x for x in tokens if x not in LLM_STRATEGIES]
    if unknown:
        raise ValueError(
            f"Unknown feature strategies in --llm_feature_strategies: {unknown}. "
            f"Allowed: {LLM_STRATEGIES}"
        )
    deduped: List[str] = []
    seen = set()
    for tok in tokens:
        if tok in seen:
            continue
        seen.add(tok)
        deduped.append(tok)
    return deduped


def _feature_strategy_from_name(feature_name: str) -> str:
    text = str(feature_name)
    if "__" not in text:
        return ""
    return text.split("__")[-1]


def _filter_layer_feature_map(
    layer_features: Dict[str, Any],
    layer_order: Sequence[str],
    requested_keys: Sequence[str],
) -> Tuple[Dict[str, Any], List[str]]:
    requested_set = set(str(x) for x in requested_keys)
    filtered_order = [str(k) for k in layer_order if str(k) in requested_set]
    filtered_features = {str(k): v for k, v in layer_features.items() if str(k) in requested_set}
    return filtered_features, filtered_order


def _gather_feature_rows_from_store(feature_store: Any, row_indices: np.ndarray) -> np.ndarray:
    idx = np.asarray(row_indices, dtype=np.int64)
    if torch.is_tensor(feature_store):
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=feature_store.device)
        return feature_store.index_select(0, idx_t).to(torch.float32).cpu().numpy()
    if isinstance(feature_store, np.ndarray):
        return np.asarray(feature_store[idx], dtype=np.float32)
    arr = np.asarray(feature_store)
    return np.asarray(arr[idx], dtype=np.float32)


def _maybe_warn_preextracted_cache_model_mismatch(
    payload: Dict,
    requested_model_path: str,
    cache_path: Path,
    family_label: str,
) -> None:
    cached_model_path = str(payload.get("model_path", "") or "").strip()
    requested = str(requested_model_path or "").strip()
    if cached_model_path and requested and cached_model_path != requested:
        warnings.warn(
            f"{family_label} pre-extracted cache model_path differs from requested --model_path. "
            f"cache={cached_model_path}, requested={requested}, cache_path={cache_path}"
        )


def _load_preextracted_contrastive_subset(
    vlm_key: str,
    conversations: Sequence[List[Dict]],
    include_attention_probes: bool,
    include_mlp_probes: bool,
    requested_model_path: str,
    include_residual_probes: bool = True,
    llm_feature_strategies: Optional[Sequence[str]] = None,
    use_additional_feature_preextract_cache: Optional[bool] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], List[str], Path]:
    family = _preextract_family_for_vlm(vlm_key)
    if family not in {"qwen", "glm"}:
        raise ValueError(f"Pre-extracted contrastive cache is unsupported for vlm_key='{vlm_key}'")

    use_additional_cache = _resolve_additional_preextract_cache_selection(
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        explicit_use_additional_feature_cache=use_additional_feature_preextract_cache,
    )
    resolved_path = (
        Path(cache_path)
        if cache_path is not None
        else _preextracted_contrastive_path_for_vlm(
            vlm_key=vlm_key,
            use_additional_feature_cache=bool(use_additional_cache),
        )
    )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Missing {family.upper()} pre-extracted contrastive activations cache. "
            f"Expected: {resolved_path}. Generate it via extract_{family}_activations.py."
        )

    payload = torch.load(resolved_path)
    expected_cache_type = f"{family}_preextracted_contrastive_features"
    if str(payload.get("cache_type", "")) not in {expected_cache_type}:
        raise RuntimeError(
            f"Invalid cache_type for {family.upper()} contrastive pre-extracted cache: "
            f"{payload.get('cache_type')}"
        )
    _maybe_warn_preextracted_cache_model_mismatch(
        payload=payload,
        requested_model_path=requested_model_path,
        cache_path=resolved_path,
        family_label=family.upper(),
    )

    layer_order = [str(x) for x in payload.get("layer_order", [])]
    requested_keys = _resolve_requested_feature_keys(
        layer_order=layer_order,
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        llm_feature_strategies=llm_feature_strategies,
    )
    if not requested_keys:
        raise RuntimeError(f"{family.upper()} contrastive cache does not expose requested feature families.")

    signature_keys = [str(x) for x in payload.get("signature_keys", [])]
    sig_to_idx = {sig: idx for idx, sig in enumerate(signature_keys)}
    required_indices: List[int] = []
    missing_signatures: List[str] = []
    for conv in conversations:
        sig_key = _signature_key_from_conv(conv)
        idx = sig_to_idx.get(sig_key)
        if idx is None:
            missing_signatures.append(sig_key)
            continue
        required_indices.append(int(idx))
    if missing_signatures:
        raise RuntimeError(
            f"{family.upper()} contrastive pre-extracted cache is missing required conversations. "
            f"missing_count={len(missing_signatures)} cache_path={resolved_path}"
        )

    source_features = payload.get("layer_features", {})
    out: Dict[str, np.ndarray] = {}
    for key in requested_keys:
        if key not in source_features:
            raise RuntimeError(
                f"Feature '{key}' missing from {family.upper()} contrastive cache at {resolved_path}"
            )
        arr = np.asarray(source_features[key], dtype=np.float32)
        out[key] = arr[np.asarray(required_indices, dtype=np.int64)]
    return out, requested_keys, resolved_path


def _load_preextracted_contrastive_with_without_subset(
    vlm_key: str,
    with_conversations: Sequence[List[Dict]],
    without_conversations: Sequence[List[Dict]],
    include_attention_probes: bool,
    include_mlp_probes: bool,
    requested_model_path: str,
    include_residual_probes: bool = True,
    llm_feature_strategies: Optional[Sequence[str]] = None,
    use_additional_feature_preextract_cache: Optional[bool] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], Path]:
    family = _preextract_family_for_vlm(vlm_key)
    if family not in {"qwen", "glm"}:
        raise ValueError(f"Pre-extracted contrastive cache is unsupported for vlm_key='{vlm_key}'")

    use_additional_cache = _resolve_additional_preextract_cache_selection(
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        explicit_use_additional_feature_cache=use_additional_feature_preextract_cache,
    )
    resolved_path = (
        Path(cache_path)
        if cache_path is not None
        else _preextracted_contrastive_path_for_vlm(
            vlm_key=vlm_key,
            use_additional_feature_cache=bool(use_additional_cache),
        )
    )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Missing {family.upper()} pre-extracted contrastive activations cache. "
            f"Expected: {resolved_path}. Generate it via extract_{family}_activations.py."
        )

    payload = torch.load(resolved_path)
    expected_cache_type = f"{family}_preextracted_contrastive_features"
    if str(payload.get("cache_type", "")) not in {expected_cache_type}:
        raise RuntimeError(
            f"Invalid cache_type for {family.upper()} contrastive pre-extracted cache: "
            f"{payload.get('cache_type')}"
        )
    _maybe_warn_preextracted_cache_model_mismatch(
        payload=payload,
        requested_model_path=requested_model_path,
        cache_path=resolved_path,
        family_label=family.upper(),
    )

    layer_order = [str(x) for x in payload.get("layer_order", [])]
    requested_keys = _resolve_requested_feature_keys(
        layer_order=layer_order,
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        llm_feature_strategies=llm_feature_strategies,
    )
    if not requested_keys:
        raise RuntimeError(f"{family.upper()} contrastive cache does not expose requested feature families.")

    with_sig_keys = [str(x) for x in payload.get("signature_keys", [])]
    with_sig_to_idx = {sig: idx for idx, sig in enumerate(with_sig_keys)}
    required_with_idx: List[int] = []
    for conv in with_conversations:
        sig_key = _signature_key_from_conv(conv)
        idx = with_sig_to_idx.get(sig_key)
        if idx is None:
            raise RuntimeError(
                f"{family.upper()} contrastive cache is missing required with-image conversation signatures. "
                f"cache_path={resolved_path}"
            )
        required_with_idx.append(int(idx))

    without_sig_keys = [str(x) for x in payload.get("without_image_signature_keys", [])]
    without_sig_to_idx = {sig: idx for idx, sig in enumerate(without_sig_keys)}
    if (not without_sig_to_idx) or ("without_image_layer_features" not in payload):
        raise RuntimeError(
            f"{family.upper()} contrastive cache does not include without-image tensors/signatures. "
            f"cache_path={resolved_path}"
        )
    required_without_idx: List[int] = []
    for conv in without_conversations:
        sig_key = _signature_key_from_conv(conv)
        idx = without_sig_to_idx.get(sig_key)
        if idx is None:
            raise RuntimeError(
                f"{family.upper()} contrastive cache is missing required without-image conversation signatures. "
                f"cache_path={resolved_path}"
            )
        required_without_idx.append(int(idx))

    with_source_features = payload.get("layer_features", {})
    without_source_features = payload.get("without_image_layer_features", {})
    with_out: Dict[str, np.ndarray] = {}
    without_out: Dict[str, np.ndarray] = {}
    for key in requested_keys:
        if key not in with_source_features:
            raise RuntimeError(
                f"Feature '{key}' missing from {family.upper()} contrastive with-image cache at {resolved_path}"
            )
        if key not in without_source_features:
            raise RuntimeError(
                f"Feature '{key}' missing from {family.upper()} contrastive without-image cache at {resolved_path}"
            )
        with_arr = np.asarray(with_source_features[key], dtype=np.float32)
        without_arr = np.asarray(without_source_features[key], dtype=np.float32)
        with_out[key] = with_arr[np.asarray(required_with_idx, dtype=np.int64)]
        without_out[key] = without_arr[np.asarray(required_without_idx, dtype=np.int64)]

    return with_out, without_out, requested_keys, resolved_path


def _load_preextracted_all_examples_subset(
    vlm_key: str,
    with_conversations: Sequence[List[Dict]],
    without_conversations: Optional[Sequence[List[Dict]]],
    require_without: bool,
    include_attention_probes: bool,
    include_mlp_probes: bool,
    requested_model_path: str,
    include_residual_probes: bool = True,
    llm_feature_strategies: Optional[Sequence[str]] = None,
    use_additional_feature_preextract_cache: Optional[bool] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], Path]:
    family = _preextract_family_for_vlm(vlm_key)
    if family not in {"qwen", "glm"}:
        raise ValueError(f"Pre-extracted all-examples cache is unsupported for vlm_key='{vlm_key}'")

    use_additional_cache = _resolve_additional_preextract_cache_selection(
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        explicit_use_additional_feature_cache=use_additional_feature_preextract_cache,
    )
    resolved_path = (
        Path(cache_path)
        if cache_path is not None
        else _preextracted_all_examples_path_for_vlm(
            vlm_key=vlm_key,
            use_additional_feature_cache=bool(use_additional_cache),
        )
    )
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Missing {family.upper()} pre-extracted all-examples activations cache. "
            f"Expected: {resolved_path}. Generate it via extract_{family}_activations.py."
        )

    payload = torch.load(resolved_path)
    expected_cache_type = f"{family}_preextracted_all_examples_features"
    if str(payload.get("cache_type", "")) not in {expected_cache_type}:
        raise RuntimeError(
            f"Invalid cache_type for {family.upper()} all-examples pre-extracted cache: "
            f"{payload.get('cache_type')}"
        )
    _maybe_warn_preextracted_cache_model_mismatch(
        payload=payload,
        requested_model_path=requested_model_path,
        cache_path=resolved_path,
        family_label=family.upper(),
    )

    layer_order = [str(x) for x in payload.get("layer_order", [])]
    requested_keys = _resolve_requested_feature_keys(
        layer_order=layer_order,
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        llm_feature_strategies=llm_feature_strategies,
    )
    if not requested_keys:
        raise RuntimeError(f"{family.upper()} all-examples cache does not expose requested feature families.")

    with_payload = payload.get("with_image", {})
    with_sig_keys = [str(x) for x in with_payload.get("signature_keys", [])]
    with_sig_to_idx = {sig: idx for idx, sig in enumerate(with_sig_keys)}
    required_with_idx: List[int] = []
    for conv in with_conversations:
        sig_key = _signature_key_from_conv(conv)
        idx = with_sig_to_idx.get(sig_key)
        if idx is None:
            raise RuntimeError(
                f"{family.upper()} all-examples cache missing required with-image conversation signature. "
                f"cache_path={resolved_path}"
            )
        required_with_idx.append(int(idx))

    with_source_features = with_payload.get("layer_features", {})
    with_out: Dict[str, np.ndarray] = {}
    for key in requested_keys:
        if key not in with_source_features:
            raise RuntimeError(
                f"Feature '{key}' missing from {family.upper()} with-image cache at {resolved_path}"
            )
        arr = np.asarray(with_source_features[key], dtype=np.float32)
        with_out[key] = arr[np.asarray(required_with_idx, dtype=np.int64)]

    without_out: Dict[str, np.ndarray] = {}
    if bool(require_without):
        if without_conversations is None:
            raise ValueError("without_conversations must be provided when require_without=True.")
        without_payload = payload.get("without_image", {})
        without_sig_keys = [str(x) for x in without_payload.get("signature_keys", [])]
        without_sig_to_idx = {sig: idx for idx, sig in enumerate(without_sig_keys)}
        required_without_idx: List[int] = []
        for conv in without_conversations:
            sig_key = _signature_key_from_conv(conv)
            idx = without_sig_to_idx.get(sig_key)
            if idx is None:
                raise RuntimeError(
                    f"{family.upper()} all-examples cache missing required without-image conversation signature. "
                    f"cache_path={resolved_path}"
                )
            required_without_idx.append(int(idx))

        without_source_features = without_payload.get("layer_features", {})
        for key in requested_keys:
            if key not in without_source_features:
                raise RuntimeError(
                    f"Feature '{key}' missing from {family.upper()} without-image cache at {resolved_path}"
                )
            arr = np.asarray(without_source_features[key], dtype=np.float32)
            without_out[key] = arr[np.asarray(required_without_idx, dtype=np.int64)]

    return with_out, without_out, requested_keys, resolved_path


def _load_qwen_preextracted_contrastive_subset(
    conversations: Sequence[List[Dict]],
    include_attention_probes: bool,
    include_mlp_probes: bool,
    requested_model_path: str,
    include_residual_probes: bool = True,
    llm_feature_strategies: Optional[Sequence[str]] = None,
    use_additional_feature_preextract_cache: Optional[bool] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], List[str], Path]:
    return _load_preextracted_contrastive_subset(
        vlm_key="qwen3_vl_32b_instruct",
        conversations=conversations,
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        llm_feature_strategies=llm_feature_strategies,
        use_additional_feature_preextract_cache=use_additional_feature_preextract_cache,
        requested_model_path=requested_model_path,
        cache_path=cache_path,
    )


def _load_qwen_preextracted_all_examples_subset(
    with_conversations: Sequence[List[Dict]],
    without_conversations: Optional[Sequence[List[Dict]]],
    require_without: bool,
    include_attention_probes: bool,
    include_mlp_probes: bool,
    requested_model_path: str,
    include_residual_probes: bool = True,
    llm_feature_strategies: Optional[Sequence[str]] = None,
    use_additional_feature_preextract_cache: Optional[bool] = None,
    cache_path: Optional[Path] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[str], Path]:
    return _load_preextracted_all_examples_subset(
        vlm_key="qwen3_vl_32b_instruct",
        with_conversations=with_conversations,
        without_conversations=without_conversations,
        require_without=bool(require_without),
        include_attention_probes=bool(include_attention_probes),
        include_mlp_probes=bool(include_mlp_probes),
        include_residual_probes=bool(include_residual_probes),
        llm_feature_strategies=llm_feature_strategies,
        use_additional_feature_preextract_cache=use_additional_feature_preextract_cache,
        requested_model_path=requested_model_path,
        cache_path=cache_path,
    )


def _build_holdout_pool(
    responses: List[Dict],
    seen_signatures: set,
    selected_benchmark: Optional[str],
    allowed_benchmarks: Sequence[str],
    include_short_response_filter: bool,
    image_lookup_uid: Dict[Tuple[str, str], List[bytes]],
    image_lookup_qid: Dict[Tuple[str, str], List[bytes]],
) -> Tuple[List[Dict], List[Dict], int]:
    pool_true = []
    pool_false = []
    seen_pool_signatures = set()
    skipped_short_response_count = 0

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
        sig = _conversation_signature_from_response_row(row)
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
        if not _norm_text(with_image_response):
            continue
        if include_short_response_filter:
            if core._count_tokens(with_image_response) < MIN_RESPONSE_TOKENS:
                skipped_short_response_count += 1
                continue

        conv = core._make_vllm_messages(
            prompt_text=row.get("prompt_text", ""),
            image_bytes_list=imgs,
            system_prompt=row.get("system_prompt", ""),
        )
        conv.append({"role": "assistant", "content": with_image_response})
        item = {
            "dataset": ds,
            "conversation": conv,
            "mirage_like": bool(mirage_like),
        }
        if mirage_like is True:
            pool_true.append(item)
        else:
            pool_false.append(item)
        seen_pool_signatures.add(sig)
    return pool_true, pool_false, int(skipped_short_response_count)


def _filter_holdout_pool_to_available_signatures(
    pool_true: Sequence[Dict],
    pool_false: Sequence[Dict],
    available_signature_keys: set[str],
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    malformed_examples: List[Dict[str, Any]] = []
    filtered_true: List[Dict] = []
    for item in pool_true:
        sig_key = _signature_key_from_holdout_item(item)
        if sig_key is None:
            malformed_examples.append(
                {
                    "class": "mirage_true",
                    "dataset": str(item.get("dataset", "unknown")),
                    "keys": sorted([str(k) for k in item.keys()]),
                }
            )
            continue
        if sig_key in available_signature_keys:
            filtered_true.append(item)
    filtered_false: List[Dict] = []
    for item in pool_false:
        sig_key = _signature_key_from_holdout_item(item)
        if sig_key is None:
            malformed_examples.append(
                {
                    "class": "mirage_false",
                    "dataset": str(item.get("dataset", "unknown")),
                    "keys": sorted([str(k) for k in item.keys()]),
                }
            )
            continue
        if sig_key in available_signature_keys:
            filtered_false.append(item)
    if malformed_examples:
        preview = malformed_examples[:5]
        raise RuntimeError(
            "Malformed holdout pool items: missing required with-image conversation key "
            "('conversation' or 'with_conversation'). "
            f"count={len(malformed_examples)} sample={preview}"
        )
    stats = {
        "mirage_true_before": int(len(pool_true)),
        "mirage_false_before": int(len(pool_false)),
        "mirage_true_after": int(len(filtered_true)),
        "mirage_false_after": int(len(filtered_false)),
        "mirage_true_dropped": int(len(pool_true) - len(filtered_true)),
        "mirage_false_dropped": int(len(pool_false) - len(filtered_false)),
    }
    return filtered_true, filtered_false, stats


def _holdout_pool_counts_by_benchmark(
    pool_true: Sequence[Dict],
    pool_false: Sequence[Dict],
) -> Dict[str, Dict[str, int]]:
    true_counts: Dict[str, int] = defaultdict(int)
    false_counts: Dict[str, int] = defaultdict(int)
    for item in pool_true:
        true_counts[str(item.get("dataset", "unknown"))] += 1
    for item in pool_false:
        false_counts[str(item.get("dataset", "unknown"))] += 1
    benchmarks = sorted(set(true_counts.keys()) | set(false_counts.keys()))
    out: Dict[str, Dict[str, int]] = {}
    for ds in benchmarks:
        t_ct = int(true_counts.get(ds, 0))
        f_ct = int(false_counts.get(ds, 0))
        out[str(ds)] = {
            "mirage_true": t_ct,
            "mirage_false": f_ct,
            "pair_cap": int(min(t_ct, f_ct)),
        }
    return out


def _plan_balanced_holdout_selection(
    pool_true: Sequence[Dict],
    pool_false: Sequence[Dict],
    requested_num_true: int,
    requested_num_false: int,
) -> Dict[str, Any]:
    counts_by_benchmark = _holdout_pool_counts_by_benchmark(pool_true=pool_true, pool_false=pool_false)
    represented = sorted(
        ds for ds, info in counts_by_benchmark.items()
        if int(info.get("mirage_true", 0)) > 0 and int(info.get("mirage_false", 0)) > 0
    )
    if not represented:
        raise RuntimeError(
            "No held-out benchmarks contain both mirage_true and mirage_false candidates after filtering."
        )

    pair_caps_by_benchmark = {
        str(ds): int(counts_by_benchmark[ds]["pair_cap"])
        for ds in represented
    }
    total_pair_cap = int(sum(pair_caps_by_benchmark.values()))
    if total_pair_cap <= 0:
        raise RuntimeError("Held-out pool has zero balanced pair capacity across represented benchmarks.")

    requested_true_i = max(0, int(requested_num_true))
    requested_false_i = max(0, int(requested_num_false))
    desired_per_class_balanced = int(min(requested_true_i, requested_false_i))
    selected_examples_per_class = int(min(desired_per_class_balanced, total_pair_cap))
    if selected_examples_per_class <= 0:
        raise RuntimeError(
            "Held-out balanced selection resolved to zero examples per class. "
            f"requested_true={requested_true_i}, requested_false={requested_false_i}, "
            f"total_pair_cap={total_pair_cap}"
        )

    selected_pairs_by_benchmark = {str(ds): 0 for ds in represented}
    raw_alloc = {
        str(ds): (
            float(selected_examples_per_class) * float(pair_caps_by_benchmark[ds]) / float(total_pair_cap)
        )
        for ds in represented
    }
    floor_alloc = {str(ds): int(np.floor(raw_alloc[ds])) for ds in represented}
    for ds in represented:
        selected_pairs_by_benchmark[ds] = int(min(pair_caps_by_benchmark[ds], floor_alloc[ds]))
    assigned = int(sum(selected_pairs_by_benchmark.values()))
    leftover = int(selected_examples_per_class - assigned)
    if leftover > 0:
        remainder_order = sorted(
            represented,
            key=lambda ds: (raw_alloc[ds] - floor_alloc[ds]),
            reverse=True,
        )
        for ds in remainder_order:
            if leftover <= 0:
                break
            if selected_pairs_by_benchmark[ds] < pair_caps_by_benchmark[ds]:
                selected_pairs_by_benchmark[ds] += 1
                leftover -= 1
    selected_examples_per_class = int(sum(selected_pairs_by_benchmark.values()))

    return {
        "selection_policy": "benchmark_cap_proportional_balanced_per_class",
        "requested_num_holdout_mirage_true": int(requested_true_i),
        "requested_num_holdout_mirage_false": int(requested_false_i),
        "requested_num_holdout_per_class_balanced": int(desired_per_class_balanced),
        "selected_num_holdout_per_class_balanced": int(selected_examples_per_class),
        "selected_num_holdout_total": int(2 * selected_examples_per_class),
        "num_classes": 2,
        "represented_benchmarks_with_both_classes": [str(ds) for ds in represented],
        "pair_caps_by_benchmark": {k: int(v) for k, v in pair_caps_by_benchmark.items()},
        "selected_pairs_by_benchmark": {k: int(v) for k, v in selected_pairs_by_benchmark.items()},
        "counts_by_benchmark_before_selection": counts_by_benchmark,
    }


def _select_holdout_examples_balanced_by_benchmark(
    pool_true: Sequence[Dict],
    pool_false: Sequence[Dict],
    selected_pairs_by_benchmark: Dict[str, int],
    seed: int,
) -> Tuple[List[Dict], Dict[str, Dict[str, int]]]:
    rng = random.Random(int(seed))
    true_by_benchmark: Dict[str, List[Dict]] = defaultdict(list)
    false_by_benchmark: Dict[str, List[Dict]] = defaultdict(list)
    for item in pool_true:
        true_by_benchmark[str(item.get("dataset", "unknown"))].append(item)
    for item in pool_false:
        false_by_benchmark[str(item.get("dataset", "unknown"))].append(item)

    selected: List[Dict] = []
    selected_counts_by_benchmark: Dict[str, Dict[str, int]] = {}
    for ds in sorted(selected_pairs_by_benchmark.keys()):
        k = int(selected_pairs_by_benchmark[ds])
        if k <= 0:
            continue
        true_candidates = list(true_by_benchmark.get(ds, []))
        false_candidates = list(false_by_benchmark.get(ds, []))
        if len(true_candidates) < k or len(false_candidates) < k:
            raise RuntimeError(
                "Insufficient held-out candidates for balanced benchmark selection. "
                f"dataset={ds}, need_per_class={k}, "
                f"available_true={len(true_candidates)}, available_false={len(false_candidates)}"
            )
        chosen_true = true_candidates if len(true_candidates) == k else rng.sample(true_candidates, k)
        chosen_false = false_candidates if len(false_candidates) == k else rng.sample(false_candidates, k)
        selected.extend(chosen_true)
        selected.extend(chosen_false)
        selected_counts_by_benchmark[str(ds)] = {
            "mirage_true": int(len(chosen_true)),
            "mirage_false": int(len(chosen_false)),
        }
    rng.shuffle(selected)
    return selected, selected_counts_by_benchmark

def _normalize_glm_pil_image_for_extraction(
    image: Image.Image,
    max_edge: int = GLM_IMAGE_MAX_EDGE,
    min_edge: int = GLM_IMAGE_MIN_EDGE,
) -> Image.Image:
    width, height = image.size
    out = image
    if max(width, height) > int(max_edge):
        scale = float(max_edge) / float(max(width, height))
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        out = out.resize((new_width, new_height), resample=PIL_LANCZOS)
        width, height = out.size
    if min(width, height) < int(min_edge):
        padded_width = max(width, int(min_edge))
        padded_height = max(height, int(min_edge))
        canvas = Image.new("RGB", (padded_width, padded_height), color=(0, 0, 0))
        paste_xy = ((padded_width - width) // 2, (padded_height - height) // 2)
        canvas.paste(out, paste_xy)
        out = canvas
    return out


def _decode_data_url_to_pil(url: str, model_key: Optional[str] = None) -> Image.Image:
    if not url.startswith("data:"):
        raise ValueError("Only data URLs are supported in contrastive artifact.")
    parts = url.split(",", 1)
    if len(parts) != 2:
        raise ValueError("Malformed data URL.")
    payload = parts[1]
    image_bytes = base64.b64decode(payload)
    with Image.open(io.BytesIO(image_bytes)) as img:
        # Some datasets include palette+transparency images; convert safely.
        if str(getattr(img, "mode", "")).upper() == "P" and "transparency" in getattr(img, "info", {}):
            img = img.convert("RGBA")
        decoded = img.convert("RGB")
    resolved_model_key = str(model_key or "").strip().lower()
    if resolved_model_key == "glm_4_6v_flash":
        decoded = _normalize_glm_pil_image_for_extraction(
            decoded,
            max_edge=int(GLM_IMAGE_MAX_EDGE),
            min_edge=int(GLM_IMAGE_MIN_EDGE),
        )
    return decoded


def _to_model_messages(conversation: List[Dict], model_key: Optional[str] = None) -> List[Dict]:
    messages: List[Dict] = []
    resolved_model_key = str(model_key or "").strip().lower()
    for msg in conversation:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if isinstance(content, list):
            converted: List[Dict] = []
            for item in content:
                item_type = item.get("type")
                if item_type == "text":
                    converted.append({"type": "text", "text": item.get("text", "")})
                elif item_type == "image_url":
                    url = (item.get("image_url") or {}).get("url", "")
                    converted.append({"type": "image", "image": _decode_data_url_to_pil(url, model_key=resolved_model_key)})
                else:
                    raise ValueError(f"Unsupported content item type: {item_type}")
            messages.append({"role": role, "content": converted})
            continue

        raise ValueError(f"Unsupported message content type: {type(content)}")

    return messages


def _to_ovis_messages(conversation: List[Dict]) -> List[Dict]:
    # Backward-compatible alias used by other scripts.
    return _to_model_messages(conversation=conversation, model_key="ovis")


def _resolve_model_name_or_path(model) -> str:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("_name_or_path", "name_or_path"):
            value = str(getattr(cfg, attr, "") or "").strip()
            if value:
                return value
    for attr in ("name_or_path", "model_name_or_path"):
        value = str(getattr(model, attr, "") or "").strip()
        if value:
            return value
    return ""


def _infer_vlm_key_from_model(model) -> str:
    # Ovis exposes a dedicated preprocess_inputs method and nested llm/visual_tokenizer.
    if hasattr(model, "preprocess_inputs") and hasattr(model, "llm") and hasattr(model, "visual_tokenizer"):
        return "ovis"

    name = _resolve_model_name_or_path(model).lower()
    if "qwen3-vl" in name or "qwen3_vl" in name:
        return "qwen3_vl_32b_instruct"
    if "glm-4.6v" in name or "glm_4_6v" in name:
        return "glm_4_6v_flash"
    return DEFAULT_VLM


def _ensure_processor_chat_template(processor, model_key: str, model_path: str) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return

    template = getattr(tokenizer, "chat_template", None)
    if template:
        return

    path_obj = Path(str(model_path))
    if not path_obj.exists():
        return

    if model_key == "glm_4_6v_flash":
        template_path = path_obj / "chat_template.jinja"
        if template_path.exists():
            tokenizer.chat_template = template_path.read_text(encoding="utf-8")
            return

    if model_key == "qwen3_vl_32b_instruct":
        template_path = path_obj / "chat_template.json"
        if template_path.exists():
            payload = json.loads(template_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("chat_template"):
                tokenizer.chat_template = str(payload["chat_template"])


def _get_processor_for_model(model, model_key: str):
    model_path = _resolve_model_name_or_path(model)
    cache_key = f"{model_key}::{model_path}"
    if cache_key in _PROCESSOR_CACHE:
        return _PROCESSOR_CACHE[cache_key]

    if not model_path:
        raise RuntimeError("Could not determine model name/path for processor loading.")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    _ensure_processor_chat_template(processor=processor, model_key=model_key, model_path=model_path)
    _PROCESSOR_CACHE[cache_key] = processor
    return processor


def _resolve_special_ids(model, model_key: str) -> set:
    text_tokenizer = getattr(model, "text_tokenizer", None)
    if text_tokenizer is not None and hasattr(text_tokenizer, "all_special_ids"):
        return set(int(x) for x in text_tokenizer.all_special_ids)

    if model_key == "ovis":
        return set()

    processor = _get_processor_for_model(model=model, model_key=model_key)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "all_special_ids"):
        return set(int(x) for x in tokenizer.all_special_ids)
    return set()


def _prepare_inputs(model, messages: List[Dict], model_key: Optional[str] = None) -> Dict[str, torch.Tensor]:
    resolved_model_key = str(model_key or _infer_vlm_key_from_model(model)).strip().lower()
    input_device = _resolve_model_input_device(model)
    if resolved_model_key == "ovis":
        input_ids, pixel_values, grid_thws = model.preprocess_inputs(
            messages=messages,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if input_ids is None:
            raise RuntimeError("Ovis preprocess_inputs returned None input_ids.")
        input_ids = input_ids.to(device=input_device)
        pixel_values = pixel_values.to(device=input_device) if pixel_values is not None else None
        grid_thws = grid_thws.to(device=input_device) if grid_thws is not None else None
        attention_mask = torch.ne(input_ids, model.text_tokenizer.pad_token_id).to(device=input_ids.device)
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "grid_thws": grid_thws,
            "attention_mask": attention_mask,
        }

    processor = _get_processor_for_model(model=model, model_key=resolved_model_key)
    chat_template_kwargs: Dict[str, Any] = {
        "add_generation_prompt": False,
        "tokenize": False,
        # Explicitly disable reasoning/thinking across supported models.
        "enable_thinking": False,
    }
    chat_text = processor.apply_chat_template(messages, **chat_template_kwargs)

    images: List[Image.Image] = []
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if str(item.get("type", "")) == "image":
                image_obj = item.get("image")
                if image_obj is not None:
                    images.append(image_obj)

    proc_kwargs: Dict[str, Any] = {"text": [chat_text], "return_tensors": "pt"}
    if images:
        proc_kwargs["images"] = images
    batch = processor(**proc_kwargs)

    device = input_device
    inputs: Dict[str, Any] = {}
    for key, value in dict(batch).items():
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
        else:
            inputs[key] = value

    if "attention_mask" not in inputs and "input_ids" in inputs:
        tokenizer = getattr(processor, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad_id is None:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=inputs["input_ids"].device)
        else:
            inputs["attention_mask"] = (inputs["input_ids"] != int(pad_id)).to(device=inputs["input_ids"].device)
    return inputs


def _force_attention_backend(model, backend: str) -> None:
    # Propagate desired backend to all known config objects.
    configs = []
    if hasattr(model, "config"):
        configs.append(model.config)
    if hasattr(model, "llm") and hasattr(model.llm, "config"):
        configs.append(model.llm.config)
    if hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "config"):
        configs.append(model.llm.model.config)

    for cfg in configs:
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = backend
        if hasattr(cfg, "attn_implementation"):
            cfg.attn_implementation = backend


def _mean_or_zero(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.device != hidden.device:
        mask = mask.to(device=hidden.device)
    if mask.dtype != torch.bool:
        mask = mask.to(dtype=torch.bool)
    idx = torch.where(mask)[0]
    if idx.numel() == 0:
        return torch.zeros(hidden.shape[-1], dtype=hidden.dtype, device=hidden.device)
    return hidden[idx].mean(dim=0)


def _compute_llm_strategy_features(
    hidden_states: torch.Tensor,
    input_ids_1d: torch.Tensor,
    special_ids: set,
) -> Dict[str, torch.Tensor]:
    # hidden_states: [seq, hidden]
    token_ids = input_ids_1d
    if token_ids.device != hidden_states.device:
        token_ids = token_ids.to(device=hidden_states.device, non_blocking=True)
    if token_ids.shape[0] != hidden_states.shape[0]:
        common_len = int(min(int(token_ids.shape[0]), int(hidden_states.shape[0])))
        token_ids = token_ids[:common_len]
        hidden_states = hidden_states[:common_len]

    visual_mask = token_ids == -300
    text_mask = token_ids >= 0
    for sid in special_ids:
        text_mask = text_mask & (token_ids != sid)
    all_nonspecial_mask = text_mask | visual_mask

    visual_mean = _mean_or_zero(hidden_states, visual_mask)
    last_token = hidden_states[-1]
    text_mean = _mean_or_zero(hidden_states, text_mask)
    all_nonspecial_mean = _mean_or_zero(hidden_states, all_nonspecial_mask)

    return {
        "current_visual_plus_last": torch.cat([visual_mean, last_token], dim=0),
        "text_nonspecial_mean": text_mean,
        "all_nonspecial_mean": all_nonspecial_mean,
    }


def _resolve_num_attention_heads(attn_module, model) -> Optional[int]:
    for key in (
        "num_heads",
        "num_attention_heads",
        "n_heads",
        "n_head",
        "num_query_heads",
    ):
        v = getattr(attn_module, key, None)
        if v is not None:
            try:
                n = int(v)
                if n > 0:
                    return n
            except (TypeError, ValueError):
                pass

    cfgs = []
    if hasattr(model, "config"):
        cfgs.append(model.config)
    if hasattr(model, "llm") and hasattr(model.llm, "config"):
        cfgs.append(model.llm.config)
    if hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "config"):
        cfgs.append(model.llm.model.config)
    for cfg in cfgs:
        for key in (
            "num_attention_heads",
            "num_heads",
            "n_head",
            "n_heads",
            "num_query_heads",
        ):
            v = getattr(cfg, key, None)
            if v is not None:
                try:
                    n = int(v)
                    if n > 0:
                        return n
                except (TypeError, ValueError):
                    pass

    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is not None:
        try:
            head_dim = int(head_dim)
        except (TypeError, ValueError):
            head_dim = None
    if head_dim and head_dim > 0:
        q_proj = getattr(attn_module, "q_proj", None)
        if hasattr(q_proj, "out_features"):
            q_out = int(q_proj.out_features)
            if q_out % head_dim == 0:
                return q_out // head_dim
        if isinstance(q_proj, torch.nn.Module) and hasattr(q_proj, "weight"):
            q_out = int(q_proj.weight.shape[0])
            if q_out % head_dim == 0:
                return q_out // head_dim

        hidden_size = getattr(attn_module, "hidden_size", None)
        if hidden_size is None and hasattr(model, "llm") and hasattr(model.llm, "model") and hasattr(model.llm.model, "config"):
            hidden_size = getattr(model.llm.model.config, "hidden_size", None)
        if hidden_size is not None:
            try:
                hidden_size = int(hidden_size)
                if hidden_size > 0 and hidden_size % head_dim == 0:
                    return hidden_size // head_dim
            except (TypeError, ValueError):
                pass

    return None


def _resolve_attention_output_projection(attn_module) -> Optional[torch.nn.Module]:
    for name in ("o_proj", "out_proj", "dense", "proj"):
        mod = getattr(attn_module, name, None)
        if isinstance(mod, torch.nn.Module):
            return mod
    return None


def _get_nested_attr(obj: Any, path: Sequence[str]) -> Any:
    out = obj
    for part in path:
        if not hasattr(out, part):
            return None
        out = getattr(out, part)
    return out


def _resolve_llm_layer_modules(model) -> Tuple[List[Any], Optional[Any], str]:
    layer_paths = [
        ("llm", "model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "layers"),
        ("transformer", "layers"),
        ("transformer", "h"),
        ("layers",),
    ]
    for path in layer_paths:
        layers = _get_nested_attr(model, path)
        if layers is None:
            continue
        try:
            layer_list = list(layers)
        except TypeError:
            continue
        if not layer_list:
            continue
        if not hasattr(layer_list[0], "register_forward_hook"):
            continue
        parent = _get_nested_attr(model, path[:-1]) if len(path) > 1 else None
        return layer_list, parent, ".".join(path)
    raise RuntimeError("Could not locate transformer layer modules for hook registration.")


def _run_model_forward(model, inputs: Dict[str, Any], model_key: str) -> None:
    kwargs = dict(inputs)
    kwargs.setdefault("output_attentions", False)
    kwargs.setdefault("return_dict", True)

    # Ovis uses dedicated multimodal kwargs.
    if model_key == "ovis":
        preferred = {
            "input_ids": kwargs.get("input_ids"),
            "attention_mask": kwargs.get("attention_mask"),
            "pixel_values": kwargs.get("pixel_values"),
            "grid_thws": kwargs.get("grid_thws"),
            "return_dict": kwargs.get("return_dict", True),
        }
        with torch.inference_mode():
            model.forward(**preferred)
        return

    try:
        with torch.inference_mode():
            model.forward(**kwargs)
        return
    except TypeError:
        pass

    sig = inspect.signature(model.forward)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var_kwargs:
        with torch.inference_mode():
            model.forward(**kwargs)
        return

    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    with torch.inference_mode():
        model.forward(**filtered)


def _extract_sample_features_only(
    model,
    messages: List[Dict],
    include_additional_attention_mlp_probes: bool = False,
    include_attention_probes: bool = False,
    include_mlp_probes: bool = False,
    include_residual_probes: bool = True,
    model_key: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Extract residual features, optionally plus attention-head/post-attention/MLP families."""
    resolved_model_key = str(model_key or _infer_vlm_key_from_model(model)).strip().lower()
    inputs = _prepare_inputs(model=model, messages=messages, model_key=resolved_model_key)
    input_ids_1d = inputs["input_ids"][0]
    special_ids = _resolve_special_ids(model=model, model_key=resolved_model_key)

    collected: Dict[str, torch.Tensor] = {}
    handles = []
    include_attention = bool(include_attention_probes or include_additional_attention_mlp_probes)
    include_mlp = bool(include_mlp_probes or include_additional_attention_mlp_probes)

    def _make_vision_hook(name: str):
        def hook(_module, _inp, out):
            x = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if isinstance(x, tuple):
                x = x[0]
            if x.dim() == 2:
                feat = x.mean(dim=0)
            elif x.dim() == 3:
                feat = x.mean(dim=(0, 1))
            else:
                feat = x.reshape(-1).mean().unsqueeze(0)
            collected[name] = feat.detach()
        return hook

    def _make_projector_hook(name: str):
        def hook(_module, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            if x.dim() == 2:
                feat = x.mean(dim=0)
            elif x.dim() == 3:
                feat = x.mean(dim=(0, 1))
            else:
                feat = x.reshape(-1).mean().unsqueeze(0)
            collected[name] = feat.detach()
        return hook

    def _make_llm_hook(base_name: str):
        def hook(_module, _inp, out):
            x = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
            if isinstance(x, tuple):
                x = x[0]
            if x.dim() != 3:
                return
            hidden = x[0]
            feats = _compute_llm_strategy_features(hidden, input_ids_1d, special_ids)
            for strategy, vec in feats.items():
                collected[f"{base_name}__{strategy}"] = vec.detach()
        return hook

    def _make_post_attention_hook(base_name: str):
        def hook(_module, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            x = x.last_hidden_state if hasattr(x, "last_hidden_state") else x
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return
            hidden = x[0]
            feats = _compute_llm_strategy_features(hidden, input_ids_1d, special_ids)
            for strategy, vec in feats.items():
                collected[f"{base_name}/post_attention__{strategy}"] = vec.detach()
        return hook

    def _make_mlp_hook(base_name: str):
        def hook(_module, _inp, out):
            x = out[0] if isinstance(out, tuple) else out
            x = x.last_hidden_state if hasattr(x, "last_hidden_state") else x
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return
            hidden = x[0]
            feats = _compute_llm_strategy_features(hidden, input_ids_1d, special_ids)
            for strategy, vec in feats.items():
                collected[f"{base_name}/mlp__{strategy}"] = vec.detach()
        return hook

    def _make_attention_head_hook(base_name: str, num_heads: int):
        def hook(_module, inp, _out):
            if not inp:
                return
            x = inp[0]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                return
            hidden = x[0]
            hidden_size = int(hidden.shape[-1])
            if hidden_size % int(num_heads) != 0:
                raise RuntimeError(
                    f"Cannot split attention projection input for {base_name}: "
                    f"hidden_size={hidden_size}, num_heads={num_heads}"
                )
            head_dim = hidden_size // int(num_heads)
            for head_idx in range(int(num_heads)):
                hs = head_idx * head_dim
                he = (head_idx + 1) * head_dim
                head_hidden = hidden[:, hs:he]
                feats = _compute_llm_strategy_features(head_hidden, input_ids_1d, special_ids)
                for strategy, vec in feats.items():
                    collected[f"{base_name}/attention_head_{head_idx + 1}__{strategy}"] = vec.detach()
        return hook

    try:
        if resolved_model_key == "ovis":
            if bool(include_residual_probes):
                for layer in range(len(model.visual_tokenizer.vit.vision_model.encoder.layers)):
                    h = model.visual_tokenizer.vit.vision_model.encoder.layers[layer].register_forward_hook(
                        _make_vision_hook(f"vision_encoder/layer_{layer + 1}")
                    )
                    handles.append(h)

                handles.append(
                    model.visual_tokenizer.vit.vision_model.register_forward_hook(
                        _make_vision_hook("vision_encoder/post_layer_norm")
                    )
                )
                handles.append(model.vte.register_forward_hook(_make_projector_hook("projector")))

            llm_layers = list(model.llm.model.layers)
            post_norm_module = model.llm.model
        else:
            llm_layers, post_norm_module, _ = _resolve_llm_layer_modules(model)

        for layer_idx, layer_module in enumerate(llm_layers):
            layer_name = f"language_model/layer_{layer_idx + 1}"

            if bool(include_residual_probes):
                handles.append(layer_module.register_forward_hook(_make_llm_hook(layer_name)))

            if include_attention and hasattr(layer_module, "self_attn"):
                attn_module = layer_module.self_attn
                num_heads = _resolve_num_attention_heads(attn_module=attn_module, model=model)
                if num_heads is not None:
                    handles.append(attn_module.register_forward_hook(_make_post_attention_hook(layer_name)))
                    proj_module = _resolve_attention_output_projection(attn_module)
                    if proj_module is not None:
                        handles.append(
                            proj_module.register_forward_hook(_make_attention_head_hook(layer_name, int(num_heads)))
                        )

            if include_mlp and hasattr(layer_module, "mlp"):
                handles.append(layer_module.mlp.register_forward_hook(_make_mlp_hook(layer_name)))

        if bool(include_residual_probes) and post_norm_module is not None and hasattr(post_norm_module, "register_forward_hook"):
            handles.append(post_norm_module.register_forward_hook(_make_llm_hook("language_model/post_layer_norm")))

        _run_model_forward(model=model, inputs=inputs, model_key=resolved_model_key)
    finally:
        for h in handles:
            h.remove()

    return collected


def _layer_sort_key(name: str) -> Tuple[int, int, int, int, str]:
    base = name.split("__")[0]
    if base == "projector":
        return (1, 0, name)
    if base.startswith("vision_encoder/"):
        comp = 0
    elif base.startswith("language_model/"):
        comp = 2
    else:
        comp = 3

    m = re.search(r"layer_(\d+)", base)
    if m:
        layer_num = int(m.group(1))
    elif base == "language_model/post_layer_norm":
        layer_num = 10**9
    else:
        layer_num = 10**8

    sub_rank = 0
    head_num = 0
    if "/post_attention" in base:
        sub_rank = 1
    elif "/attention_head_" in base:
        sub_rank = 2
        mh = re.search(r"/attention_head_(\d+)$", base)
        if mh:
            head_num = int(mh.group(1))
    elif "/mlp" in base:
        sub_rank = 3
    elif "post_layer_norm" in base:
        sub_rank = 4

    return (comp, layer_num, sub_rank, head_num, name)


def _is_attention_probe_feature(feature_name: str) -> bool:
    base = str(feature_name).split("__")[0]
    return ("/attention_head_" in base) or ("/post_attention" in base)


def _is_mlp_probe_feature(feature_name: str) -> bool:
    base = str(feature_name).split("__")[0]
    return "/mlp" in base


def _is_additional_attention_mlp_feature(feature_name: str) -> bool:
    return _is_attention_probe_feature(feature_name) or _is_mlp_probe_feature(feature_name)


def _feature_base_name(feature_name: str) -> str:
    return str(feature_name).split("__")[0]


def _best_feature_by_base_from_validation(
    per_feature_results: Dict[str, Dict],
) -> Dict[str, Dict[str, Any]]:
    best_by_base: Dict[str, Dict[str, Any]] = {}
    for feature_name, info in per_feature_results.items():
        base_name = _feature_base_name(feature_name)
        score = float(info.get("mean_validation_accuracy_at_best_c", float("-inf")))
        current = best_by_base.get(base_name)
        if (
            current is None
            or score > float(current.get("best_validation_accuracy", float("-inf")))
            or (
                score == float(current.get("best_validation_accuracy", float("-inf")))
                and str(feature_name) < str(current.get("best_feature", ""))
            )
        ):
            best_by_base[base_name] = {
                "best_feature": str(feature_name),
                "best_validation_accuracy": float(score),
            }
    return best_by_base


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
    preproc, X_train_s = _fit_feature_preprocessor(
        X_train=X_train,
        normalize_features=normalize_features,
        pca_components=pca_components,
    )
    X_val_s = _apply_feature_preprocessor(X_val, preproc)
    X_eval_s = _apply_feature_preprocessor(X_eval, preproc)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = torch.nn.Linear(X_train_s.shape[1], 1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    weight_decay = 0.0 if c_value == 0.0 else (1.0 / c_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_t = torch.from_numpy(X_train_s).to(device=device, dtype=torch.float32)
    y_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32).unsqueeze(-1)
    x_val_t = torch.from_numpy(X_val_s).to(device=device, dtype=torch.float32)
    y_val_t = torch.from_numpy(y_val).to(device=device, dtype=torch.float32).unsqueeze(-1)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    model.train()
    for epoch in range(int(epochs)):
        logits = model(x_t)
        loss = loss_fn(logits, y_t)
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
        x_eval_t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            logits = model(x_eval_t).squeeze(-1)
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
            init_seed = _probe_init_seed(
                split_seed=int(split_seed),
                c_value=float(c),
                init_idx=int(init_idx),
            )
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
            if y_test.size > 0:
                test_acc = float((test_pred == y_test).mean())
                class0_mask = y_test == 0
                class1_mask = y_test == 1
                class0_test_acc = float((test_pred[class0_mask] == y_test[class0_mask]).mean()) if class0_mask.any() else None
                class1_test_acc = float((test_pred[class1_mask] == y_test[class1_mask]).mean()) if class1_mask.any() else None
                benchmark_test_acc, benchmark_class0_test_acc, benchmark_class1_test_acc = _compute_benchmark_test_metrics(
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

            init_row = {
                "init_index": int(init_idx),
                "init_seed": int(init_seed),
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
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
                or (val_acc > best_init["validation_accuracy"])
                or (
                    val_acc == best_init["validation_accuracy"]
                    and train_acc > best_init["train_accuracy"]
                )
                or (
                    val_acc == best_init["validation_accuracy"]
                    and train_acc == best_init["train_accuracy"]
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
            "validation_accuracy": float(best_init["validation_accuracy"]),
            "test_accuracy": (float(best_init["test_accuracy"]) if best_init["test_accuracy"] is not None else None),
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
            or (float(best_init["validation_accuracy"]) > best["validation_accuracy_at_best_c"])
            or (
                float(best_init["validation_accuracy"]) == best["validation_accuracy_at_best_c"]
                and float(c) < float(best["best_c"])
            )
        ):
            best = {
                "best_c": float(c),
                "best_train_accuracy": float(best_init["train_accuracy"]),
                "validation_accuracy_at_best_c": float(best_init["validation_accuracy"]),
                "test_accuracy_at_best_c": (
                    float(best_init["test_accuracy"]) if best_init["test_accuracy"] is not None else None
                ),
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
        "selection_metric": "validation_accuracy",
        **best,
        "sweep": sweep,
    }


def _fit_fixed_c_with_multi_init(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_eval: np.ndarray,
    split_seed: int,
    c_value: float,
    args: argparse.Namespace,
) -> Dict:
    num_inits = int(args.probe_num_initializations) if bool(args.multi_init_probe_selection) else 1
    if num_inits < 1:
        raise ValueError("--probe_num_initializations must be >= 1.")

    best = None
    for init_idx in range(num_inits):
        init_seed = _probe_init_seed(
            split_seed=int(split_seed),
            c_value=float(c_value),
            init_idx=int(init_idx),
        )
        out = _fit_probe_with_early_stopping(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_eval=X_eval,
            seed=init_seed,
            epochs=int(args.probe_epochs),
            lr=float(args.probe_lr),
            c_value=float(c_value),
            normalize_features=bool(args.normalize_features),
            pca_components=int(args.pca_components),
            early_stopping_patience=int(args.early_stopping_patience),
            early_stopping_min_delta=float(args.early_stopping_min_delta),
        )
        train_pred = out["train_pred"]
        val_pred = out["val_pred"]
        eval_pred = out["eval_pred"]
        train_acc = float((train_pred == y_train).mean())
        val_acc = float((val_pred == y_val).mean())

        candidate = {
            "init_index": int(init_idx),
            "init_seed": int(init_seed),
            "train_pred": train_pred,
            "val_pred": val_pred,
            "eval_pred": eval_pred,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "best_epoch": int(out["best_epoch"]),
            "best_val_loss": float(out["best_val_loss"]),
        }
        if (
            best is None
            or (val_acc > best["validation_accuracy"])
            or (
                val_acc == best["validation_accuracy"]
                and train_acc > best["train_accuracy"]
            )
            or (
                val_acc == best["validation_accuracy"]
                and train_acc == best["train_accuracy"]
                and int(init_idx) < int(best["init_index"])
            )
        ):
            best = candidate
    return best


def _fit_feature_preprocessor(
    X_train: np.ndarray,
    normalize_features: bool,
    pca_components: int,
) -> Tuple[Dict, np.ndarray]:
    X = np.asarray(X_train, dtype=np.float32)
    preproc: Dict = {
        "normalize_features": bool(normalize_features),
        "pca_components": int(max(0, pca_components)),
        "mean": None,
        "std": None,
        "pca_mean": None,
        "pca_components_matrix": None,
    }

    if normalize_features:
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        X = (X - mean) / std
        preproc["mean"] = mean
        preproc["std"] = std

    if int(pca_components) > 0:
        k = min(int(pca_components), int(X.shape[0]), int(X.shape[1]))
        if k >= 1:
            pca_mean = X.mean(axis=0, keepdims=True)
            X_centered = X - pca_mean
            # Full SVD is fine at this dataset scale; no sklearn dependency required.
            _u, _s, vt = np.linalg.svd(X_centered, full_matrices=False)
            components = vt[:k].T.astype(np.float32, copy=False)  # [d, k]
            X = X_centered @ components
            preproc["pca_mean"] = pca_mean
            preproc["pca_components_matrix"] = components

    return preproc, X


def _apply_feature_preprocessor(X: np.ndarray, preproc: Dict) -> np.ndarray:
    out = np.asarray(X, dtype=np.float32)
    if preproc.get("normalize_features", False):
        mean = preproc.get("mean")
        std = preproc.get("std")
        out = (out - mean) / std

    components = preproc.get("pca_components_matrix")
    if components is not None:
        pca_mean = preproc.get("pca_mean")
        out = (out - pca_mean) @ components
    return out


def _compute_benchmark_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    benchmark_labels: Optional[Sequence[str]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    if benchmark_labels is None:
        return {}, {}, {}
    if len(benchmark_labels) != int(len(y_true)):
        raise ValueError(
            "benchmark_labels length must match y_true length. "
            f"Got {len(benchmark_labels)} vs {len(y_true)}."
        )

    by_benchmark_mask: Dict[str, np.ndarray] = {}
    for idx, benchmark in enumerate(benchmark_labels):
        key = str(benchmark) if str(benchmark) else "unknown"
        if key not in by_benchmark_mask:
            by_benchmark_mask[key] = np.zeros(len(y_true), dtype=bool)
        by_benchmark_mask[key][idx] = True

    benchmark_test_accuracy: Dict[str, float] = {}
    benchmark_class0_test_accuracy: Dict[str, float] = {}
    benchmark_class1_test_accuracy: Dict[str, float] = {}
    for benchmark in sorted(by_benchmark_mask.keys()):
        bench_mask = by_benchmark_mask[benchmark]
        if not bench_mask.any():
            continue
        bench_true = y_true[bench_mask]
        bench_pred = y_pred[bench_mask]
        benchmark_test_accuracy[benchmark] = float((bench_pred == bench_true).mean())

        class0_mask = bench_true == 0
        if class0_mask.any():
            benchmark_class0_test_accuracy[benchmark] = float(
                (bench_pred[class0_mask] == bench_true[class0_mask]).mean()
            )

        class1_mask = bench_true == 1
        if class1_mask.any():
            benchmark_class1_test_accuracy[benchmark] = float(
                (bench_pred[class1_mask] == bench_true[class1_mask]).mean()
            )

    return benchmark_test_accuracy, benchmark_class0_test_accuracy, benchmark_class1_test_accuracy


def _mean_dict_metrics_from_seed_runs(seed_runs: Sequence[Dict], key: str) -> Dict[str, float]:
    values_by_benchmark: Dict[str, List[float]] = defaultdict(list)
    for run in seed_runs:
        if not isinstance(run, dict):
            continue
        per_benchmark = run.get(key)
        if not isinstance(per_benchmark, dict):
            continue
        for benchmark, value in per_benchmark.items():
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            values_by_benchmark[str(benchmark)].append(value_f)

    return {
        benchmark: float(np.mean(vals))
        for benchmark, vals in sorted(values_by_benchmark.items())
        if vals
    }


def _macro_average_metric_dict(metric_by_benchmark: Optional[Dict[str, float]]) -> Optional[float]:
    if not isinstance(metric_by_benchmark, dict):
        return None
    vals = [float(v) for v in metric_by_benchmark.values() if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _infer_benchmark_from_pair(pair: Dict) -> str:
    system_prompt = ""
    for key in ("non_mirage_conversation", "mirage_conversation"):
        conv = pair.get(key, [])
        if conv and isinstance(conv, list) and conv[0].get("role") == "system":
            system_prompt = str(conv[0].get("content", ""))
            break

    s = system_prompt.lower()
    if ("microvqa" in s) or ("microscopy image" in s):
        return "microvqa"
    if ("medxpertqa-mm" in s) or ("medical professional" in s):
        return "medxpertqa_mm"
    if ("mmmu-pro" in s) or ("multiple academic disciplines" in s):
        return "mmmu_pro"
    if ("vqa-rad" in s) or ("radiologist" in s) or ("radiology image" in s):
        return "vqa_rad"
    return "unknown"


def _split_pair_benchmark_stratified_validation(
    pair_ids: np.ndarray,
    pair_benchmarks: List[str],
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, List[int]]]:
    # Fixed validation policy:
    # - exactly 9 validation pairs total
    # - 3 random pairs per represented benchmark
    # - expected represented benchmark count is 3
    unique_pairs = sorted(set(int(x) for x in pair_ids.tolist()))
    if not unique_pairs:
        raise ValueError("No pairs available for splitting.")

    benchmark_to_pairs: Dict[str, List[int]] = defaultdict(list)
    for pid in unique_pairs:
        if pid < 0 or pid >= len(pair_benchmarks):
            raise ValueError(f"Pair id {pid} out of range for pair_benchmarks (len={len(pair_benchmarks)}).")
        benchmark_to_pairs[pair_benchmarks[pid]].append(pid)

    represented = sorted([b for b in benchmark_to_pairs.keys() if b != "unknown"])
    if len(represented) != 3:
        counts = {k: len(v) for k, v in benchmark_to_pairs.items()}
        raise ValueError(
            "Expected exactly 3 represented benchmarks in contrastive pairs, "
            f"found {len(represented)}: {represented}. Pair counts={counts}"
        )

    rng = np.random.default_rng(seed)
    selected_by_benchmark: Dict[str, List[int]] = {}
    test_pairs: List[int] = []
    for benchmark in represented:
        candidates = sorted(benchmark_to_pairs[benchmark])
        if len(candidates) < 3:
            raise ValueError(
                f"Benchmark '{benchmark}' has only {len(candidates)} pairs; need at least 3 "
                "for fixed validation split."
            )
        chosen = sorted(rng.choice(candidates, size=3, replace=False).tolist())
        selected_by_benchmark[benchmark] = chosen
        test_pairs.extend(chosen)

    test_pairs = sorted(test_pairs)
    test_pairs_set = set(test_pairs)
    test_mask = np.array([int(pid) in test_pairs_set for pid in pair_ids], dtype=bool)
    train_mask = ~test_mask
    return train_mask, test_mask, test_pairs, selected_by_benchmark


def _group_pair_ids_by_benchmark(pair_ids: List[int], pair_benchmarks: List[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for pid in pair_ids:
        if pid < 0 or pid >= len(pair_benchmarks):
            continue
        grouped[pair_benchmarks[pid]].append(int(pid))
    return {k: sorted(v) for k, v in grouped.items()}


def _split_pair_fixed_validation_count(
    pair_ids: np.ndarray,
    num_validation_pairs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    unique_pairs = sorted(set(int(x) for x in pair_ids.tolist()))
    if len(unique_pairs) < 2:
        raise ValueError(
            "Need at least 2 pairs to form train/validation split. "
            f"Got {len(unique_pairs)} pair(s)."
        )
    rng = np.random.default_rng(seed)
    n_val = min(max(1, int(num_validation_pairs)), len(unique_pairs) - 1)
    val_pairs = sorted(rng.choice(unique_pairs, size=n_val, replace=False).tolist())
    val_pairs_set = set(val_pairs)
    val_mask = np.array([int(pid) in val_pairs_set for pid in pair_ids], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask, val_pairs


def _split_pair_benchmark_fraction_validation(
    pair_ids: np.ndarray,
    pair_benchmarks: List[str],
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, List[int]]]:
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError(f"Expected 0 < val_fraction < 1. Got {val_fraction}.")

    unique_pairs = sorted(set(int(x) for x in pair_ids.tolist()))
    if not unique_pairs:
        raise ValueError("No pairs available for splitting.")

    benchmark_to_pairs: Dict[str, List[int]] = defaultdict(list)
    for pid in unique_pairs:
        if pid < 0 or pid >= len(pair_benchmarks):
            raise ValueError(f"Pair id {pid} out of range for pair_benchmarks (len={len(pair_benchmarks)}).")
        benchmark_to_pairs[pair_benchmarks[pid]].append(pid)

    represented = sorted(benchmark_to_pairs.keys())
    if not represented:
        raise ValueError("No benchmark buckets available for fraction-based split.")

    rng = np.random.default_rng(seed)
    selected_by_benchmark: Dict[str, List[int]] = {}
    val_pairs: List[int] = []
    for benchmark in represented:
        candidates = sorted(benchmark_to_pairs[benchmark])
        if len(candidates) < 2:
            raise ValueError(
                f"Benchmark '{benchmark}' has only {len(candidates)} pair(s); need at least 2 "
                "for a 90/10-style train/validation split."
            )
        n = len(candidates)
        n_val = max(1, int(round(n * float(val_fraction))))
        if n_val >= n:
            n_val = n - 1
        chosen = sorted(rng.choice(candidates, size=n_val, replace=False).tolist())
        selected_by_benchmark[benchmark] = chosen
        val_pairs.extend(chosen)

    val_pairs = sorted(val_pairs)
    val_pairs_set = set(val_pairs)
    val_mask = np.array([int(pid) in val_pairs_set for pid in pair_ids], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask, val_pairs, selected_by_benchmark


def _split_pair_holdout(
    pair_ids: np.ndarray,
    num_test_pairs: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Backward-compatible wrapper for older scripts.

    The previous API used `num_test_pairs`; this is now deprecated and ignored.
    The split is fixed to benchmark-stratified validation (3 pairs per represented benchmark).
    """
    unique_pairs = sorted(set(int(x) for x in pair_ids.tolist()))
    if not unique_pairs:
        raise ValueError("No pairs available for splitting.")

    # Best-effort: recover pair-level benchmark labels from default artifact and use the
    # benchmark-stratified split.
    default_pairs_path = _resolve_model_scoped_artifact_path(
        base_path=Path("./tmp_artifacts/contrastive_conversation_pairs.json"),
        vlm_key=DEFAULT_VLM,
        include_gen_prefix_dirs=True,
    )
    if default_pairs_path.exists():
        try:
            with open(default_pairs_path, "r", encoding="utf-8") as f:
                default_pairs = json.load(f)
            if isinstance(default_pairs, list):
                max_pid = max(unique_pairs)
                if max_pid < len(default_pairs):
                    pair_benchmarks = [_infer_benchmark_from_pair(p) for p in default_pairs]
                    train_mask, val_mask, val_pairs, _ = _split_pair_benchmark_stratified_validation(
                        pair_ids=pair_ids,
                        pair_benchmarks=pair_benchmarks,
                        seed=seed,
                    )
                    return train_mask, val_mask, val_pairs
        except Exception:
            pass

    # Legacy fallback: random pair holdout.
    rng = np.random.default_rng(seed)
    n_val = min(max(1, num_test_pairs), len(unique_pairs) - 1) if len(unique_pairs) > 1 else 1
    val_pairs = sorted(rng.choice(unique_pairs, size=n_val, replace=False).tolist())
    val_set = set(val_pairs)
    val_mask = np.array([int(pid) in val_set for pid in pair_ids], dtype=bool)
    train_mask = ~val_mask
    return train_mask, val_mask, val_pairs


def _attention_difference_threshold_classifier(
    attention_scores_by_sample: Dict[str, Dict[str, Dict[str, float]]],
    sample_names: List[str],
    y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> Dict[str, Dict]:
    layer_names = sorted(
        {layer for sample in attention_scores_by_sample.values() for layer in sample.keys()},
        key=lambda s: int(re.search(r"layer_(\d+)", s).group(1)),
    )

    results: Dict[str, Dict] = {}
    for layer in layer_names:
        train_non = []
        train_mir = []
        for i, name in enumerate(sample_names):
            if not train_mask[i]:
                continue
            metrics = attention_scores_by_sample.get(name, {}).get(layer)
            if metrics is None:
                continue
            ratio = float(metrics.get("image_focus_ratio", 0.0))
            if y[i] == 0:
                train_non.append(ratio)
            else:
                train_mir.append(ratio)

        mu_non = float(np.mean(train_non)) if train_non else 0.0
        mu_mir = float(np.mean(train_mir)) if train_mir else 0.0
        midpoint = 0.5 * (mu_non + mu_mir)

        def _predict(ratio: float) -> int:
            if mu_mir >= mu_non:
                return 1 if ratio >= midpoint else 0
            return 1 if ratio <= midpoint else 0

        train_preds = []
        train_true = []
        test_preds = []
        test_true = []
        for i, name in enumerate(sample_names):
            metrics = attention_scores_by_sample.get(name, {}).get(layer)
            if metrics is None:
                continue
            pred = _predict(float(metrics.get("image_focus_ratio", 0.0)))
            if train_mask[i]:
                train_preds.append(pred)
                train_true.append(int(y[i]))
            elif test_mask[i]:
                test_preds.append(pred)
                test_true.append(int(y[i]))

        train_acc = float((np.asarray(train_preds) == np.asarray(train_true)).mean()) if train_true else 0.0
        test_acc = float((np.asarray(test_preds) == np.asarray(test_true)).mean()) if test_true else 0.0
        results[layer] = {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_non_mirage_focus_mean": mu_non,
            "train_mirage_focus_mean": mu_mir,
            "threshold_midpoint": midpoint,
        }

    best_layer = None
    best_test = -1.0
    for layer, info in results.items():
        if info["test_accuracy"] > best_test:
            best_test = info["test_accuracy"]
            best_layer = layer

    return {
        "per_layer": results,
        "best_layer_by_test_accuracy": best_layer,
        "best_layer_test_accuracy": best_test,
    }


def _build_image_lookup_from_responses(
    mirage_root: Path,
    responses: List[Dict],
) -> Tuple[Dict[Tuple[str, str], List[bytes]], Dict[Tuple[str, str], List[bytes]]]:
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


def _select_holdout_examples(
    pool_true: List[Dict],
    pool_false: List[Dict],
    num_true: int,
    num_false: int,
    seed: int,
) -> List[Dict]:
    rng = random.Random(int(seed))
    if not pool_true:
        raise RuntimeError("No unseen mirage_like=true holdout candidates.")
    if not pool_false:
        raise RuntimeError("No unseen mirage_like=false holdout candidates.")

    selected_true = pool_true if len(pool_true) <= int(num_true) else rng.sample(pool_true, int(num_true))
    selected_false = pool_false if len(pool_false) <= int(num_false) else rng.sample(pool_false, int(num_false))
    selected = list(selected_true) + list(selected_false)
    rng.shuffle(selected)
    return selected


def _split_masks_for_seed(
    pair_ids_arr: np.ndarray,
    pair_benchmarks: List[str],
    selected_benchmark: Optional[str],
    seed: int,
    vlm_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, List[int]]]:
    qwen_val_fraction = _contrastive_validation_fraction_for_vlm(vlm_key)
    if qwen_val_fraction is not None:
        train_mask, val_mask, val_pairs, val_pairs_by_benchmark = _split_pair_benchmark_fraction_validation(
            pair_ids=pair_ids_arr,
            pair_benchmarks=pair_benchmarks,
            val_fraction=float(qwen_val_fraction),
            seed=int(seed),
        )
        return train_mask, val_mask, val_pairs, val_pairs_by_benchmark

    single_benchmark_validation_pairs = _single_benchmark_validation_pairs_for_vlm(vlm_key)
    if selected_benchmark is not None:
        train_mask, val_mask, val_pairs = _split_pair_fixed_validation_count(
            pair_ids=pair_ids_arr,
            num_validation_pairs=int(single_benchmark_validation_pairs),
            seed=int(seed),
        )
        return train_mask, val_mask, val_pairs, {selected_benchmark: [int(x) for x in val_pairs]}
    if _is_glm_vlm(vlm_key):
        train_mask, val_mask, val_pairs = _split_pair_fixed_validation_count(
            pair_ids=pair_ids_arr,
            num_validation_pairs=int(_all_mode_validation_pairs_for_vlm(vlm_key)),
            seed=int(seed),
        )
        val_pairs_by_benchmark = _group_pair_ids_by_benchmark(
            pair_ids=[int(x) for x in val_pairs],
            pair_benchmarks=pair_benchmarks,
        )
        return train_mask, val_mask, val_pairs, val_pairs_by_benchmark
    train_mask, val_mask, val_pairs, val_pairs_by_benchmark = _split_pair_benchmark_stratified_validation(
        pair_ids=pair_ids_arr,
        pair_benchmarks=pair_benchmarks,
        seed=int(seed),
    )
    return train_mask, val_mask, val_pairs, val_pairs_by_benchmark


def _label_counts_by_benchmark_and_class(
    y: np.ndarray,
    benchmark_labels: Sequence[str],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, int]]:
    y_arr = np.asarray(y, dtype=np.int64)
    if mask is None:
        mask_arr = np.ones(len(y_arr), dtype=bool)
    else:
        mask_arr = np.asarray(mask, dtype=bool)
    if len(mask_arr) != len(y_arr):
        raise ValueError("mask length must match y length.")
    if len(benchmark_labels) != len(y_arr):
        raise ValueError("benchmark_labels length must match y length.")

    out: Dict[str, Dict[str, int]] = defaultdict(lambda: {"class0": 0, "class1": 0, "total": 0})
    for idx, keep in enumerate(mask_arr.tolist()):
        if not bool(keep):
            continue
        benchmark = str(benchmark_labels[idx]) if str(benchmark_labels[idx]) else "unknown"
        label = int(y_arr[idx])
        if label == 0:
            out[benchmark]["class0"] += 1
        else:
            out[benchmark]["class1"] += 1
        out[benchmark]["total"] += 1
    return {
        str(k): {
            "class0": int(v["class0"]),
            "class1": int(v["class1"]),
            "total": int(v["total"]),
        }
        for k, v in sorted(out.items())
    }

def main() -> None:
    args = parse_args()
    args.pairs_path = str(
        _canonicalize_input_arg_for_vlm(
            path_value=args.pairs_path,
            vlm_key=str(args.vlm),
            artifact_name="contrastive_conversation_pairs.json",
        )
    )
    args.neutral_pairs_path = str(
        _canonicalize_input_arg_for_vlm(
            path_value=args.neutral_pairs_path,
            vlm_key=str(args.vlm),
            artifact_name="contrastive_conversation_pairs_neutral_as_non_mirage.json",
        )
    )
    args.responses_path = str(
        _canonicalize_input_arg_for_vlm(
            path_value=args.responses_path,
            vlm_key=str(args.vlm),
            artifact_name="responses.json",
        )
    )
    selected_benchmark = _resolve_selected_benchmark(args)
    supported_contrastive_benchmarks = _supported_contrastive_benchmarks_for_vlm(str(args.vlm))
    if selected_benchmark is not None and selected_benchmark not in supported_contrastive_benchmarks:
        raise ValueError(
            f"Benchmark '{selected_benchmark}' is not supported for --vlm {args.vlm}. "
            f"Supported benchmarks: {supported_contrastive_benchmarks}"
        )
    model_path = _resolve_model_path(args)

    save_dir = _scope_default_path(Path(args.save_dir), vlm=str(args.vlm), marker="/contrastive_probe_results")
    cache_path = _scope_default_path(
        Path(args.features_cache_path),
        vlm=str(args.vlm),
        marker="contrastive_pair_layer_features_with_attn.pt",
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = Path(args.neutral_pairs_path if args.neutral_as_non_mirage_pairs else args.pairs_path)
    # Prefer the latest model-scoped artifact when available (including default VLM),
    # and fall back to legacy unscoped paths for backward compatibility.
    pairs_path = _resolve_model_scoped_artifact_path(
        base_path=pairs_path,
        vlm_key=str(args.vlm),
        include_gen_prefix_dirs=True,
    )
    with open(pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if selected_benchmark is not None:
        pairs = [p for p in pairs if _infer_benchmark_from_pair(p) == selected_benchmark]
    else:
        supported_set = set(supported_contrastive_benchmarks)
        pairs = [p for p in pairs if _infer_benchmark_from_pair(p) in supported_set]
    if not pairs:
        raise ValueError("No pairs found after benchmark filtering.")

    labels: List[int] = []
    pair_ids: List[int] = []
    sample_names: List[str] = []
    conversations: List[List[Dict]] = []
    pair_benchmarks: List[str] = []
    skipped_short_training_pairs_count = 0
    kept_pair_id = 0
    for pair in pairs:
        non_conv = pair["non_mirage_conversation"]
        mirage_conv = pair["mirage_conversation"]
        if bool(args.exclude_short_responses_in_training_pairs):
            too_short_with_image_response = False
            if _conversation_has_image_input(non_conv):
                _, _, non_assistant = _conversation_signature_from_conv(non_conv)
                if core._count_tokens(non_assistant) < MIN_RESPONSE_TOKENS:
                    too_short_with_image_response = True
            if _conversation_has_image_input(mirage_conv):
                _, _, mirage_assistant = _conversation_signature_from_conv(mirage_conv)
                if core._count_tokens(mirage_assistant) < MIN_RESPONSE_TOKENS:
                    too_short_with_image_response = True
            if too_short_with_image_response:
                skipped_short_training_pairs_count += 1
                continue
        pair_benchmarks.append(_infer_benchmark_from_pair(pair))
        conversations.append(non_conv)
        labels.append(0)
        pair_ids.append(kept_pair_id)
        sample_names.append(f"pair_{kept_pair_id}_non_mirage")
        conversations.append(mirage_conv)
        labels.append(1)
        pair_ids.append(kept_pair_id)
        sample_names.append(f"pair_{kept_pair_id}_mirage")
        kept_pair_id += 1
    if not conversations:
        raise ValueError(
            "No contrastive training pairs remain after short-response filtering. "
            "Re-run with --no_exclude_short_responses_in_training_pairs to keep short pairs."
        )

    requested_include_attention = bool(args.include_attention_probes or args.include_additional_attention_mlp_probes)
    requested_include_mlp = bool(args.include_mlp_probes or args.include_additional_attention_mlp_probes)
    requested_include_residual = bool(getattr(args, "include_residual_probes", True))
    requested_llm_feature_strategies = _parse_requested_llm_feature_strategies(
        getattr(args, "llm_feature_strategies", ",".join(LLM_STRATEGIES))
    )
    requested_include_additional = bool(requested_include_attention or requested_include_mlp)
    additional_feature_experiment_mode = _is_additional_feature_experiment_mode(
        include_attention_probes=bool(requested_include_attention),
        include_mlp_probes=bool(requested_include_mlp),
        include_residual_probes=bool(requested_include_residual),
    )
    if additional_feature_experiment_mode:
        print(
            "Additional-feature experiment mode detected; applying runtime policy: "
            "num_split_seeds=3, single-init probes, num_eval_seeds=3, "
            "fixed_feature=text_nonspecial_mean."
        )
        args.num_split_seeds = 3
        args.multi_init_probe_selection = False
        args.probe_num_initializations = 1
        args.num_eval_seeds = 3

    if _uses_preextracted_activation_store(str(args.vlm)):
        preextract_family = _preextract_family_for_vlm(str(args.vlm)).upper()
        if bool(args.force_reextract):
            warnings.warn(
                f"--force_reextract is ignored for {preextract_family}. "
                "Loading pre-extracted activations cache instead."
            )
        layer_features, layer_order, cache_path = _load_preextracted_contrastive_subset(
            vlm_key=str(args.vlm),
            conversations=conversations,
            include_attention_probes=bool(requested_include_attention),
            include_mlp_probes=bool(requested_include_mlp),
            include_residual_probes=bool(requested_include_residual),
            llm_feature_strategies=requested_llm_feature_strategies,
            use_additional_feature_preextract_cache=getattr(
                args,
                "use_additional_feature_preextract_cache",
                None,
            ),
            requested_model_path=str(model_path),
        )
    else:
        use_cache = cache_path.exists() and (not args.force_reextract) and (selected_benchmark is None)
        if cache_path.exists() and (not args.force_reextract) and (selected_benchmark is not None):
            warnings.warn(
                "Ignoring --features_cache_path cache in benchmark-specific mode to avoid sample-order mismatch. "
                "Re-extracting features."
            )

        if use_cache:
            payload = torch.load(cache_path)
            cached_model_path = str(payload.get("model_path", ""))
            expected_glm_image_normalization = _is_glm_vlm(str(args.vlm))
            cached_glm_image_normalization = bool(payload.get("glm_image_normalization_applied", False))
            if cached_model_path and cached_model_path != str(model_path):
                warnings.warn("Cached model_path differs from requested model; re-extracting features.")
                use_cache = False
            elif expected_glm_image_normalization and (not cached_glm_image_normalization):
                warnings.warn(
                    "Cached GLM features predate GLM image normalization for extraction; re-extracting features."
                )
                use_cache = False
            else:
                cached_names = [str(x) for x in payload.get("sample_names", [])]
                cached_version = int(payload.get("feature_extraction_version", 1))
                if ("include_attention_probes" in payload) or ("include_mlp_probes" in payload):
                    cached_include_attention = bool(payload.get("include_attention_probes", False))
                    cached_include_mlp = bool(payload.get("include_mlp_probes", False))
                else:
                    cached_include_additional = bool(payload.get("include_additional_attention_mlp_probes", False))
                    cached_include_attention = cached_include_additional
                    cached_include_mlp = cached_include_additional

                if cached_names != sample_names:
                    warnings.warn("Cached sample ordering does not match selected pairs. Re-extracting features.")
                    use_cache = False
                elif requested_include_attention and (not cached_include_attention):
                    warnings.warn("Cache missing attention probes required by this run. Re-extracting features.")
                    use_cache = False
                elif requested_include_mlp and (not cached_include_mlp):
                    warnings.warn("Cache missing MLP probes required by this run. Re-extracting features.")
                    use_cache = False
                elif requested_include_additional and (cached_version != FEATURE_EXTRACTION_VERSION):
                    warnings.warn("Cache extraction version is stale for requested probe families. Re-extracting.")
                    use_cache = False
                else:
                    layer_features = payload["layer_features"]
                    labels = payload["labels"]
                    pair_ids = payload["pair_ids"]
                    sample_names = payload["sample_names"]
                    layer_order = payload["layer_order"]
                    if not requested_include_attention:
                        layer_order = [k for k in layer_order if not _is_attention_probe_feature(k)]
                        layer_features = {k: v for k, v in layer_features.items() if not _is_attention_probe_feature(k)}
                    if not requested_include_mlp:
                        layer_order = [k for k in layer_order if not _is_mlp_probe_feature(k)]
                        layer_features = {k: v for k, v in layer_features.items() if not _is_mlp_probe_feature(k)}
                    requested_keys = _resolve_requested_feature_keys(
                        layer_order=layer_order,
                        include_attention_probes=bool(requested_include_attention),
                        include_mlp_probes=bool(requested_include_mlp),
                        include_residual_probes=bool(requested_include_residual),
                        llm_feature_strategies=requested_llm_feature_strategies,
                    )
                    layer_features, layer_order = _filter_layer_feature_map(
                        layer_features=layer_features,
                        layer_order=layer_order,
                        requested_keys=requested_keys,
                    )

        if not use_cache:
            model = load_vlm_for_extraction(
                model_path=model_path,
                attn_implementation=args.attn_implementation,
                device_map_raw=str(getattr(args, "device_map", "")),
                max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
                max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
            )
            _force_attention_backend(model, args.attn_implementation)

            layer_features = defaultdict(list)
            layer_order = []
            seen_layers = set()
            for conv in tqdm(conversations, desc="Extracting activations"):
                ovis_messages = _to_ovis_messages(conv)
                sample_feats = _extract_sample_features_only(
                    model=model,
                    messages=ovis_messages,
                    include_additional_attention_mlp_probes=False,
                    include_attention_probes=requested_include_attention,
                    include_mlp_probes=requested_include_mlp,
                    include_residual_probes=requested_include_residual,
                    model_key=str(args.vlm),
                )
                for key, value in sample_feats.items():
                    if key not in seen_layers:
                        seen_layers.add(key)
                        layer_order.append(key)
                    layer_features[key].append(value.to(torch.float32).cpu().numpy())

            layer_order = sorted(layer_order, key=_layer_sort_key)
            requested_keys = _resolve_requested_feature_keys(
                layer_order=layer_order,
                include_attention_probes=bool(requested_include_attention),
                include_mlp_probes=bool(requested_include_mlp),
                include_residual_probes=bool(requested_include_residual),
                llm_feature_strategies=requested_llm_feature_strategies,
            )
            layer_features, layer_order = _filter_layer_feature_map(
                layer_features=layer_features,
                layer_order=layer_order,
                requested_keys=requested_keys,
            )
            torch.save(
                {
                    "layer_features": layer_features,
                    "labels": labels,
                    "pair_ids": pair_ids,
                    "sample_names": sample_names,
                    "layer_order": layer_order,
                    "feature_extraction_version": int(FEATURE_EXTRACTION_VERSION),
                    "include_attention_probes": bool(requested_include_attention),
                    "include_mlp_probes": bool(requested_include_mlp),
                    "include_residual_probes": bool(requested_include_residual),
                    "include_additional_attention_mlp_probes": bool(requested_include_additional),
                    "llm_feature_strategies": list(requested_llm_feature_strategies),
                    "glm_image_normalization_applied": bool(_is_glm_vlm(str(args.vlm))),
                    "vlm": str(args.vlm),
                    "model_path": model_path,
                },
                cache_path,
            )

    if not layer_order:
        raise RuntimeError(
            "No features remain after applying probe-family/strategy filters. "
            "Check --include_*_probes and --llm_feature_strategies."
        )
    if additional_feature_experiment_mode:
        original_feature_count = len(layer_order)
        strategy_matches = [
            str(feature_name)
            for feature_name in layer_order
            if (
                _feature_strategy_from_name(str(feature_name)) == "text_nonspecial_mean"
                or str(feature_name) == "text_nonspecial_mean"
            )
        ]
        if not strategy_matches:
            raise RuntimeError(
                "Additional-feature runtime policy could not find feature "
                "'text_nonspecial_mean' in extracted features."
            )
        selected_additional_feature_names = sorted(strategy_matches, key=_layer_sort_key)
        layer_features, layer_order = _filter_layer_feature_map(
            layer_features=layer_features,
            layer_order=layer_order,
            requested_keys=selected_additional_feature_names,
        )
        print(
            "Additional-feature runtime policy: fixed feature selection kept "
            f"{len(layer_order)}/{original_feature_count} features for "
            "text_nonspecial_mean."
        )

    y = np.asarray(labels, dtype=np.int64)
    pair_ids_arr = np.asarray(pair_ids, dtype=np.int64)
    benchmark_labels = np.asarray([str(pair_benchmarks[int(pid)]) for pid in pair_ids_arr.tolist()], dtype=object)
    reg_values = [float(x.strip()) for x in args.regularization_values.split(",") if x.strip()]
    if not reg_values:
        raise ValueError("--regularization_values must contain at least one value.")
    if 0.0 not in reg_values:
        raise ValueError("--regularization_values must include 0.")
    if int(args.num_split_seeds) < 1:
        raise ValueError("--num_split_seeds must be >= 1.")

    split_seed_rng = np.random.default_rng(int(args.seed))
    split_seeds = split_seed_rng.choice(
        1_000_000_000,
        size=int(args.num_split_seeds),
        replace=False,
    ).astype(np.int64).tolist()

    split_payloads = []
    for split_seed in split_seeds:
        train_mask, val_mask, val_pairs, val_pairs_by_benchmark = _split_masks_for_seed(
            pair_ids_arr=pair_ids_arr,
            pair_benchmarks=pair_benchmarks,
            selected_benchmark=selected_benchmark,
            seed=int(split_seed),
            vlm_key=str(args.vlm),
        )
        split_payloads.append(
            {
                "split_seed": int(split_seed),
                "train_mask": train_mask,
                "val_mask": val_mask,
                "test_mask": np.zeros(len(val_mask), dtype=bool),
                "validation_pairs": [int(x) for x in val_pairs],
                "validation_pairs_by_benchmark": {k: [int(x) for x in v] for k, v in val_pairs_by_benchmark.items()},
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
                    "train": _label_counts_by_benchmark_and_class(
                        y=y,
                        benchmark_labels=benchmark_labels.tolist(),
                        mask=train_mask,
                    ),
                    "val": _label_counts_by_benchmark_and_class(
                        y=y,
                        benchmark_labels=benchmark_labels.tolist(),
                        mask=val_mask,
                    ),
                },
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
            run["validation_pairs"] = split_payload["validation_pairs"]
            run["validation_pairs_by_benchmark"] = split_payload["validation_pairs_by_benchmark"]
            run["split_sizes"] = split_payload["split_sizes"]
            seed_runs.append(run)

        train_scores = [float(r["best_train_accuracy"]) for r in seed_runs]
        val_scores = [float(r["validation_accuracy_at_best_c"]) for r in seed_runs]
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
        benchmark_scores = _mean_dict_metrics_from_seed_runs(
            seed_runs=seed_runs,
            key="benchmark_test_accuracy_at_best_c",
        )
        benchmark_class0_scores = _mean_dict_metrics_from_seed_runs(
            seed_runs=seed_runs,
            key="benchmark_class0_test_accuracy_at_best_c",
        )
        benchmark_class1_scores = _mean_dict_metrics_from_seed_runs(
            seed_runs=seed_runs,
            key="benchmark_class1_test_accuracy_at_best_c",
        )
        best_c_values = [float(r["best_c"]) for r in seed_runs]
        best_c_mode = sorted(best_c_values)[0] if not best_c_values else max(
            sorted(set(best_c_values)),
            key=lambda c: (best_c_values.count(c), -c),
        )
        per_feature_results[feature_name] = {
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
        for strategy in LLM_STRATEGIES:
            key = f"{llm_layer}__{strategy}"
            if key in per_feature_results:
                strategy_acc[strategy] = float(per_feature_results[key]["mean_validation_accuracy_at_best_c"])
        best_strategy = None
        best_acc = -1.0
        for strategy in LLM_STRATEGIES:
            if strategy not in strategy_acc:
                continue
            score = float(strategy_acc[strategy])
            if score > best_acc:
                best_acc = score
                best_strategy = strategy
        best_feature = f"{llm_layer}__{best_strategy}" if best_strategy is not None else None
        llm_strategy_results[llm_layer] = {
            "strategy_validation_accuracies": strategy_acc,
            "best_strategy": best_strategy,
            "best_feature": best_feature,
            "best_validation_accuracy": best_acc,
        }

    selected_residual_features = sorted(
        set(
            info["best_feature"]
            for info in llm_strategy_results.values()
            if info.get("best_feature")
        )
    )
    additional_target_results: Dict[str, Dict[str, Any]] = {}
    selected_additional_features: List[str] = []
    if additional_feature_experiment_mode:
        additional_target_results = _best_feature_by_base_from_validation(per_feature_results)
        selected_additional_features = sorted(
            set(
                info["best_feature"]
                for info in additional_target_results.values()
                if info.get("best_feature")
            )
        )

    holdout_feature_groups: Dict[str, Dict[str, Any]] = {}
    if bool(getattr(args, "heldout_eval_all_features", False)):
        holdout_feature_groups = {
            str(feature_name): {
                "best_feature": str(feature_name),
                "best_validation_accuracy": float(
                    per_feature_results[feature_name].get("mean_validation_accuracy_at_best_c", float("nan"))
                ),
                "best_strategy": (
                    str(feature_name).split("__", 1)[1]
                    if "__" in str(feature_name)
                    else None
                ),
                "selection_scope": "all_features",
            }
            for feature_name in sorted(per_feature_results.keys(), key=_layer_sort_key)
        }
    else:
        if selected_residual_features:
            holdout_feature_groups = {
                str(layer_name): {
                    "best_feature": str(llm_strategy_results[layer_name]["best_feature"]),
                    "best_validation_accuracy": float(llm_strategy_results[layer_name]["best_validation_accuracy"]),
                    "best_strategy": llm_strategy_results[layer_name].get("best_strategy"),
                    "selection_scope": "llm_residual_layer",
                }
                for layer_name in llm_layers
                if llm_strategy_results.get(layer_name, {}).get("best_feature")
            }
        elif selected_additional_features:
            holdout_feature_groups = {
                str(base_name): {
                    "best_feature": str(info["best_feature"]),
                    "best_validation_accuracy": float(info["best_validation_accuracy"]),
                    "best_strategy": (
                        str(info["best_feature"]).split("__", 1)[1]
                        if "__" in str(info["best_feature"])
                        else None
                    ),
                    "selection_scope": "llm_additional_target",
                }
                for base_name, info in sorted(additional_target_results.items(), key=lambda kv: _layer_sort_key(kv[0]))
                if info.get("best_feature")
            }

    selected_holdout_features = sorted(
        set(
            info["best_feature"]
            for info in holdout_feature_groups.values()
            if info.get("best_feature")
        )
    )

    responses_path = _resolve_responses_path_for_vlm(
        base_responses_path=Path(args.responses_path),
        vlm_key=str(args.vlm),
    )

    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    mirage_root = REPO_ROOT.resolve()
    image_lookup_uid, image_lookup_qid = _build_image_lookup_from_responses(mirage_root=mirage_root, responses=responses)
    seen_signatures = {_conversation_signature_from_conv(c) for c in conversations}
    pool_true, pool_false, skipped_short_response_count = _build_holdout_pool(
        responses=responses,
        seen_signatures=seen_signatures,
        selected_benchmark=selected_benchmark,
        allowed_benchmarks=supported_contrastive_benchmarks,
        include_short_response_filter=bool(args.exclude_short_responses_in_holdout),
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
    )
    raw_holdout_pool_sizes = {
        "mirage_true": int(len(pool_true)),
        "mirage_false": int(len(pool_false)),
    }
    holdout_pool_counts_by_benchmark_before_preextract_filter = _holdout_pool_counts_by_benchmark(
        pool_true=pool_true,
        pool_false=pool_false,
    )
    holdout_pool_counts_by_benchmark_after_preextract_filter = dict(holdout_pool_counts_by_benchmark_before_preextract_filter)
    holdout_selection_plan_before_preextract_filter = _plan_balanced_holdout_selection(
        pool_true=pool_true,
        pool_false=pool_false,
        requested_num_true=int(args.num_holdout_mirage_true),
        requested_num_false=int(args.num_holdout_mirage_false),
    )
    holdout_selection_plan_after_preextract_filter = dict(holdout_selection_plan_before_preextract_filter)
    holdout_pool_filter_to_preextract_cache = {
        "mirage_true_before": int(len(pool_true)),
        "mirage_false_before": int(len(pool_false)),
        "mirage_true_after": int(len(pool_true)),
        "mirage_false_after": int(len(pool_false)),
        "mirage_true_dropped": 0,
        "mirage_false_dropped": 0,
    }
    holdout_selected_sizes_by_seed: Dict[str, Dict[str, Any]] = {}

    eval_seeds = [int(args.seed) + i for i in range(int(args.num_eval_seeds))]
    holdout_payloads_by_seed: Dict[int, Dict] = {}
    preextracted_holdout_feature_store: Optional[Dict[str, Any]] = None
    if selected_holdout_features:
        if _uses_preextracted_activation_store(str(args.vlm)):
            preextract_family_raw = _preextract_family_for_vlm(str(args.vlm))
            preextract_family = str(preextract_family_raw).upper()
            use_additional_cache = _resolve_additional_preextract_cache_selection(
                include_attention_probes=bool(requested_include_attention),
                include_mlp_probes=bool(requested_include_mlp),
                include_residual_probes=bool(requested_include_residual),
                explicit_use_additional_feature_cache=getattr(
                    args,
                    "use_additional_feature_preextract_cache",
                    None,
                ),
            )
            preextracted_all_cache_path = _preextracted_all_examples_path_for_vlm(
                vlm_key=str(args.vlm),
                use_additional_feature_cache=bool(use_additional_cache),
            )
            if not preextracted_all_cache_path.exists():
                raise FileNotFoundError(
                    f"Missing {preextract_family} pre-extracted all-examples activations cache. "
                    f"Expected: {preextracted_all_cache_path}. Generate it via "
                    f"extract_{preextract_family_raw}_activations.py."
                )
            preextracted_all_payload = torch.load(preextracted_all_cache_path)
            expected_cache_type = f"{preextract_family_raw}_preextracted_all_examples_features"
            if str(preextracted_all_payload.get("cache_type", "")) not in {expected_cache_type}:
                raise RuntimeError(
                    f"Invalid cache_type for {preextract_family} all-examples pre-extracted cache: "
                    f"{preextracted_all_payload.get('cache_type')}"
                )
            _maybe_warn_preextracted_cache_model_mismatch(
                payload=preextracted_all_payload,
                requested_model_path=str(model_path),
                cache_path=preextracted_all_cache_path,
                family_label=preextract_family,
            )
            with_payload = preextracted_all_payload.get("with_image", {})
            with_signature_keys = [str(x) for x in with_payload.get("signature_keys", [])]
            sig_to_idx = {sig: idx for idx, sig in enumerate(with_signature_keys)}
            preextracted_holdout_feature_store = with_payload.get("layer_features", {})
            for feature in selected_holdout_features:
                if str(feature) not in preextracted_holdout_feature_store:
                    raise RuntimeError(
                        f"Required held-out feature '{feature}' not present in {preextract_family} "
                        f"all-examples cache: {preextracted_all_cache_path}"
                    )
            del preextracted_all_payload

            pool_true, pool_false, holdout_pool_filter_to_preextract_cache = _filter_holdout_pool_to_available_signatures(
                pool_true=pool_true,
                pool_false=pool_false,
                available_signature_keys=set(sig_to_idx.keys()),
            )
            dropped_total = int(
                holdout_pool_filter_to_preextract_cache["mirage_true_dropped"]
                + holdout_pool_filter_to_preextract_cache["mirage_false_dropped"]
            )
            if dropped_total > 0:
                print(
                    f"Filtered held-out pool to examples present in {preextract_family} all-examples cache: "
                    f"dropped_true={holdout_pool_filter_to_preextract_cache['mirage_true_dropped']}, "
                    f"dropped_false={holdout_pool_filter_to_preextract_cache['mirage_false_dropped']}, "
                    f"kept_true={holdout_pool_filter_to_preextract_cache['mirage_true_after']}, "
                    f"kept_false={holdout_pool_filter_to_preextract_cache['mirage_false_after']}"
                )
            if not pool_true or not pool_false:
                raise RuntimeError(
                    f"After filtering held-out pool to {preextract_family} all-examples cache signatures, "
                    f"insufficient candidates remain (true={len(pool_true)}, false={len(pool_false)}). "
                    f"cache_path={preextracted_all_cache_path}. Rebuild pre-extracted all-examples cache with "
                    "a larger --all_examples_max_samples (or <=0 for no cap)."
                )
            holdout_pool_counts_by_benchmark_after_preextract_filter = _holdout_pool_counts_by_benchmark(
                pool_true=pool_true,
                pool_false=pool_false,
            )
            holdout_selection_plan_after_preextract_filter = _plan_balanced_holdout_selection(
                pool_true=pool_true,
                pool_false=pool_false,
                requested_num_true=int(args.num_holdout_mirage_true),
                requested_num_false=int(args.num_holdout_mirage_false),
            )

            for eval_seed in tqdm(eval_seeds, desc=f"Loading held-out features from {preextract_family} cache"):
                selected_examples, selected_counts_by_benchmark = _select_holdout_examples_balanced_by_benchmark(
                    pool_true=pool_true,
                    pool_false=pool_false,
                    selected_pairs_by_benchmark={
                        str(k): int(v)
                        for k, v in dict(holdout_selection_plan_after_preextract_filter["selected_pairs_by_benchmark"]).items()
                    },
                    seed=int(eval_seed),
                )
                y_holdout = np.asarray(
                    [1 if bool(item["mirage_like"]) else 0 for item in selected_examples],
                    dtype=np.int64,
                )
                benchmark_labels_holdout = [str(item.get("dataset", "unknown")) for item in selected_examples]
                holdout_indices: List[int] = []
                for item in selected_examples:
                    sig = _signature_key_from_holdout_item(item)
                    if sig is None:
                        raise RuntimeError(
                            "Held-out example missing required with-image conversation "
                            "key ('conversation' or 'with_conversation')."
                        )
                    idx = sig_to_idx.get(sig)
                    if idx is None:
                        raise RuntimeError(
                            f"Held-out example signature missing in {preextract_family} all-examples cache "
                            f"after pre-filtering. cache_path={preextracted_all_cache_path}"
                        )
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
                }
                holdout_selected_sizes_by_seed[str(int(eval_seed))] = {
                    "num_examples_total": int(len(y_holdout)),
                    "num_examples_per_class": {
                        "mirage_true": int((y_holdout == 1).sum()),
                        "mirage_false": int((y_holdout == 0).sum()),
                    },
                    "selected_counts_by_benchmark": selected_counts_by_benchmark,
                }
        else:
            eval_model = load_vlm_for_extraction(
                model_path=model_path,
                attn_implementation=args.attn_implementation,
                device_map_raw=str(getattr(args, "device_map", "")),
                max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
                max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
            )
            _force_attention_backend(eval_model, args.attn_implementation)
            holdout_pool_counts_by_benchmark_after_preextract_filter = _holdout_pool_counts_by_benchmark(
                pool_true=pool_true,
                pool_false=pool_false,
            )
            holdout_selection_plan_after_preextract_filter = _plan_balanced_holdout_selection(
                pool_true=pool_true,
                pool_false=pool_false,
                requested_num_true=int(args.num_holdout_mirage_true),
                requested_num_false=int(args.num_holdout_mirage_false),
            )

            for eval_seed in tqdm(eval_seeds, desc="Extracting held-out features"):
                selected_examples, selected_counts_by_benchmark = _select_holdout_examples_balanced_by_benchmark(
                    pool_true=pool_true,
                    pool_false=pool_false,
                    selected_pairs_by_benchmark={
                        str(k): int(v)
                        for k, v in dict(holdout_selection_plan_after_preextract_filter["selected_pairs_by_benchmark"]).items()
                    },
                    seed=int(eval_seed),
                )
                y_holdout = np.asarray(
                    [1 if bool(item["mirage_like"]) else 0 for item in selected_examples],
                    dtype=np.int64,
                )
                benchmark_labels_holdout = [str(item.get("dataset", "unknown")) for item in selected_examples]
                feature_arrays = {f: [] for f in selected_holdout_features}
                for item in selected_examples:
                    sample_feats = _extract_sample_features_only(
                        model=eval_model,
                        messages=_to_ovis_messages(item["conversation"]),
                        include_additional_attention_mlp_probes=False,
                        include_attention_probes=bool(requested_include_attention),
                        include_mlp_probes=bool(requested_include_mlp),
                        include_residual_probes=bool(requested_include_residual),
                        model_key=str(args.vlm),
                    )
                    for feature in selected_holdout_features:
                        if feature not in sample_feats:
                            raise RuntimeError(f"Missing held-out feature '{feature}' during extraction.")
                        feature_arrays[feature].append(sample_feats[feature].to(torch.float32).cpu().numpy())
                holdout_payloads_by_seed[int(eval_seed)] = {
                    "y_holdout": y_holdout,
                    "benchmark_labels_holdout": benchmark_labels_holdout,
                    "features": {
                        f: np.asarray(v, dtype=np.float32) for f, v in feature_arrays.items()
                    },
                    "num_examples": int(len(y_holdout)),
                    "selected_counts_by_benchmark": selected_counts_by_benchmark,
                    "num_examples_per_class": {
                        "mirage_true": int((y_holdout == 1).sum()),
                        "mirage_false": int((y_holdout == 0).sum()),
                    },
                }
                holdout_selected_sizes_by_seed[str(int(eval_seed))] = {
                    "num_examples_total": int(len(y_holdout)),
                    "num_examples_per_class": {
                        "mirage_true": int((y_holdout == 1).sum()),
                        "mirage_false": int((y_holdout == 0).sum()),
                    },
                    "selected_counts_by_benchmark": selected_counts_by_benchmark,
                }

    residual_layer_holdout_results = {}
    for group_name, group_info in sorted(holdout_feature_groups.items(), key=lambda kv: _layer_sort_key(kv[0])):
        best_feature = str(group_info.get("best_feature", ""))
        if not best_feature:
            continue
        layer_runs = []
        X_all = np.asarray(layer_features[best_feature], dtype=np.float32)
        for eval_idx, eval_seed in enumerate(eval_seeds):
            split_payload = split_payloads[eval_idx % len(split_payloads)]
            split_seed = int(split_payload["split_seed"])
            run_by_seed = {
                int(r["split_seed"]): r for r in per_feature_results[best_feature]["seed_runs"]
            }
            if split_seed not in run_by_seed:
                raise RuntimeError(f"Missing split run for split_seed={split_seed}, feature={best_feature}.")
            split_run = run_by_seed[split_seed]
            best_c = float(split_run["best_c"])

            train_mask = split_payload["train_mask"]
            val_mask = split_payload["val_mask"]
            X_train = X_all[train_mask]
            y_train = y[train_mask]
            X_val = X_all[val_mask]
            y_val = y[val_mask]

            holdout_payload = holdout_payloads_by_seed[int(eval_seed)]
            if "features" in holdout_payload:
                X_holdout = holdout_payload["features"][best_feature]
            else:
                holdout_indices = holdout_payload.get("holdout_indices")
                if holdout_indices is None:
                    raise RuntimeError(
                        f"Held-out payload for seed={eval_seed} is missing both 'features' "
                        "and 'holdout_indices'."
                    )
                if preextracted_holdout_feature_store is None:
                    raise RuntimeError(
                        "Internal error: preextracted_holdout_feature_store is unavailable "
                        "for index-based held-out feature loading."
                    )
                if best_feature not in preextracted_holdout_feature_store:
                    raise RuntimeError(
                        f"Held-out feature '{best_feature}' is missing from pre-extracted "
                        "all-examples feature store."
                    )
                X_holdout = _gather_feature_rows_from_store(
                    feature_store=preextracted_holdout_feature_store[best_feature],
                    row_indices=np.asarray(holdout_indices, dtype=np.int64),
                )
            y_holdout = holdout_payload["y_holdout"]
            benchmark_labels_holdout = holdout_payload.get("benchmark_labels_holdout", [])
            fit = _fit_fixed_c_with_multi_init(
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
            (
                benchmark_test_acc,
                benchmark_class0_test_acc,
                benchmark_class1_test_acc,
            ) = _compute_benchmark_test_metrics(
                y_true=y_holdout,
                y_pred=holdout_pred,
                benchmark_labels=benchmark_labels_holdout,
            )
            layer_runs.append(
                {
                    "eval_seed": int(eval_seed),
                    "split_seed": int(split_seed),
                    "feature": best_feature,
                    "best_c": best_c,
                    "train_accuracy": float(fit["train_accuracy"]),
                    "validation_accuracy": float(fit["validation_accuracy"]),
                    "test_accuracy": holdout_acc,
                    "test_accuracy_mirage_true": float((holdout_pred[class1_mask] == y_holdout[class1_mask]).mean()) if class1_mask.any() else None,
                    "test_accuracy_mirage_false": float((holdout_pred[class0_mask] == y_holdout[class0_mask]).mean()) if class0_mask.any() else None,
                    "benchmark_test_accuracy": benchmark_test_acc,
                    "benchmark_class0_test_accuracy": benchmark_class0_test_acc,
                    "benchmark_class1_test_accuracy": benchmark_class1_test_acc,
                    "macro_benchmark_test_accuracy": _macro_average_metric_dict(benchmark_test_acc),
                    "macro_benchmark_class0_test_accuracy": _macro_average_metric_dict(benchmark_class0_test_acc),
                    "macro_benchmark_class1_test_accuracy": _macro_average_metric_dict(benchmark_class1_test_acc),
                    "num_holdout_examples": int(len(y_holdout)),
                    "num_holdout_examples_per_class": holdout_payload.get("num_examples_per_class"),
                    "num_holdout_examples_by_benchmark": holdout_payload.get("selected_counts_by_benchmark"),
                    "best_init_index": int(fit["init_index"]),
                    "best_init_seed": int(fit["init_seed"]),
                }
            )

        residual_layer_holdout_results[str(group_name)] = {
            "best_feature": best_feature,
            "selection_scope": str(group_info.get("selection_scope") or ""),
            "best_strategy": group_info.get("best_strategy"),
            "best_validation_accuracy": (
                float(group_info.get("best_validation_accuracy"))
                if group_info.get("best_validation_accuracy") is not None
                else None
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

    attention_probe_results = {
        k: {
            "num_split_seeds": int(v["num_split_seeds"]),
            "split_seeds": v["split_seeds"],
            "selection_metric": v["selection_metric"],
            "best_c": float(v["best_c"]),
            "best_c_values_by_split": v["best_c_values_by_split"],
            "mean_train_accuracy_at_best_c": float(v["mean_train_accuracy_at_best_c"]),
            "mean_validation_accuracy_at_best_c": float(v["mean_validation_accuracy_at_best_c"]),
            "mean_test_accuracy_at_best_c": (
                float(v["mean_test_accuracy_at_best_c"])
                if v.get("mean_test_accuracy_at_best_c") is not None
                else None
            ),
            "seed_runs": v["seed_runs"],
        }
        for k, v in sorted(per_feature_results.items(), key=lambda kv: _layer_sort_key(kv[0]))
        if _is_additional_attention_mlp_feature(k)
    }
    best_attention_feature = None
    best_attention_val = -1.0
    for k, v in attention_probe_results.items():
        score = float(v["mean_validation_accuracy_at_best_c"])
        if score > best_attention_val:
            best_attention_val = score
            best_attention_feature = k

    all_feature_results = [
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
        for k, v in sorted(per_feature_results.items(), key=lambda kv: _layer_sort_key(kv[0]))
    ]

    training_pair_counts_by_benchmark: Dict[str, int] = defaultdict(int)
    for bench in pair_benchmarks:
        training_pair_counts_by_benchmark[str(bench)] += 1
    training_sample_counts_by_benchmark_and_class = _label_counts_by_benchmark_and_class(
        y=y,
        benchmark_labels=benchmark_labels.tolist(),
        mask=None,
    )

    summary_payload = {
        "num_pairs": int(len(set(pair_ids))),
        "num_samples": int(len(y)),
        "num_class0_samples": int((y == 0).sum()),
        "num_class1_samples": int((y == 1).sum()),
        "training_pair_counts_by_benchmark": {
            str(k): int(v) for k, v in sorted(training_pair_counts_by_benchmark.items())
        },
        "training_sample_counts_by_benchmark_and_class": training_sample_counts_by_benchmark_and_class,
        "selected_benchmark": selected_benchmark,
        "training_mode": (
            f"pair_{selected_benchmark}_validation_multiseed"
            if selected_benchmark is not None
            else (
                "pair_fixed_count_validation_multiseed_glm"
                if _is_glm_vlm(str(args.vlm))
                else "pair_benchmark_stratified_validation_multiseed"
            )
        ),
        "num_split_seeds": int(args.num_split_seeds),
        "split_seeds": [int(s) for s in split_seeds],
        "regularization_sweep_c_values": reg_values,
        "hyperparam_selection_metric": "validation_accuracy",
        "probe_epochs": int(args.probe_epochs),
        "probe_lr": float(args.probe_lr),
        "multi_init_probe_selection": bool(args.multi_init_probe_selection),
        "probe_num_initializations": int(args.probe_num_initializations),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "include_attention_probes": bool(requested_include_attention),
        "include_mlp_probes": bool(requested_include_mlp),
        "include_residual_probes": bool(requested_include_residual),
        "include_additional_attention_mlp_probes": bool(requested_include_additional),
        "use_additional_feature_preextract_cache": getattr(
            args, "use_additional_feature_preextract_cache", None
        ),
        "requested_llm_feature_strategies": list(requested_llm_feature_strategies),
        "llm_strategies": LLM_STRATEGIES,
        "feature_extraction_version": int(FEATURE_EXTRACTION_VERSION),
        "num_attention_activation_features": int(len(attention_probe_results)),
        "num_eval_seeds": int(args.num_eval_seeds),
        "eval_seeds": [int(s) for s in eval_seeds],
        "heldout_eval_all_features": bool(getattr(args, "heldout_eval_all_features", False)),
        "num_holdout_mirage_true": int(args.num_holdout_mirage_true),
        "num_holdout_mirage_false": int(args.num_holdout_mirage_false),
        "exclude_short_responses_in_training_pairs": bool(args.exclude_short_responses_in_training_pairs),
        "num_training_pairs_skipped_short_responses": int(skipped_short_training_pairs_count),
        "min_response_tokens_required_for_training_pairs": (
            int(MIN_RESPONSE_TOKENS) if bool(args.exclude_short_responses_in_training_pairs) else None
        ),
        "exclude_short_responses_in_holdout": bool(args.exclude_short_responses_in_holdout),
        "num_candidates_skipped_short_responses": int(skipped_short_response_count),
        "holdout_pool_sizes_before_preextract_filter": raw_holdout_pool_sizes,
        "holdout_pool_sizes_by_benchmark_before_preextract_filter": holdout_pool_counts_by_benchmark_before_preextract_filter,
        "holdout_selection_plan_before_preextract_filter": holdout_selection_plan_before_preextract_filter,
        "holdout_pool_sizes": {
            "mirage_true": int(len(pool_true)),
            "mirage_false": int(len(pool_false)),
        },
        "holdout_pool_sizes_by_benchmark_after_preextract_filter": holdout_pool_counts_by_benchmark_after_preextract_filter,
        "holdout_selection_plan_after_preextract_filter": holdout_selection_plan_after_preextract_filter,
        "holdout_pool_filter_to_preextract_cache": holdout_pool_filter_to_preextract_cache,
        "holdout_selected_sizes_by_seed": holdout_selected_sizes_by_seed,
    }
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

    vlm_tag = str(args.vlm)
    all_features_path = save_dir / f"{vlm_tag}_all_feature_probe_accuracies.json"
    llm_strategy_path = save_dir / f"{vlm_tag}_llm_layer_best_strategy_results.json"
    llm_holdout_path = save_dir / f"{vlm_tag}_llm_residual_layer_heldout_eval.json"
    attention_raw_path = save_dir / f"{vlm_tag}_llm_attention_scores_by_sample.json"
    attention_summary_path = save_dir / f"{vlm_tag}_llm_attention_mirage_vs_non_summary.json"
    attention_classifier_path = save_dir / f"{vlm_tag}_attention_difference_classifier_results.json"
    attention_probe_path = save_dir / f"{vlm_tag}_attention_layer_probe_accuracies.json"
    config_path = save_dir / f"{vlm_tag}_run_config.json"

    with open(all_features_path, "w", encoding="utf-8") as f:
        json.dump(all_feature_results, f, indent=2, ensure_ascii=False)
    with open(llm_strategy_path, "w", encoding="utf-8") as f:
        json.dump(llm_strategy_results, f, indent=2, ensure_ascii=False)
    with open(llm_holdout_path, "w", encoding="utf-8") as f:
        json.dump(residual_layer_holdout_results, f, indent=2, ensure_ascii=False)
    with open(attention_raw_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "deprecated": True,
                "note": (
                    "Legacy attention-weight metrics were removed. "
                    "This run extracts attention/MLP activations as probe features instead."
                ),
                "feature_extraction_version": int(FEATURE_EXTRACTION_VERSION),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(attention_summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "deprecated": True,
                "note": (
                    "Legacy mirage-vs-non attention-focus summary is no longer produced. "
                    f"See {vlm_tag}_attention_layer_probe_accuracies.json for activation-family probe results."
                ),
                "num_attention_activation_features": int(len(attention_probe_results)),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(attention_classifier_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "deprecated": True,
                "note": "Legacy threshold classifier over attention-weight metrics is no longer used.",
                "best_layer_by_test_accuracy": best_attention_feature,
                "best_layer_test_accuracy": float(best_attention_val) if best_attention_feature is not None else None,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(attention_probe_path, "w", encoding="utf-8") as f:
        json.dump(attention_probe_results, f, indent=2, ensure_ascii=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                **summary_payload,
                "pairs_path": str(pairs_path),
                "responses_path": str(responses_path),
                "repo_root": str(mirage_root),
                "neutral_as_non_mirage_pairs": bool(args.neutral_as_non_mirage_pairs),
                "vqa_only_pairs": bool(args.vqa_only_pairs),
                "mmmu_only_pairs": bool(args.mmmu_only_pairs),
                "medxpert_only_pairs": bool(args.medxpert_only_pairs),
                "single_benchmark_validation_pairs": int(
                    _single_benchmark_validation_pairs_for_vlm(str(args.vlm))
                ),
                "all_mode_validation_pairs": int(_all_mode_validation_pairs_for_vlm(str(args.vlm))),
                "qwen_validation_fraction_if_applicable": _contrastive_validation_fraction_for_vlm(str(args.vlm)),
                "supported_contrastive_benchmarks": supported_contrastive_benchmarks,
                "vlm": str(args.vlm),
                "model_path": model_path,
                "attn_implementation": args.attn_implementation,
                "device_map": str(getattr(args, "device_map", "")),
                "max_memory_per_gpu_gib": float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
                "max_memory_cpu_gib": float(getattr(args, "max_memory_cpu_gib", 0.0)),
                "features_cache_path": str(cache_path),
                "normalize_features": bool(args.normalize_features),
                "pca_components": int(args.pca_components),
                "split_details": split_details_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Saved feature accuracies: {all_features_path}")
    print(f"Saved LLM strategy best-per-layer: {llm_strategy_path}")
    print(f"Saved held-out feature-group evaluation: {llm_holdout_path}")
    print(f"Saved legacy attention placeholders: {attention_raw_path}, {attention_summary_path}, {attention_classifier_path}")
    print(f"Saved attention-layer probe accuracies: {attention_probe_path}")
    print(f"Saved run config: {config_path}")
    if llm_layers:
        print("\nBest strategy per LLM residual layer (mean validation accuracy across split seeds):")
        for layer in llm_layers:
            info = llm_strategy_results[layer]
            print(f"{layer}: {info['best_strategy']} (validation_accuracy={info['best_validation_accuracy']:.4f})")
    print("\nHeld-out feature-group results (mean train/validation/test across eval seeds):")
    for group_name, info in sorted(residual_layer_holdout_results.items(), key=lambda kv: _layer_sort_key(kv[0])):
        print(
            f"{group_name}: train={info['train_accuracy_mean']:.4f}, "
            f"val={info['validation_accuracy_mean']:.4f}, "
            f"test={info['test_accuracy_mean']:.4f} "
            f"(feature={info['best_feature']})"
        )


if __name__ == "__main__":
    main()
