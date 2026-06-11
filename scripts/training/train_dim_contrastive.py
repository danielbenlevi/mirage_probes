#!/usr/bin/env python3
import argparse
import base64
import importlib
import inspect
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.training.train_log_reg_contrastive as pair_core

DATA_ROOT = Path("./tmp_artifacts")
FEATURE_EXTRACTION_VERSION = 4
SINGLE_BENCHMARK_VALIDATION_PAIRS = 5
MIN_RESPONSE_TOKENS = int(getattr(pair_core, "MIN_RESPONSE_TOKENS", 10))

VLM_MODEL_PATHS = {
    "ovis": "AIDC-AI/Ovis2.5-2B",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
    "glm_4_6v_flash": "zai-org/GLM-4.6V-Flash",
}

VLM_ALIASES = {
    "ovis": "ovis",
    "qwen": "qwen3_vl_32b_instruct",
    "qwen3_vl_32b_instruct": "qwen3_vl_32b_instruct",
    "qwen3-vl-32b-instruct": "qwen3_vl_32b_instruct",
    "qwen/qwen3-vl-32b-instruct": "qwen3_vl_32b_instruct",
    "glm": "glm_4_6v_flash",
    "glm_4_6v_flash": "glm_4_6v_flash",
    "glm-4.6v-flash": "glm_4_6v_flash",
    "zai-org/glm-4.6v": "glm_4_6v_flash",
    "zai-org/glm-4.6v-flash": "glm_4_6v_flash",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train LLM-only DIM token probes from contrastive pairs with a fixed pair-level split. "
            "Supports Ovis/Qwen3-VL/GLM-4.6V-Flash via shared resolver if available."
        )
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default="ovis",
        choices=sorted(VLM_MODEL_PATHS.keys()),
        help="Vision-language model key.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional explicit HF/local model path. Overrides resolver/default for selected --vlm.",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default=None,
        help="Optional explicit contrastive pair artifact path.",
    )
    parser.add_argument(
        "--neutral_pairs_path",
        type=str,
        default=None,
        help="Optional explicit neutral-as-non-mirage pair artifact path.",
    )
    parser.add_argument(
        "--neutral_as_non_mirage_pairs",
        action="store_true",
        help="Use neutral-inclusive contrastive pairs artifact for training.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument(
        "--vqa_only_pairs",
        action="store_true",
        help="Train/eval only VQA-RAD pairs; uses fixed validation-pair count (default 5, GLM default 2).",
    )
    parser.add_argument(
        "--mmmu_only_pairs",
        action="store_true",
        help="Train/eval only MMMU-Pro pairs; uses fixed validation-pair count (default 5, GLM default 2).",
    )
    parser.add_argument(
        "--microvqa_only_pairs",
        action="store_true",
        help="Train/eval only MicroVQA pairs; uses fixed validation-pair count (default 5, GLM default 2).",
    )
    parser.add_argument(
        "--medxpertqa_only_pairs",
        action="store_true",
        help="Train/eval only MedXpertQA-MM pairs; uses fixed validation-pair count (default 5, GLM default 2).",
    )
    parser.add_argument(
        "--single_benchmark_validation_pairs",
        type=int,
        default=SINGLE_BENCHMARK_VALIDATION_PAIRS,
        help="Validation-pair count when running a single benchmark mode (defaults to 5; GLM defaults to 2).",
    )
    parser.add_argument(
        "--num_test_pairs",
        type=int,
        default=4,
        help=(
            "Deprecated and ignored for benchmark-stratified multi-benchmark mode. "
            "Used only for backward-compatible run config metadata."
        ),
    )
    parser.add_argument(
        "--regularization_values",
        type=str,
        default="0,0.01,0.1,1.0",
        help="Comma-separated C values for sweep (must include 0).",
    )
    parser.add_argument("--probe_epochs", type=int, default=800)
    parser.add_argument("--probe_lr", type=float, default=0.03)
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
        default=None,
        help="Optional explicit cache path for token-level layer features.",
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
    parser.add_argument("--force_reextract", action="store_true")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Optional explicit save directory.",
    )
    pair_core.add_model_loading_args(parser)
    return parser.parse_args()


def _normalize_vlm_key(key: str) -> str:
    k = str(key).strip().lower()
    if k in VLM_ALIASES:
        return VLM_ALIASES[k]
    raise ValueError(f"Unsupported --vlm value: {key}")


def _find_shared_resolver():
    resolver_attr_names = (
        "resolve_vlm_model_spec",
        "_resolve_vlm_model_spec",
        "resolve_model_spec",
        "resolve_vlm_model",
    )
    for name in resolver_attr_names:
        fn = getattr(pair_core, name, None)
        if callable(fn):
            return fn

    module_candidates = [
        "vlm_model_resolver",
        "model_resolver",
        "vlm_resolver",
        "common.model_resolver",
        "utils.model_resolver",
    ]
    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        for name in resolver_attr_names:
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn
    return None


def _resolve_model_spec(vlm: str, model_path_override: Optional[str]) -> Dict[str, str]:
    model_key = _normalize_vlm_key(vlm)
    default_model_path = str(VLM_MODEL_PATHS[model_key])
    shared_default_path_resolver = getattr(pair_core, "_default_model_path_for_vlm", None)
    if callable(shared_default_path_resolver):
        try:
            default_model_path = str(shared_default_path_resolver(model_key))
        except Exception:
            pass
    resolver = _find_shared_resolver()
    if resolver is not None:
        for kwargs in (
            {"vlm": model_key, "model_path_override": model_path_override},
            {"model_key": model_key, "model_path_override": model_path_override},
            {"vlm": model_key, "model_path": model_path_override},
            {"model_key": model_key, "model_path": model_path_override},
        ):
            try:
                resolved = resolver(**kwargs)
            except TypeError:
                continue
            except Exception:
                resolved = None
            if resolved is None:
                continue
            if isinstance(resolved, dict):
                resolved_key = _normalize_vlm_key(str(resolved.get("model_key", model_key)))
                resolved_path = str(resolved.get("model_path") or resolved.get("hf_model_id") or "")
                if not resolved_path:
                    resolved_path = str(model_path_override or default_model_path)
                return {"model_key": resolved_key, "model_path": resolved_path}
            if isinstance(resolved, (tuple, list)) and len(resolved) >= 2:
                resolved_key = _normalize_vlm_key(str(resolved[0]))
                resolved_path = str(resolved[1] or model_path_override or default_model_path)
                return {"model_key": resolved_key, "model_path": resolved_path}

    return {
        "model_key": model_key,
        "model_path": str(model_path_override or default_model_path),
    }


def _select_default_input_path(explicit_path: Optional[str], model_key: str, filename: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    legacy = DATA_ROOT / filename
    return pair_core._resolve_model_scoped_artifact_path(
        base_path=legacy,
        vlm_key=str(model_key),
        include_gen_prefix_dirs=True,
    )


def _select_default_output_path(explicit_path: Optional[str], model_key: str, filename: str) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return DATA_ROOT / model_key / filename


def _resolve_selected_benchmark(args: argparse.Namespace) -> Optional[str]:
    selected: List[str] = []
    if args.vqa_only_pairs:
        selected.append("vqa_rad")
    if args.mmmu_only_pairs:
        selected.append("mmmu_pro")
    if args.microvqa_only_pairs:
        selected.append("microvqa")
    if args.medxpertqa_only_pairs:
        selected.append("medxpertqa_mm")
    if len(selected) > 1:
        raise ValueError(f"Select at most one benchmark-only mode; got {selected}")
    return selected[0] if selected else None


def _layer_sort_key(name: str) -> Tuple[int, int, str]:
    base = name.split("__")[0]
    comp = 0 if base.startswith("language_model/") else 1
    m = re.search(r"layer_(\d+)", base)
    if m:
        layer_num = int(m.group(1))
    elif "post_layer_norm" in base:
        layer_num = 10**9
    else:
        layer_num = 10**8
    return (comp, layer_num, name)


def _parse_regularization_values(spec: str) -> List[float]:
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) != 4:
        raise ValueError("--regularization_values must contain exactly 4 values.")
    if 0.0 not in vals:
        raise ValueError("--regularization_values must include 0.")
    return vals


def _fit_probe_and_predict_local(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
    c_value: float,
    normalize_features: bool,
    pca_components: int,
) -> np.ndarray:
    preproc, X_train_s = pair_core._fit_feature_preprocessor(
        X_train=X_train,
        normalize_features=normalize_features,
        pca_components=pca_components,
    )
    X_eval_s = pair_core._apply_feature_preprocessor(X_eval, preproc)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model = torch.nn.Linear(X_train_s.shape[1], 1).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    weight_decay = 0.0 if float(c_value) == 0.0 else (1.0 / float(c_value))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=weight_decay)

    x_train_t = torch.from_numpy(X_train_s).to(device=device, dtype=torch.float32)
    y_train_t = torch.from_numpy(y_train).to(device=device, dtype=torch.float32).unsqueeze(-1)
    model.train()
    for _ in range(int(epochs)):
        logits = model(x_train_t)
        loss = loss_fn(logits, y_train_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    x_eval_t = torch.from_numpy(X_eval_s).to(device=device, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        logits = model(x_eval_t).squeeze(-1)
        preds = (torch.sigmoid(logits) >= 0.5).to(torch.int64).cpu().numpy()
    return preds


def _decode_data_url_to_pil(url: str, model_key: str = "") -> Image.Image:
    shared_decoder = getattr(pair_core, "_decode_data_url_to_pil", None)
    if callable(shared_decoder):
        for kwargs in (
            {"url": url, "model_key": model_key},
            {"url": url},
        ):
            try:
                return shared_decoder(**kwargs)
            except TypeError:
                continue
            except Exception:
                break
    if not str(url).startswith("data:"):
        raise ValueError("Only data URLs are supported in contrastive artifact.")
    parts = str(url).split(",", 1)
    if len(parts) != 2:
        raise ValueError("Malformed data URL.")
    payload = parts[1]
    image_bytes = base64.b64decode(payload)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _to_generic_hf_messages(conversation: List[Dict], model_key: str) -> List[Dict]:
    messages: List[Dict] = []
    for msg in conversation:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
            continue
        if isinstance(content, list):
            converted: List[Dict] = []
            for item in content:
                item_type = str((item or {}).get("type", ""))
                if item_type == "text":
                    converted.append({"type": "text", "text": str((item or {}).get("text", ""))})
                elif item_type == "image_url":
                    url = str(((item or {}).get("image_url") or {}).get("url", ""))
                    converted.append({"type": "image", "image": _decode_data_url_to_pil(url, model_key=model_key)})
                elif item_type == "image":
                    converted.append({"type": "image", "image": (item or {}).get("image")})
            if converted:
                messages.append({"role": role, "content": converted})
            continue
        raise ValueError(f"Unsupported message content type: {type(content)}")
    return messages


def _to_model_messages(conversation: List[Dict], model_key: str) -> List[Dict]:
    converter = getattr(pair_core, "_to_model_messages", None)
    if callable(converter):
        for kwargs in (
            {"conversation": conversation, "model_key": model_key},
            {"conversation": conversation, "vlm": model_key},
        ):
            try:
                return converter(**kwargs)
            except TypeError:
                continue
            except Exception:
                break
    if model_key == "ovis" and hasattr(pair_core, "_to_ovis_messages"):
        return pair_core._to_ovis_messages(conversation)
    return _to_generic_hf_messages(conversation, model_key=model_key)


def _move_tensor_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _prepare_model_inputs(
    model,
    messages: List[Dict],
    model_key: str,
    model_path: str,
    prep_state: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], Any]:
    prepare_fn = getattr(pair_core, "_prepare_inputs", None)
    if callable(prepare_fn):
        attempts = [
            ((model, messages), {"model_key": model_key}),
            ((model, messages), {"vlm": model_key}),
            ((model, messages), {}),
        ]
        for call_args, call_kwargs in attempts:
            try:
                inputs = prepare_fn(*call_args, **call_kwargs)
                tokenizer = getattr(model, "text_tokenizer", None)
                return inputs, tokenizer
            except TypeError:
                continue
            except Exception:
                break

    device = next(model.parameters()).device
    processor = prep_state.get("processor")
    if processor is None:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and not getattr(tokenizer, "chat_template", None):
            model_dir = Path(str(model_path))
            if model_key == "glm_4_6v_flash":
                jinja_path = model_dir / "chat_template.jinja"
                if model_dir.exists() and jinja_path.exists():
                    tokenizer.chat_template = jinja_path.read_text(encoding="utf-8")
        prep_state["processor"] = processor

    chat_template_kwargs: Dict[str, Any] = {
        "add_generation_prompt": False,
        "tokenize": False,
        # Explicitly disable thinking/reasoning where supported by template.
        "enable_thinking": False,
    }
    chat_text = processor.apply_chat_template(messages, **chat_template_kwargs)
    images: List[Image.Image] = []
    for msg in messages:
        for item in msg.get("content", []) if isinstance(msg.get("content", []), list) else []:
            if str(item.get("type", "")) == "image":
                image_obj = item.get("image")
                if image_obj is not None:
                    images.append(image_obj)

    proc_kwargs: Dict[str, Any] = {"text": [chat_text], "return_tensors": "pt"}
    if images:
        proc_kwargs["images"] = images

    batch = processor(**proc_kwargs)
    inputs = _move_tensor_batch_to_device(dict(batch), device=device)
    if "attention_mask" not in inputs and "input_ids" in inputs:
        tokenizer = getattr(processor, "tokenizer", None)
        pad_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
        if pad_id is None:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], device=inputs["input_ids"].device)
        else:
            inputs["attention_mask"] = (inputs["input_ids"] != int(pad_id)).to(device=inputs["input_ids"].device)
    return inputs, getattr(processor, "tokenizer", None)


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


def _run_model_forward(model, inputs: Dict[str, Any]) -> None:
    kwargs = dict(inputs)
    kwargs.setdefault("output_attentions", False)
    kwargs.setdefault("return_dict", True)
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


def _extract_llm_token_features(
    model,
    messages: List[Dict],
    model_key: str,
    model_path: str,
    prep_state: Dict[str, Any],
) -> Tuple[Dict[str, torch.Tensor], int]:
    inputs, tokenizer = _prepare_model_inputs(
        model=model,
        messages=messages,
        model_key=model_key,
        model_path=model_path,
        prep_state=prep_state,
    )

    input_ids = inputs.get("input_ids")
    if input_ids is None or input_ids.dim() < 2:
        raise RuntimeError("Model input preparation did not produce valid input_ids.")

    input_ids_1d = input_ids[0]
    special_ids = set()
    if tokenizer is None:
        tokenizer = getattr(model, "text_tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "all_special_ids"):
        special_ids = set(int(x) for x in tokenizer.all_special_ids)

    text_mask = input_ids_1d >= 0
    for sid in special_ids:
        text_mask = text_mask & (input_ids_1d != sid)
    text_idx = torch.where(text_mask)[0]

    layers, post_norm_module, layer_path = _resolve_llm_layer_modules(model)
    collected: Dict[str, torch.Tensor] = {}
    handles = []

    def _make_llm_hook(base_name: str):
        def hook(_module, _inp, out):
            x = out
            if hasattr(x, "last_hidden_state"):
                x = x.last_hidden_state
            elif isinstance(x, tuple):
                x = x[0]
            if not torch.is_tensor(x) or x.dim() != 3:
                return
            hidden = x[0]
            if text_idx.numel() == 0:
                token_feats = torch.zeros((0, hidden.shape[-1]), dtype=torch.float16)
            else:
                token_feats = hidden[text_idx].detach().to(torch.float16).cpu()
            collected[base_name] = token_feats

        return hook

    try:
        for layer_idx, layer in enumerate(layers):
            h = layer.register_forward_hook(_make_llm_hook(f"language_model/layer_{layer_idx + 1}"))
            handles.append(h)
        if post_norm_module is not None and hasattr(post_norm_module, "register_forward_hook"):
            handles.append(post_norm_module.register_forward_hook(_make_llm_hook("language_model/post_layer_norm")))

        _run_model_forward(model=model, inputs=inputs)
    finally:
        for h in handles:
            h.remove()

    prep_state["layer_path"] = layer_path
    return collected, int(text_idx.numel())


def _torch_load(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cache_compatibility_check(
    payload: Dict[str, Any],
    model_key: str,
    model_path: str,
    pairs_path: Path,
    sample_names: List[str],
) -> Tuple[bool, str]:
    if int(payload.get("feature_extraction_version", -1)) != FEATURE_EXTRACTION_VERSION:
        return False, "feature_extraction_version mismatch"
    if str(payload.get("model_key", "")) != model_key:
        return False, "model_key mismatch"
    if str(payload.get("model_path", "")) != model_path:
        return False, "model_path mismatch"
    if str(payload.get("pairs_source_path", "")) != str(pairs_path):
        return False, "pairs_source_path mismatch"
    cached_samples = payload.get("sample_names")
    if cached_samples is None or list(cached_samples) != list(sample_names):
        return False, "sample_names mismatch"
    required_keys = ("layer_features", "layer_labels", "layer_pair_ids", "layer_sample_names", "layer_order")
    for key in required_keys:
        if key not in payload:
            return False, f"cache missing '{key}'"
    return True, ""


def _group_pair_ids_by_benchmark(pair_ids: List[int], pair_benchmarks: List[str]) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for pid in pair_ids:
        if pid < 0 or pid >= len(pair_benchmarks):
            continue
        grouped[pair_benchmarks[pid]].append(int(pid))
    return {k: sorted(v) for k, v in grouped.items()}


def _choose_pair_split(
    pair_ids: np.ndarray,
    pair_benchmarks: List[str],
    seed: int,
    selected_benchmark: Optional[str],
    single_benchmark_validation_pairs: int,
    vlm_key: str,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, List[int]], str]:
    qwen_val_fraction = pair_core._contrastive_validation_fraction_for_vlm(vlm_key)
    if qwen_val_fraction is not None:
        train_mask, val_mask, val_pairs, by_benchmark = pair_core._split_pair_benchmark_fraction_validation(
            pair_ids=pair_ids,
            pair_benchmarks=pair_benchmarks,
            val_fraction=float(qwen_val_fraction),
            seed=int(seed),
        )
        return train_mask, val_mask, val_pairs, by_benchmark, "benchmark_fraction_qwen_90_10"

    if selected_benchmark is not None:
        train_mask, val_mask, val_pairs = pair_core._split_pair_fixed_validation_count(
            pair_ids=pair_ids,
            num_validation_pairs=int(single_benchmark_validation_pairs),
            seed=int(seed),
        )
        return train_mask, val_mask, val_pairs, {selected_benchmark: [int(x) for x in val_pairs]}, "fixed_count"

    if pair_core._is_glm_vlm(vlm_key):
        train_mask, val_mask, val_pairs = pair_core._split_pair_fixed_validation_count(
            pair_ids=pair_ids,
            num_validation_pairs=int(pair_core._all_mode_validation_pairs_for_vlm(vlm_key)),
            seed=int(seed),
        )
        by_benchmark = _group_pair_ids_by_benchmark(
            pair_ids=[int(x) for x in val_pairs],
            pair_benchmarks=pair_benchmarks,
        )
        return train_mask, val_mask, val_pairs, by_benchmark, "fixed_count_glm_all_mode"

    represented = sorted({b for b in pair_benchmarks if b != "unknown"})
    if len(represented) == 3:
        train_mask, val_mask, val_pairs, by_benchmark = pair_core._split_pair_benchmark_stratified_validation(
            pair_ids=pair_ids,
            pair_benchmarks=pair_benchmarks,
            seed=int(seed),
        )
        return train_mask, val_mask, val_pairs, by_benchmark, "benchmark_stratified"

    train_mask, val_mask, val_pairs = pair_core._split_pair_fixed_validation_count(
        pair_ids=pair_ids,
        num_validation_pairs=int(single_benchmark_validation_pairs),
        seed=int(seed),
    )
    by_benchmark = _group_pair_ids_by_benchmark(pair_ids=[int(x) for x in val_pairs], pair_benchmarks=pair_benchmarks)
    return train_mask, val_mask, val_pairs, by_benchmark, "fixed_count_fallback"


def main() -> None:
    args = parse_args()
    spec = _resolve_model_spec(vlm=args.vlm, model_path_override=args.model_path)
    model_key = spec["model_key"]
    model_path = spec["model_path"]
    supported_benchmarks = pair_core._supported_contrastive_benchmarks_for_vlm(model_key)
    selected_benchmark = _resolve_selected_benchmark(args)
    if selected_benchmark is not None and selected_benchmark not in supported_benchmarks:
        raise ValueError(
            f"Benchmark '{selected_benchmark}' is not supported for --vlm {model_key}. "
            f"Supported benchmarks: {supported_benchmarks}"
        )
    single_benchmark_validation_pairs = int(args.single_benchmark_validation_pairs)
    if (
        pair_core._is_glm_vlm(model_key)
        and single_benchmark_validation_pairs == int(SINGLE_BENCHMARK_VALIDATION_PAIRS)
    ):
        single_benchmark_validation_pairs = int(pair_core._single_benchmark_validation_pairs_for_vlm(model_key))

    pairs_path = _select_default_input_path(args.pairs_path, model_key, "contrastive_conversation_pairs.json")
    neutral_pairs_path = _select_default_input_path(
        args.neutral_pairs_path,
        model_key,
        "contrastive_conversation_pairs_neutral_as_non_mirage.json",
    )
    cache_path = _select_default_output_path(args.features_cache_path, model_key, "contrastive_pair_llm_token_features.pt")
    save_dir = _select_default_output_path(args.save_dir, model_key, "contrastive_probe_results_dim")
    save_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    active_pairs_path = neutral_pairs_path if args.neutral_as_non_mirage_pairs else pairs_path
    with open(active_pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise ValueError(f"No pairs found in {active_pairs_path}")

    if selected_benchmark is not None:
        pairs = [p for p in pairs if pair_core._infer_benchmark_from_pair(p) == selected_benchmark]
        if not pairs:
            raise ValueError(
                f"No pairs found for benchmark '{selected_benchmark}' after filtering in {active_pairs_path}"
            )
    else:
        supported_set = set(supported_benchmarks)
        pairs = [p for p in pairs if pair_core._infer_benchmark_from_pair(p) in supported_set]
        if not pairs:
            raise ValueError(
                f"No supported contrastive pairs found for --vlm {model_key} in {active_pairs_path}. "
                f"Expected benchmarks: {supported_benchmarks}"
            )

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
            if pair_core._conversation_has_image_input(non_conv):
                _, _, non_assistant = pair_core._conversation_signature_from_conv(non_conv)
                if len(re.findall(r"\S+", non_assistant or "")) < MIN_RESPONSE_TOKENS:
                    too_short_with_image_response = True
            if pair_core._conversation_has_image_input(mirage_conv):
                _, _, mirage_assistant = pair_core._conversation_signature_from_conv(mirage_conv)
                if len(re.findall(r"\S+", mirage_assistant or "")) < MIN_RESPONSE_TOKENS:
                    too_short_with_image_response = True
            if too_short_with_image_response:
                skipped_short_training_pairs_count += 1
                continue

        pair_benchmarks.append(pair_core._infer_benchmark_from_pair(pair))
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

    cache_compatible = False
    if cache_path.exists() and not args.force_reextract:
        payload = _torch_load(cache_path)
        cache_compatible, reason = _cache_compatibility_check(
            payload=payload,
            model_key=model_key,
            model_path=model_path,
            pairs_path=active_pairs_path,
            sample_names=sample_names,
        )
        if not cache_compatible:
            print(f"[cache] Ignoring incompatible cache ({reason}): {cache_path}")
    else:
        payload = None

    if cache_compatible and payload is not None:
        layer_features = payload["layer_features"]
        layer_labels = payload["layer_labels"]
        layer_pair_ids = payload["layer_pair_ids"]
        layer_sample_names = payload["layer_sample_names"]
        layer_order = payload["layer_order"]
        token_counts_per_sample = payload.get("token_counts_per_sample", {})
    else:
        model = pair_core.load_vlm_for_extraction(
            model_path,
            attn_implementation=None,
            device_map_raw=str(getattr(args, "device_map", "")),
            max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
            max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
        )

        layer_features: Dict[str, List[np.ndarray]] = defaultdict(list)
        layer_labels: Dict[str, List[np.ndarray]] = defaultdict(list)
        layer_pair_ids: Dict[str, List[np.ndarray]] = defaultdict(list)
        layer_sample_names: Dict[str, List[str]] = defaultdict(list)
        token_counts_per_sample: Dict[str, int] = {}
        prep_state: Dict[str, Any] = {}

        seen_layers = set()
        layer_order: List[str] = []

        for i, conv in enumerate(tqdm(conversations, desc="Extracting LLM token features")):
            messages = _to_model_messages(conv, model_key=model_key)
            sample_feats, token_count = _extract_llm_token_features(
                model=model,
                messages=messages,
                model_key=model_key,
                model_path=model_path,
                prep_state=prep_state,
            )
            token_counts_per_sample[sample_names[i]] = int(token_count)

            for layer_name, tokens in sample_feats.items():
                if layer_name not in seen_layers:
                    seen_layers.add(layer_name)
                    layer_order.append(layer_name)
                if tokens.shape[0] == 0:
                    continue
                n_tok = int(tokens.shape[0])
                layer_features[layer_name].append(tokens.numpy())
                layer_labels[layer_name].append(np.full(n_tok, labels[i], dtype=np.int64))
                layer_pair_ids[layer_name].append(np.full(n_tok, pair_ids[i], dtype=np.int64))
                layer_sample_names[layer_name].extend([sample_names[i]] * n_tok)

        layer_order = sorted(layer_order, key=_layer_sort_key)
        torch.save(
            {
                "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
                "model_key": model_key,
                "model_path": model_path,
                "pairs_source_path": str(active_pairs_path),
                "sample_names": list(sample_names),
                "layer_features": dict(layer_features),
                "layer_labels": dict(layer_labels),
                "layer_pair_ids": dict(layer_pair_ids),
                "layer_sample_names": dict(layer_sample_names),
                "layer_order": list(layer_order),
                "token_counts_per_sample": token_counts_per_sample,
                "layer_source_path": prep_state.get("layer_path", ""),
            },
            cache_path,
        )

    responses_path = pair_core._resolve_responses_path_for_vlm(
        base_responses_path=Path(args.responses_path),
        vlm_key=model_key,
    )
    with open(responses_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    mirage_root = REPO_ROOT.resolve()
    image_lookup_uid, image_lookup_qid = pair_core._build_image_lookup_from_responses(
        mirage_root=mirage_root,
        responses=responses,
    )
    seen_signatures = {pair_core._conversation_signature_from_conv(c) for c in conversations}
    pool_true, pool_false, skipped_short_holdout_candidates = pair_core._build_holdout_pool(
        responses=responses,
        seen_signatures=seen_signatures,
        selected_benchmark=selected_benchmark,
        allowed_benchmarks=supported_benchmarks,
        include_short_response_filter=bool(args.exclude_short_responses_in_holdout),
        image_lookup_uid=image_lookup_uid,
        image_lookup_qid=image_lookup_qid,
    )
    raw_holdout_pool_sizes = {
        "mirage_true": int(len(pool_true)),
        "mirage_false": int(len(pool_false)),
    }
    holdout_selection_plan = pair_core._plan_balanced_holdout_selection(
        pool_true=pool_true,
        pool_false=pool_false,
        requested_num_true=int(args.num_holdout_mirage_true),
        requested_num_false=int(args.num_holdout_mirage_false),
    )
    eval_seeds = [int(args.seed) + i for i in range(int(args.num_eval_seeds))]
    holdout_payloads_by_seed: Dict[int, Dict[str, Any]] = {}
    holdout_selected_sizes_by_seed: Dict[str, Dict[str, Any]] = {}
    eval_model = pair_core.load_vlm_for_extraction(
        model_path,
        attn_implementation=None,
        device_map_raw=str(getattr(args, "device_map", "")),
        max_memory_per_gpu_gib=float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
        max_memory_cpu_gib=float(getattr(args, "max_memory_cpu_gib", 0.0)),
    )
    holdout_prep_state: Dict[str, Any] = {}
    for eval_seed in tqdm(eval_seeds, desc="Extracting unseen holdout DIM features", unit="seed", dynamic_ncols=True):
        selected_examples, selected_counts_by_benchmark = pair_core._select_holdout_examples_balanced_by_benchmark(
            pool_true=pool_true,
            pool_false=pool_false,
            selected_pairs_by_benchmark={
                str(k): int(v)
                for k, v in dict(holdout_selection_plan["selected_pairs_by_benchmark"]).items()
            },
            seed=int(eval_seed),
        )
        holdout_layer_features: Dict[str, List[np.ndarray]] = defaultdict(list)
        holdout_layer_labels: Dict[str, List[np.ndarray]] = defaultdict(list)
        holdout_layer_benchmarks: Dict[str, List[str]] = defaultdict(list)
        for item in selected_examples:
            messages = _to_model_messages(item["conversation"], model_key=model_key)
            sample_feats, _token_count = _extract_llm_token_features(
                model=eval_model,
                messages=messages,
                model_key=model_key,
                model_path=model_path,
                prep_state=holdout_prep_state,
            )
            label = 1 if bool(item.get("mirage_like")) else 0
            bench = str(item.get("dataset", "unknown"))
            for layer_name, tokens in sample_feats.items():
                if int(tokens.shape[0]) <= 0:
                    continue
                arr = tokens.numpy()
                n_tok = int(arr.shape[0])
                holdout_layer_features[layer_name].append(np.asarray(arr, dtype=np.float32))
                holdout_layer_labels[layer_name].append(np.full(n_tok, int(label), dtype=np.int64))
                holdout_layer_benchmarks[layer_name].extend([bench] * n_tok)
        holdout_payloads_by_seed[int(eval_seed)] = {
            "features": {
                str(layer): np.concatenate(vals, axis=0) if vals else np.zeros((0, 0), dtype=np.float32)
                for layer, vals in holdout_layer_features.items()
            },
            "labels": {
                str(layer): np.concatenate(vals, axis=0) if vals else np.zeros((0,), dtype=np.int64)
                for layer, vals in holdout_layer_labels.items()
            },
            "benchmarks": {
                str(layer): [str(x) for x in vals]
                for layer, vals in holdout_layer_benchmarks.items()
            },
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
        }
        holdout_selected_sizes_by_seed[str(int(eval_seed))] = {
            "num_examples_total": int(len(selected_examples)),
            "num_examples_per_class": {
                "mirage_true": int(sum(1 for x in selected_examples if bool(x.get("mirage_like")))),
                "mirage_false": int(sum(1 for x in selected_examples if not bool(x.get("mirage_like")))),
            },
            "selected_counts_by_benchmark": selected_counts_by_benchmark,
        }

    reg_values = _parse_regularization_values(args.regularization_values)
    per_layer_results: Dict[str, Dict] = {}

    for layer_name in tqdm(layer_order, desc="Training LLM DIM token probes"):
        feats_list = layer_features.get(layer_name, [])
        labels_list = layer_labels.get(layer_name, [])
        pair_ids_list = layer_pair_ids.get(layer_name, [])
        if not feats_list or not labels_list or not pair_ids_list:
            continue

        X = np.concatenate([np.asarray(x, dtype=np.float32) for x in feats_list], axis=0)
        y = np.concatenate([np.asarray(x, dtype=np.int64) for x in labels_list], axis=0)
        pids = np.concatenate([np.asarray(x, dtype=np.int64) for x in pair_ids_list], axis=0)

        train_mask, val_mask, validation_pairs, validation_pairs_by_benchmark, split_strategy = _choose_pair_split(
            pair_ids=pids,
            pair_benchmarks=pair_benchmarks,
            seed=int(args.seed),
            selected_benchmark=selected_benchmark,
            single_benchmark_validation_pairs=int(single_benchmark_validation_pairs),
            vlm_key=model_key,
        )
        if not train_mask.any() or not val_mask.any():
            continue

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        val_benchmark_labels = [str(pair_benchmarks[int(pid)]) for pid in pids[val_mask].tolist()]

        best_c = None
        best_train_acc = -1.0
        best_val_acc = -1.0
        best_class0_val_acc = None
        best_class1_val_acc = None
        best_benchmark_val_acc: Dict[str, float] = {}
        best_benchmark_class0_val_acc: Dict[str, float] = {}
        best_benchmark_class1_val_acc: Dict[str, float] = {}
        sweep_results = []
        for c in reg_values:
            train_pred = _fit_probe_and_predict_local(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_train,
                seed=int(pair_core._probe_init_seed(int(args.seed), float(c), 0)),
                epochs=int(args.probe_epochs),
                lr=float(args.probe_lr),
                c_value=float(c),
                normalize_features=bool(args.normalize_features),
                pca_components=int(args.pca_components),
            )
            train_acc = float((train_pred == y_train).mean())

            val_pred = _fit_probe_and_predict_local(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_val,
                seed=int(pair_core._probe_init_seed(int(args.seed), float(c), 0)),
                epochs=int(args.probe_epochs),
                lr=float(args.probe_lr),
                c_value=float(c),
                normalize_features=bool(args.normalize_features),
                pca_components=int(args.pca_components),
            )
            val_acc = float((val_pred == y_val).mean())
            class0_mask = y_val == 0
            class1_mask = y_val == 1
            class0_val_acc = float((val_pred[class0_mask] == y_val[class0_mask]).mean()) if class0_mask.any() else None
            class1_val_acc = float((val_pred[class1_mask] == y_val[class1_mask]).mean()) if class1_mask.any() else None
            benchmark_val_acc, benchmark_class0_val_acc, benchmark_class1_val_acc = pair_core._compute_benchmark_test_metrics(
                y_true=y_val,
                y_pred=val_pred,
                benchmark_labels=val_benchmark_labels,
            )

            sweep_results.append(
                {
                    "c_value": float(c),
                    "train_accuracy": train_acc,
                    "validation_accuracy": val_acc,
                    "class0_validation_accuracy": class0_val_acc,
                    "class1_validation_accuracy": class1_val_acc,
                    "benchmark_validation_accuracy": benchmark_val_acc,
                    "benchmark_class0_validation_accuracy": benchmark_class0_val_acc,
                    "benchmark_class1_validation_accuracy": benchmark_class1_val_acc,
                }
            )
            if (val_acc > best_val_acc) or (val_acc == best_val_acc and (best_c is None or c < best_c)):
                best_c = float(c)
                best_train_acc = train_acc
                best_val_acc = val_acc
                best_class0_val_acc = class0_val_acc
                best_class1_val_acc = class1_val_acc
                best_benchmark_val_acc = benchmark_val_acc
                best_benchmark_class0_val_acc = benchmark_class0_val_acc
                best_benchmark_class1_val_acc = benchmark_class1_val_acc

        heldout_seed_runs: List[Dict[str, Any]] = []
        for eval_seed in eval_seeds:
            hp = holdout_payloads_by_seed.get(int(eval_seed), {})
            X_holdout = np.asarray((hp.get("features", {}) or {}).get(layer_name, np.zeros((0, X.shape[1]), dtype=np.float32)))
            y_holdout = np.asarray((hp.get("labels", {}) or {}).get(layer_name, np.zeros((0,), dtype=np.int64)))
            holdout_benchmarks = [str(x) for x in ((hp.get("benchmarks", {}) or {}).get(layer_name, []) or [])]
            if X_holdout.size == 0 or y_holdout.size == 0:
                continue
            holdout_pred = _fit_probe_and_predict_local(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_holdout,
                seed=int(pair_core._probe_init_seed(int(args.seed), float(best_c), 0)),
                epochs=int(args.probe_epochs),
                lr=float(args.probe_lr),
                c_value=float(best_c),
                normalize_features=bool(args.normalize_features),
                pca_components=int(args.pca_components),
            )
            holdout_acc = float((holdout_pred == y_holdout).mean())
            h_c0_mask = y_holdout == 0
            h_c1_mask = y_holdout == 1
            h_c0 = float((holdout_pred[h_c0_mask] == y_holdout[h_c0_mask]).mean()) if h_c0_mask.any() else None
            h_c1 = float((holdout_pred[h_c1_mask] == y_holdout[h_c1_mask]).mean()) if h_c1_mask.any() else None
            h_bench, h_bench_c0, h_bench_c1 = pair_core._compute_benchmark_test_metrics(
                y_true=y_holdout,
                y_pred=holdout_pred,
                benchmark_labels=holdout_benchmarks,
            )
            heldout_seed_runs.append(
                {
                    "eval_seed": int(eval_seed),
                    "test_accuracy": float(holdout_acc),
                    "test_accuracy_mirage_false": h_c0,
                    "test_accuracy_mirage_true": h_c1,
                    "benchmark_test_accuracy": h_bench,
                    "benchmark_class0_test_accuracy": h_bench_c0,
                    "benchmark_class1_test_accuracy": h_bench_c1,
                    "num_holdout_tokens": int(len(y_holdout)),
                }
            )

        heldout_test_scores = [float(r["test_accuracy"]) for r in heldout_seed_runs if r.get("test_accuracy") is not None]
        heldout_class0_scores = [
            float(r["test_accuracy_mirage_false"]) for r in heldout_seed_runs if r.get("test_accuracy_mirage_false") is not None
        ]
        heldout_class1_scores = [
            float(r["test_accuracy_mirage_true"]) for r in heldout_seed_runs if r.get("test_accuracy_mirage_true") is not None
        ]
        heldout_benchmark_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=heldout_seed_runs,
            key="benchmark_test_accuracy",
        )
        heldout_benchmark_class0_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=heldout_seed_runs,
            key="benchmark_class0_test_accuracy",
        )
        heldout_benchmark_class1_scores = pair_core._mean_dict_metrics_from_seed_runs(
            seed_runs=heldout_seed_runs,
            key="benchmark_class1_test_accuracy",
        )

        per_layer_results[layer_name] = {
            "probe_type": "llm_dim_token_probe",
            "feature_definition": "individual_text_only_non_special_tokens",
            "selection_metric": "validation_accuracy",
            "split_strategy": split_strategy,
            "best_c": best_c,
            "best_train_accuracy": float(best_train_acc),
            "validation_accuracy_at_best_c": float(best_val_acc),
            "test_accuracy_at_best_c": (float(np.mean(heldout_test_scores)) if heldout_test_scores else None),
            "class0_test_accuracy_at_best_c": (float(np.mean(heldout_class0_scores)) if heldout_class0_scores else None),
            "class1_test_accuracy_at_best_c": (float(np.mean(heldout_class1_scores)) if heldout_class1_scores else None),
            "benchmark_validation_accuracy_at_best_c": best_benchmark_val_acc,
            "benchmark_class0_validation_accuracy_at_best_c": best_benchmark_class0_val_acc,
            "benchmark_class1_validation_accuracy_at_best_c": best_benchmark_class1_val_acc,
            "benchmark_test_accuracy_at_best_c": heldout_benchmark_scores,
            "benchmark_class0_test_accuracy_at_best_c": heldout_benchmark_class0_scores,
            "benchmark_class1_test_accuracy_at_best_c": heldout_benchmark_class1_scores,
            "num_tokens_total": int(X.shape[0]),
            "num_tokens_train": int(X_train.shape[0]),
            "num_tokens_validation": int(X_val.shape[0]),
            "num_tokens_test_mean_across_eval_seeds": (
                float(np.mean([int(r["num_holdout_tokens"]) for r in heldout_seed_runs])) if heldout_seed_runs else None
            ),
            "num_pairs_validation": int(len(validation_pairs)),
            "validation_pair_ids": [int(x) for x in validation_pairs],
            "validation_pairs_by_benchmark": {k: [int(x) for x in v] for k, v in validation_pairs_by_benchmark.items()},
            "class_balance_total": {
                "non_mirage_tokens": int((y == 0).sum()),
                "mirage_tokens": int((y == 1).sum()),
            },
            "heldout_seed_runs": heldout_seed_runs,
            "sweep": sweep_results,
        }

    results_name = f"{model_key}_llm_dim_token_probe_accuracies.json"
    out_path = save_dir / results_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(per_layer_results, f, indent=2, ensure_ascii=False)

    config_name = f"{model_key}_llm_dim_token_probe_run_config.json"
    config_path = save_dir / config_name
    run_config = {
        "vlm": model_key,
        "model_path": model_path,
        "device_map": str(getattr(args, "device_map", "")),
        "max_memory_per_gpu_gib": float(getattr(args, "max_memory_per_gpu_gib", 0.0)),
        "max_memory_cpu_gib": float(getattr(args, "max_memory_cpu_gib", 0.0)),
        "pairs_path": str(active_pairs_path),
        "responses_path": str(responses_path),
        "repo_root": str(mirage_root),
        "neutral_as_non_mirage_pairs": bool(args.neutral_as_non_mirage_pairs),
        "selected_benchmark": selected_benchmark,
        "single_benchmark_validation_pairs": int(single_benchmark_validation_pairs),
        "all_mode_validation_pairs": int(pair_core._all_mode_validation_pairs_for_vlm(model_key)),
        "qwen_validation_fraction_if_applicable": pair_core._contrastive_validation_fraction_for_vlm(model_key),
        "supported_contrastive_benchmarks": supported_benchmarks,
        "seed": int(args.seed),
        "max_pairs": int(args.max_pairs),
        "exclude_short_responses_in_training_pairs": bool(args.exclude_short_responses_in_training_pairs),
        "min_response_tokens_required_for_training_pairs": (
            int(MIN_RESPONSE_TOKENS) if bool(args.exclude_short_responses_in_training_pairs) else None
        ),
        "num_training_pairs_skipped_short_responses": int(skipped_short_training_pairs_count),
        "exclude_short_responses_in_holdout": bool(args.exclude_short_responses_in_holdout),
        "num_holdout_candidates_skipped_short_responses": int(skipped_short_holdout_candidates),
        "num_holdout_mirage_true": int(args.num_holdout_mirage_true),
        "num_holdout_mirage_false": int(args.num_holdout_mirage_false),
        "num_eval_seeds": int(args.num_eval_seeds),
        "eval_seeds": [int(x) for x in eval_seeds],
        "holdout_pool_sizes_before_selection": raw_holdout_pool_sizes,
        "holdout_selection_plan": holdout_selection_plan,
        "holdout_selected_sizes_by_seed": holdout_selected_sizes_by_seed,
        "num_test_pairs": int(args.num_test_pairs),
        "regularization_values": reg_values,
        "probe_epochs": int(args.probe_epochs),
        "probe_lr": float(args.probe_lr),
        "normalize_features": bool(args.normalize_features),
        "pca_components": int(args.pca_components),
        "features_cache_path": str(cache_path),
        "save_dir": str(save_dir),
        "feature_extraction_version": FEATURE_EXTRACTION_VERSION,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "vlm": model_key,
                "layers_trained": len(per_layer_results),
                "results_path": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
