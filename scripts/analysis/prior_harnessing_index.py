#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.data.gen_mutations_get_responses as core
import scripts.training.train_log_reg_contrastive as pair_core


DEFAULT_DATASETS = ("vqa_rad", "mmmu_pro", "medxpertqa_mm")
DEFAULT_VLM_KEY = "ovis"
DEFAULT_Q_NULL_TEXT = "What is the Answer?"
DEFAULT_ANSWER_PREFIX = "[["


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Prior Harnessing Index (PHI) from existing VLM response artifacts without "
            "regenerating responses. PHI(Q)=logp(a*|Q)-logp(a*|Q_null), where Q_null defaults to "
            f"'{DEFAULT_Q_NULL_TEXT}'."
        )
    )
    parser.add_argument(
        "--responses_path",
        type=str,
        default=str(Path("tmp_artifacts") / "responses.json"),
        help=(
            "Base responses artifact path. Resolved to a model-scoped artifact by --vlm "
            "(e.g., Ovis/Qwen renamed artifacts) when available."
        ),
    )
    parser.add_argument(
        "--mirage_root",
        type=str,
        default=".",
        help="Repository root used for dataset/image lookup.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets to evaluate.",
    )
    parser.add_argument(
        "--num_questions_per_dataset",
        type=int,
        default=200,
        help="Maximum number of questions per dataset after class-target sampling.",
    )
    parser.add_argument(
        "--target_mirage_per_dataset",
        type=int,
        default=100,
        help="Target number of mirage-like rows (mirage_like==True) per dataset.",
    )
    parser.add_argument(
        "--target_non_mirage_per_dataset",
        type=int,
        default=100,
        help="Target number of non-mirage rows (mirage_like==False) per dataset.",
    )
    parser.add_argument(
        "--original_only",
        action="store_true",
        default=False,
        help="Use only rows marked as original question variants.",
    )
    parser.add_argument(
        "--no_original_only",
        dest="original_only",
        action="store_false",
        help="Allow non-original variants.",
    )
    parser.add_argument(
        "--shuffle_before_select",
        action="store_true",
        help="Shuffle candidate rows per dataset before selecting first N.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--q_null_text",
        type=str,
        default=DEFAULT_Q_NULL_TEXT,
        help="Dummy question text for Q_null.",
    )
    parser.add_argument(
        "--answer_prefix",
        type=str,
        default=DEFAULT_ANSWER_PREFIX,
        help="Assistant prefill prefix used before scoring the answer token.",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default=DEFAULT_VLM_KEY,
        choices=sorted(pair_core.VLM_MODEL_PATHS.keys()),
        help="VLM key for model loading and model-scoped response artifact resolution.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Optional explicit model path override.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path("tmp_artifacts") / "prior_harnessing_index"),
    )
    parser.add_argument(
        "--score_batch_size",
        type=int,
        default=1,
        help="Micro-batch size for PHI logprob scoring forward passes.",
    )
    pair_core.add_model_loading_args(parser)
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_original_row(row: Dict[str, Any]) -> bool:
    if row.get("is_original") is True:
        return True
    return str(row.get("variant_id", "")).strip().lower() == "original"


def _dataset_rows(
    all_rows: Sequence[Dict[str, Any]],
    dataset_name: str,
    original_only: bool,
) -> List[Dict[str, Any]]:
    rows = [r for r in all_rows if str(r.get("dataset", "")) == str(dataset_name)]
    if original_only:
        rows = [r for r in rows if _is_original_row(r)]
    dedup: List[Dict[str, Any]] = []
    seen: set = set()
    for row in rows:
        key = (
            str(row.get("dataset", "")),
            str(row.get("unique_id", "")),
            str(row.get("question_id", "")),
            str(row.get("variant_id", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(row)
    return dedup


def _row_mirage_label(row: Dict[str, Any]) -> Optional[bool]:
    wo = row.get("without_image", {}) or {}
    val = wo.get("mirage_like")
    if val is True:
        return True
    if val is False:
        return False
    return None


def _select_rows_with_non_mirage_priority(
    rows: Sequence[Dict[str, Any]],
    max_total: int,
    target_mirage: int,
    target_non_mirage: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    non_mirage_rows: List[Dict[str, Any]] = []
    mirage_rows: List[Dict[str, Any]] = []
    neutral_rows: List[Dict[str, Any]] = []
    for row in rows:
        label = _row_mirage_label(row)
        if label is False:
            non_mirage_rows.append(row)
        elif label is True:
            mirage_rows.append(row)
        else:
            neutral_rows.append(row)

    if max_total <= 0:
        selected_mirage = min(max(0, int(target_mirage)), len(mirage_rows))
        selected_non = min(max(0, int(target_non_mirage)), len(non_mirage_rows))
        selected = list(mirage_rows[:selected_mirage]) + list(non_mirage_rows[:selected_non])
        stats = {
            "selected_total": int(len(selected)),
            "selected_mirage": int(selected_mirage),
            "selected_non_mirage": int(selected_non),
            "available_total": int(len(rows)),
            "available_mirage": int(len(mirage_rows)),
            "available_non_mirage": int(len(non_mirage_rows)),
            "available_neutral": int(len(neutral_rows)),
            "target_mirage": int(target_mirage),
            "target_non_mirage": int(target_non_mirage),
            "selected_other": int(len(selected) - selected_non - selected_mirage),
            "selected_neutral": 0,
        }
        return selected, stats

    mir_target = max(0, int(target_mirage))
    non_target = max(0, int(target_non_mirage))
    max_total = int(max_total)
    if (mir_target + non_target) > max_total:
        half = max_total // 2
        mir_target = min(mir_target, half)
        non_target = min(non_target, max_total - mir_target)

    take_mirage = min(mir_target, len(mirage_rows))
    take_non = min(non_target, len(non_mirage_rows))
    selected = list(mirage_rows[:take_mirage]) + list(non_mirage_rows[:take_non])
    if len(selected) > max_total:
        selected = selected[:max_total]
        take_mirage = sum(int(_row_mirage_label(r) is True) for r in selected)
        take_non = sum(int(_row_mirage_label(r) is False) for r in selected)

    stats = {
        "selected_total": int(len(selected)),
        "selected_mirage": int(take_mirage),
        "selected_non_mirage": int(take_non),
        "available_total": int(len(rows)),
        "available_mirage": int(len(mirage_rows)),
        "available_non_mirage": int(len(non_mirage_rows)),
        "available_neutral": int(len(neutral_rows)),
        "target_mirage": int(mir_target),
        "target_non_mirage": int(non_target),
        "selected_other": int(len(selected) - take_non - take_mirage),
        "selected_neutral": 0,
    }
    return selected, stats


def _build_image_lookup(
    mirage_root: Path,
    datasets: Sequence[str],
) -> Tuple[Dict[Tuple[str, str], List[bytes]], Dict[Tuple[str, str], List[bytes]]]:
    uid_lookup: Dict[Tuple[str, str], List[bytes]] = {}
    qid_lookup: Dict[Tuple[str, str], List[bytes]] = {}
    for ds in datasets:
        items = core._load_dataset_items(mirage_root=mirage_root, dataset_name=str(ds))
        for item in items:
            uid = str(item.get("unique_id", ""))
            qid = str(item.get("question_id", ""))
            images = list(item.get("images", []) or [])
            uid_lookup[(str(ds), uid)] = images
            if qid:
                qid_lookup[(str(ds), qid)] = images
    return uid_lookup, qid_lookup


def _resolve_images_for_row(
    row: Dict[str, Any],
    uid_lookup: Dict[Tuple[str, str], List[bytes]],
    qid_lookup: Dict[Tuple[str, str], List[bytes]],
) -> Optional[List[bytes]]:
    ds = str(row.get("dataset", ""))
    uid = str(row.get("unique_id", ""))
    qid = str(row.get("question_id", ""))
    imgs = uid_lookup.get((ds, uid))
    if imgs is None:
        imgs = qid_lookup.get((ds, qid))
    if imgs is None:
        return None
    return list(imgs)


def _extract_first_open_token(text: str) -> Optional[str]:
    raw = str(text or "").strip()
    if not raw:
        return None
    for tok in raw.split():
        token = tok.strip().strip("\"'`[](){}.,;:!?")
        if token:
            return token
    return None


def _resolve_target_token(row: Dict[str, Any]) -> Optional[str]:
    task_type = str(row.get("task_type", "")).strip().lower()
    if task_type == "mcq":
        gt = str(row.get("ground_truth_normalized", "") or row.get("ground_truth_raw", "")).strip().upper()
        if gt and gt[0] in core.CHOICE_LETTERS:
            return gt[0]
        return None
    gt_raw = str(row.get("ground_truth_raw", "")).strip()
    return _extract_first_open_token(gt_raw)


def _build_prompt_text(question_text: str, options_text: str) -> str:
    q = str(question_text or "").strip()
    opts = str(options_text or "").strip()
    if opts:
        return f"{q}\n\n{opts}"
    return q


def _build_q_null_prompt(row: Dict[str, Any], q_null_text: str) -> str:
    return _build_prompt_text(question_text=q_null_text, options_text=str(row.get("options", "")))


def _append_assistant_turn(conversation: List[Dict[str, Any]], assistant_text: str) -> List[Dict[str, Any]]:
    conv = list(conversation)
    conv.append({"role": "assistant", "content": str(assistant_text)})
    return conv


def _sequence_length(inputs: Dict[str, Any]) -> int:
    attention_mask = inputs.get("attention_mask")
    if torch.is_tensor(attention_mask):
        return int(attention_mask[0].sum().item())
    input_ids = inputs.get("input_ids")
    if torch.is_tensor(input_ids):
        return int(input_ids.shape[1])
    return 0


def _run_forward_get_logits(model, inputs: Dict[str, Any]) -> torch.Tensor:
    kwargs = dict(inputs)
    kwargs.setdefault("return_dict", True)

    try:
        with torch.inference_mode():
            output = model.forward(**kwargs)
    except TypeError:
        sig = inspect.signature(model.forward)
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_var_kwargs:
            with torch.inference_mode():
                output = model.forward(**kwargs)
        else:
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            with torch.inference_mode():
                output = model.forward(**filtered)

    logits = getattr(output, "logits", None)
    if logits is None:
        if isinstance(output, (tuple, list)) and output:
            logits = output[0]
    if logits is None:
        raise RuntimeError("Model forward pass did not return logits.")
    return logits


def _score_first_assistant_token_logprob(
    model,
    model_key: str,
    conversation_pre_response: List[Dict[str, Any]],
    target_token_text: str,
    answer_prefix: str,
) -> Dict[str, Any]:
    prefix = str(answer_prefix or "")
    conv_prefix = _append_assistant_turn(conversation_pre_response, prefix)
    conv_target = _append_assistant_turn(conversation_pre_response, f"{prefix}{target_token_text}")

    messages_empty = pair_core._to_model_messages(conversation=conv_prefix, model_key=model_key)
    messages_target = pair_core._to_model_messages(conversation=conv_target, model_key=model_key)

    inputs_empty = pair_core._prepare_inputs(model=model, messages=messages_empty, model_key=model_key)
    inputs_target = pair_core._prepare_inputs(model=model, messages=messages_target, model_key=model_key)

    len_empty = _sequence_length(inputs_empty)
    len_target = _sequence_length(inputs_target)
    if len_target <= len_empty:
        raise RuntimeError(
            f"Target assistant content added no tokens after prefix (len_empty={len_empty}, len_target={len_target}, "
            f"target_token_text={target_token_text!r})."
        )
    if len_empty <= 0:
        raise RuntimeError(f"Unexpected empty prompt token length for target {target_token_text!r}.")

    token_pos = int(len_empty)
    pred_pos = int(token_pos - 1)
    input_ids = inputs_target["input_ids"]
    if pred_pos >= int(input_ids.shape[1]) - 1:
        raise RuntimeError("Token position out of bounds while scoring target token.")

    target_token_id = int(input_ids[0, token_pos].item())

    logits = _run_forward_get_logits(model=model, inputs=inputs_target)
    step_logits = logits[0, pred_pos, :].to(torch.float32)
    log_probs = torch.log_softmax(step_logits, dim=-1)
    logp = float(log_probs[target_token_id].item())

    decoded_token = None
    try:
        if hasattr(model, "text_tokenizer") and model.text_tokenizer is not None:
            decoded_token = str(model.text_tokenizer.decode([target_token_id], skip_special_tokens=False))
        else:
            processor = pair_core._get_processor_for_model(model=model, model_key=model_key)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                decoded_token = str(tokenizer.decode([target_token_id], skip_special_tokens=False))
    except Exception:
        decoded_token = None

    return {
        "logp": logp,
        "target_token_id": int(target_token_id),
        "decoded_token": decoded_token,
        "answer_prefix": prefix,
        "prompt_seq_len": int(len_empty),
        "target_seq_len": int(len_target),
    }


def _prepare_assistant_token_scoring_item(
    model,
    model_key: str,
    conversation_pre_response: List[Dict[str, Any]],
    target_token_text: str,
    answer_prefix: str,
) -> Dict[str, Any]:
    prefix = str(answer_prefix or "")
    conv_prefix = _append_assistant_turn(conversation_pre_response, prefix)
    conv_target = _append_assistant_turn(conversation_pre_response, f"{prefix}{target_token_text}")

    messages_empty = pair_core._to_model_messages(conversation=conv_prefix, model_key=model_key)
    messages_target = pair_core._to_model_messages(conversation=conv_target, model_key=model_key)
    inputs_empty = pair_core._prepare_inputs(model=model, messages=messages_empty, model_key=model_key)
    inputs_target = pair_core._prepare_inputs(model=model, messages=messages_target, model_key=model_key)

    len_empty = _sequence_length(inputs_empty)
    len_target = _sequence_length(inputs_target)
    if len_target <= len_empty:
        raise RuntimeError(
            f"Target assistant content added no tokens after prefix (len_empty={len_empty}, len_target={len_target}, "
            f"target_token_text={target_token_text!r})."
        )
    if len_empty <= 0:
        raise RuntimeError(f"Unexpected empty prompt token length for target {target_token_text!r}.")

    token_pos = int(len_empty)
    pred_pos = int(token_pos - 1)
    input_ids = inputs_target["input_ids"]
    if pred_pos >= int(input_ids.shape[1]) - 1:
        raise RuntimeError("Token position out of bounds while scoring target token.")
    target_token_id = int(input_ids[0, token_pos].item())

    return {
        "inputs_target": inputs_target,
        "input_keys_signature": tuple(sorted(str(k) for k in inputs_target.keys())),
        "pred_pos": int(pred_pos),
        "target_token_id": int(target_token_id),
        "prompt_seq_len": int(len_empty),
        "target_seq_len": int(len_target),
        "answer_prefix": prefix,
    }


def _pad_2d_right(tensor: torch.Tensor, target_len: int, pad_value: float) -> torch.Tensor:
    if tensor.dim() != 2:
        raise ValueError(f"Expected 2D tensor for padding, got shape={tuple(tensor.shape)}")
    cur = int(tensor.shape[1])
    if cur == target_len:
        return tensor
    if cur > target_len:
        return tensor[:, :target_len]
    pad = torch.full(
        (int(tensor.shape[0]), int(target_len - cur)),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, pad], dim=1)


def _collate_inputs_for_batch(input_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not input_items:
        raise ValueError("Cannot collate empty input_items.")
    if len(input_items) == 1:
        return dict(input_items[0])

    first = input_items[0]
    out: Dict[str, Any] = {}
    keys = set(first.keys())
    for item in input_items[1:]:
        if set(item.keys()) != keys:
            raise RuntimeError(
                "Batch collation requires identical input key sets per sample. "
                f"Found mismatch: base={sorted(keys)} vs sample={sorted(item.keys())}"
            )

    max_seq = 0
    for item in input_items:
        inp = item.get("input_ids")
        if not torch.is_tensor(inp):
            raise RuntimeError("Each batch item must contain tensor input_ids.")
        max_seq = max(max_seq, int(inp.shape[1]))

    for key in sorted(keys):
        values = [item[key] for item in input_items]
        if all(torch.is_tensor(v) for v in values):
            tensors = [v for v in values if torch.is_tensor(v)]
            if key in {"input_ids", "attention_mask"}:
                if key == "input_ids":
                    pad_val = 0
                else:
                    pad_val = 0
                padded = [_pad_2d_right(t, target_len=max_seq, pad_value=pad_val) for t in tensors]
                out[key] = torch.cat(padded, dim=0)
                continue

            # Multimodal side tensors are usually flattened across samples by processor;
            # concatenating dim 0 mirrors processor batch behavior.
            if key in {"pixel_values", "grid_thws", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}:
                out[key] = torch.cat(tensors, dim=0)
                continue

            # Common case: batched [1, ...] tensors.
            if all(int(t.shape[0]) == 1 for t in tensors):
                out[key] = torch.cat(tensors, dim=0)
                continue

            # Fallback for same-shape tensors.
            same_shape = all(tuple(t.shape) == tuple(tensors[0].shape) for t in tensors)
            if same_shape:
                out[key] = torch.stack(tensors, dim=0)
                continue

            # Last resort: concatenate dim 0 if remaining dims match.
            compatible = all(tuple(t.shape[1:]) == tuple(tensors[0].shape[1:]) for t in tensors)
            if compatible:
                out[key] = torch.cat(tensors, dim=0)
                continue

            raise RuntimeError(f"Could not collate tensor key={key} with shapes {[tuple(t.shape) for t in tensors]}")
        else:
            # Keep first non-tensor value for stable kwargs like booleans/None.
            out[key] = values[0]
    return out


def _score_assistant_token_logprobs_batch(
    model,
    model_key: str,
    conversations_pre_response: Sequence[List[Dict[str, Any]]],
    target_token_texts: Sequence[str],
    answer_prefix: str,
    batch_size: int,
    progress_desc: str = "",
) -> List[Dict[str, Any]]:
    if len(conversations_pre_response) != len(target_token_texts):
        raise ValueError("conversations_pre_response and target_token_texts must have equal length.")
    n = len(conversations_pre_response)
    if n == 0:
        return []

    prepared: List[Dict[str, Any]] = []
    for conv, token_text in zip(conversations_pre_response, target_token_texts):
        prepared.append(
            _prepare_assistant_token_scoring_item(
                model=model,
                model_key=model_key,
                conversation_pre_response=conv,
                target_token_text=token_text,
                answer_prefix=answer_prefix,
            )
        )

    results: List[Optional[Dict[str, Any]]] = [None] * n
    bsz = max(1, int(batch_size))

    # Avoid mixing samples with different processor/model input-key signatures
    # (e.g., with-image vs no-image tensor sets), which can mis-collate multimodal inputs.
    sig_to_indices: Dict[Tuple[str, ...], List[int]] = {}
    for idx, item in enumerate(prepared):
        sig = tuple(item["input_keys_signature"])
        sig_to_indices.setdefault(sig, []).append(idx)

    total_batches = sum((len(indices) + bsz - 1) // bsz for indices in sig_to_indices.values())
    pbar = tqdm(
        total=total_batches,
        desc=(progress_desc or "PHI scoring"),
        unit="batch",
        dynamic_ncols=True,
    )
    for _sig, indices in sig_to_indices.items():
        for group_start in range(0, len(indices), bsz):
            group_idx = indices[group_start : group_start + bsz]
            chunk = [prepared[i] for i in group_idx]
            batch_inputs = _collate_inputs_for_batch([c["inputs_target"] for c in chunk])
            logits = _run_forward_get_logits(model=model, inputs=batch_inputs)

            for j, item in enumerate(chunk):
                pred_pos = int(item["pred_pos"])
                target_token_id = int(item["target_token_id"])
                step_logits = logits[j, pred_pos, :].to(torch.float32)
                log_probs = torch.log_softmax(step_logits, dim=-1)
                logp = float(log_probs[target_token_id].item())
                decoded_token = None
                try:
                    if hasattr(model, "text_tokenizer") and model.text_tokenizer is not None:
                        decoded_token = str(model.text_tokenizer.decode([target_token_id], skip_special_tokens=False))
                    else:
                        processor = pair_core._get_processor_for_model(model=model, model_key=model_key)
                        tokenizer = getattr(processor, "tokenizer", None)
                        if tokenizer is not None:
                            decoded_token = str(tokenizer.decode([target_token_id], skip_special_tokens=False))
                except Exception:
                    decoded_token = None

                results[group_idx[j]] = {
                    "logp": logp,
                    "target_token_id": int(target_token_id),
                    "decoded_token": decoded_token,
                    "answer_prefix": str(item["answer_prefix"]),
                    "prompt_seq_len": int(item["prompt_seq_len"]),
                    "target_seq_len": int(item["target_seq_len"]),
                }
            pbar.update(1)
    pbar.close()

    finalized: List[Dict[str, Any]] = []
    for idx, r in enumerate(results):
        if r is None:
            raise RuntimeError(f"Missing batched score result at index={idx}.")
        finalized.append(r)
    return finalized


def _safe_mean(values: Sequence[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _pearson_corr(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    den = denx * deny
    if den <= 0.0:
        return None
    return float(num / den)


def _auc_from_scores(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(scores) != len(labels) or len(scores) == 0:
        return None
    pos = [float(s) for s, y in zip(scores, labels) if int(y) == 1]
    neg = [float(s) for s, y in zip(scores, labels) if int(y) == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    total = float(len(pos) * len(neg))
    for ps in pos:
        for ns in neg:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return float(wins / total)


def _dataset_summary(dataset_name: str, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    phi_wo = [r.get("phi_without_image") for r in rows if r.get("phi_without_image") is not None]
    phi_w = [r.get("phi_with_image") for r in rows if r.get("phi_with_image") is not None]

    pairs: List[Tuple[float, int]] = []
    for r in rows:
        phi = r.get("phi_without_image")
        mirage = r.get("mirage_like")
        if phi is None or mirage not in (True, False):
            continue
        if not math.isfinite(float(phi)):
            continue
        pairs.append((float(phi), 1 if bool(mirage) else 0))

    phi_vals = [p[0] for p in pairs]
    low_phi_vals = [-p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    phi_mirage = [p[0] for p in pairs if p[1] == 1]
    phi_non = [p[0] for p in pairs if p[1] == 0]

    mirage_count = sum(int(r.get("mirage_like") is True) for r in rows)
    non_count = sum(int(r.get("mirage_like") is False) for r in rows)
    neutral_count = sum(int(r.get("mirage_like") not in (True, False)) for r in rows)

    return {
        "dataset": dataset_name,
        "num_rows": int(len(rows)),
        "num_rows_with_phi_without": int(len(phi_wo)),
        "num_rows_with_phi_with": int(len(phi_w)),
        "mirage_like_count": int(mirage_count),
        "non_mirage_count": int(non_count),
        "neutral_or_missing_mirage_label_count": int(neutral_count),
        "phi_without_mean": _safe_mean(phi_wo),
        "phi_with_mean": _safe_mean(phi_w),
        "phi_without_mean_mirage": _safe_mean(phi_mirage),
        "phi_without_mean_non_mirage": _safe_mean(phi_non),
        "classified_rows_for_correlation": int(len(pairs)),
        "corr_phi_vs_mirage": _pearson_corr(phi_vals, labels),
        "corr_low_phi_vs_mirage": _pearson_corr(low_phi_vals, labels),
        "auc_low_phi_predicts_mirage": _auc_from_scores(low_phi_vals, labels),
    }


def _resolve_model_path(args: argparse.Namespace) -> str:
    if str(args.model_path).strip():
        return str(args.model_path).strip()
    return str(pair_core._default_model_path_for_vlm(str(args.vlm)))


def _resolve_device_map_for_run(args: argparse.Namespace, vlm_key: str) -> str:
    explicit = str(getattr(args, "device_map", "") or "").strip()
    if explicit:
        return explicit
    # Qwen3-VL-32B typically requires model sharding for inference workloads.
    if str(vlm_key).strip().lower() == "qwen3_vl_32b_instruct":
        return "auto"
    return ""


def main() -> None:
    args = parse_args()
    vlm_key = str(args.vlm).strip().lower()
    resolved_device_map = _resolve_device_map_for_run(args=args, vlm_key=vlm_key)

    responses_path = pair_core._resolve_responses_path_for_vlm(
        base_responses_path=Path(args.responses_path).expanduser(),
        vlm_key=vlm_key,
    ).resolve()
    if not responses_path.exists():
        raise FileNotFoundError(f"responses_path not found: {responses_path}")

    datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    invalid = [d for d in datasets if d not in DEFAULT_DATASETS]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}. Supported: {DEFAULT_DATASETS}")

    save_root = Path(args.save_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = save_root / f"phi_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    all_rows = _load_json(responses_path)
    if not isinstance(all_rows, list):
        raise ValueError(f"Expected list at responses_path, got: {type(all_rows)}")

    selected_rows: Dict[str, List[Dict[str, Any]]] = {}
    selection_stats_by_dataset: Dict[str, Dict[str, int]] = {}
    import random

    rng = random.Random(int(args.seed))
    for ds in datasets:
        rows = _dataset_rows(all_rows=all_rows, dataset_name=ds, original_only=bool(args.original_only))
        if bool(args.shuffle_before_select):
            rows = list(rows)
            rng.shuffle(rows)
        take_n = int(args.num_questions_per_dataset)
        selected, stats = _select_rows_with_non_mirage_priority(
            rows=rows,
            max_total=take_n,
            target_mirage=int(args.target_mirage_per_dataset),
            target_non_mirage=int(args.target_non_mirage_per_dataset),
        )
        if len(selected) == 0:
            raise RuntimeError(f"No rows selected for dataset={ds}.")
        selected_rows[ds] = selected
        selection_stats_by_dataset[ds] = stats

    mirage_root = Path(args.mirage_root).expanduser().resolve()
    uid_lookup, qid_lookup = _build_image_lookup(mirage_root=mirage_root, datasets=datasets)

    model_path = _resolve_model_path(args)
    model = pair_core.load_vlm_for_extraction(
        model_path=model_path,
        attn_implementation=str(args.attn_implementation),
        device_map_raw=str(resolved_device_map),
        max_memory_per_gpu_gib=float(args.max_memory_per_gpu_gib),
        max_memory_cpu_gib=float(args.max_memory_cpu_gib),
    )
    pair_core._force_attention_backend(model, str(args.attn_implementation))

    per_dataset_records: Dict[str, List[Dict[str, Any]]] = {}
    global_records: List[Dict[str, Any]] = []
    missing_image_rows: List[Dict[str, Any]] = []
    missing_target_rows: List[Dict[str, Any]] = []
    scoring_errors: List[Dict[str, Any]] = []

    for ds in datasets:
        records: List[Dict[str, Any]] = []
        rows = selected_rows[ds]
        valid_rows: List[Dict[str, Any]] = []
        context_with_q: List[List[Dict[str, Any]]] = []
        context_without_q: List[List[Dict[str, Any]]] = []
        context_with_qnull: List[List[Dict[str, Any]]] = []
        context_without_qnull: List[List[Dict[str, Any]]] = []
        target_tokens: List[str] = []

        for row in tqdm(rows, desc=f"{ds}: PHI prep", unit="q", dynamic_ncols=True):
            target_token = _resolve_target_token(row)
            if not target_token:
                missing_target_rows.append(
                    {
                        "dataset": ds,
                        "unique_id": row.get("unique_id", ""),
                        "question_id": row.get("question_id", ""),
                        "reason": "missing_target_token",
                    }
                )
                continue

            images = _resolve_images_for_row(row=row, uid_lookup=uid_lookup, qid_lookup=qid_lookup)
            if images is None:
                missing_image_rows.append(
                    {
                        "dataset": ds,
                        "unique_id": row.get("unique_id", ""),
                        "question_id": row.get("question_id", ""),
                        "reason": "missing_images_lookup",
                    }
                )
                continue

            system_prompt = str(row.get("system_prompt", ""))
            prompt_q = str(row.get("prompt_text", ""))
            prompt_qnull = _build_q_null_prompt(row=row, q_null_text=str(args.q_null_text))

            conv_with_q = core._make_vllm_messages(prompt_text=prompt_q, image_bytes_list=images, system_prompt=system_prompt)
            conv_without_q = core._make_vllm_messages(prompt_text=prompt_q, image_bytes_list=None, system_prompt=system_prompt)
            conv_with_qnull = core._make_vllm_messages(
                prompt_text=prompt_qnull, image_bytes_list=images, system_prompt=system_prompt
            )
            conv_without_qnull = core._make_vllm_messages(
                prompt_text=prompt_qnull, image_bytes_list=None, system_prompt=system_prompt
            )
            valid_rows.append(row)
            target_tokens.append(str(target_token))
            context_with_q.append(conv_with_q)
            context_without_q.append(conv_without_q)
            context_with_qnull.append(conv_with_qnull)
            context_without_qnull.append(conv_without_qnull)

        if valid_rows:
            try:
                score_with_q_all = _score_assistant_token_logprobs_batch(
                    model=model,
                    model_key=vlm_key,
                    conversations_pre_response=context_with_q,
                    target_token_texts=target_tokens,
                    answer_prefix=str(args.answer_prefix),
                    batch_size=int(args.score_batch_size),
                    progress_desc=f"{ds}: PHI with-image Q",
                )
                score_without_q_all = _score_assistant_token_logprobs_batch(
                    model=model,
                    model_key=vlm_key,
                    conversations_pre_response=context_without_q,
                    target_token_texts=target_tokens,
                    answer_prefix=str(args.answer_prefix),
                    batch_size=int(args.score_batch_size),
                    progress_desc=f"{ds}: PHI without-image Q",
                )
                score_with_qnull_all = _score_assistant_token_logprobs_batch(
                    model=model,
                    model_key=vlm_key,
                    conversations_pre_response=context_with_qnull,
                    target_token_texts=target_tokens,
                    answer_prefix=str(args.answer_prefix),
                    batch_size=int(args.score_batch_size),
                    progress_desc=f"{ds}: PHI with-image Q_null",
                )
                score_without_qnull_all = _score_assistant_token_logprobs_batch(
                    model=model,
                    model_key=vlm_key,
                    conversations_pre_response=context_without_qnull,
                    target_token_texts=target_tokens,
                    answer_prefix=str(args.answer_prefix),
                    batch_size=int(args.score_batch_size),
                    progress_desc=f"{ds}: PHI without-image Q_null",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Batched scoring failed for dataset={ds}. "
                    "Try reducing --score_batch_size (e.g., to 1 or 2)."
                ) from exc
        else:
            score_with_q_all = []
            score_without_q_all = []
            score_with_qnull_all = []
            score_without_qnull_all = []

        for row, score_with_q, score_without_q, score_with_qnull, score_without_qnull in zip(
            valid_rows,
            score_with_q_all,
            score_without_q_all,
            score_with_qnull_all,
            score_without_qnull_all,
        ):

            logp_with_q = float(score_with_q["logp"])
            logp_without_q = float(score_without_q["logp"])
            logp_with_qnull = float(score_with_qnull["logp"])
            logp_without_qnull = float(score_without_qnull["logp"])

            phi_with = float(logp_with_q - logp_with_qnull)
            phi_without = float(logp_without_q - logp_without_qnull)

            without_payload = row.get("without_image", {}) or {}
            record = {
                "dataset": ds,
                "task_type": str(row.get("task_type", "")),
                "unique_id": str(row.get("unique_id", "")),
                "question_id": str(row.get("question_id", "")),
                "variant_id": str(row.get("variant_id", "")),
                "is_original": bool(row.get("is_original", False)),
                "category": row.get("category", ""),
                "target_token_text": str(target_token),
                "prompt_text_q": prompt_q,
                "prompt_text_q_null": prompt_qnull,
                "mirage_like": without_payload.get("mirage_like"),
                "mirage_label": without_payload.get("mirage_label"),
                "without_image_acknowledged_missing_or_uncertain": without_payload.get(
                    "acknowledged_missing_or_uncertain"
                ),
                "with_image_response": (row.get("with_image", {}) or {}).get("response"),
                "without_image_response": without_payload.get("response"),
                "logp_with_image_q": logp_with_q,
                "logp_with_image_q_null": logp_with_qnull,
                "logp_without_image_q": logp_without_q,
                "logp_without_image_q_null": logp_without_qnull,
                "phi_with_image": phi_with,
                "phi_without_image": phi_without,
                "phi_without_image_negated": float(-phi_without),
                "score_meta_with_q": score_with_q,
                "score_meta_without_q": score_without_q,
                "score_meta_with_q_null": score_with_qnull,
                "score_meta_without_q_null": score_without_qnull,
            }
            records.append(record)
            global_records.append(record)

        per_dataset_records[ds] = records

    per_dataset_summary = {ds: _dataset_summary(ds, rows) for ds, rows in per_dataset_records.items()}
    global_summary = _dataset_summary("all", global_records)

    run_summary: Dict[str, Any] = {
        "vlm": str(args.vlm),
        "model_path": model_path,
        "responses_path": str(responses_path),
        "mirage_root": str(mirage_root),
        "datasets": datasets,
        "num_questions_per_dataset_requested": int(args.num_questions_per_dataset),
        "target_mirage_per_dataset": int(args.target_mirage_per_dataset),
        "target_non_mirage_per_dataset": int(args.target_non_mirage_per_dataset),
        "original_only": bool(args.original_only),
        "shuffle_before_select": bool(args.shuffle_before_select),
        "seed": int(args.seed),
        "q_null_text": str(args.q_null_text),
        "answer_prefix": str(args.answer_prefix),
        "score_batch_size": int(args.score_batch_size),
        "attn_implementation": str(args.attn_implementation),
        "device_map_requested": str(args.device_map),
        "device_map_resolved": str(resolved_device_map),
        "max_memory_per_gpu_gib": float(args.max_memory_per_gpu_gib),
        "max_memory_cpu_gib": float(args.max_memory_cpu_gib),
        "selected_counts_per_dataset": {ds: int(len(selected_rows[ds])) for ds in datasets},
        "selection_stats_by_dataset": selection_stats_by_dataset,
        "scored_counts_per_dataset": {ds: int(len(per_dataset_records.get(ds, []))) for ds in datasets},
        "missing_image_rows_count": int(len(missing_image_rows)),
        "missing_target_rows_count": int(len(missing_target_rows)),
        "scoring_errors_count": int(len(scoring_errors)),
        "per_dataset": per_dataset_summary,
        "global": global_summary,
    }

    with (save_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)
    with (save_dir / "phi_rows.json").open("w", encoding="utf-8") as f:
        json.dump(global_records, f, indent=2, ensure_ascii=False)
    with (save_dir / "phi_rows_by_dataset.json").open("w", encoding="utf-8") as f:
        json.dump(per_dataset_records, f, indent=2, ensure_ascii=False)
    with (save_dir / "missing_image_rows.json").open("w", encoding="utf-8") as f:
        json.dump(missing_image_rows, f, indent=2, ensure_ascii=False)
    with (save_dir / "missing_target_rows.json").open("w", encoding="utf-8") as f:
        json.dump(missing_target_rows, f, indent=2, ensure_ascii=False)
    with (save_dir / "scoring_errors.json").open("w", encoding="utf-8") as f:
        json.dump(scoring_errors, f, indent=2, ensure_ascii=False)

    print(json.dumps(run_summary, indent=2, ensure_ascii=False))
    print(f"Saved PHI outputs to: {save_dir}")


if __name__ == "__main__":
    main()
