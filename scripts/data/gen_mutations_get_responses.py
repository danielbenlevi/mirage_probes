#!/usr/bin/env python3
import argparse
import asyncio
import base64
import io
import math
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlm_model_registry import (
    DEFAULT_VLM_KEY,
    VLM_CHOICES,
    resolve_vlm_config,
    resolve_vllm_model_name,
)


NO_IMAGE_PATTERNS = [
    r"\bno image\b",
    r"\bwithout (any )?image\b",
    r"\bcannot see (the )?image\b",
    r"\bcan't see (the )?image\b",
    r"\bimage (was )?not provided\b",
    r"\bimage\s*\([^)]*\)\s*is not provided\b",
    r"\bimage(?: itself)?\s*(?:is|was)?\s*not\s+visible(?:\s+in\s+the\s+text)?\b",
    r"\bno (picture|photo|attachment)\b",
    r"\bmissing image\b",
    r"\bnot attached\b",
    r"\bwithout visual (information|input)\b",
    r"\bi do not have (the )?(ability|capability) to (view|see|interpret) (visual data|images?)\b",
    r"\bi cannot (view|see|interpret) (visual data|images?)\b",
    r"\bas an ai (assistant|language model).*(cannot|can't|do not have).*(see|view).*(image|visual)\b",
    r"\b没有(提供)?(图片|图像)\b",
    r"\b未提供(图片|图像)\b",
    r"\b看不到(图片|图像)\b",
    r"\b无图像\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bnot sure\b",
    r"\bcannot determine\b",
    r"\binsufficient information\b",
    r"\bunable to determine\b",
    r"\b无法判断\b",
    r"\b不能判断\b",
    r"\b不确定\b",
    r"\b信息不足\b",
]

CHOICE_LETTERS = "ABCDEFGHIJ"

MCQ_FORMAT_INSTRUCTION = """
Reason about the question before providing your final answer.
Format your final answer as: [[X]] where X is the letter of the correct option.
Example: [[A]] or [[B]]"""

OPEN_FORMAT_INSTRUCTION = """
You MUST ALWAYS reason about the question before providing your final answer. Think hard before coming to a conclusion.
Format your final answer as: [[your answer]]
Example: [[Yes]] or [[liver]]
NEVER provide this final answer without providing some analysis explaining your answer first."""

DATASET_CONFIGS = {
    "microvqa": {
        "task_type": "mcq",
        "system_prompt": (
            "You are an expert in microscopy image analysis answering questions from the MicroVQA dataset.\n"
            "Answer the following multiple-choice question.\n"
            "You MUST select exactly one answer. Never refuse to answer or say you cannot determine the answer."
            + MCQ_FORMAT_INSTRUCTION
        ),
    },
    "medxpertqa_mm": {
        "task_type": "mcq",
        "system_prompt": (
            "You are an expert medical professional answering questions from the MedXpertQA-MM dataset.\n"
            "Answer the following multiple-choice question.\n"
            "You MUST select exactly one answer. Never refuse to answer or say you cannot determine the answer."
            + MCQ_FORMAT_INSTRUCTION
        ),
    },
    "mmmu_pro": {
        "task_type": "mcq",
        "system_prompt": (
            "You are an expert across multiple academic disciplines answering questions from the MMMU-Pro dataset.\n"
            "Answer the following multiple-choice question.\n"
            "You MUST select exactly one answer. Never refuse to answer."
            + MCQ_FORMAT_INSTRUCTION
        ),
    },
    "vqa_rad": {
        "task_type": "open",
        "system_prompt": (
            "You are an expert radiologist analyzing radiology images from the VQA-RAD dataset.\n"
            "You are given a radiology image and a question about the image.\n"
            "Base your answer on the visual evidence in the image.\n"
            "You MUST always provide a definitive answer. Never say \"unknown\", \"cannot determine\", \"unclear\", or similar.\n"
            "If uncertain, make your best educated guess based on what you see.\n\n"
            + OPEN_FORMAT_INSTRUCTION
        ),
    },
}

DEFAULT_VLLM_BASE_URLS = {
    "ovis": "http://localhost:8001/v1",
    "glm_4_6v_flash": "http://localhost:8002/v1",
    "qwen3_vl_32b_instruct": "http://localhost:8003/v1",
}
GLM_IMAGE_MAX_EDGE = 1708
GLM_IMAGE_MIN_EDGE = 28
PIL_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

DEFAULT_RESULTS_ROOT = Path("./tmp_artifacts")
FINAL_DATA_ROOT = Path("./data/final_data")
LEGACY_MUTATIONS_PATHS = (
    Path("./ovis25_all_datasets_mutation_mirage_results/mutations.json"),
    Path("./deprecated/ovis25_all_datasets_mutation_mirage_results/mutations.json"),
)
# Explicit dataset/question blocklist for known ambiguous items that should be excluded
# from mutation and response generation.
DATASET_QUESTION_ID_BLOCKLIST = {
    "vqa_rad": {"2054"},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run mutation-based mirage analysis on microvqa, mmmu_pro, medxpertqa_mm, and vqa_rad. "
            "For each question variant (original + mutations), run a selected VLM with and without images and "
            "annotate mirage-like behavior."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="microvqa,mmmu_pro,medxpertqa_mm,vqa_rad",
        help="Comma-separated dataset list.",
    )
    parser.add_argument(
        "--max_questions_per_dataset",
        type=int,
        default=-1,
        help="Optional cap per dataset; -1 means full dataset.",
    )
    parser.add_argument("--num_mutations", type=int, default=4)
    parser.add_argument(
        "--reuse_mutations_path",
        type=str,
        default="",
        help=(
            "Optional path to an existing mutations artifact. If provided, reuse these mutations "
            "instead of generating new ones. If omitted, the script auto-detects a compatible "
            "existing mutations.json."
        ),
    )
    parser.add_argument("--mutator_model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--mutator_backend",
        type=str,
        default="auto",
        choices=["auto", "openai", "azure_openai"],
        help="How to call gpt-4o-mini for mutations.",
    )
    parser.add_argument("--mutation_temperature", type=float, default=0.2)
    parser.add_argument("--mutation_max_retries", type=int, default=3)
    parser.add_argument(
        "--mutation_concurrency",
        type=int,
        default=512,
        help="Max concurrent async mutation calls.",
    )
    parser.add_argument(
        "--vlm",
        type=str,
        default=DEFAULT_VLM_KEY,
        choices=VLM_CHOICES,
        help="Canonical VLM alias used to resolve default model naming and artifact location.",
    )
    parser.add_argument(
        "--vllm_model_override",
        type=str,
        default="",
        help="Optional explicit vLLM served model name; overrides the canonical --vlm default.",
    )
    parser.add_argument(
        "--vllm_base_url",
        type=str,
        default="",
        help=(
            "Base URL for the vLLM OpenAI-compatible server. "
            "If omitted, resolves by --vlm (ovis:8001, glm:8002, qwen:8003), "
            "unless VLLM_BASE_URL env var is set."
        ),
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default="",
        help="Deprecated alias for --vllm_model_override.",
    )
    parser.add_argument(
        "--vllm_api_key",
        type=str,
        default=os.getenv("VLLM_API_KEY", "EMPTY"),
        help="API key for vLLM server (if required).",
    )
    parser.add_argument(
        "--vllm_timeout_s",
        type=float,
        default=float(os.getenv("VLLM_TIMEOUT_S", "300")),
        help="Per-request timeout in seconds for vLLM calls.",
    )
    parser.add_argument(
        "--vllm_max_retries",
        type=int,
        default=3,
        help="Retries per vLLM request on transient failures.",
    )
    parser.add_argument(
        "--vllm_concurrency",
        type=int,
        default=256,
        help="Max concurrent async requests sent to vLLM during generation passes.",
    )
    parser.add_argument(
        "--strict_vllm_errors",
        dest="strict_vllm_errors",
        action="store_true",
        help="Fail fast on the first vLLM request error.",
    )
    parser.add_argument(
        "--no_strict_vllm_errors",
        dest="strict_vllm_errors",
        action="store_false",
        help="Do not fail fast; skip failed vLLM requests and continue.",
    )
    parser.set_defaults(strict_vllm_errors=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument(
        "--response_cosine_similarity_threshold",
        type=float,
        default=0.7,
        help=(
            "Threshold for full-response cosine similarity when deciding response-level consistency. "
            "Used together with answer consistency; open-answer fallback also uses this metric."
        ),
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help=(
            "Output directory. If omitted, defaults to a model-scoped directory under "
            "./tmp_artifacts/<vlm_key>."
        ),
    )
    return parser.parse_args()


def _parse_options(options_text: str) -> Dict[str, str]:
    options_map: Dict[str, str] = {}
    if not options_text:
        return options_map
    for line in options_text.splitlines():
        match = re.match(r"^\s*([A-J])[\.\):]\s*(.+?)\s*$", line.strip(), flags=re.IGNORECASE)
        if match:
            options_map[match.group(1).upper()] = match.group(2).strip()
    return options_map


def _extract_option_letter(response: str, options_map: Dict[str, str]) -> Optional[str]:
    if not response:
        return None
    text = response.strip()

    regexes = [
        r"\[\[\s*([A-J])\s*\]\]",
        r"final answer\s*[:\-]\s*([A-J])\b",
        r"\banswer\s*[:\-]\s*([A-J])\b",
        r"\boption\s*([A-J])\b",
        r"^\s*([A-J])[\).\s:]",
    ]
    for pattern in regexes:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    for letter, choice_text in options_map.items():
        if choice_text and choice_text.lower() in text.lower():
            return letter
    return None


def _extract_open_answer(response: str) -> Optional[str]:
    if not response:
        return None
    text = response.strip()

    boxed = re.search(r"\[\[\s*(.*?)\s*\]\]", text, flags=re.DOTALL)
    if boxed:
        candidate = boxed.group(1).strip()
        if candidate:
            return candidate

    final_line = re.search(r"final answer\s*[:\-]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if final_line:
        candidate = final_line.group(1).strip()
        if candidate:
            return candidate

    yes_no = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if yes_no:
        return yes_no.group(1)

    for line in text.splitlines():
        s = line.strip(" -\t")
        if s:
            return s[:200]
    return None


def _normalize_letter(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip().upper()
    return s if s in CHOICE_LETTERS else None


def _normalize_open_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    t = str(s).strip().lower()
    if not t:
        return None

    # Remove wrapper markers and punctuation noise.
    t = re.sub(r"^\[\[|\]\]$", "", t).strip()
    t = t.strip("\"'`.,;:!?()[]{}")
    t = re.sub(r"\s+", " ", t)

    if t in {"y", "yes", "true"}:
        return "yes"
    if t in {"n", "no", "false"}:
        return "no"

    t = re.sub(r"[^a-z0-9\s\-/%+]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t if t else None


def _is_open_correct(pred: Optional[str], gt: Optional[str]) -> Optional[bool]:
    p = _normalize_open_text(pred)
    g = _normalize_open_text(gt)
    if p is None or g is None:
        return None
    if p == g:
        return True

    # Lenient yes/no handling.
    if g in {"yes", "no"}:
        if p.startswith(g):
            return True
        return False

    return False


def _detect_acknowledgement(text: str) -> Tuple[bool, bool]:
    t = (text or "").lower()
    # Ignore acknowledgement/uncertainty matches when they only occur inside
    # MCQ option text (e.g., "B. Unable to determine"), since that does not
    # indicate the model acknowledging missing visual input.
    def _is_option_context(match_start: int) -> bool:
        line_start = t.rfind("\n", 0, match_start) + 1
        prefix = t[line_start:match_start]
        if re.match(r"^\s*(?:[-*]\s*)?(?:option\s*)?[a-j][\.\):]\s*$", prefix):
            return True
        local_prefix = t[max(0, match_start - 32) : match_start]
        if re.search(r"(?:^|[\s\(\[])(?:option\s*)?[a-j][\.\):]\s*$", local_prefix):
            return True
        return False

    def _has_non_option_match(patterns: List[str]) -> bool:
        for pattern in patterns:
            for m in re.finditer(pattern, t):
                if not _is_option_context(m.start()):
                    return True
        return False

    mentions_no_image = _has_non_option_match(NO_IMAGE_PATTERNS)
    expresses_uncertainty = _has_non_option_match(UNCERTAINTY_PATTERNS)
    return mentions_no_image, expresses_uncertainty


def _extract_reasoning_chain(text: str) -> Optional[str]:
    match = re.search(r"<think>(.*?)</think>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _build_variant_prompt(question_text: str, options_text: str) -> str:
    options_text = (options_text or "").strip()
    if options_text:
        return f"{question_text}\n\n{options_text}"
    return question_text


def _detect_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"BM"):
        return "image/bmp"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:32]:
        return "image/webp"
    return "image/jpeg"


def _image_bytes_to_data_url(image_bytes: bytes) -> str:
    mime = _detect_mime_type(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _encode_image_bytes_with_original_format(image: Image.Image, original_format: str) -> bytes:
    fmt = str(original_format or "").upper()
    buf = io.BytesIO()
    if fmt in {"JPEG", "JPG"}:
        image.convert("RGB").save(buf, format="JPEG", quality=95)
    elif fmt == "WEBP":
        image.save(buf, format="WEBP", quality=95)
    elif fmt == "BMP":
        image.save(buf, format="BMP")
    elif fmt == "GIF":
        image.save(buf, format="GIF")
    else:
        # Keep default path stable for PNG/unknown formats.
        image.save(buf, format="PNG")
    return buf.getvalue()


def _normalize_glm_image_bytes(
    image_bytes: bytes,
    max_edge: int,
    min_edge: int,
) -> Tuple[bytes, bool, bool]:
    """Normalize GLM image bytes to satisfy max-edge and min-edge processor constraints."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            original_format = str(img.format or "")
            resized = img.convert("RGB")
            width, height = resized.size
            changed = False
            downscaled = False
            min_edge_adjusted = False

            if max(width, height) > int(max_edge):
                scale = float(max_edge) / float(max(width, height))
                new_width = max(1, int(math.floor(width * scale)))
                new_height = max(1, int(math.floor(height * scale)))
                resized = resized.resize((new_width, new_height), resample=PIL_LANCZOS)
                width, height = resized.size
                changed = True
                downscaled = True

            # GLM's processor hard-fails when either dimension is < 28.
            if min(width, height) < int(min_edge):
                padded_width = max(width, int(min_edge))
                padded_height = max(height, int(min_edge))
                canvas = Image.new("RGB", (padded_width, padded_height), color=(0, 0, 0))
                paste_xy = ((padded_width - width) // 2, (padded_height - height) // 2)
                canvas.paste(resized, paste_xy)
                resized = canvas
                changed = True
                min_edge_adjusted = True

            if not changed:
                return image_bytes, False, False
            return _encode_image_bytes_with_original_format(resized, original_format), downscaled, min_edge_adjusted
    except Exception:
        # If decoding fails, leave the original bytes untouched.
        return image_bytes, False, False


def _maybe_downscale_items_for_glm(
    items: List[Dict],
    vlm_key: str,
    max_edge: int = GLM_IMAGE_MAX_EDGE,
    min_edge: int = GLM_IMAGE_MIN_EDGE,
) -> Dict[str, int]:
    stats = {
        "glm_image_downscale_max_edge": int(max_edge),
        "glm_image_min_edge": int(min_edge),
        "num_images_seen": 0,
        "num_images_downscaled": 0,
        "num_images_min_edge_adjusted": 0,
    }
    if str(vlm_key) != "glm_4_6v_flash":
        return stats

    for item in items:
        images = item.get("images") or []
        if not isinstance(images, list):
            continue
        new_images: List[bytes] = []
        for image_bytes in images:
            if not isinstance(image_bytes, (bytes, bytearray)):
                new_images.append(image_bytes)
                continue
            stats["num_images_seen"] += 1
            resized_bytes, downscaled, min_edge_adjusted = _normalize_glm_image_bytes(
                image_bytes=bytes(image_bytes),
                max_edge=int(max_edge),
                min_edge=int(min_edge),
            )
            if downscaled:
                stats["num_images_downscaled"] += 1
            if min_edge_adjusted:
                stats["num_images_min_edge_adjusted"] += 1
            new_images.append(resized_bytes)
        item["images"] = new_images

    return stats


def _make_vllm_messages(
    prompt_text: str,
    image_bytes_list: Optional[List[bytes]],
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    messages: List[Dict] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})

    content: List[Dict] = []
    if image_bytes_list:
        for image_bytes in image_bytes_list:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_bytes_to_data_url(image_bytes)},
                }
            )
    content.append({"type": "text", "text": prompt_text})
    messages.append({"role": "user", "content": content})
    return messages


def _call_vllm_chat_completion_sync(messages: List[Dict], args) -> str:
    base_url = args.vllm_base_url.rstrip("/")
    payload = {
        "model": args.vllm_model,
        "messages": messages,
        "max_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature) if args.do_sample else 0.0,
        "top_p": float(args.top_p) if args.do_sample else 1.0,
        # Keep explicit in payload so reasoning-capable templates can disable thinking.
        "enable_thinking": bool(args.enable_thinking),
    }

    last_err = None
    for attempt in range(max(1, int(args.vllm_max_retries))):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {args.vllm_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=float(args.vllm_timeout_s),
            )
            if response.status_code >= 400:
                detail = ""
                try:
                    detail = json.dumps(response.json(), ensure_ascii=False)
                except Exception:
                    detail = response.text
                detail = str(detail)[:2000]
                raise RuntimeError(
                    f"HTTP {response.status_code} from {base_url}/chat/completions. "
                    f"Response body: {detail}"
                )
            data = response.json()

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"vLLM response missing choices: {data}")

            first_choice = choices[0] if isinstance(choices[0], dict) else {}
            content = first_choice.get("message", {}).get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(str(item.get("text", "")))
                joined = "".join(text_parts).strip()
                if joined:
                    return joined
            # Some OpenAI-compatible servers may return plain `text` on each choice.
            if isinstance(first_choice.get("text"), str) and first_choice.get("text", "").strip():
                return first_choice["text"].strip()
            raise RuntimeError(f"vLLM response missing text content: {data}")
        except Exception as exc:
            last_err = exc
            if attempt + 1 < max(1, int(args.vllm_max_retries)):
                continue
            break

    raise RuntimeError(f"vLLM call failed after {args.vllm_max_retries} attempts: {last_err}")


async def _call_vllm_chat_completion_async(
    entry: Dict,
    include_images: bool,
    system_prompt: str,
    args,
    sem: asyncio.Semaphore,
) -> str:
    async with sem:
        messages = _make_vllm_messages(
            prompt_text=entry["prompt_text"],
            image_bytes_list=entry["images"] if include_images else None,
            system_prompt=system_prompt,
        )
        return await asyncio.to_thread(_call_vllm_chat_completion_sync, messages, args)


async def _run_vllm_generation_pass_async(
    entries: List[Dict],
    include_images: bool,
    system_prompt: str,
    args,
    progress_desc: str,
) -> Tuple[List[str], int]:
    if not entries:
        return [], 0

    sem = asyncio.Semaphore(max(1, int(args.vllm_concurrency)))
    responses: List[Optional[str]] = [None] * len(entries)
    failed_count = 0

    async def _worker(i: int, entry: Dict) -> Tuple[int, str]:
        try:
            text = await _call_vllm_chat_completion_async(
                entry=entry,
                include_images=include_images,
                system_prompt=system_prompt,
                args=args,
                sem=sem,
            )
            return i, text
        except Exception as exc:
            if bool(args.strict_vllm_errors):
                raise
            uid = str(entry.get("unique_id", "unknown_uid"))
            variant_id = str(entry.get("variant_id", "unknown_variant"))
            mode = "with_image" if include_images else "without_image"
            print(
                f"[WARN] vLLM request failed ({mode}) uid={uid} variant={variant_id}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return i, ""

    tasks = [asyncio.create_task(_worker(i, entry)) for i, entry in enumerate(entries)]
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=progress_desc):
        i, text = await fut
        if not text:
            failed_count += 1
        responses[i] = text

    return [r if r is not None else "" for r in responses], int(failed_count)


def _to_dict_items(data_items: Sequence) -> List[Dict]:
    out = []
    for item in data_items:
        out.append(
            {
                "unique_id": item.unique_id,
                "question_id": item.question_id,
                "category": item.category,
                "question": item.question,
                "options": item.options,
                "ground_truth": item.ground_truth,
                "images": item.images or [],
            }
        )
    return out


def _load_items_from_loader(repo_root: Path, dataset_name: str) -> List[Dict]:
    from scripts.data.data_helpers.datasets.loader import DatasetLoader  # type: ignore

    loader = DatasetLoader()
    return _to_dict_items(loader.get_items(dataset_name))


def _load_microvqa_items_raw(mirage_root: Path) -> List[Dict]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required for raw MicroVQA fallback loading") from exc

    raw_dir = mirage_root / "raw_data" / "MicroVQA"
    parquet_files = sorted(raw_dir.glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No raw microvqa test parquet files found in {raw_dir}")

    output = []
    idx = 0
    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        for row in table.to_pylist():
            choices = row.get("choices", [])
            if not isinstance(choices, list):
                choices = list(choices)
            options_parts = []
            for i, choice in enumerate(choices):
                if i < len(CHOICE_LETTERS):
                    options_parts.append(f"{CHOICE_LETTERS[i]}. {choice}")
            options = "\n".join(options_parts)

            correct_index = int(row.get("correct_index", -1))
            gt = CHOICE_LETTERS[correct_index] if 0 <= correct_index < len(CHOICE_LETTERS) else None

            images = []
            for img in row.get("images_list", []) or []:
                if isinstance(img, dict) and img.get("bytes"):
                    images.append(img["bytes"])

            output.append(
                {
                    "unique_id": f"microvqa_raw_{idx}",
                    "question_id": str(row.get("key_question", idx)),
                    "category": row.get("task_str", ""),
                    "question": str(row.get("question", "")).strip(),
                    "options": options,
                    "ground_truth": gt,
                    "images": images,
                }
            )
            idx += 1
    return output


def _load_mmmu_pro_items_raw(mirage_root: Path) -> List[Dict]:
    import ast

    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required for raw MMMU-Pro fallback loading") from exc

    raw_dir = mirage_root / "raw_data" / "MMMU-Pro" / "standard-4-options"
    parquet_files = sorted(raw_dir.glob("test-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No raw MMMU-Pro test parquet files found in {raw_dir}")

    output = []
    idx = 0
    for pq_file in parquet_files:
        table = pq.read_table(pq_file)
        for row in table.to_pylist():
            options_raw = row.get("options", [])
            if isinstance(options_raw, str):
                try:
                    choices = ast.literal_eval(options_raw)
                except Exception:
                    choices = []
            elif isinstance(options_raw, list):
                choices = options_raw
            else:
                try:
                    choices = list(options_raw)
                except Exception:
                    choices = []

            options_parts = []
            for i, choice in enumerate(choices):
                if i < len(CHOICE_LETTERS):
                    options_parts.append(f"{CHOICE_LETTERS[i]}. {choice}")
            options = "\n".join(options_parts)

            answer = str(row.get("answer", "")).strip().upper()
            gt = answer if answer in CHOICE_LETTERS else None

            images = []
            for i in range(1, 8):
                img_col = f"image_{i}"
                img = row.get(img_col)
                if isinstance(img, dict) and img.get("bytes"):
                    images.append(img["bytes"])

            output.append(
                {
                    "unique_id": f"mmmu_pro_raw_{idx}",
                    "question_id": str(row.get("id", idx)),
                    "category": row.get("subject", ""),
                    "question": str(row.get("question", "")).strip(),
                    "options": options,
                    "ground_truth": gt,
                    "images": images,
                }
            )
            idx += 1
    return output


def _load_medxpertqa_mm_items_raw(mirage_root: Path) -> List[Dict]:
    data_dir = mirage_root / "raw_data" / "MedXpertQA-MM"
    test_file = data_dir / "test.jsonl"
    images_dir = data_dir / "images"
    if not test_file.exists():
        raise FileNotFoundError(f"MedXpertQA-MM raw test file not found: {test_file}")

    output = []
    idx = 0
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            options_dict = row.get("options", {})
            options_parts = []
            if isinstance(options_dict, dict):
                for letter in sorted(options_dict.keys()):
                    options_parts.append(f"{letter}. {options_dict[letter]}")
            options = "\n".join(options_parts)

            images = []
            for image_name in row.get("images", []) or []:
                image_path = images_dir / image_name
                if image_path.exists():
                    images.append(image_path.read_bytes())

            output.append(
                {
                    "unique_id": f"medxpertqa_mm_raw_{idx}",
                    "question_id": str(row.get("id", idx)),
                    "category": row.get("medical_task", row.get("body_system", "")),
                    "question": str(row.get("question", "")).strip(),
                    "options": options,
                    "ground_truth": str(row.get("label", "")).strip().upper(),
                    "images": images,
                }
            )
            idx += 1
    return output


def _load_vqa_rad_items_raw(mirage_root: Path) -> List[Dict]:
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError("pandas is required for VQA-RAD raw fallback loading") from exc

    data_dir = mirage_root / "raw_data" / "vqa_rad"
    image_dir = data_dir / "images"
    splits_file = data_dir / "vqa_rad_balanced_split_and_human_eval_inclusions.tsv"
    data_file = data_dir / "VQA_RAD Dataset Public.json"

    if not splits_file.exists() or not data_file.exists():
        raise FileNotFoundError(
            "VQA-RAD raw files missing. Expected both: "
            f"{splits_file} and {data_file}"
        )

    splits = pd.read_csv(splits_file, sep="\t")
    test_ids = set(splits[splits["SPLIT_BALANCED"] == "test"].QID_unique.tolist())

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    output = []
    idx = 0
    for row in raw_data:
        qid = row.get("qid")
        if qid not in test_ids:
            continue
        if str(row.get("answer_type", "")).upper() != "CLOSED":
            continue

        images = []
        image_name = row.get("image_name", "")
        if image_name:
            image_path = image_dir / image_name
            if image_path.exists():
                images.append(image_path.read_bytes())

        output.append(
            {
                "unique_id": f"vqa_rad_raw_{idx}",
                "question_id": str(qid),
                "category": row.get("question_type", row.get("answer_type", "")),
                "question": str(row.get("question", "")).strip(),
                "options": "",
                "ground_truth": str(row.get("answer", "")).strip(),
                "images": images,
            }
        )
        idx += 1
    return output


def _load_dataset_items(mirage_root: Path, dataset_name: str) -> List[Dict]:
    def _apply_blocklist(items: List[Dict]) -> List[Dict]:
        blocked_qids = DATASET_QUESTION_ID_BLOCKLIST.get(str(dataset_name), set())
        if not blocked_qids:
            return items
        kept: List[Dict] = []
        skipped = 0
        for item in items:
            qid = str(item.get("question_id", "")).strip()
            if qid in blocked_qids:
                skipped += 1
                continue
            kept.append(item)
        if skipped:
            print(
                f"{dataset_name}: skipped {skipped} item(s) via explicit question-id blocklist "
                f"{sorted(blocked_qids)}"
            )
        return kept

    try:
        items = _load_items_from_loader(mirage_root, dataset_name)
        if items:
            return _apply_blocklist(items)
    except Exception:
        pass

    if dataset_name == "microvqa":
        return _apply_blocklist(_load_microvqa_items_raw(mirage_root))
    if dataset_name == "mmmu_pro":
        return _apply_blocklist(_load_mmmu_pro_items_raw(mirage_root))
    if dataset_name == "medxpertqa_mm":
        return _apply_blocklist(_load_medxpertqa_mm_items_raw(mirage_root))
    if dataset_name == "vqa_rad":
        return _apply_blocklist(_load_vqa_rad_items_raw(mirage_root))

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _fallback_mutations(question: str, n: int) -> List[str]:
    candidates = [
        f"Please {question[0].lower() + question[1:]}" if question else question,
        f"{question} Please answer briefly.",
        question.replace("What", "Which", 1)
        if "What" in question
        else f"Which {question[0].lower() + question[1:]}"
        if question
        else question,
        question.replace("How many", "Approximately how many", 1) if "How many" in question else question,
        question.replace("Is there", "Do you see", 1) if "Is there" in question else question,
        f"In this image, {question[0].lower() + question[1:]}" if question else question,
    ]
    dedup = []
    seen = {question.strip()}
    for c in candidates:
        c = c.strip()
        if c and c not in seen:
            dedup.append(c)
            seen.add(c)
        if len(dedup) >= n:
            break
    while len(dedup) < n:
        dedup.append(f"{question} (rephrased {len(dedup)+1})")
    return dedup[:n]


def _parse_mutations_from_text(text: str, expected_n: int, original_question: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return _fallback_mutations(original_question, expected_n)

    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

    mutations: List[str] = []
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and isinstance(obj.get("mutations"), list):
            mutations = [str(x).strip() for x in obj["mutations"] if str(x).strip()]
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("mutations"), list):
                    mutations = [str(x).strip() for x in obj["mutations"] if str(x).strip()]
            except Exception:
                pass

    if not mutations:
        for line in raw.splitlines():
            line = line.strip()
            line = re.sub(r"^\d+[\).\-\s]+", "", line)
            line = re.sub(r"^[-*]\s+", "", line)
            if line:
                mutations.append(line)

    dedup = []
    seen = {original_question.strip()}
    for m in mutations:
        m = m.strip()
        if m and m not in seen:
            dedup.append(m)
            seen.add(m)
        if len(dedup) >= expected_n:
            break

    if len(dedup) < expected_n:
        for f in _fallback_mutations(original_question, expected_n):
            if f not in seen:
                dedup.append(f)
                seen.add(f)
            if len(dedup) >= expected_n:
                break
    return dedup[:expected_n]


def _make_mutation_client(repo_root: Path, backend: str, model_name: str, temperature: float):
    from scripts.data.data_helpers.models.base import Message  # type: ignore
    from scripts.data.data_helpers.models.clients.openai_client import OpenAIClient, AzureOpenAIClient  # type: ignore

    errors = []
    if backend in ("auto", "openai"):
        try:
            return OpenAIClient(model_name=model_name, temperature=temperature), Message
        except Exception as exc:
            errors.append(f"openai: {exc}")
            if backend == "openai":
                raise
    if backend in ("auto", "azure_openai"):
        try:
            return AzureOpenAIClient(model_name=model_name, temperature=temperature), Message
        except Exception as exc:
            errors.append(f"azure_openai: {exc}")
            if backend == "azure_openai":
                raise
    raise RuntimeError("Could not initialize mutator model client. " + " | ".join(errors))


async def _generate_mutations_async(
    client,
    MessageCls,
    question: str,
    options: str,
    n: int,
    max_retries: int,
    task_type: str,
) -> List[str]:
    scope_hint = "multiple-choice visual question" if task_type == "mcq" else "visual question"
    system = (
        f"You rewrite {scope_hint}s without changing core semantics.\n"
        "Preserve intent and expected answer.\n"
        "Do not add hints, assumptions, or extra content.\n"
        "Return ONLY valid JSON with this exact schema: {\"mutations\": [\"...\", ...]}."
    )

    if task_type == "mcq" and (options or "").strip():
        user = (
            f"Original question:\n{question}\n\n"
            f"Options:\n{options}\n\n"
            f"Create exactly {n} minimal rephrased mutations of only the question text.\n"
            "Each mutation must remain standalone and answer-equivalent."
        )
    else:
        user = (
            f"Original question:\n{question}\n\n"
            f"Create exactly {n} minimal rephrased mutations of this question text.\n"
            "Each mutation must remain standalone and answer-equivalent."
        )

    messages = [MessageCls(role="system", content=system), MessageCls(role="user", content=user)]

    for _ in range(max_retries):
        resp = await client.generate(messages)
        if not resp.skipped and not resp.error and resp.content:
            muts = _parse_mutations_from_text(resp.content, n, question)
            if len(muts) == n:
                return muts

    return _fallback_mutations(question, n)


async def _generate_all_mutations_for_items_async(
    client,
    MessageCls,
    items: List[Dict],
    n: int,
    max_retries: int,
    task_type: str,
    concurrency: int,
    progress_desc: str,
) -> Dict[str, List[str]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _worker(item: Dict) -> Tuple[str, List[str]]:
        async with sem:
            try:
                muts = await _generate_mutations_async(
                    client=client,
                    MessageCls=MessageCls,
                    question=item["question"],
                    options=item.get("options", ""),
                    n=n,
                    max_retries=max_retries,
                    task_type=task_type,
                )
            except Exception:
                muts = _fallback_mutations(item["question"], n)
            return item["unique_id"], muts

    tasks = [asyncio.create_task(_worker(item)) for item in items]
    outputs: Dict[str, List[str]] = {}

    for fut in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=progress_desc,
    ):
        uid, muts = await fut
        outputs[uid] = muts

    return outputs


def _compute_correctness(task_type: str, pred: Optional[str], gt: Optional[str]) -> Optional[bool]:
    if task_type == "mcq":
        p = _normalize_letter(pred)
        g = _normalize_letter(gt)
        if p is None or g is None:
            return None
        return p == g
    return _is_open_correct(pred, gt)


def _normalize_ground_truth(task_type: str, gt_raw: Optional[str]) -> Optional[str]:
    if task_type == "mcq":
        return _normalize_letter(gt_raw)
    return _normalize_open_text(gt_raw)


def _text_to_token_counter(text: str) -> Counter:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return Counter(tokens)


def _cosine_similarity_text(a: str, b: str) -> float:
    ca = _text_to_token_counter(a)
    cb = _text_to_token_counter(b)
    if not ca or not cb:
        return 0.0

    shared = set(ca.keys()) & set(cb.keys())
    dot = float(sum(ca[t] * cb[t] for t in shared))
    norm_a = math.sqrt(float(sum(v * v for v in ca.values())))
    norm_b = math.sqrt(float(sum(v * v for v in cb.values())))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _responses_similar_enough(
    task_type: str,
    with_resp: str,
    no_resp: str,
    with_pred: Optional[str],
    no_pred: Optional[str],
    cosine_threshold: float,
) -> Tuple[bool, float]:
    # Mirage labeling policy:
    # - mirage_like=True requires BOTH:
    #   (a) no missing-image acknowledgement, and
    #   (b) with-image and without-image responses are answer-consistent AND
    #       pass the cosine similarity threshold.
    # - mirage_like=False requires acknowledgement.
    # - otherwise mirage_like=None (neutral).
    #
    # Full-response cosine similarity is retained as a fallback for open tasks
    # when parsed final answers are unavailable.
    cosine_sim = _cosine_similarity_text(with_resp, no_resp)

    cosine_ok = cosine_sim >= float(cosine_threshold)

    if task_type == "mcq":
        wp = _normalize_letter(with_pred)
        np = _normalize_letter(no_pred)
        answer_consistent = bool(wp is not None and np is not None and wp == np)
        return bool(answer_consistent and cosine_ok), cosine_sim

    wp = _normalize_open_text(with_pred)
    np = _normalize_open_text(no_pred)
    if wp is not None and np is not None:
        answer_consistent = bool(wp == np)
        return bool(answer_consistent and cosine_ok), cosine_sim

    # Open-answer fallback when extraction/normalization fails:
    # rely on full-response cosine similarity only.
    return bool(cosine_ok), cosine_sim


def _load_reused_mutations(path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"reuse mutations file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("reuse mutations file must be a JSON object")

    # Supports both:
    # 1) global format: {dataset_name: {unique_id: [mutations...]}}
    # 2) per-dataset format: {unique_id: [mutations...]}
    if payload and all(isinstance(v, list) for v in payload.values()):
        return {"__single_dataset__": payload}

    normalized: Dict[str, Dict[str, List[str]]] = {}
    for dataset_name, ds_payload in payload.items():
        if not isinstance(ds_payload, dict):
            continue
        ds_map: Dict[str, List[str]] = {}
        for uid, muts in ds_payload.items():
            if isinstance(muts, list):
                ds_map[str(uid)] = [str(m).strip() for m in muts if str(m).strip()]
        normalized[str(dataset_name)] = ds_map
    return normalized


def _build_auto_reuse_candidates(save_dir: Path, datasets: Sequence[str]) -> List[Path]:
    candidates: List[Path] = []
    seen: set = set()

    def _add(path: Path) -> None:
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    # Prefer current run directory first (reruns), then shared historical locations.
    _add(save_dir / "mutations.json")
    _add(save_dir.parent / "mutations.json")
    _add(FINAL_DATA_ROOT / "mutations.json")
    _add(DEFAULT_RESULTS_ROOT / "mutations.json")
    _add(DEFAULT_RESULTS_ROOT / "ovis" / "mutations.json")
    if len(datasets) == 1:
        _add(DEFAULT_RESULTS_ROOT / datasets[0] / "mutations.json")
    for path in LEGACY_MUTATIONS_PATHS:
        _add(path)
    return candidates


def _payload_covers_requested_datasets(
    payload: Dict[str, Dict[str, List[str]]],
    datasets: Sequence[str],
) -> bool:
    if "__single_dataset__" in payload:
        # Single-dataset artifacts are only valid when exactly one dataset is requested.
        return len(datasets) == 1
    return all(str(ds) in payload for ds in datasets)


def _resolve_reused_mutations(
    explicit_path: str,
    save_dir: Path,
    datasets: Sequence[str],
) -> Tuple[Optional[Path], Optional[Dict[str, Dict[str, List[str]]]]]:
    explicit = str(explicit_path or "").strip()
    if explicit:
        p = Path(explicit).expanduser()
        return p, _load_reused_mutations(p)

    for candidate in _build_auto_reuse_candidates(save_dir=save_dir, datasets=datasets):
        if not candidate.exists():
            continue
        try:
            payload = _load_reused_mutations(candidate)
        except Exception:
            continue
        if _payload_covers_requested_datasets(payload=payload, datasets=datasets):
            return candidate, payload

    return None, None


def _select_reused_mutations_for_uid(
    question: str,
    uid: str,
    reused_for_dataset: Dict[str, List[str]],
    n: int,
) -> List[str]:
    if uid not in reused_for_dataset:
        raise KeyError(f"Missing reused mutations for uid={uid}")

    muts = [m.strip() for m in reused_for_dataset[uid] if isinstance(m, str) and m.strip()]
    dedup: List[str] = []
    seen = {question.strip()}
    for m in muts:
        if m not in seen:
            dedup.append(m)
            seen.add(m)

    if len(dedup) < n:
        raise ValueError(
            f"Reused mutations for uid={uid} has only {len(dedup)} unique entries, "
            f"but --num_mutations={n}."
        )
    return dedup[:n]


def main():
    args = parse_args()
    vlm_cfg = resolve_vlm_config(args.vlm)
    legacy_override = str(args.vllm_model).strip()
    explicit_override = str(args.vllm_model_override).strip()
    if explicit_override and legacy_override:
        raise ValueError("Specify only one of --vllm_model_override or --vllm_model (deprecated).")
    model_override = explicit_override or legacy_override or os.getenv("VLLM_MODEL_NAME", "").strip()
    args.vllm_model = resolve_vllm_model_name(vlm_cfg.key, override_model_name=model_override)
    explicit_base_url = str(args.vllm_base_url or "").strip()
    env_base_url = str(os.getenv("VLLM_BASE_URL", "")).strip()
    if explicit_base_url:
        args.vllm_base_url = explicit_base_url
    elif env_base_url:
        args.vllm_base_url = env_base_url
    else:
        args.vllm_base_url = DEFAULT_VLLM_BASE_URLS.get(vlm_cfg.key, "http://localhost:8001/v1")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    invalid = [d for d in datasets if d not in DATASET_CONFIGS]
    if invalid:
        raise ValueError(f"Unsupported dataset(s): {invalid}. Supported: {sorted(DATASET_CONFIGS.keys())}")

    if str(args.save_path).strip():
        save_dir = Path(args.save_path).expanduser().resolve()
    else:
        save_dir = DEFAULT_RESULTS_ROOT / vlm_cfg.key
    save_dir.mkdir(parents=True, exist_ok=True)

    repo_root = REPO_ROOT.resolve()

    reused_mutations_path, reused_mutations_all = _resolve_reused_mutations(
        explicit_path=args.reuse_mutations_path,
        save_dir=save_dir,
        datasets=datasets,
    )
    args.reuse_mutations_path = str(reused_mutations_path) if reused_mutations_path is not None else ""
    if reused_mutations_all is None:
        mutator_client, MessageCls = _make_mutation_client(
            repo_root=repo_root,
            backend=args.mutator_backend,
            model_name=args.mutator_model,
            temperature=args.mutation_temperature,
        )
    else:
        print(f"Reusing mutations artifact: {args.reuse_mutations_path}")

    all_results: List[Dict] = []
    all_grouped: List[Dict] = []
    all_flagged: List[Dict] = []
    all_mutations_by_dataset: Dict[str, Dict[str, List[str]]] = {}
    per_dataset_summary: Dict[str, Dict] = {}

    for dataset_name in datasets:
        cfg = DATASET_CONFIGS[dataset_name]
        task_type = cfg["task_type"]
        system_prompt = cfg["system_prompt"]

        ds_dir = save_dir / dataset_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        items = _load_dataset_items(repo_root, dataset_name)
        if args.max_questions_per_dataset > 0:
            items = items[: args.max_questions_per_dataset]
        if not items:
            raise ValueError(f"No items loaded for dataset: {dataset_name}")
        glm_resize_stats = _maybe_downscale_items_for_glm(
            items=items,
            vlm_key=vlm_cfg.key,
        )
        if (
            int(glm_resize_stats.get("num_images_downscaled", 0)) > 0
            or int(glm_resize_stats.get("num_images_min_edge_adjusted", 0)) > 0
        ):
            print(
                f"{dataset_name}: glm image normalization "
                f"{glm_resize_stats['num_images_downscaled']}/{glm_resize_stats['num_images_seen']} "
                f"downscaled to max edge {glm_resize_stats['glm_image_downscale_max_edge']}; "
                f"{glm_resize_stats['num_images_min_edge_adjusted']}/{glm_resize_stats['num_images_seen']} "
                f"adjusted to min edge {glm_resize_stats['glm_image_min_edge']}"
            )

        if reused_mutations_all is not None:
            reused_for_dataset = reused_mutations_all.get(dataset_name)
            if reused_for_dataset is None:
                reused_for_dataset = reused_mutations_all.get("__single_dataset__")
            if reused_for_dataset is None:
                raise KeyError(
                    f"Dataset '{dataset_name}' not found in reused mutations file: {args.reuse_mutations_path}"
                )
            all_mutations = {}
            for item in items:
                uid = item["unique_id"]
                all_mutations[uid] = _select_reused_mutations_for_uid(
                    question=item["question"],
                    uid=uid,
                    reused_for_dataset=reused_for_dataset,
                    n=args.num_mutations,
                )
        else:
            all_mutations = asyncio.run(
                _generate_all_mutations_for_items_async(
                    client=mutator_client,
                    MessageCls=MessageCls,
                    items=items,
                    n=args.num_mutations,
                    max_retries=args.mutation_max_retries,
                    task_type=task_type,
                    concurrency=args.mutation_concurrency,
                    progress_desc=f"{dataset_name}: generating mutations",
                )
            )
        all_mutations_by_dataset[dataset_name] = all_mutations

        results: List[Dict] = []
        per_base_question: Dict[str, Dict] = {}
        variant_entries: List[Dict] = []

        for item in items:
            gt_raw = item.get("ground_truth")
            gt_norm = _normalize_ground_truth(task_type, gt_raw)
            uid = item["unique_id"]

            per_base_question[uid] = {
                "dataset": dataset_name,
                "task_type": task_type,
                "vlm_key": vlm_cfg.key,
                "vllm_model": args.vllm_model,
                "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
                "unique_id": uid,
                "question_id": item["question_id"],
                "base_question": item["question"],
                "ground_truth_raw": gt_raw,
                "ground_truth_normalized": gt_norm,
                "image_count": len(item.get("images", [])),
                "num_variants": 0,
                "mirage_like_variants": 0,
                "non_mirage_variants": 0,
                "neutral_variants": 0,
                "mirage_flip_flag": False,
                "variants": [],
            }

            variants = [{"variant_id": "original", "question_text": item["question"], "is_original": True}]
            for i, m in enumerate(all_mutations[uid], start=1):
                variants.append({"variant_id": f"mutation_{i}", "question_text": m, "is_original": False})

            options_map = _parse_options(item.get("options", "")) if task_type == "mcq" else {}
            for v in variants:
                variant_entries.append(
                    {
                        "dataset": dataset_name,
                        "task_type": task_type,
                        "unique_id": uid,
                        "question_id": item["question_id"],
                        "category": item.get("category", ""),
                        "ground_truth_raw": gt_raw,
                        "ground_truth_normalized": gt_norm,
                        "options": item.get("options", ""),
                        "options_map": options_map,
                        "image_count": len(item.get("images", [])),
                        "images": item.get("images", []),
                        "variant_id": v["variant_id"],
                        "is_original": v["is_original"],
                        "variant_question_text": v["question_text"],
                        "prompt_text": _build_variant_prompt(v["question_text"], item.get("options", "")),
                    }
                )

        with_image_responses, with_image_failed_vllm_request_count = asyncio.run(
            _run_vllm_generation_pass_async(
                entries=variant_entries,
                include_images=True,
                system_prompt=system_prompt,
                args=args,
                progress_desc=f"{dataset_name}: with-image vLLM inference",
            )
        )
        without_image_responses, without_image_failed_vllm_request_count = asyncio.run(
            _run_vllm_generation_pass_async(
                entries=variant_entries,
                include_images=False,
                system_prompt=system_prompt,
                args=args,
                progress_desc=f"{dataset_name}: without-image vLLM inference",
            )
        )

        short_response_pair_count = 0
        short_with_image_response_count = 0
        short_without_image_response_count = 0
        missing_response_pair_count = 0
        missing_with_image_response_count = 0
        missing_without_image_response_count = 0
        for entry, with_resp, no_resp in zip(variant_entries, with_image_responses, without_image_responses):
            # Keep every pair and annotate missing/short responses for downstream filtering.
            with_token_count = _count_tokens(with_resp)
            no_token_count = _count_tokens(no_resp)
            with_missing = not bool(with_resp)
            no_missing = not bool(no_resp)
            with_short = with_token_count < 10
            no_short = no_token_count < 10

            if with_missing:
                missing_with_image_response_count += 1
            if no_missing:
                missing_without_image_response_count += 1
            if with_missing or no_missing:
                missing_response_pair_count += 1

            if with_short:
                short_with_image_response_count += 1
            if no_short:
                short_without_image_response_count += 1
            if with_short or no_short:
                short_response_pair_count += 1

            if task_type == "mcq":
                with_pred = _extract_option_letter(with_resp, entry["options_map"])
                no_pred = _extract_option_letter(no_resp, entry["options_map"])
            else:
                with_pred = _extract_open_answer(with_resp)
                no_pred = _extract_open_answer(no_resp)

            with_correct = _compute_correctness(task_type, with_pred, entry["ground_truth_raw"])
            no_correct = _compute_correctness(task_type, no_pred, entry["ground_truth_raw"])

            mentions_no_image, expresses_uncertainty = _detect_acknowledgement(no_resp)
            acknowledged = mentions_no_image or expresses_uncertainty
            responses_similar, response_cosine_similarity = _responses_similar_enough(
                task_type=task_type,
                with_resp=with_resp,
                no_resp=no_resp,
                with_pred=with_pred,
                no_pred=no_pred,
                cosine_threshold=float(args.response_cosine_similarity_threshold),
            )
            if with_missing or no_missing:
                mirage_like = None
                mirage_label = "neutral_missing_response"
            elif acknowledged:
                mirage_like: Optional[bool] = False
                mirage_label = "non_mirage_acknowledged"
            elif responses_similar:
                mirage_like = True
                mirage_label = "mirage_like_consistent"
            else:
                mirage_like = None
                mirage_label = "neutral_no_ack_but_response_shift"

            record = {
                "dataset": entry["dataset"],
                "task_type": entry["task_type"],
                "vlm_key": vlm_cfg.key,
                "vllm_model": args.vllm_model,
                "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
                "unique_id": entry["unique_id"],
                "question_id": entry["question_id"],
                "category": entry["category"],
                "ground_truth_raw": entry["ground_truth_raw"],
                "ground_truth_normalized": entry["ground_truth_normalized"],
                "options": entry["options"],
                "image_count": entry["image_count"],
                "variant_id": entry["variant_id"],
                "is_original": entry["is_original"],
                "variant_question_text": entry["variant_question_text"],
                "prompt_text": entry["prompt_text"],
                "system_prompt": system_prompt,
                "with_image": {
                    "response": with_resp,
                    "response_token_count": int(with_token_count),
                    "response_missing": bool(with_missing),
                    "response_too_short": bool(with_short),
                    "reasoning_chain": _extract_reasoning_chain(with_resp),
                    "pred": with_pred,
                    "pred_normalized": _normalize_letter(with_pred) if task_type == "mcq" else _normalize_open_text(with_pred),
                    "correct": with_correct,
                },
                "without_image": {
                    "response": no_resp,
                    "response_token_count": int(no_token_count),
                    "response_missing": bool(no_missing),
                    "response_too_short": bool(no_short),
                    "reasoning_chain": _extract_reasoning_chain(no_resp),
                    "pred": no_pred,
                    "pred_normalized": _normalize_letter(no_pred) if task_type == "mcq" else _normalize_open_text(no_pred),
                    "correct": no_correct,
                    "mentions_no_image": bool(mentions_no_image),
                    "expresses_uncertainty": bool(expresses_uncertainty),
                    "acknowledged_missing_or_uncertain": bool(acknowledged),
                    "response_similarity_with_image": bool(responses_similar),
                    "response_cosine_similarity_with_image": float(response_cosine_similarity),
                    "mirage_like": mirage_like,
                    "mirage_label": mirage_label,
                },
                "pair_has_missing_response": bool(with_missing or no_missing),
                "pair_has_short_response": bool(with_short or no_short),
            }
            results.append(record)
            all_results.append(record)
            per_base_question[entry["unique_id"]]["variants"].append(record)

        for grouped_record in per_base_question.values():
            variant_records = grouped_record["variants"]
            mirage_like_variants = sum(int(r["without_image"]["mirage_like"] is True) for r in variant_records)
            non_mirage_variants = sum(int(r["without_image"]["mirage_like"] is False) for r in variant_records)
            neutral_variants = sum(int(r["without_image"]["mirage_like"] is None) for r in variant_records)
            grouped_record["num_variants"] = len(variant_records)
            grouped_record["mirage_like_variants"] = mirage_like_variants
            grouped_record["non_mirage_variants"] = non_mirage_variants
            grouped_record["neutral_variants"] = neutral_variants
            grouped_record["mirage_flip_flag"] = bool(mirage_like_variants > 0 and non_mirage_variants > 0)
            all_grouped.append(grouped_record)

        grouped = list(per_base_question.values())
        flagged = [g for g in grouped if g["mirage_flip_flag"]]
        all_flagged.extend(flagged)

        with_image_records = [r["with_image"] for r in results]
        without_image_records = [r["without_image"] for r in results]

        with_correct_vals = [r["correct"] for r in with_image_records if r["correct"] is not None]
        without_correct_vals = [r["correct"] for r in without_image_records if r["correct"] is not None]
        without_mirage_like_count = sum(int(r["mirage_like"] is True) for r in without_image_records)
        without_non_mirage_count = sum(int(r["mirage_like"] is False) for r in without_image_records)
        without_neutral_count = sum(int(r["mirage_like"] is None) for r in without_image_records)
        without_classified_count = without_mirage_like_count + without_non_mirage_count

        summary = {
            "dataset": dataset_name,
            "task_type": task_type,
            "vlm_key": vlm_cfg.key,
            "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
            "num_base_questions": len(items),
            "num_mutations_per_question": args.num_mutations,
            "total_generated_variants": len(variant_entries),
            "total_variants": len(results),
            "kept_all_variants": True,
            "short_response_pair_count": int(short_response_pair_count),
            "short_with_image_response_count": int(short_with_image_response_count),
            "short_without_image_response_count": int(short_without_image_response_count),
            "missing_response_pair_count": int(missing_response_pair_count),
            "missing_with_image_response_count": int(missing_with_image_response_count),
            "missing_without_image_response_count": int(missing_without_image_response_count),
            # Backward-compatible legacy fields; variants are no longer dropped.
            "skipped_short_response_count": 0,
            "skipped_failed_response_count": 0,
            "with_image_failed_vllm_request_count": int(with_image_failed_vllm_request_count),
            "without_image_failed_vllm_request_count": int(without_image_failed_vllm_request_count),
            "total_failed_vllm_request_count": int(
                with_image_failed_vllm_request_count + without_image_failed_vllm_request_count
            ),
            "mutator_model": args.mutator_model,
            "mutator_backend": args.mutator_backend,
            "reuse_mutations_path": args.reuse_mutations_path or None,
            "vllm_model": args.vllm_model,
            "vllm_base_url": args.vllm_base_url,
            "vllm_concurrency": int(args.vllm_concurrency),
            "strict_vllm_errors": bool(args.strict_vllm_errors),
            "enable_thinking": bool(args.enable_thinking),
            "glm_image_downscale_max_edge": int(glm_resize_stats["glm_image_downscale_max_edge"]),
            "glm_image_min_edge": int(glm_resize_stats["glm_image_min_edge"]),
            "glm_images_seen_for_downscale_check": int(glm_resize_stats["num_images_seen"]),
            "glm_images_downscaled": int(glm_resize_stats["num_images_downscaled"]),
            "glm_images_min_edge_adjusted": int(glm_resize_stats["num_images_min_edge_adjusted"]),
            "response_cosine_similarity_threshold": float(args.response_cosine_similarity_threshold),
            "mirage_flip_questions_count": len(flagged),
            "mirage_flip_questions_rate": (len(flagged) / len(items)) if items else None,
            "without_image_mirage_like_count": without_mirage_like_count,
            "without_image_non_mirage_acknowledged_count": without_non_mirage_count,
            "without_image_neutral_count": without_neutral_count,
            "without_image_classified_count": without_classified_count,
            "without_image_mirage_like_rate": (
                without_mirage_like_count / without_classified_count
                if without_classified_count
                else None
            ),
            "without_image_mentions_no_image_count": sum(int(r["mentions_no_image"]) for r in without_image_records),
            "without_image_uncertainty_count": sum(int(r["expresses_uncertainty"]) for r in without_image_records),
            "with_image_accuracy": (
                sum(int(v) for v in with_correct_vals) / len(with_correct_vals)
                if with_correct_vals
                else None
            ),
            "without_image_accuracy": (
                sum(int(v) for v in without_correct_vals) / len(without_correct_vals)
                if without_correct_vals
                else None
            ),
            "with_image_scored_count": len(with_correct_vals),
            "without_image_scored_count": len(without_correct_vals),
        }

        per_dataset_summary[dataset_name] = summary
        run_metadata = {
            "vlm_key": vlm_cfg.key,
            "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
            "vllm_model": args.vllm_model,
            "vllm_base_url": args.vllm_base_url,
            "save_dir": str(ds_dir),
        }

        with open(ds_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(ds_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(run_metadata, f, indent=2, ensure_ascii=False)
        with open(ds_dir / "responses.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(ds_dir / "grouped_by_base_question.json", "w", encoding="utf-8") as f:
            json.dump(grouped, f, indent=2, ensure_ascii=False)
        with open(ds_dir / "mirage_flip_questions.json", "w", encoding="utf-8") as f:
            json.dump(flagged, f, indent=2, ensure_ascii=False)
        with open(ds_dir / "mutations.json", "w", encoding="utf-8") as f:
            json.dump(all_mutations, f, indent=2, ensure_ascii=False)

    overall_summary = {
        "datasets": datasets,
        "num_datasets": len(datasets),
        "vlm_key": vlm_cfg.key,
        "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
        "num_mutations_per_question": args.num_mutations,
        "mutator_model": args.mutator_model,
        "mutator_backend": args.mutator_backend,
        "reuse_mutations_path": args.reuse_mutations_path or None,
        "vllm_model": args.vllm_model,
        "vllm_base_url": args.vllm_base_url,
        "vllm_concurrency": int(args.vllm_concurrency),
        "enable_thinking": bool(args.enable_thinking),
        "glm_image_downscale_max_edge": int(GLM_IMAGE_MAX_EDGE),
        "glm_image_min_edge": int(GLM_IMAGE_MIN_EDGE),
        "overall_glm_images_seen_for_downscale_check": sum(
            int(s.get("glm_images_seen_for_downscale_check", 0)) for s in per_dataset_summary.values()
        ),
        "overall_glm_images_downscaled": sum(
            int(s.get("glm_images_downscaled", 0)) for s in per_dataset_summary.values()
        ),
        "overall_glm_images_min_edge_adjusted": sum(
            int(s.get("glm_images_min_edge_adjusted", 0)) for s in per_dataset_summary.values()
        ),
        "response_cosine_similarity_threshold": float(args.response_cosine_similarity_threshold),
        "total_base_questions": sum(s["num_base_questions"] for s in per_dataset_summary.values()),
        "total_variants": sum(s["total_variants"] for s in per_dataset_summary.values()),
        "total_mirage_flip_questions": sum(s["mirage_flip_questions_count"] for s in per_dataset_summary.values()),
        "overall_without_image_mirage_like_count": sum(
            s["without_image_mirage_like_count"] for s in per_dataset_summary.values()
        ),
        "overall_without_image_non_mirage_acknowledged_count": sum(
            s["without_image_non_mirage_acknowledged_count"] for s in per_dataset_summary.values()
        ),
        "overall_without_image_neutral_count": sum(
            s["without_image_neutral_count"] for s in per_dataset_summary.values()
        ),
        "overall_short_response_pair_count": sum(
            int(s.get("short_response_pair_count", 0)) for s in per_dataset_summary.values()
        ),
        "overall_short_with_image_response_count": sum(
            int(s.get("short_with_image_response_count", 0)) for s in per_dataset_summary.values()
        ),
        "overall_short_without_image_response_count": sum(
            int(s.get("short_without_image_response_count", 0)) for s in per_dataset_summary.values()
        ),
        "overall_missing_response_pair_count": sum(
            int(s.get("missing_response_pair_count", 0)) for s in per_dataset_summary.values()
        ),
        "overall_missing_with_image_response_count": sum(
            int(s.get("missing_with_image_response_count", 0)) for s in per_dataset_summary.values()
        ),
        "overall_missing_without_image_response_count": sum(
            int(s.get("missing_without_image_response_count", 0)) for s in per_dataset_summary.values()
        ),
        "overall_failed_vllm_request_count": sum(
            int(s.get("total_failed_vllm_request_count", 0)) for s in per_dataset_summary.values()
        ),
        "strict_vllm_errors": bool(args.strict_vllm_errors),
        "per_dataset": per_dataset_summary,
    }
    run_metadata = {
        "vlm_key": vlm_cfg.key,
        "vlm_hf_repo_id": vlm_cfg.hf_repo_id,
        "vllm_model": args.vllm_model,
        "vllm_base_url": args.vllm_base_url,
        "save_dir": str(save_dir),
    }

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)
    with open(save_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)
    with open(save_dir / "responses.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(save_dir / "grouped_by_base_question.json", "w", encoding="utf-8") as f:
        json.dump(all_grouped, f, indent=2, ensure_ascii=False)
    with open(save_dir / "mirage_flip_questions.json", "w", encoding="utf-8") as f:
        json.dump(all_flagged, f, indent=2, ensure_ascii=False)
    with open(save_dir / "mutations.json", "w", encoding="utf-8") as f:
        json.dump(all_mutations_by_dataset, f, indent=2, ensure_ascii=False)

    print(json.dumps(overall_summary, indent=2, ensure_ascii=False))
    print(f"Saved outputs to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
