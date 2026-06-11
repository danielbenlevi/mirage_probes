#!/usr/bin/env python3
"""Analyze text confounds in Ovis/Qwen mirage probe data.

This script intentionally uses only the generated assistant text that is passed
into activation extraction: with_image.response for all-examples rows and the
assistant turn in each contrastive conversation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SUPPORTED_BENCHMARKS = ("vqa_rad", "mmmu_pro", "medxpertqa_mm")
DEFAULT_ROOT = Path(".")
MIN_RESPONSE_TOKENS = 10

PHRASE_GROUPS: Dict[str, List[str]] = {
    "uncertainty": [
        "cannot determine",
        "cannot be determined",
        "can't determine",
        "can't tell",
        "not possible",
        "impossible",
        "insufficient",
        "uncertain",
        "unsure",
        "unable to determine",
        "definitively determine",
        "image alone",
        "without more information",
        "without additional",
    ],
    "answer_boilerplate": [
        "the correct answer is",
        "correct answer is",
        "answer is",
        "therefore",
        "final answer",
        "[[",
    ],
    "reasoning_scaffold": [
        "we need",
        "to determine",
        "the question asks",
        "among the options",
        "option a",
        "option b",
        "option c",
        "option d",
        "option e",
    ],
    "image_grounding": [
        "based on the image",
        "based on the provided image",
        "in the image",
        "the image shows",
        "shown in the image",
        "visible",
        "appears",
        "visual evidence",
    ],
    "hedging": [
        "likely",
        "suggests",
        "consistent with",
        "however",
        "may",
        "could",
    ],
    "radiology_terms": [
        "ct",
        "mri",
        "x-ray",
        "radiograph",
        "contrast",
        "lesion",
        "mass",
    ],
}

PHRASES = sorted({phrase for phrases in PHRASE_GROUPS.values() for phrase in phrases})
OPTION_LIST_RE = re.compile(r"(?m)^\s*(?:[-*]\s*)?(?:[A-E][.)]|\([A-E]\))\s+")
FINAL_BRACKET_RE = re.compile(r"\[\[\s*([^\]]+?)\s*\]\]")
WORD_RE = re.compile(r"[A-Za-z0-9_\[\]]+")


Example = Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare label-correlated textual separators in Ovis and Qwen generated "
            "responses used by all-examples and contrastive probe trainers."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output_json", type=Path, default=Path("tmp_artifacts/confound_analysis.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--target_examples_per_class", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument(
        "--settings",
        choices=("all_examples", "contrastive", "both"),
        default="all_examples",
        help=(
            "Which generated-response artifacts to analyze. The default targets the "
            "all-examples confound question; use 'both' to include image-heavy "
            "contrastive pair files as well."
        ),
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokens(text: str) -> List[str]:
    return WORD_RE.findall(str(text or "").lower())


def token_ngrams(text: str) -> List[str]:
    words = tokens(text)
    return words + [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]


def example_ngrams(ex: Example) -> set[str]:
    cached = ex.get("_ngram_set")
    if cached is None:
        cached = set(token_ngrams(str(ex["text"])))
        ex["_ngram_set"] = cached
    return cached


def row_label(row: Dict[str, Any]) -> Optional[int]:
    mirage_like = (row.get("without_image") or {}).get("mirage_like")
    if mirage_like is True:
        return 1
    if mirage_like is False:
        return 0
    return None


def assistant_text(conversation: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(list(conversation or [])):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


def infer_benchmark_from_pair(pair: Dict[str, Any]) -> str:
    system_prompt = ""
    for key in ("non_mirage_conversation", "mirage_conversation"):
        conv = pair.get(key, [])
        if conv and isinstance(conv, list) and conv[0].get("role") == "system":
            system_prompt = str(conv[0].get("content", ""))
            break
    prompt = system_prompt.lower()
    if ("microvqa" in prompt) or ("microscopy image" in prompt):
        return "microvqa"
    if ("medxpertqa-mm" in prompt) or ("medical professional" in prompt):
        return "medxpertqa_mm"
    if ("mmmu-pro" in prompt) or ("multiple academic disciplines" in prompt):
        return "mmmu_pro"
    if ("vqa-rad" in prompt) or ("radiologist" in prompt) or ("radiology image" in prompt):
        return "vqa_rad"
    return "unknown"


def load_all_examples(path: Path, model: str) -> List[Example]:
    rows = json.load(open(path, "r", encoding="utf-8"))
    examples: List[Example] = []
    for row in rows:
        dataset = str(row.get("dataset", ""))
        if dataset not in SUPPORTED_BENCHMARKS:
            continue
        label = row_label(row)
        if label is None:
            continue
        text = str((row.get("with_image") or {}).get("response", ""))
        if len(text.split()) < MIN_RESPONSE_TOKENS:
            continue
        uid = str(row.get("unique_id", ""))
        qid = str(row.get("question_id", ""))
        examples.append(
            {
                "model": model,
                "setting": "all_examples",
                "dataset": dataset,
                "label": int(label),
                "text": text,
                "group_id": f"{dataset}::{uid or qid}",
                "unique_id": uid,
                "question_id": qid,
                "variant_id": str(row.get("variant_id", "")),
                "pred": (row.get("with_image") or {}).get("pred_normalized"),
                "correct": (row.get("with_image") or {}).get("correct"),
            }
        )
    return examples


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False
    with open(path, "r", encoding="utf-8") as f:
        while not finished:
            chunk = f.read(chunk_size)
            if chunk:
                buffer += chunk
            elif not buffer.strip():
                break

            idx = 0
            while True:
                while idx < len(buffer) and buffer[idx].isspace():
                    idx += 1
                if not started:
                    if idx >= len(buffer):
                        break
                    if buffer[idx] != "[":
                        raise ValueError(f"Expected JSON array in {path}")
                    started = True
                    idx += 1
                    continue
                while idx < len(buffer) and (buffer[idx].isspace() or buffer[idx] == ","):
                    idx += 1
                if idx < len(buffer) and buffer[idx] == "]":
                    finished = True
                    buffer = buffer[idx + 1 :]
                    break
                if idx >= len(buffer):
                    break
                try:
                    obj, end = decoder.raw_decode(buffer, idx)
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected objects inside JSON array in {path}")
                yield obj
                idx = end

            if idx > 0:
                buffer = buffer[idx:]
            if not chunk and not finished:
                if buffer.strip():
                    raise ValueError(f"Unexpected trailing/incomplete JSON while reading {path}")
                break


def load_contrastive_pairs(path: Path, model: str) -> List[Example]:
    examples: List[Example] = []
    kept_pair_id = 0
    for raw_pair_id, pair in enumerate(iter_json_array(path)):
        dataset = infer_benchmark_from_pair(pair)
        if dataset not in SUPPORTED_BENCHMARKS:
            continue
        non_text = assistant_text(pair.get("non_mirage_conversation", []))
        mirage_text = assistant_text(pair.get("mirage_conversation", []))
        if len(non_text.split()) < MIN_RESPONSE_TOKENS or len(mirage_text.split()) < MIN_RESPONSE_TOKENS:
            continue
        for label, key, text in (
            (0, "non_mirage_variant_id", non_text),
            (1, "mirage_variant_id", mirage_text),
        ):
            examples.append(
                {
                    "model": model,
                    "setting": "contrastive",
                    "dataset": dataset,
                    "label": label,
                    "text": text,
                    "group_id": f"{model}::contrastive::{kept_pair_id}",
                    "raw_pair_id": raw_pair_id,
                    "pair_id": kept_pair_id,
                    "unique_id": str(pair.get("source_unique_id", "")),
                    "question_id": str(pair.get("source_question_id", "")),
                    "variant_id": str(pair.get(key, "")),
                }
            )
        kept_pair_id += 1
    return examples


def filter_eval_pool_remove_train_overlap(
    train_examples: Sequence[Example],
    eval_pool: Sequence[Example],
) -> Tuple[List[Example], Dict[str, int]]:
    seen_uid: set[Tuple[str, str]] = set()
    seen_qid: set[Tuple[str, str]] = set()
    for ex in train_examples:
        ds = str(ex.get("dataset", ""))
        uid = str(ex.get("unique_id", "")).strip()
        qid = str(ex.get("question_id", "")).strip()
        if ds and uid:
            seen_uid.add((ds, uid))
        if ds and qid:
            seen_qid.add((ds, qid))

    kept: List[Example] = []
    dropped_uid = 0
    dropped_qid = 0
    for ex in eval_pool:
        ds = str(ex.get("dataset", ""))
        uid = str(ex.get("unique_id", "")).strip()
        qid = str(ex.get("question_id", "")).strip()
        if ds and uid and (ds, uid) in seen_uid:
            dropped_uid += 1
            continue
        if ds and qid and (ds, qid) in seen_qid:
            dropped_qid += 1
            continue
        kept.append(ex)

    return kept, {
        "before": int(len(eval_pool)),
        "after": int(len(kept)),
        "dropped_total": int(len(eval_pool) - len(kept)),
        "dropped_uid_overlap": int(dropped_uid),
        "dropped_qid_overlap": int(dropped_qid),
    }


def class_counts(examples: Sequence[Example]) -> Dict[str, int]:
    counts = Counter(int(ex["label"]) for ex in examples)
    return {"class0": int(counts.get(0, 0)), "class1": int(counts.get(1, 0))}


def rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom else 0.0


def phrase_hit(text: str, phrase: str) -> bool:
    return phrase in normalize_text(text).lower()


def summarize_text(examples: Sequence[Example], top_k: int) -> Dict[str, Any]:
    by_label = {0: [ex for ex in examples if int(ex["label"]) == 0], 1: [ex for ex in examples if int(ex["label"]) == 1]}
    label_summary: Dict[str, Any] = {}
    for label, label_examples in by_label.items():
        lengths = [len(tokens(ex["text"])) for ex in label_examples]
        finals = [final_answer(ex["text"]) for ex in label_examples]
        label_summary[f"class{label}"] = {
            "n": len(label_examples),
            "mean_tokens": mean(lengths) if lengths else None,
            "median_tokens": median(lengths) if lengths else None,
            "option_list_rate": rate(sum(bool(OPTION_LIST_RE.search(ex["text"])) for ex in label_examples), len(label_examples)),
            "final_bracket_rate": rate(sum(bool(FINAL_BRACKET_RE.search(ex["text"])) for ex in label_examples), len(label_examples)),
            "final_answer_counts": dict(Counter(x for x in finals if x).most_common(10)),
            "phrase_group_rates": phrase_group_rates(label_examples),
        }

    phrase_skews = []
    for phrase in PHRASES:
        r0 = rate(sum(phrase_hit(ex["text"], phrase) for ex in by_label[0]), len(by_label[0]))
        r1 = rate(sum(phrase_hit(ex["text"], phrase) for ex in by_label[1]), len(by_label[1]))
        phrase_skews.append({"phrase": phrase, "class0_rate": r0, "class1_rate": r1, "diff_class1_minus_class0": r1 - r0})

    return {
        "counts": class_counts(examples),
        "overall_phrase_group_rates": phrase_group_rates(examples),
        "by_label": label_summary,
        "largest_phrase_skews": sorted(phrase_skews, key=lambda x: abs(x["diff_class1_minus_class0"]), reverse=True)[:top_k],
        "top_ngrams_class1": top_ngrams(examples, target_label=1, top_k=top_k),
        "top_ngrams_class0": top_ngrams(examples, target_label=0, top_k=top_k),
    }


def phrase_group_rates(examples: Sequence[Example]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for group_name, phrases in PHRASE_GROUPS.items():
        out[group_name] = rate(
            sum(any(phrase_hit(ex["text"], phrase) for phrase in phrases) for ex in examples),
            len(examples),
        )
    return out


def final_answer(text: str) -> str:
    matches = FINAL_BRACKET_RE.findall(text)
    if not matches:
        return ""
    return normalize_text(matches[-1])[:40]


def top_ngrams(examples: Sequence[Example], target_label: int, top_k: int) -> List[Dict[str, Any]]:
    by_label = {0: [], 1: []}
    rng = random.Random(0)
    for label in (0, 1):
        label_examples = [ex for ex in examples if int(ex["label"]) == label]
        if len(label_examples) > 2000:
            label_examples = rng.sample(label_examples, 2000)
        by_label[label] = label_examples

    doc_freq = {0: Counter(), 1: Counter()}
    for label in (0, 1):
        for ex in by_label[label]:
            grams = set(example_ngrams(ex))
            grams = {g for g in grams if len(g) > 1 and not g.isdigit()}
            doc_freq[label].update(grams)

    n0 = len(by_label[0])
    n1 = len(by_label[1])
    rows = []
    for gram in set(doc_freq[0]) | set(doc_freq[1]):
        c0 = int(doc_freq[0][gram])
        c1 = int(doc_freq[1][gram])
        if c0 + c1 < max(4, int(0.02 * (n0 + n1))):
            continue
        p0 = (c0 + 0.5) / (n0 + 1.0)
        p1 = (c1 + 0.5) / (n1 + 1.0)
        log_odds = math.log(p1 / (1.0 - p1)) - math.log(p0 / (1.0 - p0))
        score = log_odds if target_label == 1 else -log_odds
        rows.append((score, gram, c0, c1, log_odds))
    rows.sort(reverse=True)
    return [
        {
            "ngram": gram,
            "class0_doc_count": c0,
            "class1_doc_count": c1,
            "log_odds_class1_vs_class0": log_odds,
        }
        for _score, gram, c0, c1, log_odds in rows[:top_k]
    ]


def trainer_like_all_examples_selection(
    examples: Sequence[Example],
    seed: int,
    target_examples_per_class: int,
    datasets: Optional[Sequence[str]] = None,
) -> Tuple[List[Example], Dict[str, Any]]:
    rng = random.Random(seed)
    dataset_set = set(datasets or SUPPORTED_BENCHMARKS)
    grouped: Dict[Tuple[str, str, int], List[Example]] = defaultdict(list)
    for ex in examples:
        if ex["dataset"] not in dataset_set:
            continue
        grouped[(str(ex["dataset"]), str(ex["group_id"]), int(ex["label"]))].append(ex)

    picked_by_label_dataset: Dict[Tuple[int, str], List[Example]] = defaultdict(list)
    for (dataset, _group_id, label), rows in grouped.items():
        picked_by_label_dataset[(label, dataset)].append(rng.choice(rows))

    represented = [
        dataset
        for dataset in sorted(dataset_set)
        if len(picked_by_label_dataset[(0, dataset)]) >= 3 and len(picked_by_label_dataset[(1, dataset)]) >= 3
    ]
    caps = {
        dataset: min(len(picked_by_label_dataset[(0, dataset)]), len(picked_by_label_dataset[(1, dataset)]))
        for dataset in represented
    }
    alloc = allocate_by_capacity(caps, target_examples_per_class)

    selected: List[Example] = []
    for dataset, count in alloc.items():
        for label in (0, 1):
            selected.extend(rng.sample(picked_by_label_dataset[(label, dataset)], count))
    rng.shuffle(selected)
    return selected, {"represented_datasets": represented, "selected_per_dataset_per_class": alloc}


def _holdout_pool_counts_by_benchmark(
    pool_class0: Sequence[Example],
    pool_class1: Sequence[Example],
) -> Dict[str, Dict[str, int]]:
    class0_counts: Dict[str, int] = defaultdict(int)
    class1_counts: Dict[str, int] = defaultdict(int)
    for item in pool_class0:
        class0_counts[str(item.get("dataset", "unknown"))] += 1
    for item in pool_class1:
        class1_counts[str(item.get("dataset", "unknown"))] += 1
    benchmarks = sorted(set(class0_counts.keys()) | set(class1_counts.keys()))
    out: Dict[str, Dict[str, int]] = {}
    for ds in benchmarks:
        c0 = int(class0_counts.get(ds, 0))
        c1 = int(class1_counts.get(ds, 0))
        out[str(ds)] = {
            "class0": c0,
            "class1": c1,
            "pair_cap": int(min(c0, c1)),
        }
    return out


def _plan_balanced_holdout_selection(
    pool_class0: Sequence[Example],
    pool_class1: Sequence[Example],
    requested_per_class: int,
) -> Dict[str, Any]:
    counts_by_benchmark = _holdout_pool_counts_by_benchmark(pool_class0=pool_class0, pool_class1=pool_class1)
    represented = sorted(
        ds for ds, info in counts_by_benchmark.items()
        if int(info.get("class0", 0)) > 0 and int(info.get("class1", 0)) > 0
    )
    if not represented:
        return {
            "selected_num_per_class": 0,
            "selected_pairs_by_benchmark": {},
            "counts_by_benchmark_before_selection": counts_by_benchmark,
            "represented_benchmarks_with_both_classes": [],
        }

    pair_caps_by_benchmark = {
        str(ds): int(counts_by_benchmark[ds]["pair_cap"])
        for ds in represented
    }
    total_pair_cap = int(sum(pair_caps_by_benchmark.values()))
    selected_per_class = int(min(max(0, int(requested_per_class)), total_pair_cap))
    if selected_per_class <= 0:
        return {
            "selected_num_per_class": 0,
            "selected_pairs_by_benchmark": {},
            "counts_by_benchmark_before_selection": counts_by_benchmark,
            "represented_benchmarks_with_both_classes": represented,
        }

    selected_pairs_by_benchmark = {str(ds): 0 for ds in represented}
    raw_alloc = {
        str(ds): (
            float(selected_per_class) * float(pair_caps_by_benchmark[ds]) / float(total_pair_cap)
        )
        for ds in represented
    }
    floor_alloc = {str(ds): int(math.floor(raw_alloc[ds])) for ds in represented}
    for ds in represented:
        selected_pairs_by_benchmark[ds] = int(min(pair_caps_by_benchmark[ds], floor_alloc[ds]))
    assigned = int(sum(selected_pairs_by_benchmark.values()))
    leftover = int(selected_per_class - assigned)
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

    selected_per_class = int(sum(selected_pairs_by_benchmark.values()))
    return {
        "selected_num_per_class": selected_per_class,
        "selected_pairs_by_benchmark": {k: int(v) for k, v in selected_pairs_by_benchmark.items()},
        "counts_by_benchmark_before_selection": counts_by_benchmark,
        "represented_benchmarks_with_both_classes": represented,
    }


def _select_holdout_examples_balanced_by_benchmark(
    pool_class0: Sequence[Example],
    pool_class1: Sequence[Example],
    selected_pairs_by_benchmark: Dict[str, int],
    seed: int,
) -> Tuple[List[Example], Dict[str, Dict[str, int]]]:
    rng = random.Random(int(seed))
    class0_by_benchmark: Dict[str, List[Example]] = defaultdict(list)
    class1_by_benchmark: Dict[str, List[Example]] = defaultdict(list)
    for item in pool_class0:
        class0_by_benchmark[str(item.get("dataset", "unknown"))].append(item)
    for item in pool_class1:
        class1_by_benchmark[str(item.get("dataset", "unknown"))].append(item)

    selected: List[Example] = []
    selected_counts_by_benchmark: Dict[str, Dict[str, int]] = {}
    for ds in sorted(selected_pairs_by_benchmark.keys()):
        k = int(selected_pairs_by_benchmark[ds])
        if k <= 0:
            continue
        c0_candidates = list(class0_by_benchmark.get(ds, []))
        c1_candidates = list(class1_by_benchmark.get(ds, []))
        if len(c0_candidates) < k or len(c1_candidates) < k:
            continue
        chosen_c0 = c0_candidates if len(c0_candidates) == k else rng.sample(c0_candidates, k)
        chosen_c1 = c1_candidates if len(c1_candidates) == k else rng.sample(c1_candidates, k)
        selected.extend(chosen_c0)
        selected.extend(chosen_c1)
        selected_counts_by_benchmark[str(ds)] = {
            "class0": int(len(chosen_c0)),
            "class1": int(len(chosen_c1)),
        }
    rng.shuffle(selected)
    return selected, selected_counts_by_benchmark


def benchmark_balanced_subset(
    examples: Sequence[Example],
    seed: int,
    target_per_class: int,
    datasets: Optional[Sequence[str]] = None,
) -> Tuple[List[Example], Dict[str, Any]]:
    dataset_set = set(datasets or SUPPORTED_BENCHMARKS)
    filtered = [ex for ex in examples if str(ex.get("dataset")) in dataset_set]
    pool_class0 = [ex for ex in filtered if int(ex["label"]) == 0]
    pool_class1 = [ex for ex in filtered if int(ex["label"]) == 1]
    plan = _plan_balanced_holdout_selection(
        pool_class0=pool_class0,
        pool_class1=pool_class1,
        requested_per_class=int(target_per_class),
    )
    selected, selected_counts = _select_holdout_examples_balanced_by_benchmark(
        pool_class0=pool_class0,
        pool_class1=pool_class1,
        selected_pairs_by_benchmark=dict(plan.get("selected_pairs_by_benchmark") or {}),
        seed=int(seed),
    )
    meta = dict(plan)
    meta["selected_counts_by_benchmark"] = selected_counts
    meta["selected_total"] = int(len(selected))
    return selected, meta


def balanced_subset(examples: Sequence[Example], seed: int, max_per_class: Optional[int] = None) -> List[Example]:
    rng = random.Random(seed)
    by_label = {0: [ex for ex in examples if int(ex["label"]) == 0], 1: [ex for ex in examples if int(ex["label"]) == 1]}
    n = min(len(by_label[0]), len(by_label[1]))
    if max_per_class is not None:
        n = min(n, int(max_per_class))
    if n <= 0:
        return []
    out = rng.sample(by_label[0], n) + rng.sample(by_label[1], n)
    rng.shuffle(out)
    return out


def allocate_by_capacity(caps: Dict[str, int], target_per_class: int) -> Dict[str, int]:
    caps = {k: int(v) for k, v in caps.items() if int(v) >= 3}
    if not caps:
        return {}
    target = min(int(target_per_class), sum(caps.values()))
    base = {dataset: 3 for dataset in caps}
    remaining = target - (3 * len(caps))
    if remaining <= 0:
        return {dataset: min(3, caps[dataset]) for dataset in caps}
    extra_caps = {dataset: caps[dataset] - 3 for dataset in caps}
    extra_total = sum(extra_caps.values())
    alloc = dict(base)
    if extra_total <= 0:
        return alloc
    raw = {dataset: remaining * extra_caps[dataset] / extra_total for dataset in caps}
    floors = {dataset: int(math.floor(raw[dataset])) for dataset in caps}
    for dataset in caps:
        alloc[dataset] += floors[dataset]
    leftover = remaining - sum(floors.values())
    for dataset in sorted(caps, key=lambda d: raw[d] - floors[d], reverse=True):
        if leftover <= 0:
            break
        if alloc[dataset] < caps[dataset]:
            alloc[dataset] += 1
            leftover -= 1
    return alloc


def stratified_split(examples: Sequence[Example], seed: int, test_fraction: float = 0.2) -> Tuple[List[Example], List[Example]]:
    rng = random.Random(seed)
    train: List[Example] = []
    test: List[Example] = []
    by_label = {0: [], 1: []}
    for ex in examples:
        by_label[int(ex["label"])].append(ex)
    for label_examples in by_label.values():
        label_examples = list(label_examples)
        rng.shuffle(label_examples)
        n_test = max(1, int(round(len(label_examples) * test_fraction)))
        if n_test >= len(label_examples):
            n_test = max(1, len(label_examples) - 1)
        test.extend(label_examples[:n_test])
        train.extend(label_examples[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def group_split(examples: Sequence[Example], seed: int, test_fraction: float = 0.2) -> Tuple[List[Example], List[Example]]:
    rng = random.Random(seed)
    groups = sorted({str(ex["group_id"]) for ex in examples})
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    test_groups = set(groups[:n_test])
    train = [ex for ex in examples if str(ex["group_id"]) not in test_groups]
    test = [ex for ex in examples if str(ex["group_id"]) in test_groups]
    return train, test


def train_nb(train_examples: Sequence[Example]) -> Dict[str, Any]:
    doc_freq = {0: Counter(), 1: Counter()}
    label_counts = Counter(int(ex["label"]) for ex in train_examples)
    vocab = set()
    for ex in train_examples:
        label = int(ex["label"])
        grams = example_ngrams(ex)
        doc_freq[label].update(grams)
        vocab.update(grams)
    vocab_list = sorted(vocab)
    denominators = {
        label: sum(doc_freq[label].values()) + len(vocab_list)
        for label in (0, 1)
    }
    total = len(train_examples)
    priors = {
        label: math.log((label_counts[label] + 1.0) / (total + 2.0))
        for label in (0, 1)
    }
    return {
        "vocab": set(vocab_list),
        "doc_freq": doc_freq,
        "denominators": denominators,
        "priors": priors,
    }


def predict_nb(model: Dict[str, Any], text: str) -> int:
    grams = set(token_ngrams(text)) & model["vocab"]
    scores = {}
    for label in (0, 1):
        score = model["priors"][label]
        denom = model["denominators"][label]
        freq = model["doc_freq"][label]
        for gram in grams:
            score += math.log((freq[gram] + 1.0) / denom)
        scores[label] = score
    return 1 if scores[1] > scores[0] else 0


def predict_nb_example(model: Dict[str, Any], ex: Example) -> int:
    grams = example_ngrams(ex) & model["vocab"]
    scores = {}
    for label in (0, 1):
        score = model["priors"][label]
        denom = model["denominators"][label]
        freq = model["doc_freq"][label]
        for gram in grams:
            score += math.log((freq[gram] + 1.0) / denom)
        scores[label] = score
    return 1 if scores[1] > scores[0] else 0


def evaluate_nb(train_examples: Sequence[Example], test_examples: Sequence[Example]) -> Dict[str, Any]:
    if not train_examples or not test_examples:
        return {"accuracy": None}
    model = train_nb(train_examples)
    correct = 0
    by_label = {0: [0, 0], 1: [0, 0]}
    by_dataset: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for ex in test_examples:
        pred = predict_nb_example(model, ex)
        label = int(ex["label"])
        hit = int(pred == label)
        correct += hit
        by_label[label][0] += hit
        by_label[label][1] += 1
        by_dataset[str(ex["dataset"])][0] += hit
        by_dataset[str(ex["dataset"])][1] += 1
    return {
        "accuracy": correct / len(test_examples),
        "n_train": len(train_examples),
        "n_test": len(test_examples),
        "class0_accuracy": by_label[0][0] / by_label[0][1] if by_label[0][1] else None,
        "class1_accuracy": by_label[1][0] / by_label[1][1] if by_label[1][1] else None,
        "benchmark_accuracy": {
            dataset: vals[0] / vals[1]
            for dataset, vals in sorted(by_dataset.items())
            if vals[1]
        },
    }


def aggregate_runs(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    accs = [float(run["accuracy"]) for run in runs if run.get("accuracy") is not None]
    if not accs:
        return {"accuracy_mean": None, "accuracy_std": None, "runs": list(runs)}
    out: Dict[str, Any] = {
        "accuracy_mean": mean(accs),
        "accuracy_std": pstdev(accs) if len(accs) > 1 else 0.0,
        "runs": list(runs),
    }
    for key in ("class0_accuracy", "class1_accuracy"):
        vals = [float(run[key]) for run in runs if run.get(key) is not None]
        out[f"{key}_mean"] = mean(vals) if vals else None
    datasets = sorted({d for run in runs for d in (run.get("benchmark_accuracy") or {}).keys()})
    out["benchmark_accuracy_mean"] = {
        dataset: mean(float(run["benchmark_accuracy"][dataset]) for run in runs if dataset in (run.get("benchmark_accuracy") or {}))
        for dataset in datasets
    }
    return out


def text_probe_suite(
    examples: Sequence[Example],
    setting: str,
    seed: int,
    num_seeds: int,
    target_examples_per_class: int,
    contrastive_eval_pool: Optional[Sequence[Example]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    eval_pool = list(contrastive_eval_pool or [])
    for dataset in SUPPORTED_BENCHMARKS:
        dataset_examples = [ex for ex in examples if ex["dataset"] == dataset]
        runs = []
        for idx in range(num_seeds):
            run_seed = seed + idx
            if setting == "all_examples":
                selected, _meta = benchmark_balanced_subset(
                    dataset_examples,
                    seed=run_seed,
                    target_per_class=target_examples_per_class,
                    datasets=[dataset],
                )
                train, test = stratified_split(selected, seed=run_seed)
            else:
                train = list(dataset_examples)
                test_pool = [ex for ex in eval_pool if ex["dataset"] == dataset]
                test, _meta = benchmark_balanced_subset(
                    test_pool,
                    seed=run_seed,
                    target_per_class=target_examples_per_class,
                    datasets=[dataset],
                )
            runs.append(evaluate_nb(train, test))
        out[f"within_{dataset}"] = aggregate_runs(runs)

    if setting == "all_examples":
        all_runs = []
        for idx in range(num_seeds):
            run_seed = seed + idx
            selected, meta = benchmark_balanced_subset(
                examples,
                seed=run_seed,
                target_per_class=target_examples_per_class,
            )
            train, test = stratified_split(selected, seed=run_seed)
            run = evaluate_nb(train, test)
            run["selection_meta"] = meta
            all_runs.append(run)
        out["all_benchmarks_mixed"] = aggregate_runs(all_runs)
        out["cross_benchmark_transfer"] = cross_benchmark_transfer(
            examples,
            seed,
            num_seeds,
            target_examples_per_class,
            setting=setting,
            contrastive_eval_pool=contrastive_eval_pool,
        )
    else:
        all_runs = []
        for idx in range(num_seeds):
            run_seed = seed + idx
            train = list(examples)
            test, meta = benchmark_balanced_subset(
                eval_pool,
                seed=run_seed,
                target_per_class=target_examples_per_class,
            )
            run = evaluate_nb(train, test)
            run["selection_meta"] = meta
            all_runs.append(run)
        out["all_benchmarks_mixed"] = aggregate_runs(all_runs)
        out["cross_benchmark_transfer"] = cross_benchmark_transfer(
            examples,
            seed,
            num_seeds,
            target_examples_per_class,
            setting=setting,
            contrastive_eval_pool=contrastive_eval_pool,
        )
    return out


def cross_benchmark_transfer(
    examples: Sequence[Example],
    seed: int,
    num_seeds: int,
    target_examples_per_class: int,
    setting: str,
    contrastive_eval_pool: Optional[Sequence[Example]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    eval_pool = list(contrastive_eval_pool or [])
    for train_dataset in SUPPORTED_BENCHMARKS:
        for test_dataset in SUPPORTED_BENCHMARKS:
            if train_dataset == test_dataset:
                continue
            train_datasets = (train_dataset,)
            name = f"train_{train_dataset}_test_{test_dataset}"
            runs = []
            train_pool = [ex for ex in examples if ex["dataset"] in set(train_datasets)]
            test_pool = [ex for ex in examples if ex["dataset"] == test_dataset]
            for idx in range(num_seeds):
                run_seed = seed + idx
                if setting == "all_examples":
                    train_selected, _ = benchmark_balanced_subset(
                        train_pool,
                        seed=run_seed,
                        target_per_class=target_examples_per_class,
                        datasets=train_datasets,
                    )
                    test_selected, _ = benchmark_balanced_subset(
                        test_pool,
                        seed=run_seed,
                        target_per_class=target_examples_per_class,
                        datasets=[test_dataset],
                    )
                    test_balanced = list(test_selected)
                else:
                    train_selected = list(train_pool)
                    eval_test_pool = [ex for ex in eval_pool if ex["dataset"] == test_dataset]
                    test_balanced, _ = benchmark_balanced_subset(
                        eval_test_pool,
                        seed=run_seed,
                        target_per_class=target_examples_per_class,
                        datasets=[test_dataset],
                    )
                runs.append(evaluate_nb(train_selected, test_balanced))
            out[name] = aggregate_runs(runs)
    return out


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.root)
    def _first_existing(candidates: Sequence[Path]) -> Path:
        for p in candidates:
            if Path(p).exists():
                return Path(p)
        return Path(candidates[0])

    data_paths = {
        "ovis": {
            "all_examples": _first_existing([
                root / "data/final_data/ovis_all_responses.json",
                root / "tmp_artifacts/responses.json",
            ]),
            "contrastive": _first_existing([
                root / "data/final_data/ovis_contrastive.json",
                root / "tmp_artifacts/contrastive_conversation_pairs.json",
            ]),
        },
        "qwen3_vl_32b_instruct": {
            "all_examples": _first_existing([
                root / "data/final_data/qwen_all_responses.json",
                root / "tmp_artifacts/gen_qwen3_vl_32b_instruct_/responses.json",
                root / "tmp_artifacts/responses_qwen3_vl_32b_instruct.json",
            ]),
            "contrastive": _first_existing([
                root / "data/final_data/qwen_contrastive.json",
                root / "tmp_artifacts/qwen3_vl_32b_instruct/contrastive_conversation_pairs.json",
                root / "tmp_artifacts/contrastive_conversation_pairs_qwen3_vl_32b_instruct.json",
            ]),
        },
    }

    report: Dict[str, Any] = {
        "config": {
            "root": str(root),
            "seed": int(args.seed),
            "num_seeds": int(args.num_seeds),
            "target_examples_per_class": int(args.target_examples_per_class),
            "supported_benchmarks": list(SUPPORTED_BENCHMARKS),
            "settings": str(args.settings),
            "note": "All-examples text is with_image.response only; contrastive text is assistant response in with-image conversations.",
        },
        "models": {},
    }

    selected_settings = ("all_examples", "contrastive") if args.settings == "both" else (str(args.settings),)
    for model_name, paths in data_paths.items():
        model_report: Dict[str, Any] = {}
        loaders = {
            "all_examples": load_all_examples,
            "contrastive": load_contrastive_pairs,
        }
        for setting in selected_settings:
            loader = loaders[setting]
            path = paths[setting]
            examples = loader(path, model_name)
            contrastive_eval_pool: Optional[List[Example]] = None
            overlap_filter_stats: Optional[Dict[str, int]] = None
            if setting == "contrastive":
                raw_eval_pool = load_all_examples(paths["all_examples"], model_name)
                contrastive_eval_pool, overlap_filter_stats = filter_eval_pool_remove_train_overlap(
                    train_examples=examples,
                    eval_pool=raw_eval_pool,
                )
            setting_report: Dict[str, Any] = {
                "num_examples": len(examples),
                "counts_by_benchmark": {
                    dataset: class_counts([ex for ex in examples if ex["dataset"] == dataset])
                    for dataset in SUPPORTED_BENCHMARKS
                },
                "text_summary_by_benchmark": {
                    dataset: summarize_text([ex for ex in examples if ex["dataset"] == dataset], top_k=int(args.top_k))
                    for dataset in SUPPORTED_BENCHMARKS
                },
                "text_probe_results": text_probe_suite(
                    examples,
                    setting=setting,
                    seed=int(args.seed),
                    num_seeds=int(args.num_seeds),
                    target_examples_per_class=int(args.target_examples_per_class),
                    contrastive_eval_pool=contrastive_eval_pool,
                ),
            }
            if overlap_filter_stats is not None:
                setting_report["contrastive_eval_overlap_filter"] = overlap_filter_stats
            setting_report["overall_text_summary"] = summarize_text(examples, top_k=int(args.top_k))
            model_report[setting] = setting_report
            del examples
            gc.collect()
        report["models"][model_name] = model_report
    return report


def print_compact_summary(report: Dict[str, Any]) -> None:
    print("Text confound analysis")
    print(report["config"]["note"])
    for model_name, model_report in report["models"].items():
        print(f"\nMODEL {model_name}")
        for setting in ("all_examples", "contrastive"):
            if setting not in model_report:
                continue
            setting_report = model_report[setting]
            print(f"  {setting}: n={setting_report['num_examples']}")
            for dataset in SUPPORTED_BENCHMARKS:
                counts = setting_report["counts_by_benchmark"][dataset]
                probe = setting_report["text_probe_results"].get(f"within_{dataset}", {})
                acc = probe.get("accuracy_mean")
                acc_s = "NA" if acc is None else f"{acc:.3f}"
                print(f"    {dataset}: {counts}, text_probe_acc={acc_s}")
                skews = setting_report["text_summary_by_benchmark"][dataset]["largest_phrase_skews"][:5]
                skew_text = "; ".join(
                    f"{row['phrase']}({row['diff_class1_minus_class0']:+.2f})"
                    for row in skews
                )
                print(f"      phrase_skews: {skew_text}")
            mixed = setting_report["text_probe_results"].get("all_benchmarks_mixed", {})
            mixed_acc = mixed.get("accuracy_mean")
            if mixed_acc is not None:
                print(f"    all_benchmarks_mixed text_probe_acc={mixed_acc:.3f}")
                by_benchmark = mixed.get("benchmark_accuracy_mean", {})
                if by_benchmark:
                    print(f"      mixed_test_by_benchmark={format_float_dict(by_benchmark)}")
            transfer = setting_report["text_probe_results"].get("cross_benchmark_transfer")
            if transfer:
                print("    cross_benchmark_transfer:")
                for name, result in transfer.items():
                    acc = result.get("accuracy_mean")
                    print(f"      {name}: {'NA' if acc is None else f'{acc:.3f}'}")


def format_float_dict(values: Dict[str, float]) -> str:
    return "{" + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(values.items())) + "}"


def main() -> None:
    args = parse_args()
    report = build_report(args)
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = Path(args.root) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print_compact_summary(report)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
