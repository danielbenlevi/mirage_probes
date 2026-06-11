#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SUMMARY_MODES: Tuple[str, ...] = ("vqa_rad", "mmmu_pro", "medxpertqa_mm", "all")
PRIMARY_BENCHMARK_ORDER: Tuple[str, ...] = ("vqa_rad", "mmmu_pro", "medxpertqa_mm")

DEFAULT_RESIDUAL_MD = Path("./results/results_summary/results_summary.md")
DEFAULT_MLP_MD = Path("./results/results_summary/results_summary_mlp.md")
DEFAULT_ATT_MD = Path("./results/results_summary/results_summary_att.md")

TABLE_OUTPUT_DEFAULTS: Dict[str, Path] = {
    "residual": DEFAULT_RESIDUAL_MD,
    "mlp": DEFAULT_MLP_MD,
    "attention": DEFAULT_ATT_MD,
}

TABLE_FAMILY_KEYS: Dict[str, str] = {
    "residual": "residual",
    "mlp": "mlp",
    "attention": "attention_all_max",
}

TABLE_PHASE_LABELS: Dict[str, str] = {
    "residual": "Residual",
    "mlp": "MLP",
    "attention": "Attention",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one summary markdown table at a time from a matched pair of OVIS/QWEN run directories."
        )
    )
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        choices=tuple(TABLE_OUTPUT_DEFAULTS.keys()),
        help="Which table to generate.",
    )
    parser.add_argument(
        "--ovis_run",
        type=Path,
        required=True,
        help="Run root for OVIS for the selected table target.",
    )
    parser.add_argument(
        "--qwen_run",
        type=Path,
        required=True,
        help="Run root for QWEN for the selected table target.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=None,
        help="Optional output markdown path. Defaults to the standard location for the selected table.",
    )
    return parser.parse_args()


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float_dict(value) -> Optional[Dict[str, float]]:
    if not isinstance(value, dict):
        return None
    out: Dict[str, float] = {}
    for key, val in value.items():
        f = _safe_float(val)
        if f is None:
            continue
        out[str(key)] = float(f)
    return out


def _first_present_dict(d: Dict, keys: Iterable[str]) -> Optional[Dict[str, float]]:
    for key in keys:
        if key in d:
            value = _safe_float_dict(d.get(key))
            if value is not None:
                return value
    return None


def _rows_from_all_feature(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("feature_probe_accuracies"), list):
        return [r for r in payload["feature_probe_accuracies"] if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _mean_dict_metrics_from_seed_runs(seed_runs, key: str) -> Optional[Dict[str, float]]:
    if not isinstance(seed_runs, list) or not seed_runs:
        return None
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for run in seed_runs:
        if not isinstance(run, dict):
            continue
        d = run.get(key)
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            fv = _safe_float(v)
            if fv is None:
                continue
            kk = str(k)
            sums[kk] = sums.get(kk, 0.0) + float(fv)
            counts[kk] = counts.get(kk, 0) + 1
    if not counts:
        return None
    return {k: sums[k] / counts[k] for k in counts}


def _mean_scalar_metrics_from_seed_runs(seed_runs, key: str) -> Optional[float]:
    if not isinstance(seed_runs, list) or not seed_runs:
        return None
    vals: List[float] = []
    for run in seed_runs:
        if not isinstance(run, dict):
            continue
        fv = _safe_float(run.get(key))
        if fv is None:
            continue
        vals.append(float(fv))
    if not vals:
        return None
    return float(mean(vals))


def _row_seed_runs(row: Dict) -> List[Dict]:
    for key in ("heldout_seed_runs", "seed_runs"):
        v = row.get(key)
        if isinstance(v, list) and v:
            return [x for x in v if isinstance(x, dict)]
    return []


def _row_has_heldout_like_evidence(row: Dict) -> bool:
    seed_runs = _row_seed_runs(row)
    if not seed_runs:
        return False
    for sr in seed_runs:
        if _safe_float(sr.get("test_accuracy")) is not None:
            return True
        if _safe_float(sr.get("test_accuracy_at_best_c")) is not None:
            return True
    return False


def _heldout_metrics_from_seed_runs(heldout_seed_runs) -> Optional[Dict]:
    if not isinstance(heldout_seed_runs, list) or not heldout_seed_runs:
        return None
    test_scores: List[float] = []
    for run in heldout_seed_runs:
        if not isinstance(run, dict):
            continue
        t = _safe_float(run.get("test_accuracy"))
        if t is not None:
            test_scores.append(float(t))
    benchmark_test = _mean_dict_metrics_from_seed_runs(heldout_seed_runs, key="benchmark_test_accuracy")
    if not test_scores:
        return None
    return {
        "test_accuracy": float(sum(test_scores) / len(test_scores)),
        "benchmark_test_accuracy": benchmark_test,
    }


def _load_llm_holdout_feature_metrics(path: Path) -> Dict[str, Dict]:
    parent = path.parent
    candidates = sorted(parent.glob("*llm_residual_layer_heldout_eval*.json"))
    if not candidates:
        return {}
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    out: Dict[str, Dict] = {}
    for _layer_name, info in payload.items():
        if not isinstance(info, dict):
            continue
        feature = str(info.get("best_feature") or "")
        if not feature:
            continue
        test_acc = _safe_float(info.get("test_accuracy_mean"))
        if test_acc is None:
            continue
        benchmark_test = _mean_dict_metrics_from_seed_runs(info.get("seed_runs"), key="benchmark_test_accuracy")
        out[feature] = {
            "test_accuracy": float(test_acc),
            "benchmark_test_accuracy": benchmark_test,
        }
    return out


def _pick_test_accuracy(
    row: Dict,
    heldout_by_feature: Dict[str, Dict],
    require_heldout: bool,
) -> Optional[float]:
    feature = str(row.get("feature") or "")
    if feature and feature in heldout_by_feature:
        return _safe_float(heldout_by_feature[feature].get("test_accuracy"))

    if require_heldout and not _row_has_heldout_like_evidence(row):
        return None

    for key in (
        "mean_test_accuracy_at_best_c",
        "test_accuracy_at_best_c",
        "best_test_accuracy",
        "test_accuracy",
    ):
        if key in row:
            val = _safe_float(row.get(key))
            if val is not None:
                return val

    seed_runs = _row_seed_runs(row)
    for seed_key in ("test_accuracy_at_best_c", "test_accuracy"):
        v = _mean_scalar_metrics_from_seed_runs(seed_runs, key=seed_key)
        if v is not None:
            return v

    for key in (
        "mean_validation_accuracy_at_best_c",
        "validation_accuracy_at_best_c",
        "best_validation_accuracy",
        "validation_accuracy",
    ):
        if key in row:
            val = _safe_float(row.get(key))
            if val is not None:
                return val
    return None


def _pick_benchmark_test_accuracy(row: Dict) -> Optional[Dict[str, float]]:
    direct = _first_present_dict(
        row,
        (
            "mean_benchmark_test_accuracy_at_best_c",
            "benchmark_test_accuracy_at_best_c",
            "benchmark_test_accuracy",
        ),
    )
    if direct is not None:
        return direct

    seed_runs = _row_seed_runs(row)
    for key in ("benchmark_test_accuracy_at_best_c", "benchmark_test_accuracy"):
        v = _mean_dict_metrics_from_seed_runs(seed_runs, key=key)
        if v is not None:
            return v
    return None


def _pick_heldout_benchmark_test_accuracy(
    row: Dict,
    heldout_by_feature: Dict[str, Dict],
) -> Optional[Dict[str, float]]:
    feature = str(row.get("feature") or "")
    if feature and feature in heldout_by_feature:
        v = heldout_by_feature[feature].get("benchmark_test_accuracy")
        if isinstance(v, dict):
            return {str(k): float(val) for k, val in v.items() if _safe_float(val) is not None}
    return None


def _parse_feature(feature: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    m = re.fullmatch(r"language_model/all_layers_concat__(.+)", feature)
    if m:
        return "residual", 0, m.group(1)

    m = re.fullmatch(r"language_model/all_layers_concat_mlp__(.+)", feature)
    if m:
        return "mlp", 0, m.group(1)

    m = re.fullmatch(r"language_model/all_layers_concat_post_attention__(.+)", feature)
    if m:
        return "post_attention", 0, m.group(1)

    m = re.fullmatch(r"language_model/all_layers_concat_attention_head_(\d+)__(.+)", feature)
    if m:
        head = int(m.group(1))
        suffix = m.group(2)
        return "attention_head", head, f"attention_head_{head}__{suffix}"

    m = re.fullmatch(r"language_model/layer_(\d+)__(.+)", feature)
    if m:
        return "residual", int(m.group(1)), m.group(2)

    m = re.fullmatch(r"language_model/layer_(\d+)/post_attention__(.+)", feature)
    if m:
        return "post_attention", int(m.group(1)), m.group(2)

    m = re.fullmatch(r"language_model/layer_(\d+)/mlp__(.+)", feature)
    if m:
        return "mlp", int(m.group(1)), m.group(2)

    m = re.fullmatch(r"language_model/layer_(\d+)/attention_head_(\d+)__(.+)", feature)
    if m:
        layer = int(m.group(1))
        head = int(m.group(2))
        suffix = m.group(3)
        return "attention_head", layer, f"attention_head_{head}__{suffix}"

    return None, None, None


def _infer_stage_mode(path: Path) -> Tuple[Optional[str], Optional[str]]:
    stage_key: Optional[str] = None
    mode: Optional[str] = None
    parts = list(path.parts)

    for part in parts:
        for s in (
            "logreg_contrastive",
            "logreg_all_examples",
            "mlp_contrastive",
            "mlp_all_examples",
            "concat_contrastive",
            "concat_all_examples",
            "diff_contrastive",
            "diff_all_examples",
        ):
            if s in part:
                stage_key = s
                break
        if stage_key is not None:
            break

    for part in parts:
        lowered = part.lower()
        for m in SUMMARY_MODES:
            if m == "all" and "all_examples" in lowered:
                continue
            if re.search(rf"(^|_){re.escape(m)}($|_)", part):
                mode = m
                break
        if mode is not None:
            break

    return stage_key, mode


def _candidate_families(parsed_family: Optional[str]) -> List[str]:
    if parsed_family is None:
        return []
    if parsed_family == "residual":
        return ["residual"]
    if parsed_family == "mlp":
        return ["mlp"]
    if parsed_family == "post_attention":
        return ["post_attention", "attention_all_max"]
    if parsed_family == "attention_head":
        return ["attention_head_max", "attention_all_max"]
    return []


def _collect_best_by_family(run_root: Path) -> Dict[Tuple[str, str, str], Dict]:
    best: Dict[Tuple[str, str, str], Dict] = {}
    files = sorted(run_root.rglob("*all_feature_probe_accuracies*.json"))

    for fp in files:
        stage_key, mode = _infer_stage_mode(fp)
        if stage_key is None or mode is None:
            continue

        rows = _rows_from_all_feature(fp)
        heldout = _load_llm_holdout_feature_metrics(fp)
        require_heldout = "contrastive" in stage_key

        for row in rows:
            feature = str(row.get("feature") or "")
            fam, _layer, _feature_name = _parse_feature(feature)
            candidates = _candidate_families(fam)
            if not candidates:
                continue

            test_acc = _pick_test_accuracy(row, heldout, require_heldout=require_heldout)
            benchmark_dict = _pick_heldout_benchmark_test_accuracy(row, heldout)
            if require_heldout:
                heldout_metrics = _heldout_metrics_from_seed_runs(row.get("heldout_seed_runs"))
                if heldout_metrics is None:
                    heldout_metrics = heldout.get(feature)
                if heldout_metrics is not None and heldout_metrics.get("test_accuracy") is not None:
                    test_acc = _safe_float(heldout_metrics.get("test_accuracy"))
                    benchmark_dict = heldout_metrics.get("benchmark_test_accuracy")
            if test_acc is None:
                continue
            if benchmark_dict is None and not require_heldout:
                benchmark_dict = _pick_benchmark_test_accuracy(row)

            for family_key in candidates:
                k = (stage_key, mode, family_key)
                existing = best.get(k)
                if existing is None or float(test_acc) > float(existing["test_accuracy"]):
                    best[k] = {
                        "test_accuracy": float(test_acc),
                        "feature": feature,
                        "benchmark_test_accuracy": benchmark_dict,
                        "source_path": str(fp),
                    }

    return best


def _pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{100.0 * float(x):.2f}%"


def _format_cell(metric: Optional[Dict], mode: str) -> str:
    if metric is None:
        return "N/A"
    base = _pct(metric.get("test_accuracy"))
    if mode != "all":
        return base

    bd = metric.get("benchmark_test_accuracy")
    if not isinstance(bd, dict):
        return base

    vals = []
    for ds in PRIMARY_BENCHMARK_ORDER:
        vals.append(_pct(bd.get(ds)))
    return f"{base} ({', '.join(vals)})"


def _build_section(
    model_label: str,
    phase_label: str,
    family_key: str,
    primary_best: Dict[Tuple[str, str, str], Dict],
) -> List[str]:
    phase = "contrastive" if "Contrastive" in phase_label else "all_examples"

    rows = [
        ("LogReg", "logreg", False),
        ("MLP", "mlp", False),
        ("Concat", "concat", False),
        ("Diff", "diff", False),
    ]

    lines = [
        f"### {model_label} - {phase_label}",
        "",
        "| Probing Strategy | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` | `all` |",
        "|---|---:|---:|---:|---:|",
    ]

    for row_label, base_stage, is_concat_mlp in rows:
        stage_key = f"{base_stage}_{phase}"
        cells: List[str] = []
        for mode in SUMMARY_MODES:
            metric: Optional[Dict] = None if is_concat_mlp else primary_best.get((stage_key, mode, family_key))
            cells.append(_format_cell(metric, mode))

        lines.append("| " + row_label + " | " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def _write_summary_table(
    out_path: Path,
    family_key: str,
    ovis_primary_best: Dict[Tuple[str, str, str], Dict],
    qwen_primary_best: Dict[Tuple[str, str, str], Dict],
    phase_label: str,
) -> None:
    lines: List[str] = []
    lines.extend(_build_section("OVIS", "Contrastive", family_key, ovis_primary_best))
    lines.extend(_build_section("OVIS", "All Examples", family_key, ovis_primary_best))
    lines.extend(_build_section("QWEN", "Contrastive", family_key, qwen_primary_best))
    lines.extend(_build_section("QWEN", "All Examples", family_key, qwen_primary_best))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _require_existing_dir(path: Path, flag_name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{flag_name} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{flag_name} is not a directory: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    table = str(args.table)
    ovis_run = _require_existing_dir(args.ovis_run, "--ovis_run")
    qwen_run = _require_existing_dir(args.qwen_run, "--qwen_run")
    output_md = (
        args.output_md.expanduser().resolve()
        if args.output_md is not None
        else TABLE_OUTPUT_DEFAULTS[table].resolve()
    )

    ovis_best = _collect_best_by_family(ovis_run)
    qwen_best = _collect_best_by_family(qwen_run)

    _write_summary_table(
        out_path=output_md,
        family_key=TABLE_FAMILY_KEYS[table],
        ovis_primary_best=ovis_best,
        qwen_primary_best=qwen_best,
        phase_label=TABLE_PHASE_LABELS[table],
    )

    print(json.dumps({
        "table": table,
        "output_md": str(output_md),
        "ovis_run": str(ovis_run),
        "qwen_run": str(qwen_run),
    }, indent=2))


if __name__ == "__main__":
    main()
