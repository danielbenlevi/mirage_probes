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

DEFAULT_RESIDUAL_OVIS_RUN = Path("./results/results_final/final_ovis_residual_020533")
DEFAULT_RESIDUAL_QWEN_RUN = Path("./results/results_final/final_qwen_residual_192228")
DEFAULT_ADDITIONAL_OVIS_RUN = Path("./results/results_final/final_ovis_additional_023840")
DEFAULT_ADDITIONAL_QWEN_RUN = Path("./results/results_final/final_qwen_additional_194220")
DEFAULT_EXTRA_OVIS_RUN = Path("./results/results_final/final_ovis_extra_165134")
DEFAULT_EXTRA_QWEN_RUN = Path("./results/results_final/final_qwen_extra_020208")

DEFAULT_RESIDUAL_MD = Path("./results/results_summary/results_summary.md")
DEFAULT_MLP_MD = Path("./results/results_summary/results_summary_mlp.md")
DEFAULT_ATT_MD = Path("./results/results_summary/results_summary_att.md")

# Note: this script is configured to generate summary tables under the assumption that
# contrastive logistic regression results are stored in "extra" run directories.
# It should only be used to reproduce results from the original paper.
# If you are running your own experiments with "run_full_probe_experiment.py",
# please use "analyze_tables_one.py" to generate your tables.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all three summary markdown tables (residual/mlp/attention) with unified source policy."
        )
    )
    parser.add_argument("--residual_ovis_run", type=Path, default=DEFAULT_RESIDUAL_OVIS_RUN)
    parser.add_argument("--residual_qwen_run", type=Path, default=DEFAULT_RESIDUAL_QWEN_RUN)
    parser.add_argument("--additional_ovis_run", type=Path, default=DEFAULT_ADDITIONAL_OVIS_RUN)
    parser.add_argument("--additional_qwen_run", type=Path, default=DEFAULT_ADDITIONAL_QWEN_RUN)
    parser.add_argument(
        "--extra_ovis_run",
        type=str,
        default=str(DEFAULT_EXTRA_OVIS_RUN),
        help="Optional path to Ovis extra run. If omitted/unavailable, tries latest results/results_final/final_ovis_extra_*.",
    )
    parser.add_argument(
        "--extra_qwen_run",
        type=str,
        default=str(DEFAULT_EXTRA_QWEN_RUN),
        help="Optional path to Qwen extra run. If omitted/unavailable, tries latest results/results_final/final_qwen_extra_*.",
    )
    parser.add_argument("--residual_md", type=Path, default=DEFAULT_RESIDUAL_MD)
    parser.add_argument("--mlp_md", type=Path, default=DEFAULT_MLP_MD)
    parser.add_argument("--attention_md", type=Path, default=DEFAULT_ATT_MD)
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


def _discover_heldout_override(path: Path) -> Dict[str, Dict]:
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
        feat = str(info.get("best_feature") or "")
        if not feat:
            continue
        v = _safe_float(info.get("test_accuracy_mean"))
        if v is None:
            continue
        benchmark = _mean_dict_metrics_from_seed_runs(info.get("seed_runs"), key="benchmark_test_accuracy")
        out[feat] = {
            "test_accuracy": float(v),
            "benchmark_test_accuracy": benchmark,
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


def _infer_extra_phase(path: Path) -> Optional[str]:
    s = str(path)
    if "phase_residual_standard" in s:
        return "residual"
    if "phase_additional_targets_text_nonspecial_mean" in s:
        return "additional"
    return None


def _resolve_extra_run(path_hint: str, model: str) -> Optional[Path]:
    raw = str(path_hint or "").strip()
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.exists():
            return p

    root = Path("./results/results_final")
    candidates = sorted(root.glob(f"final_{model}_extra_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


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


def _collect_best_by_family(run_root: Path, phase_filter: Optional[str]) -> Dict[Tuple[str, str, str], Dict]:
    best: Dict[Tuple[str, str, str], Dict] = {}
    files = sorted(run_root.rglob("*all_feature_probe_accuracies*.json"))

    for fp in files:
        stage_key, mode = _infer_stage_mode(fp)
        if stage_key is None or mode is None:
            continue

        if phase_filter is not None:
            phase = _infer_extra_phase(fp)
            if phase != phase_filter:
                continue

        rows = _rows_from_all_feature(fp)
        heldout = _discover_heldout_override(fp)
        require_heldout = "contrastive" in stage_key

        for row in rows:
            feature = str(row.get("feature") or "")
            fam, _layer, _feature_name = _parse_feature(feature)
            candidates = _candidate_families(fam)
            if not candidates:
                continue

            test_acc = _pick_test_accuracy(row, heldout, require_heldout=require_heldout)
            if test_acc is None:
                continue

            benchmark_dict = _pick_heldout_benchmark_test_accuracy(row, heldout)
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
    extra_best: Optional[Dict[Tuple[str, str, str], Dict]],
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
            metric: Optional[Dict] = None

            if is_concat_mlp:
                if extra_best is not None:
                    metric = extra_best.get((stage_key, mode, family_key))
            else:
                if (
                    base_stage == "logreg"
                    and phase == "contrastive"
                    and extra_best is not None
                    and extra_best.get((stage_key, mode, family_key)) is not None
                ):
                    metric = extra_best.get((stage_key, mode, family_key))
                else:
                    metric = primary_best.get((stage_key, mode, family_key))

            cells.append(_format_cell(metric, mode))

        lines.append("| " + row_label + " | " + " | ".join(cells) + " |")

    lines.append("")
    return lines


def _write_summary_table(
    out_path: Path,
    family_key: str,
    ovis_primary_best: Dict[Tuple[str, str, str], Dict],
    qwen_primary_best: Dict[Tuple[str, str, str], Dict],
    ovis_extra_best: Optional[Dict[Tuple[str, str, str], Dict]],
    qwen_extra_best: Optional[Dict[Tuple[str, str, str], Dict]],
) -> None:
    lines: List[str] = []
    lines.extend(_build_section("OVIS", "Contrastive", family_key, ovis_primary_best, ovis_extra_best))
    lines.extend(_build_section("OVIS", "All Examples", family_key, ovis_primary_best, ovis_extra_best))
    lines.extend(_build_section("QWEN", "Contrastive", family_key, qwen_primary_best, qwen_extra_best))
    lines.extend(_build_section("QWEN", "All Examples", family_key, qwen_primary_best, qwen_extra_best))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    ovis_extra_run = _resolve_extra_run(args.extra_ovis_run, model="ovis")
    qwen_extra_run = _resolve_extra_run(args.extra_qwen_run, model="qwen")

    residual_ovis_best = _collect_best_by_family(args.residual_ovis_run, phase_filter=None)
    residual_qwen_best = _collect_best_by_family(args.residual_qwen_run, phase_filter=None)
    additional_ovis_best = _collect_best_by_family(args.additional_ovis_run, phase_filter=None)
    additional_qwen_best = _collect_best_by_family(args.additional_qwen_run, phase_filter=None)

    residual_ovis_extra_best = (
        _collect_best_by_family(ovis_extra_run, phase_filter="residual") if ovis_extra_run is not None else None
    )
    additional_ovis_extra_best = (
        _collect_best_by_family(ovis_extra_run, phase_filter="additional") if ovis_extra_run is not None else None
    )
    residual_qwen_extra_best = (
        _collect_best_by_family(qwen_extra_run, phase_filter="residual") if qwen_extra_run is not None else None
    )
    additional_qwen_extra_best = (
        _collect_best_by_family(qwen_extra_run, phase_filter="additional") if qwen_extra_run is not None else None
    )

    _write_summary_table(
        out_path=args.residual_md,
        family_key="residual",
        ovis_primary_best=residual_ovis_best,
        qwen_primary_best=residual_qwen_best,
        ovis_extra_best=residual_ovis_extra_best,
        qwen_extra_best=residual_qwen_extra_best,
    )
    _write_summary_table(
        out_path=args.mlp_md,
        family_key="mlp",
        ovis_primary_best=additional_ovis_best,
        qwen_primary_best=additional_qwen_best,
        ovis_extra_best=additional_ovis_extra_best,
        qwen_extra_best=additional_qwen_extra_best,
    )
    _write_summary_table(
        out_path=args.attention_md,
        family_key="attention_all_max",
        ovis_primary_best=additional_ovis_best,
        qwen_primary_best=additional_qwen_best,
        ovis_extra_best=additional_ovis_extra_best,
        qwen_extra_best=additional_qwen_extra_best,
    )

    print(json.dumps({
        "residual_md": str(args.residual_md),
        "mlp_md": str(args.mlp_md),
        "attention_md": str(args.attention_md),
        "extra_ovis_run": (str(ovis_extra_run) if ovis_extra_run is not None else None),
        "extra_qwen_run": (str(qwen_extra_run) if qwen_extra_run is not None else None),
    }, indent=2))


if __name__ == "__main__":
    main()
