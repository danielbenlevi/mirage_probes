#!/usr/bin/env python3
"""Generate markdown tables from analyze_confounds.py reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


BENCHMARKS: Tuple[str, ...] = ("vqa_rad", "mmmu_pro", "medxpertqa_mm")
CONFIG_ROWS: Tuple[Tuple[str, str], ...] = (
    ("ovis", "contrastive"),
    ("ovis", "all_examples"),
    ("qwen3_vl_32b_instruct", "contrastive"),
    ("qwen3_vl_32b_instruct", "all_examples"),
)

DEFAULT_REPORTS: Tuple[Path, ...] = (
    Path("./tmp_artifacts/confound_analysis.json"),
    Path("./tmp_artifacts/confound_analysis_contrastive.json"),
)
DEFAULT_OUTPUT_MD = Path("./results/results_summary/confounds_tables.md")

TRANSFER_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("vqa_rad", "mmmu_pro"),
    ("vqa_rad", "medxpertqa_mm"),
    ("mmmu_pro", "vqa_rad"),
    ("mmmu_pro", "medxpertqa_mm"),
    ("medxpertqa_mm", "vqa_rad"),
    ("medxpertqa_mm", "mmmu_pro"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create markdown summary tables for text confound analysis.")
    parser.add_argument("--reports", nargs="+", type=Path, default=list(DEFAULT_REPORTS))
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--top_phrase_groups",
        type=int,
        default=6,
        help="How many phrase groups to include in each separator table.",
    )
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_pct(value: Any) -> str:
    val = _safe_float(value)
    return "NA" if val is None else f"{100.0 * val:.2f}%"


def model_title(model_name: str) -> str:
    return {
        "ovis": "OVIS",
        "qwen3_vl_32b_instruct": "QWEN3-VL-32B-Instruct",
    }.get(model_name, model_name)


def setting_title(setting: str) -> str:
    return {
        "all_examples": "All Examples",
        "contrastive": "Contrastive",
    }.get(setting, setting)


def config_title(model_name: str, setting: str) -> str:
    return f"{model_title(model_name)} {setting_title(setting)}"


def merge_reports(paths: Sequence[Path]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"models": {}}
    found = 0
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for model_name, model_report in (payload.get("models") or {}).items():
            merged_model = merged["models"].setdefault(model_name, {})
            for setting, setting_report in (model_report or {}).items():
                merged_model[setting] = setting_report
        found += 1
    if found == 0:
        joined = ", ".join(str(p) for p in paths)
        raise FileNotFoundError(f"No report files found. Checked: {joined}")
    return merged


def _setting_report(report: Dict[str, Any], model_name: str, setting: str) -> Dict[str, Any]:
    return ((report.get("models") or {}).get(model_name) or {}).get(setting) or {}


def _fmt_all_with_breakdown(mixed: Dict[str, Any]) -> str:
    overall = fmt_pct(mixed.get("accuracy_mean"))
    by_benchmark = mixed.get("benchmark_accuracy_mean") or {}
    parts = [fmt_pct(by_benchmark.get(benchmark)) for benchmark in BENCHMARKS]
    return f"{overall} ({parts[0]}, {parts[1]}, {parts[2]})"


def build_nb_summary_table(report: Dict[str, Any]) -> str:
    lines = [
        "| Model / Setting | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` | `all` |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, setting in CONFIG_ROWS:
        setting_report = _setting_report(report, model_name, setting)
        probe = setting_report.get("text_probe_results") or {}
        within = {
            benchmark: _safe_float(((probe.get(f"within_{benchmark}") or {}).get("accuracy_mean")))
            for benchmark in BENCHMARKS
        }
        mixed = probe.get("all_benchmarks_mixed") or {}
        lines.append(
            f"| {config_title(model_name, setting)} | {fmt_pct(within['vqa_rad'])} | "
            f"{fmt_pct(within['mmmu_pro'])} | {fmt_pct(within['medxpertqa_mm'])} | "
            f"{_fmt_all_with_breakdown(mixed)} |"
        )
    return "\n".join(lines)


def _transfer_key(train_benchmark: str, test_benchmark: str) -> str:
    return f"train_{train_benchmark}_test_{test_benchmark}"


def _short_benchmark_name(name: str) -> str:
    return {
        "vqa_rad": "vqa",
        "mmmu_pro": "mmmu",
        "medxpertqa_mm": "medx",
    }.get(name, name)


def build_transfer_summary_table(report: Dict[str, Any]) -> str:
    headers = [f"`{_short_benchmark_name(train)}→{_short_benchmark_name(test)}`" for train, test in TRANSFER_PAIRS]
    lines = [
        "| Model / Setting | " + " | ".join(headers) + " |",
        "|---|" + "|".join(["---:"] * len(headers)) + "|",
    ]
    for model_name, setting in CONFIG_ROWS:
        setting_report = _setting_report(report, model_name, setting)
        transfer = ((setting_report.get("text_probe_results") or {}).get("cross_benchmark_transfer")) or {}
        cells = []
        for train, test in TRANSFER_PAIRS:
            row = transfer.get(_transfer_key(train, test)) or {}
            cells.append(fmt_pct(row.get("accuracy_mean")))
        lines.append(f"| {config_title(model_name, setting)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _separator_rows(setting_report: Dict[str, Any]) -> List[Tuple[float, str, Dict[str, Tuple[float, float, float] | None]]]:
    summary_by_benchmark = setting_report.get("text_summary_by_benchmark") or {}
    per_benchmark: Dict[str, Dict[str, Tuple[float, float, float]]] = {benchmark: {} for benchmark in BENCHMARKS}
    groups: set[str] = set()
    for benchmark in BENCHMARKS:
        benchmark_summary = summary_by_benchmark.get(benchmark) or {}
        by_label = benchmark_summary.get("by_label") or {}
        c0 = (by_label.get("class0") or {}).get("phrase_group_rates") or {}
        c1 = (by_label.get("class1") or {}).get("phrase_group_rates") or {}
        for group_name in set(c0.keys()) | set(c1.keys()):
            r0 = _safe_float(c0.get(group_name))
            r1 = _safe_float(c1.get(group_name))
            if r0 is None or r1 is None:
                continue
            delta = float(r1) - float(r0)
            per_benchmark[benchmark][str(group_name)] = (float(r0), float(r1), delta)
            groups.add(str(group_name))

    rows: List[Tuple[float, str, Dict[str, Tuple[float, float, float] | None]]] = []
    for group_name in sorted(groups):
        by_benchmark = {benchmark: per_benchmark[benchmark].get(group_name) for benchmark in BENCHMARKS}
        deltas = [abs(v[2]) for v in by_benchmark.values() if v is not None]
        if not deltas:
            continue
        rows.append((max(deltas), group_name, by_benchmark))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def _fmt_separator_cell(value: Tuple[float, float, float] | None) -> str:
    if value is None:
        return "NA"
    _r0, _r1, delta = value
    return f"{delta * 100.0:+.1f}%"


def _shared_phrase_group_order(report: Dict[str, Any], top_phrase_groups: int) -> List[str]:
    score_by_group: Dict[str, float] = {}
    for model_name, setting in CONFIG_ROWS:
        setting_report = _setting_report(report, model_name, setting)
        for score, group_name, _by_benchmark in _separator_rows(setting_report):
            prev = score_by_group.get(group_name)
            if prev is None or score > prev:
                score_by_group[group_name] = score
    ordered = [name for name, _score in sorted(score_by_group.items(), key=lambda kv: (-kv[1], kv[0]))]
    if top_phrase_groups > 0:
        return ordered[:top_phrase_groups]
    return ordered


def build_phrase_separator_table(setting_report: Dict[str, Any], ordered_groups: List[str]) -> str:
    row_map = {
        group_name: by_benchmark
        for _score, group_name, by_benchmark in _separator_rows(setting_report)
    }
    lines = [
        "| Phrase Group | `vqa_rad` | `mmmu_pro` | `medxpertqa_mm` |",
        "|---|---:|---:|---:|",
    ]
    for group_name in ordered_groups:
        by_benchmark = row_map.get(group_name, {})
        lines.append(
            f"| `{group_name}` | {_fmt_separator_cell(by_benchmark.get('vqa_rad'))} | "
            f"{_fmt_separator_cell(by_benchmark.get('mmmu_pro'))} | {_fmt_separator_cell(by_benchmark.get('medxpertqa_mm'))} |"
        )
    if len(lines) == 2:
        lines.append("| NA | NA | NA | NA |")
    return "\n".join(lines)


def render_markdown(report: Dict[str, Any], top_phrase_groups: int) -> str:
    lines: List[str] = []
    lines.append("### Naive Bayes Summary")
    lines.append("")
    lines.append(build_nb_summary_table(report))
    lines.append("")
    lines.append("### Cross-Benchmark Transfer (Single-Benchmark Train/Test)")
    lines.append("")
    lines.append(build_transfer_summary_table(report))
    lines.append("")
    lines.append("### Phrase-Group Class Separators")
    lines.append("")
    lines.append("Cells show `class1_rate - class0_rate`.")
    lines.append("")
    ordered_groups = _shared_phrase_group_order(report, top_phrase_groups=top_phrase_groups)
    for model_name, setting in CONFIG_ROWS:
        setting_report = _setting_report(report, model_name, setting)
        lines.append(f"#### {config_title(model_name, setting)}")
        lines.append("")
        lines.append(build_phrase_separator_table(setting_report, ordered_groups=ordered_groups))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    report = merge_reports(args.reports)
    markdown = render_markdown(report, top_phrase_groups=int(args.top_phrase_groups))
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
