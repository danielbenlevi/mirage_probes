#!/usr/bin/env python3
"""Aggregate probe experiment outputs and generate residual-layer plots + top-k summaries."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


TEST_ACC_KEYS = (
    "mean_test_accuracy_at_best_c",
    "test_accuracy_at_best_c",
    "best_test_accuracy",
    "test_accuracy",
)
CLASS0_KEYS = (
    "mean_class0_test_accuracy_at_best_c",
    "class0_test_accuracy_at_best_c",
    "class0_test_accuracy",
    "test_accuracy_mirage_false",
)
CLASS1_KEYS = (
    "mean_class1_test_accuracy_at_best_c",
    "class1_test_accuracy_at_best_c",
    "class1_test_accuracy",
    "test_accuracy_mirage_true",
)
BENCHMARK_KEYS = (
    "mean_benchmark_test_accuracy_at_best_c",
    "benchmark_test_accuracy_at_best_c",
    "benchmark_test_accuracy",
)
BENCHMARK_CLASS0_KEYS = (
    "mean_benchmark_class0_test_accuracy_at_best_c",
    "benchmark_class0_test_accuracy_at_best_c",
    "benchmark_class0_test_accuracy",
)
BENCHMARK_CLASS1_KEYS = (
    "mean_benchmark_class1_test_accuracy_at_best_c",
    "benchmark_class1_test_accuracy_at_best_c",
    "benchmark_class1_test_accuracy",
)
_PLOT_RUNTIME_NOTES: List[str] = []


def _append_plot_note_once(note: str) -> None:
    if note not in _PLOT_RUNTIME_NOTES:
        _PLOT_RUNTIME_NOTES.append(note)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create per-experiment residual-layer test-accuracy plots and a top-k summary JSON "
            "from a run_full_probe_experiment output directory."
        )
    )
    parser.add_argument("--experiment_run_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--plot_format", type=str, default="png", choices=["png", "svg"])
    return parser.parse_args()


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_present(d: Dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if key in d:
            value = _safe_float(d.get(key))
            if value is not None:
                return value
    return None


def _derive_class_acc_from_seed_runs(row: Dict, class_keys: Tuple[str, ...]) -> Optional[float]:
    seed_runs = row.get("seed_runs")
    if not isinstance(seed_runs, list):
        return None
    vals = []
    for run in seed_runs:
        if not isinstance(run, dict):
            continue
        v = _first_present(run, class_keys)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


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


def _derive_dict_from_seed_runs(row: Dict, keys: Tuple[str, ...]) -> Optional[Dict[str, float]]:
    seed_runs = row.get("seed_runs")
    if not isinstance(seed_runs, list):
        return None
    vals: Dict[str, List[float]] = defaultdict(list)
    for run in seed_runs:
        if not isinstance(run, dict):
            continue
        d = _first_present_dict(run, keys)
        if not isinstance(d, dict):
            continue
        for benchmark, value in d.items():
            vals[str(benchmark)].append(float(value))
    if not vals:
        return None
    return {
        benchmark: float(sum(v) / len(v))
        for benchmark, v in sorted(vals.items())
        if v
    }


def _mean_dict_metrics_from_seed_runs(seed_runs: List[Dict], key: str) -> Optional[Dict[str, float]]:
    vals: Dict[str, List[float]] = defaultdict(list)
    for run in seed_runs:
        if not isinstance(run, dict):
            continue
        d = _safe_float_dict(run.get(key))
        if not isinstance(d, dict):
            continue
        for benchmark, value in d.items():
            vals[str(benchmark)].append(float(value))
    if not vals:
        return None
    return {
        benchmark: float(sum(v) / len(v))
        for benchmark, v in sorted(vals.items())
        if v
    }


def _is_contrastive_experiment(stage_key: Optional[str], config_payload: Dict, rel_parent: str) -> bool:
    sk = str(stage_key or "").lower()
    if "contrastive" in sk:
        return True
    script_name = str(config_payload.get("script_name") or "").lower()
    if "contrastive" in script_name:
        return True
    rel_parent_l = str(rel_parent or "").lower()
    return "contrastive" in rel_parent_l


def _is_logreg_contrastive_experiment(stage_key: Optional[str], config_payload: Dict, rel_parent: str) -> bool:
    sk = str(stage_key or "").lower()
    if sk == "logreg_contrastive":
        return True
    script_name = str(config_payload.get("script_name") or "").lower()
    if "train_log_reg_contrastive" in script_name:
        return True
    rel_parent_l = str(rel_parent or "").lower()
    return "logreg_contrastive" in rel_parent_l


def _heldout_metrics_from_seed_runs(heldout_seed_runs) -> Optional[Dict]:
    if not isinstance(heldout_seed_runs, list) or not heldout_seed_runs:
        return None
    test_scores = []
    class0_scores = []
    class1_scores = []
    for run in heldout_seed_runs:
        if not isinstance(run, dict):
            continue
        t = _safe_float(run.get("test_accuracy"))
        if t is not None:
            test_scores.append(float(t))
        c0 = _safe_float(run.get("test_accuracy_mirage_false"))
        if c0 is not None:
            class0_scores.append(float(c0))
        c1 = _safe_float(run.get("test_accuracy_mirage_true"))
        if c1 is not None:
            class1_scores.append(float(c1))
    benchmark_test = _mean_dict_metrics_from_seed_runs(heldout_seed_runs, key="benchmark_test_accuracy")
    benchmark_class0 = _mean_dict_metrics_from_seed_runs(
        heldout_seed_runs,
        key="benchmark_class0_test_accuracy",
    )
    benchmark_class1 = _mean_dict_metrics_from_seed_runs(
        heldout_seed_runs,
        key="benchmark_class1_test_accuracy",
    )
    if not test_scores:
        return None
    return {
        "test_accuracy": float(sum(test_scores) / len(test_scores)),
        "class0_test_accuracy": (float(sum(class0_scores) / len(class0_scores)) if class0_scores else None),
        "class1_test_accuracy": (float(sum(class1_scores) / len(class1_scores)) if class1_scores else None),
        "benchmark_test_accuracy": benchmark_test,
        "benchmark_class0_test_accuracy": benchmark_class0,
        "benchmark_class1_test_accuracy": benchmark_class1,
    }


def _load_llm_holdout_feature_metrics(artifact_path: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    candidates = sorted(artifact_path.parent.glob("*llm_residual_layer_heldout_eval*.json"))
    if not candidates:
        return out
    try:
        payload = _load_json(candidates[0])
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    for _layer_name, info in payload.items():
        if not isinstance(info, dict):
            continue
        feature = str(info.get("best_feature") or "")
        if not feature:
            continue
        test_acc = _safe_float(info.get("test_accuracy_mean"))
        if test_acc is None:
            continue
        out[feature] = {
            "test_accuracy": float(test_acc),
            "class0_test_accuracy": _safe_float(info.get("test_accuracy_mirage_false_mean")),
            "class1_test_accuracy": _safe_float(info.get("test_accuracy_mirage_true_mean")),
            "benchmark_test_accuracy": _safe_float_dict(info.get("benchmark_test_accuracy_mean")),
            "benchmark_class0_test_accuracy": _safe_float_dict(info.get("benchmark_class0_test_accuracy_mean")),
            "benchmark_class1_test_accuracy": _safe_float_dict(info.get("benchmark_class1_test_accuracy_mean")),
        }
    return out


def _parse_layer_info(feature: str) -> Dict:
    base = str(feature).split("__")[0]
    strategy = str(feature).split("__", 1)[1] if "__" in str(feature) else "default"

    layer_index = None
    layer_name = base
    feature_family = "other"
    attention_head_index = None
    m = re.search(r"language_model/layer_(\d+)", base)
    if m:
        layer_index = int(m.group(1))
        layer_name = f"layer_{layer_index}"
        if "/attention_head_" in base:
            feature_family = "attention_head"
            mh = re.search(r"/attention_head_(\d+)", base)
            if mh:
                attention_head_index = int(mh.group(1))
        elif "/post_attention" in base:
            feature_family = "post_attention"
        elif "/mlp" in base:
            feature_family = "mlp"
        else:
            feature_family = "residual"
    elif "language_model/post_layer_norm" in base:
        layer_name = "post_layer_norm"
        feature_family = "residual"
    elif "all_layers_concat_attention_head_" in base:
        feature_family = "concat_attention_head"
        mh = re.search(r"all_layers_concat_attention_head_(\d+)", base)
        if mh:
            attention_head_index = int(mh.group(1))
    elif "all_layers_concat_post_attention" in base:
        feature_family = "concat_post_attention"
    elif "all_layers_concat_mlp" in base:
        feature_family = "concat_mlp"
    elif "all_layers_concat" in base:
        feature_family = "concat_residual"

    return {
        "layer_index": layer_index,
        "layer_name": layer_name,
        "strategy": strategy,
        "feature_family": feature_family,
        "attention_head_index": attention_head_index,
    }


def _is_concat_experiment(config_payload: Dict, rows: List[Dict]) -> bool:
    feature_variant = str(config_payload.get("feature_variant") or "")
    if feature_variant == "concat_llm_residual":
        return True
    for row in rows:
        fam = str(row.get("feature_family") or "")
        if fam.startswith("concat_"):
            return True
    return False


def _is_additional_family_mode(config_payload: Dict, rows: List[Dict]) -> bool:
    include_attention = bool(config_payload.get("include_attention_probes", False))
    include_mlp = bool(config_payload.get("include_mlp_probes", False))
    include_additional = bool(config_payload.get("include_additional_attention_mlp_probes", False))
    include_residual = bool(config_payload.get("include_residual_probes", True))
    if (not include_residual) and (include_attention or include_mlp or include_additional):
        return True

    fams = {str(row.get("feature_family") or "") for row in rows}
    has_additional = bool(fams.intersection({"attention_head", "post_attention", "mlp"}))
    has_residual = ("residual" in fams)
    return bool(has_additional and (not has_residual))


def _build_additional_plot_groups(rows: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    groups: List[Tuple[str, List[Dict]]] = []
    head_ids = sorted(
        {
            int(row["attention_head_index"])
            for row in rows
            if row.get("feature_family") == "attention_head" and row.get("attention_head_index") is not None
        }
    )
    for head_id in head_ids:
        group_rows = [
            row
            for row in rows
            if row.get("feature_family") == "attention_head" and int(row.get("attention_head_index")) == int(head_id)
        ]
        if group_rows:
            groups.append((f"attention_head_{int(head_id)}", group_rows))

    post_rows = [row for row in rows if row.get("feature_family") == "post_attention"]
    if post_rows:
        groups.append(("post_attention", post_rows))

    mlp_rows = [row for row in rows if row.get("feature_family") == "mlp"]
    if mlp_rows:
        groups.append(("mlp", mlp_rows))

    return groups


def _rows_from_payload(payload) -> List[Dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = []
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("feature", str(key))
            rows.append(row)
        return rows
    return []


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_config_path(artifact_path: Path) -> Optional[Path]:
    candidates = sorted(
        set(artifact_path.parent.glob("*run_config*.json")) | set(artifact_path.parent.glob("run_config.json"))
    )
    if not candidates:
        return None
    if "llm_dim_token_probe_accuracies" in artifact_path.name:
        for c in candidates:
            if "dim_token_probe_run_config" in c.name:
                return c
    return candidates[0]


def _candidate_artifacts(root: Path) -> List[Path]:
    out = []
    for p in root.rglob("*all_feature_probe_accuracies*.json"):
        out.append(p)
    for p in root.rglob("*llm_dim_token_probe_accuracies*.json"):
        out.append(p)
    # de-duplicate and stable sort
    return sorted(set(out))


def _extract_stage_mode_from_relpath(rel_parent: str) -> Tuple[Optional[str], Optional[str]]:
    parts = rel_parent.split("/")
    stage_key = None
    mode = None
    for part in parts:
        m = re.match(r"^\d+_(.+)$", part)
        if m:
            stage_key = m.group(1)
            break
    # Try every path segment, preferring explicit benchmark<->model adjacency in either order.
    model_alt = r"(?:ovis|qwen3_vl_32b_instruct|glm_4_6v_flash)"
    mode_alt = r"(?:vqa_rad|mmmu_pro|medxpertqa_mm|all)"
    for segment in reversed(parts):
        # Pattern A: ..._<mode>_<model>_...
        mode_match = re.search(rf"_({mode_alt})_{model_alt}(?:_|$)", segment)
        if mode_match is not None:
            mode = mode_match.group(1)
            break
        # Pattern B: ..._<model>_<mode>_...
        mode_match = re.search(rf"_{model_alt}_({mode_alt})(?:_|$)", segment)
        if mode_match is not None:
            mode = mode_match.group(1)
            break
    if mode is None:
        tail = parts[-1] if parts else ""
        # Legacy fallback for non-standard names. Keep this strict enough to avoid
        # matching "_all_" inside tokens like "all_examples".
        mode_match = re.search(r"_(vqa_rad|mmmu_pro|medxpertqa_mm|all)(?:_|$)", tail)
        if mode_match:
            mode = mode_match.group(1)
    return stage_key, mode


def _build_strategy_layer_map(rows: List[Dict]) -> Dict[str, Dict[int, float]]:
    by_strategy: Dict[str, Dict[int, float]] = defaultdict(dict)
    for row in rows:
        layer_idx = row.get("layer_index")
        test_acc = row.get("test_accuracy")
        strategy = row.get("strategy", "default")
        if layer_idx is None or test_acc is None:
            continue
        prev = by_strategy[strategy].get(int(layer_idx))
        if prev is None or float(test_acc) > float(prev):
            by_strategy[strategy][int(layer_idx)] = float(test_acc)
    return by_strategy


def _plot_layer_curves_with_pil(
    by_strategy: Dict[str, Dict[int, float]],
    output_path: Path,
    plot_style: str = "default",
) -> Optional[str]:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        _append_plot_note_once(f"Pillow import failed; no plots created. error={type(exc).__name__}: {exc}")
        return None

    target_path = output_path
    if target_path.suffix.lower() != ".png":
        target_path = output_path.with_suffix(".png")
        _append_plot_note_once(
            f"matplotlib unavailable; Pillow fallback only supports PNG. Wrote '{target_path.name}' instead of '{output_path.name}'."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 1100, 520
    left, right, top, bottom = 80, 30, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    if plot_w <= 0 or plot_h <= 0:
        return None

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x_values = sorted({x for layer_map in by_strategy.values() for x in layer_map.keys()})
    y_values = [float(y) for layer_map in by_strategy.values() for y in layer_map.values()]
    if not x_values or not y_values:
        return None

    x_min, x_max = int(min(x_values)), int(max(x_values))
    if x_min == x_max:
        x_min -= 1
        x_max += 1

    y_min_data = min(y_values)
    y_max_data = max(y_values)
    y_min = min(0.0, y_min_data - 0.02)
    y_max = max(1.0, y_max_data + 0.02)
    if y_max <= y_min:
        y_max = y_min + 1.0

    def map_x(x: int) -> float:
        return left + (float(x - x_min) / float(x_max - x_min)) * plot_w

    def map_y(y: float) -> float:
        return top + (float(y_max - y) / float(y_max - y_min)) * plot_h

    # Axes
    draw.line([(left, top), (left, top + plot_h)], fill="black", width=1)
    draw.line([(left, top + plot_h), (left + plot_w, top + plot_h)], fill="black", width=1)

    # Y-axis grid and ticks.
    for i in range(6):
        frac = i / 5.0
        y_val = y_max - frac * (y_max - y_min)
        py = map_y(y_val)
        draw.line([(left, py), (left + plot_w, py)], fill=(230, 230, 230), width=1)
        draw.text((8, py - 6), f"{y_val:.2f}", fill="black", font=font)

    # X-axis ticks.
    for xv in x_values:
        px = map_x(xv)
        draw.line([(px, top + plot_h), (px, top + plot_h + 5)], fill="black", width=1)
        draw.text((px - 8, top + plot_h + 10), str(xv), fill="black", font=font)

    palette = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
    ]

    legend_x = left + 8
    legend_y = 8
    strategy_items = sorted(by_strategy.items())
    strategy_colors = {
        str(strategy): palette[idx % len(palette)]
        for idx, (strategy, _layer_map) in enumerate(strategy_items)
    }

    all_points = []
    for strategy, layer_map in strategy_items:
        for x in sorted(layer_map.keys()):
            all_points.append((int(x), float(layer_map[x]), str(strategy)))
    all_points.sort(key=lambda item: (item[0], item[2]))

    if plot_style == "logreg_contrastive_neutral_line":
        neutral_points = [(map_x(x), map_y(y)) for x, y, _strategy in all_points]
        if len(neutral_points) >= 2:
            draw.line(neutral_points, fill=(140, 140, 140), width=2)
        for x, y, strategy in all_points:
            color = strategy_colors[str(strategy)]
            px = map_x(x)
            py = map_y(y)
            draw.ellipse([(px - 3, py - 3), (px + 3, py + 3)], fill=color, outline=color)

    for idx, (strategy, layer_map) in enumerate(strategy_items):
        color = palette[idx % len(palette)]
        xs = sorted(layer_map.keys())
        points = [(map_x(x), map_y(float(layer_map[x]))) for x in xs]
        if plot_style != "logreg_contrastive_neutral_line" and len(points) >= 2:
            draw.line(points, fill=color, width=2)
        if plot_style != "logreg_contrastive_neutral_line":
            for px, py in points:
                draw.ellipse([(px - 2, py - 2), (px + 2, py + 2)], fill=color, outline=color)

        if len(by_strategy) <= 12:
            y = legend_y + idx * 14
            draw.rectangle([(legend_x, y + 3), (legend_x + 10, y + 9)], fill=color, outline=color)
            draw.text((legend_x + 14, y), str(strategy), fill="black", font=font)

    draw.text((width // 2 - 85, 8), "Test Accuracy by LLM Residual Layer", fill="black", font=font)
    draw.text((width // 2 - 60, height - 20), "LLM Residual Layer", fill="black", font=font)
    draw.text((8, top - 18), "Test Accuracy", fill="black", font=font)

    image.save(target_path)
    return str(target_path)


def _plot_layer_curves(rows: List[Dict], output_path: Path, plot_style: str = "default") -> Optional[str]:
    by_strategy = _build_strategy_layer_map(rows)

    if not by_strategy:
        return None

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        _append_plot_note_once(
            f"matplotlib unavailable; using Pillow fallback renderer. error={type(exc).__name__}: {exc}"
        )
        return _plot_layer_curves_with_pil(by_strategy=by_strategy, output_path=output_path, plot_style=plot_style)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 5))
    strategy_items = sorted(by_strategy.items())
    palette = [
        (31 / 255.0, 119 / 255.0, 180 / 255.0),
        (255 / 255.0, 127 / 255.0, 14 / 255.0),
        (44 / 255.0, 160 / 255.0, 44 / 255.0),
        (214 / 255.0, 39 / 255.0, 40 / 255.0),
        (148 / 255.0, 103 / 255.0, 189 / 255.0),
        (140 / 255.0, 86 / 255.0, 75 / 255.0),
        (227 / 255.0, 119 / 255.0, 194 / 255.0),
        (127 / 255.0, 127 / 255.0, 127 / 255.0),
        (188 / 255.0, 189 / 255.0, 34 / 255.0),
        (23 / 255.0, 190 / 255.0, 207 / 255.0),
    ]
    strategy_colors = {
        str(strategy): palette[idx % len(palette)]
        for idx, (strategy, _layer_map) in enumerate(strategy_items)
    }

    if plot_style == "logreg_contrastive_neutral_line":
        all_points = []
        for strategy, layer_map in strategy_items:
            for x in sorted(layer_map.keys()):
                all_points.append((int(x), float(layer_map[x]), str(strategy)))
        all_points.sort(key=lambda item: (item[0], item[2]))
        if len(all_points) >= 2:
            plt.plot(
                [x for x, _y, _s in all_points],
                [y for _x, y, _s in all_points],
                color=(140 / 255.0, 140 / 255.0, 140 / 255.0),
                linewidth=1.7,
                zorder=1,
            )
        for strategy, layer_map in strategy_items:
            xs = sorted(layer_map.keys())
            ys = [layer_map[x] for x in xs]
            if not xs:
                continue
            plt.scatter(xs, ys, s=24, color=strategy_colors[str(strategy)], label=strategy, zorder=3)
    else:
        for strategy, layer_map in strategy_items:
            xs = sorted(layer_map.keys())
            ys = [layer_map[x] for x in xs]
            plt.plot(xs, ys, marker="o", linewidth=1.7, markersize=3.8, label=strategy)

    plt.xlabel("LLM Residual Layer")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by LLM Residual Layer")
    if len(by_strategy) <= 12:
        plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return str(output_path)


def main() -> None:
    args = parse_args()
    run_root = Path(args.experiment_run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    output_dir = Path(args.output_dir).expanduser().resolve() if str(args.output_dir).strip() else (run_root / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _candidate_artifacts(run_root)
    experiments = []

    for artifact_path in artifacts:
        if output_dir in artifact_path.parents:
            continue
        rel_parent = str(artifact_path.parent.relative_to(run_root))
        stage_key, mode = _extract_stage_mode_from_relpath(rel_parent)
        config_path = _find_config_path(artifact_path)
        config_payload = _load_json(config_path) if config_path is not None and config_path.exists() else {}
        is_contrastive = _is_contrastive_experiment(
            stage_key=stage_key,
            config_payload=config_payload,
            rel_parent=rel_parent,
        )
        is_logreg_contrastive = _is_logreg_contrastive_experiment(
            stage_key=stage_key,
            config_payload=config_payload,
            rel_parent=rel_parent,
        )
        llm_holdout_by_feature: Dict[str, Dict] = {}
        if is_contrastive:
            llm_holdout_by_feature = _load_llm_holdout_feature_metrics(artifact_path)

        payload = _load_json(artifact_path)
        rows_raw = _rows_from_payload(payload)
        normalized_rows = []
        rows_missing_test_accuracy = 0
        for row in rows_raw:
            feature = str(row.get("feature", ""))
            if not feature:
                continue
            test_acc = _first_present(row, TEST_ACC_KEYS)
            class0_acc = _first_present(row, CLASS0_KEYS)
            class1_acc = _first_present(row, CLASS1_KEYS)
            if class0_acc is None:
                class0_acc = _derive_class_acc_from_seed_runs(row, CLASS0_KEYS)
            if class1_acc is None:
                class1_acc = _derive_class_acc_from_seed_runs(row, CLASS1_KEYS)
            benchmark_acc = _first_present_dict(row, BENCHMARK_KEYS)
            benchmark_class0_acc = _first_present_dict(row, BENCHMARK_CLASS0_KEYS)
            benchmark_class1_acc = _first_present_dict(row, BENCHMARK_CLASS1_KEYS)
            if benchmark_acc is None:
                benchmark_acc = _derive_dict_from_seed_runs(row, BENCHMARK_KEYS)
            if benchmark_class0_acc is None:
                benchmark_class0_acc = _derive_dict_from_seed_runs(row, BENCHMARK_CLASS0_KEYS)
            if benchmark_class1_acc is None:
                benchmark_class1_acc = _derive_dict_from_seed_runs(row, BENCHMARK_CLASS1_KEYS)

            if is_contrastive:
                heldout_metrics = _heldout_metrics_from_seed_runs(row.get("heldout_seed_runs"))
                if heldout_metrics is None:
                    heldout_metrics = llm_holdout_by_feature.get(feature)
                if heldout_metrics is not None and heldout_metrics.get("test_accuracy") is not None:
                    test_acc = heldout_metrics.get("test_accuracy")
                    class0_acc = heldout_metrics.get("class0_test_accuracy")
                    class1_acc = heldout_metrics.get("class1_test_accuracy")
                    benchmark_acc = heldout_metrics.get("benchmark_test_accuracy")
                    benchmark_class0_acc = heldout_metrics.get("benchmark_class0_test_accuracy")
                    benchmark_class1_acc = heldout_metrics.get("benchmark_class1_test_accuracy")

            if test_acc is None:
                rows_missing_test_accuracy += 1
                continue

            layer_info = _parse_layer_info(feature)
            normalized_rows.append(
                {
                    "feature": feature,
                    "test_accuracy": float(test_acc),
                    "class0_test_accuracy": class0_acc,
                    "class1_test_accuracy": class1_acc,
                    "benchmark_test_accuracy": benchmark_acc,
                    "benchmark_class0_test_accuracy": benchmark_class0_acc,
                    "benchmark_class1_test_accuracy": benchmark_class1_acc,
                    **layer_info,
                }
            )
        if not normalized_rows:
            if is_contrastive and rows_raw and rows_missing_test_accuracy > 0:
                _append_plot_note_once(
                    "Skipped contrastive artifact with no heldout test accuracies: "
                    f"{artifact_path}"
                )
            continue

        normalized_rows = sorted(
            normalized_rows,
            key=lambda r: (-float(r["test_accuracy"]), str(r["feature"])),
        )
        top_items = normalized_rows[: int(max(1, args.top_k))]
        experiment_id = rel_parent.replace("/", "__")
        is_concat = _is_concat_experiment(config_payload=config_payload, rows=normalized_rows)
        is_additional_mode = _is_additional_family_mode(config_payload=config_payload, rows=normalized_rows)

        plot_path = plots_dir / f"{experiment_id}_llm_residual_test_accuracy.{args.plot_format}"
        plot_style = "logreg_contrastive_neutral_line" if is_logreg_contrastive else "default"
        saved_plot = None if is_concat else _plot_layer_curves(normalized_rows, plot_path, plot_style=plot_style)

        additional_family_plot_paths: Dict[str, str] = {}
        if (not is_concat) and is_additional_mode:
            for group_key, group_rows in _build_additional_plot_groups(normalized_rows):
                group_plot_path = (
                    plots_dir
                    / f"{experiment_id}_{group_key}_cross_layer_test_accuracy.{args.plot_format}"
                )
                saved_group_plot = _plot_layer_curves(group_rows, group_plot_path, plot_style="default")
                if saved_group_plot:
                    additional_family_plot_paths[str(group_key)] = str(saved_group_plot)

        experiments.append(
            {
                "experiment_id": experiment_id,
                "stage_key": stage_key,
                "mode": mode,
                "artifact_path": str(artifact_path),
                "config_path": str(config_path) if config_path is not None else None,
                "script_name": config_payload.get("script_name"),
                "probe_type": config_payload.get("probe_type"),
                "feature_variant": config_payload.get("feature_variant"),
                "vlm": config_payload.get("vlm"),
                "num_probe_rows": int(len(normalized_rows)),
                "layer_plot_path": saved_plot,
                "additional_family_layer_plot_paths": (
                    additional_family_plot_paths if additional_family_plot_paths else {}
                ),
                "top_probes": [
                    {
                        "rank": int(i + 1),
                        "feature": row["feature"],
                        "layer_name": row["layer_name"],
                        "layer_index": row["layer_index"],
                        "strategy": row["strategy"],
                        "test_accuracy": float(row["test_accuracy"]),
                        "class0_test_accuracy_avg": row["class0_test_accuracy"],
                        "class1_test_accuracy_avg": row["class1_test_accuracy"],
                        "benchmark_test_accuracy_avg": row["benchmark_test_accuracy"],
                        "benchmark_class0_test_accuracy_avg": row["benchmark_class0_test_accuracy"],
                        "benchmark_class1_test_accuracy_avg": row["benchmark_class1_test_accuracy"],
                    }
                    for i, row in enumerate(top_items)
                ],
            }
        )

    experiments = sorted(
        experiments,
        key=lambda e: (str(e.get("stage_key") or ""), str(e.get("mode") or ""), str(e.get("experiment_id") or "")),
    )

    summary_payload = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "experiment_run_root": str(run_root),
        "output_dir": str(output_dir),
        "top_k": int(max(1, args.top_k)),
        "num_experiments": int(len(experiments)),
        "plot_runtime_notes": list(_PLOT_RUNTIME_NOTES),
        "experiments": experiments,
    }

    summary_path = output_dir / "top_probe_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    manifest = {
        "analysis_summary_json": str(summary_path),
        "plots_dir": str(plots_dir),
        "num_experiments": int(len(experiments)),
        "plot_runtime_notes": list(_PLOT_RUNTIME_NOTES),
    }
    manifest_path = output_dir / "analysis_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
