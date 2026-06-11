#!/usr/bin/env python3
"""Generate results-summary PDF in the legacy layout.

This script renders a 6-page summary:
  - One page per probing configuration
  - Rows: benchmark modes
  - Columns: OVIS / QWEN
  - Each cell contains the pre-rendered layerwise line-plot PNG from analysis outputs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat


STAGE_ORDER: List[Tuple[str, str]] = [
    ("logreg_contrastive", "LogReg Contrastive"),
    ("logreg_all_examples", "LogReg All Examples"),
    ("mlp_contrastive", "MLP Contrastive"),
    ("mlp_all_examples", "MLP All Examples"),
    ("diff_contrastive", "Diff Contrastive"),
    ("diff_all_examples", "Diff All Examples"),
]
BENCHMARK_MODES: List[str] = ["vqa_rad", "mmmu_pro", "medxpertqa_mm", "all"]


@dataclass(frozen=True)
class Layout:
    width: int = 2480
    height: int = 3508
    page_bg: Tuple[int, int, int] = (236, 238, 243)

    title_x: int = 84
    title_y: int = 64
    subtitle_y_offset: int = 56

    section_x: int = 52
    section_w: int = 2376
    section_h: int = 786
    section_gap: int = 34
    first_section_y: int = 196

    section_title_x_pad: int = 20
    section_title_y_pad: int = 14

    column_gap: int = 52
    col_card_x_pad: int = 18
    col_card_y_top: int = 68
    col_card_y_bottom: int = 108

    label_x_pad: int = 22
    label_y_from_top: int = 48

    plot_x_pad: int = 20
    plot_top_from_section: int = 130
    plot_h: int = 536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate results-summary PDF with legacy style.")
    parser.add_argument(
        "--ovis_summary",
        type=str,
        default="./results/results_final/final_ovis_residual_020533/analysis/top_probe_summary.json",
    )
    parser.add_argument(
        "--qwen_summary",
        type=str,
        default="./results/results_final/final_qwen_residual_192228/analysis/top_probe_summary.json",
    )
    parser.add_argument(
        "--output_pdf",
        type=str,
        default="./results/results_summary/results_summary_new.pdf",
    )
    parser.add_argument(
        "--style_reference_pdf",
        type=str,
        default="./results/results_summary/results_summary.pdf",
        help="Optional existing PDF used for style-comparison diagnostics.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _index_plot_paths(summary_payload: dict) -> Dict[Tuple[str, str], Path]:
    out: Dict[Tuple[str, str], Path] = {}
    for exp in summary_payload.get("experiments", []):
        if not isinstance(exp, dict):
            continue
        stage_key = str(exp.get("stage_key") or "")
        mode = str(exp.get("mode") or "")
        p = exp.get("layer_plot_path")
        if (not stage_key) or (not mode) or (not p):
            continue
        # There should be at most one plot per (stage, mode). If duplicates appear,
        # keep deterministic first-seen order.
        out.setdefault((stage_key, mode), Path(str(p)))
    return out


def _load_fonts() -> Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        return (
            ImageFont.truetype(regular, 56), # title
            ImageFont.truetype(regular, 32), # subtitle
            ImageFont.truetype(regular, 38), # section benchmark title
            ImageFont.truetype(regular, 36), # column model label
        )
    except Exception:
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback, fallback


def _render_page(
    stage_title: str,
    ovis_plots: Dict[Tuple[str, str], Path],
    qwen_plots: Dict[Tuple[str, str], Path],
    layout: Layout,
) -> Image.Image:
    title_font, subtitle_font, section_font, model_font = _load_fonts()
    img = Image.new("RGB", (layout.width, layout.height), layout.page_bg)
    draw = ImageDraw.Draw(img)

    # Header
    draw.text((layout.title_x, layout.title_y), stage_title, fill=(40, 45, 55), font=title_font)
    draw.text(
        (layout.title_x, layout.title_y + layout.subtitle_y_offset),
        "Rows: benchmark | Columns: OVIS, QWEN",
        fill=(88, 94, 106),
        font=subtitle_font,
    )

    inner_gap = layout.column_gap
    inner_w = (layout.section_w - inner_gap - 2 * layout.col_card_x_pad) // 2
    left_col_x = layout.section_x + layout.col_card_x_pad
    right_col_x = left_col_x + inner_w + inner_gap

    for ridx, mode in enumerate(BENCHMARK_MODES):
        sy = layout.first_section_y + ridx * (layout.section_h + layout.section_gap)
        sx = layout.section_x
        ex = sx + layout.section_w
        ey = sy + layout.section_h

        # Outer row card
        draw.rounded_rectangle(
            (sx, sy, ex, ey),
            radius=24,
            fill=(240, 240, 240),
            outline=(195, 198, 205),
            width=2,
        )
        draw.text(
            (sx + layout.section_title_x_pad, sy + layout.section_title_y_pad),
            f"Benchmark: {mode}",
            fill=(92, 98, 112),
            font=section_font,
        )

        card_top = sy + layout.col_card_y_top
        card_bottom = ey - layout.col_card_y_bottom
        card_h = card_bottom - card_top

        # Left and right inner cards
        for card_x in (left_col_x, right_col_x):
            draw.rectangle(
                (card_x, card_top, card_x + inner_w, card_bottom),
                fill=(240, 240, 240),
                outline=(214, 216, 221),
                width=2,
            )

        draw.text((left_col_x + layout.label_x_pad, sy + layout.label_y_from_top), "OVIS", fill=(60, 66, 76), font=model_font)
        draw.text((right_col_x + layout.label_x_pad, sy + layout.label_y_from_top), "QWEN", fill=(60, 66, 76), font=model_font)

        ovis_plot = ovis_plots.get((stage_title_to_key(stage_title), mode))
        qwen_plot = qwen_plots.get((stage_title_to_key(stage_title), mode))

        plot_y = sy + layout.plot_top_from_section
        _paste_plot(img, ovis_plot, left_col_x + layout.plot_x_pad, plot_y, inner_w - 2 * layout.plot_x_pad, layout.plot_h)
        _paste_plot(img, qwen_plot, right_col_x + layout.plot_x_pad, plot_y, inner_w - 2 * layout.plot_x_pad, layout.plot_h)

    return img


def _paste_plot(canvas: Image.Image, plot_path: Optional[Path], x: int, y: int, w: int, h: int) -> None:
    draw = ImageDraw.Draw(canvas)
    if plot_path is None or (not plot_path.exists()):
        draw.rectangle((x, y, x + w, y + h), outline=(180, 180, 180), width=2)
        draw.text((x + 14, y + 14), "Missing plot", fill=(130, 130, 130))
        return
    try:
        plot = Image.open(plot_path).convert("RGB")
    except Exception:
        draw.rectangle((x, y, x + w, y + h), outline=(180, 180, 180), width=2)
        draw.text((x + 14, y + 14), "Unreadable plot", fill=(130, 130, 130))
        return

    plot = _trim_plot_whitespace(plot)
    plot.thumbnail((w, h), Image.Resampling.LANCZOS)
    px = x + (w - plot.width) // 2
    py = y + (h - plot.height) // 2
    canvas.paste(plot, (px, py))


def _trim_plot_whitespace(plot: Image.Image) -> Image.Image:
    """Trim near-uniform outer whitespace so plots fill their slots consistently."""
    # Use top-left pixel as border reference color (works for white/light backgrounds).
    bg = Image.new("RGB", plot.size, color=plot.getpixel((0, 0)))
    diff = ImageChops.difference(plot, bg).convert("L")
    # Slight thresholding to ignore tiny JPEG noise in borders.
    mask = diff.point(lambda p: 255 if p > 8 else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return plot
    x0, y0, x1, y1 = bbox
    # Keep tight crop so chart content fills the slot more completely.
    pad = 0
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(plot.width, x1 + pad)
    y1 = min(plot.height, y1 + pad)
    cropped = plot.crop((x0, y0, x1, y1))
    return cropped if (cropped.width > 0 and cropped.height > 0) else plot


def stage_title_to_key(stage_title: str) -> str:
    for key, title in STAGE_ORDER:
        if title == stage_title:
            return key
    raise KeyError(stage_title)


def _extract_jpegs_from_pdf(pdf_path: Path) -> List[Image.Image]:
    payload = pdf_path.read_bytes()
    out: List[Image.Image] = []
    i = 0
    while True:
        s = payload.find(b"\xff\xd8", i)
        if s < 0:
            break
        e = payload.find(b"\xff\xd9", s + 2)
        if e < 0:
            break
        chunk = payload[s : e + 2]
        i = e + 2
        if len(chunk) < 50_000:
            continue
        try:
            from io import BytesIO

            im = Image.open(BytesIO(chunk)).convert("RGB")
            out.append(im)
        except Exception:
            continue
    return out


def _style_diagnostics(new_pdf: Path, ref_pdf: Path, layout: Layout) -> Dict[str, float]:
    if not ref_pdf.exists():
        return {"reference_found": 0.0}
    new_pages = _extract_jpegs_from_pdf(new_pdf)
    ref_pages = _extract_jpegs_from_pdf(ref_pdf)
    if not new_pages or not ref_pages:
        return {"reference_found": 1.0, "comparable_pages": 0.0}

    n = min(len(new_pages), len(ref_pages))
    # Build a mask that ignores plot-content rectangles and measures static UI style.
    mask_img = Image.new("L", (layout.width, layout.height), color=255)
    mask_draw = ImageDraw.Draw(mask_img)
    inner_w = (layout.section_w - layout.column_gap - 2 * layout.col_card_x_pad) // 2
    left_col_x = layout.section_x + layout.col_card_x_pad
    right_col_x = left_col_x + inner_w + layout.column_gap
    for ridx in range(len(BENCHMARK_MODES)):
        sy = layout.first_section_y + ridx * (layout.section_h + layout.section_gap)
        py = sy + layout.plot_top_from_section
        for px in (left_col_x + layout.plot_x_pad, right_col_x + layout.plot_x_pad):
            x0, y0 = px, py
            x1, y1 = px + (inner_w - 2 * layout.plot_x_pad), py + layout.plot_h
            mask_draw.rectangle((x0, y0, x1, y1), fill=0)

    diffs: List[float] = []
    for i in range(n):
        a = new_pages[i].resize((layout.width, layout.height), Image.Resampling.BILINEAR)
        b = ref_pages[i].resize((layout.width, layout.height), Image.Resampling.BILINEAR)
        diff_rgb = ImageChops.difference(a, b).convert("L")
        stat = ImageStat.Stat(diff_rgb, mask=mask_img)
        diffs.append(float(stat.mean[0]))

    return {
        "reference_found": 1.0,
        "comparable_pages": float(n),
        "mean_abs_diff_nonplot_rgb": float(sum(diffs) / len(diffs)),
        "max_abs_diff_nonplot_rgb": float(max(diffs)),
    }


def main() -> None:
    args = parse_args()
    layout = Layout()

    ovis_summary = _load_json(Path(args.ovis_summary))
    qwen_summary = _load_json(Path(args.qwen_summary))
    ovis_plots = _index_plot_paths(ovis_summary)
    qwen_plots = _index_plot_paths(qwen_summary)

    pages: List[Image.Image] = []
    for _stage_key, stage_title in STAGE_ORDER:
        pages.append(_render_page(stage_title, ovis_plots, qwen_plots, layout))

    output_pdf = Path(args.output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(output_pdf, save_all=True, append_images=pages[1:])

    diag = _style_diagnostics(output_pdf, Path(args.style_reference_pdf), layout)
    manifest = {
        "output_pdf": str(output_pdf),
        "ovis_summary": str(args.ovis_summary),
        "qwen_summary": str(args.qwen_summary),
        "num_pages": len(pages),
        "style_diagnostics": diag,
    }
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
