#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DATA_ROOT = Path("./tmp_artifacts")
DEFAULT_EXPERIMENT_ROOT = Path("./results/results_final")
DEFAULT_BENCHMARK_MODES = ("vqa_rad", "mmmu_pro", "medxpertqa_mm")
SUPPORTED_VLMS = ("ovis", "qwen3_vl_32b_instruct", "glm_4_6v_flash")
DEFAULT_GPU_IDS = ("6", "7")
GLM_SKIPPED_BENCHMARK_MODES = {"medxpertqa_mm"}
QWEN_PREEXTRACTED_REQUIRED_PATHS = (
    Path("./tmp_artifacts/qwen3_vl_32b_instruct/qwen3_vl_32b_instruct_preextracted_contrastive_features.pt"),
    Path("./tmp_artifacts/qwen3_vl_32b_instruct/qwen3_vl_32b_instruct_preextracted_all_examples_features.pt"),
)
QWEN_PREEXTRACTED_ADDITIONAL_REQUIRED_PATHS = (
    Path("./tmp_artifacts/qwen3_vl_32b_instruct/qwen3_vl_32b_instruct_preextracted_contrastive_additional_features.pt"),
    Path("./tmp_artifacts/qwen3_vl_32b_instruct/qwen3_vl_32b_instruct_preextracted_all_examples_additional_features.pt"),
)
GLM_PREEXTRACTED_REQUIRED_PATHS = (
    Path("./tmp_artifacts/glm_4_6v_flash/glm_4_6v_flash_preextracted_contrastive_features.pt"),
    Path("./tmp_artifacts/glm_4_6v_flash/glm_4_6v_flash_preextracted_all_examples_features.pt"),
)
GLM_PREEXTRACTED_ADDITIONAL_REQUIRED_PATHS = (
    Path("./tmp_artifacts/glm_4_6v_flash/glm_4_6v_flash_preextracted_contrastive_additional_features.pt"),
    Path("./tmp_artifacts/glm_4_6v_flash/glm_4_6v_flash_preextracted_all_examples_additional_features.pt"),
)


SCRIPT_CANDIDATES: Dict[str, List[str]] = {
    "logreg_contrastive": ["scripts/training/train_log_reg_contrastive.py"],
    "logreg_all_examples": ["scripts/training/train_log_reg_all_examples.py"],
    "mlp_contrastive": [
        "scripts/training/train_mlp_contrastive.py",
        "scripts/training/train_probe_mlp_contrastive.py",
        "scripts/training/train_contrastive_mlp_probes.py",
    ],
    "mlp_all_examples": [
        "scripts/training/train_mlp_all_examples.py",
        "scripts/training/train_probe_mlp_all_examples.py",
        "scripts/training/train_all_examples_mlp_probes.py",
    ],
    "concat_contrastive": [
        "scripts/training/train_concat_llm_residual_contrastive.py",
        "scripts/training/train_concat_layers_contrastive.py",
        "scripts/training/train_concat_residual_contrastive.py",
        "scripts/training/train_contrastive_concat_layers.py",
    ],
    "concat_all_examples": [
        "scripts/training/train_concat_llm_residual_all_examples.py",
        "scripts/training/train_concat_layers_all_examples.py",
        "scripts/training/train_concat_residual_all_examples.py",
        "scripts/training/train_all_examples_concat_layers.py",
    ],
    "diff_contrastive": [
        "scripts/training/train_diff_activations_contrastive.py",
        "scripts/training/train_contrastive_difference_probes.py",
        "scripts/training/train_activation_diff_contrastive.py",
    ],
    "diff_all_examples": [
        "scripts/training/train_diff_activations_all_examples.py",
        "scripts/training/train_all_examples_difference_probes.py",
        "scripts/training/train_activation_diff_all_examples.py",
    ],
}


MODE_FLAG_CANDIDATES: Dict[str, Dict[str, List[Tuple[str, ...]]]] = {
    "contrastive": {
        "vqa_rad": (
            ("--benchmark_mode", "vqa_rad"),
            ("--benchmark", "vqa_rad"),
            ("--vqa_only_pairs",),
            ("--vqa_only",),
        ),
        "mmmu_pro": (
            ("--benchmark_mode", "mmmu_pro"),
            ("--benchmark", "mmmu_pro"),
            ("--mmmu_only_pairs",),
            ("--mmmu_only",),
        ),
        "microvqa": (
            ("--benchmark_mode", "microvqa"),
            ("--benchmark", "microvqa"),
            ("--microvqa_only_pairs",),
            ("--microvqa_only",),
        ),
        "medxpertqa_mm": (
            ("--benchmark_mode", "medxpertqa_mm"),
            ("--benchmark", "medxpertqa_mm"),
            ("--medxpert_only_pairs",),
            ("--medxpertqa_only_pairs",),
            ("--medxpertqa_only",),
        ),
    },
    "all_examples": {
        "vqa_rad": (
            ("--benchmark_mode", "vqa_rad"),
            ("--benchmark", "vqa_rad"),
            ("--vqa_only_examples",),
            ("--vqa_only",),
        ),
        "mmmu_pro": (
            ("--benchmark_mode", "mmmu_pro"),
            ("--benchmark", "mmmu_pro"),
            ("--mmmu_only_examples",),
            ("--mmmu_only",),
        ),
        "microvqa": (
            ("--benchmark_mode", "microvqa"),
            ("--benchmark", "microvqa"),
            ("--microvqa_only_examples",),
            ("--microvqa_only",),
        ),
        "medxpertqa_mm": (
            ("--benchmark_mode", "medxpertqa_mm"),
            ("--benchmark", "medxpertqa_mm"),
            ("--medxpert_only_examples",),
            ("--medxpertqa_only_examples",),
            ("--medxpertqa_only",),
        ),
    },
}


@dataclass
class StageConfig:
    key: str
    family: str
    script_path: Path
    extra_args: List[str]


@dataclass
class JobSpec:
    stage_key: str
    mode: str
    command: List[str]
    save_dir: Path
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full probe experiment stages with up to 2 concurrent GPU jobs (default dry-run). "
            "Uses one --vlm selection for every stage."
        )
    )
    parser.add_argument("--vlm", type=str, choices=SUPPORTED_VLMS, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually execute jobs. Default is dry-run.")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    parser.add_argument("--repo_root", type=str, default=".")
    parser.add_argument("--experiment_root", type=str, default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--run_name_prefix", type=str, default="full_probe")
    parser.add_argument(
        "--gpus",
        type=str,
        default=",".join(DEFAULT_GPU_IDS),
        help="Comma-separated visible GPU ids. Default is hard-pinned to 6,7.",
    )
    parser.add_argument("--poll_interval_sec", type=float, default=10.0)
    parser.add_argument(
        "--benchmark_modes",
        type=str,
        default=",".join(DEFAULT_BENCHMARK_MODES),
        help="Comma-separated benchmark modes. Example: vqa_rad,mmmu_pro,medxpertqa_mm",
    )
    parser.add_argument(
        "--include_all_mode",
        dest="include_all_mode",
        action="store_true",
        default=True,
        help="Also run each stage with all-benchmark mode enabled.",
    )
    parser.add_argument(
        "--no_include_all_mode",
        dest="include_all_mode",
        action="store_false",
        help="Disable all-benchmark mode runs.",
    )
    parser.add_argument(
        "--strict_mode_flags",
        action="store_true",
        default=True,
        help="Fail if a script does not expose recognizable benchmark-mode flags.",
    )
    parser.add_argument(
        "--no_strict_mode_flags",
        dest="strict_mode_flags",
        action="store_false",
        help="Skip unsupported benchmark modes for scripts that lack flags.",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Continue executing remaining jobs after a failure.",
    )
    parser.add_argument(
        "--extra_args",
        action="append",
        default=[],
        help="Additional args appended to every job command (repeatable).",
    )
    parser.add_argument(
        "--stage_extra_args_json",
        type=str,
        default="",
        help=(
            "Optional JSON file mapping stage key to extra args. "
            "Value can be a list of arg tokens or a shell-style arg string."
        ),
    )
    parser.add_argument(
        "--analysis_script",
        type=str,
        default="scripts/analysis/analyze_full_probe_results.py",
        help="Script to run as the final analysis stage over the experiment run directory.",
    )
    parser.add_argument(
        "--analysis_output_dir_name",
        type=str,
        default="analysis",
        help="Subdirectory name under run root where final analysis artifacts are written.",
    )

    for stage_key in SCRIPT_CANDIDATES:
        parser.add_argument(
            f"--{stage_key}_script",
            type=str,
            default="",
            help=f"Override script path for stage '{stage_key}'.",
        )
    return parser.parse_args()


def _parse_extra_args(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not str(v).strip():
            continue
        out.extend(shlex.split(str(v)))
    return out


def _load_stage_extra_args(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("--stage_extra_args_json must contain an object keyed by stage key.")
    out: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            out[str(key)] = shlex.split(value)
        elif isinstance(value, list):
            out[str(key)] = [str(x) for x in value]
        else:
            raise ValueError(f"Invalid extra args format for stage '{key}'. Expected string or list.")
    return out


def _parse_modes(spec: str, include_all_mode: bool) -> List[str]:
    modes = [m.strip() for m in spec.split(",") if m.strip()]
    deduped = []
    seen = set()
    for m in modes:
        if m not in seen:
            deduped.append(m)
            seen.add(m)
    if include_all_mode and "all" not in seen:
        deduped.append("all")
    return deduped


def _parse_gpu_ids(spec: str) -> List[str]:
    gpus = [g.strip() for g in spec.split(",") if g.strip()]
    if not gpus:
        raise ValueError("At least one GPU id is required in --gpus.")
    return gpus


def _resolve_script(repo_root: Path, override: str, candidates: Sequence[str]) -> Path:
    if override:
        p = Path(override)
        return p if p.is_absolute() else (repo_root / p)
    for name in candidates:
        p = repo_root / name
        if p.exists():
            return p
    return repo_root / candidates[0]


def _extract_long_flags(script_path: Path) -> set:
    text = script_path.read_text(encoding="utf-8")
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*", text))


def _extract_local_imported_flags(script_path: Path) -> set:
    """Best-effort static fallback for thin wrapper scripts.

    If a wrapper defines args via an imported local module, collect flags from
    those sibling modules without importing/executing Python.
    """
    text = script_path.read_text(encoding="utf-8")
    imported_modules = set(re.findall(r"^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", text, flags=re.MULTILINE))
    imported_modules |= set(
        re.findall(r"^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+import\b", text, flags=re.MULTILINE)
    )

    out = set()
    for module_name in imported_modules:
        local_module_path = script_path.parent / f"{module_name}.py"
        if local_module_path.exists():
            try:
                out |= _extract_long_flags(local_module_path)
            except Exception:
                continue
    return out


def _extract_long_flags_with_help(script_path: Path, python_bin: str) -> set:
    script_flags = _extract_long_flags(script_path)
    # Trust direct --help output first when available; this reflects the actual parser.
    # Fallback to static imported-module scanning only if --help cannot be inspected.
    try:
        proc = subprocess.run(
            [str(python_bin), str(script_path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=20,
            check=False,
        )
        help_text = str(proc.stdout or "")
        help_flags = set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9_-]*", help_text))
        if help_flags:
            return script_flags | help_flags
    except Exception:
        pass
    return script_flags | _extract_local_imported_flags(script_path)


def _pick_mode_args(
    script_path: Path,
    family: str,
    mode: str,
    strict_mode_flags: bool,
    known_flags: Optional[set] = None,
) -> Optional[List[str]]:
    if mode == "all":
        return []

    flag_candidates = MODE_FLAG_CANDIDATES.get(family, {}).get(mode, ())
    if not flag_candidates:
        if strict_mode_flags:
            raise ValueError(f"No mode flag candidates configured for family='{family}', mode='{mode}'.")
        return None

    if known_flags is None:
        known_flags = _extract_long_flags(script_path)
    for candidate in flag_candidates:
        option_tokens = [tok for tok in candidate if tok.startswith("--")]
        if all(tok in known_flags for tok in option_tokens):
            return list(candidate)

    if strict_mode_flags:
        raise ValueError(
            f"Could not infer benchmark flag for mode='{mode}' in script={script_path}. "
            f"Known options looked for: {flag_candidates}"
        )
    return None


def _build_stage_configs(
    repo_root: Path,
    stage_extra_args: Dict[str, List[str]],
    args: argparse.Namespace,
) -> List[StageConfig]:
    ordered = [
        ("logreg_contrastive", "contrastive"),
        ("logreg_all_examples", "all_examples"),
        ("mlp_contrastive", "contrastive"),
        ("mlp_all_examples", "all_examples"),
        ("concat_contrastive", "contrastive"),
        ("concat_all_examples", "all_examples"),
        ("diff_contrastive", "contrastive"),
        ("diff_all_examples", "all_examples"),
    ]

    stages: List[StageConfig] = []
    for stage_key, family in ordered:
        override = getattr(args, f"{stage_key}_script")
        script_path = _resolve_script(repo_root=repo_root, override=override, candidates=SCRIPT_CANDIDATES[stage_key])
        if not script_path.exists():
            raise FileNotFoundError(
                f"Could not find script for stage '{stage_key}'. "
                f"Tried override='{override}' and candidates={SCRIPT_CANDIDATES[stage_key]}"
            )
        stages.append(
            StageConfig(
                key=stage_key,
                family=family,
                script_path=script_path,
                extra_args=list(stage_extra_args.get(stage_key, [])),
            )
        )
    return stages


def _command_to_str(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _build_job_plan(
    stages: Sequence[StageConfig],
    modes: Sequence[str],
    args: argparse.Namespace,
    run_root: Path,
) -> List[List[JobSpec]]:
    common_extra_args = _parse_extra_args(args.extra_args)
    plan: List[List[JobSpec]] = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    for stage_idx, stage in enumerate(stages):
        stage_dir = run_root / f"{stage_idx + 1:02d}_{stage.key}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_jobs: List[JobSpec] = []

        supports_run_name = "--run_name" in _extract_long_flags(stage.script_path)
        known_flags = _extract_long_flags_with_help(stage.script_path, str(args.python_bin))
        supports_save_dir = "--save_dir" in known_flags
        supports_vlm = "--vlm" in known_flags
        supports_features_cache_path = "--features_cache_path" in known_flags
        supports_heldout_eval_all_features = "--heldout_eval_all_features" in known_flags

        if not supports_save_dir:
            raise ValueError(f"Script does not expose --save_dir: {stage.script_path}")
        if not supports_vlm:
            raise ValueError(f"Script does not expose --vlm: {stage.script_path}")

        for mode_idx, mode in enumerate(modes):
            # GLM contrastive artifacts are intentionally sparse outside vqa_rad.
            # Keep all-examples stage coverage unchanged; constrain only contrastive stages.
            if (
                str(args.vlm) == "glm_4_6v_flash"
                and str(stage.family) == "contrastive"
                and str(mode) != "vqa_rad"
            ):
                continue
            mode_args = _pick_mode_args(
                script_path=stage.script_path,
                family=stage.family,
                mode=mode,
                strict_mode_flags=bool(args.strict_mode_flags),
                known_flags=known_flags,
            )
            if mode_args is None:
                continue

            run_tag = f"{stage_idx + 1:02d}_{mode_idx + 1:02d}_{stage.key}_{mode}_{args.vlm}_{timestamp}"
            save_dir = stage_dir / run_tag
            log_path = stage_dir / f"{run_tag}.log"

            cmd = [
                str(args.python_bin),
                str(stage.script_path),
                "--vlm",
                str(args.vlm),
                "--save_dir",
                str(save_dir),
            ]
            if supports_run_name:
                cmd.extend(["--run_name", run_tag])
            if supports_features_cache_path:
                cmd.extend(["--features_cache_path", str(save_dir / "features_cache.pt")])
            if stage.key == "logreg_contrastive" and supports_heldout_eval_all_features:
                cmd.append("--heldout_eval_all_features")
            cmd.extend(mode_args)
            cmd.extend(common_extra_args)
            cmd.extend(stage.extra_args)

            stage_jobs.append(
                JobSpec(
                    stage_key=stage.key,
                    mode=mode,
                    command=cmd,
                    save_dir=save_dir,
                    log_path=log_path,
                )
            )
        plan.append(stage_jobs)
    return plan


def _write_plan_file(plan: Sequence[Sequence[JobSpec]], run_root: Path, args: argparse.Namespace) -> Path:
    payload = {
        "vlm": args.vlm,
        "execute": bool(args.execute),
        "gpus": args.gpus,
        "poll_interval_sec": float(args.poll_interval_sec),
        "analysis_script": str(args.analysis_script),
        "analysis_output_dir_name": str(args.analysis_output_dir_name),
        "stages": [],
    }
    for stage_jobs in plan:
        if not stage_jobs:
            continue
        stage_payload = {
            "stage_key": stage_jobs[0].stage_key,
            "jobs": [],
        }
        for job in stage_jobs:
            stage_payload["jobs"].append(
                {
                    "mode": job.mode,
                    "save_dir": str(job.save_dir),
                    "log_path": str(job.log_path),
                    "command": job.command,
                }
            )
        payload["stages"].append(stage_payload)

    path = run_root / "plan.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def _run_stage(
    stage_jobs: Sequence[JobSpec],
    repo_root: Path,
    gpu_ids: Sequence[str],
    poll_interval_sec: float,
    continue_on_error: bool,
    start_gpu_idx: int,
) -> Tuple[List[Dict], bool, int]:
    pending = list(stage_jobs)
    running: Dict[str, Dict] = {}
    stage_results: List[Dict] = []
    failed = False
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty.")
    next_gpu_idx = int(start_gpu_idx) % len(gpu_ids)

    def _pick_next_available_gpu() -> Tuple[Optional[str], int]:
        nonlocal next_gpu_idx
        for offset in range(len(gpu_ids)):
            idx = (next_gpu_idx + offset) % len(gpu_ids)
            gpu = str(gpu_ids[idx])
            if gpu not in running:
                next_idx = (idx + 1) % len(gpu_ids)
                return gpu, next_idx
        return None, next_gpu_idx

    while pending or running:
        while pending:
            gpu, proposed_next_idx = _pick_next_available_gpu()
            if gpu is None:
                break
            next_gpu_idx = proposed_next_idx
            job = pending.pop(0)
            job.save_dir.mkdir(parents=True, exist_ok=True)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(job.log_path, "w", encoding="utf-8", buffering=1)
            log_handle.write(f"stage={job.stage_key}\n")
            log_handle.write(f"mode={job.mode}\n")
            log_handle.write(f"gpu={gpu}\n")
            log_handle.write(f"command={_command_to_str(job.command)}\n")
            log_handle.write(f"start_utc={datetime.utcnow().isoformat()}Z\n\n")
            log_handle.flush()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                job.command,
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = {
                "job": job,
                "proc": proc,
                "log_handle": log_handle,
                "start_ts": time.time(),
            }

        if not running:
            break

        time.sleep(max(0.1, poll_interval_sec))
        completed_gpus = []
        for gpu, info in running.items():
            proc = info["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            log_handle = info["log_handle"]
            elapsed = time.time() - float(info["start_ts"])
            log_handle.write(f"\nend_utc={datetime.utcnow().isoformat()}Z\n")
            log_handle.write(f"elapsed_sec={elapsed:.2f}\n")
            log_handle.write(f"return_code={rc}\n")
            log_handle.close()

            job = info["job"]
            stage_results.append(
                {
                    "stage_key": job.stage_key,
                    "mode": job.mode,
                    "gpu": gpu,
                    "command": job.command,
                    "save_dir": str(job.save_dir),
                    "log_path": str(job.log_path),
                    "elapsed_sec": elapsed,
                    "return_code": int(rc),
                }
            )
            if rc != 0:
                failed = True
            completed_gpus.append(gpu)

        for gpu in completed_gpus:
            running.pop(gpu, None)

        if failed and not continue_on_error:
            for info in running.values():
                proc = info["proc"]
                log_handle = info["log_handle"]
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log_handle.write("\nterminated_due_to_prior_failure=1\n")
                log_handle.close()
            running.clear()
            break

    return stage_results, failed, next_gpu_idx


def _run_plan_with_stage_overlap(
    plan: Sequence[Sequence[JobSpec]],
    repo_root: Path,
    gpu_ids: Sequence[str],
    poll_interval_sec: float,
    continue_on_error: bool,
    start_gpu_idx: int,
) -> Tuple[List[Dict], bool, int]:
    """Run staged jobs with limited overlap between adjacent stages.

    Rule: when a stage has exactly one running job left and no pending jobs,
    the next stage is allowed to start on a free GPU immediately.
    """
    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty.")

    pending_by_stage: List[List[JobSpec]] = [list(stage_jobs) for stage_jobs in plan]
    running_counts: List[int] = [0 for _ in pending_by_stage]
    running: Dict[str, Dict[str, Any]] = {}
    all_results: List[Dict] = []
    failed = False
    next_gpu_idx = int(start_gpu_idx) % len(gpu_ids)

    def _pick_next_available_gpu() -> Tuple[Optional[str], int]:
        nonlocal next_gpu_idx
        for offset in range(len(gpu_ids)):
            idx = (next_gpu_idx + offset) % len(gpu_ids)
            gpu = str(gpu_ids[idx])
            if gpu not in running:
                next_idx = (idx + 1) % len(gpu_ids)
                return gpu, next_idx
        return None, next_gpu_idx

    def _stage_completed(stage_idx: int) -> bool:
        return (not pending_by_stage[stage_idx]) and (running_counts[stage_idx] == 0)

    def _all_prior_stages_completed(stage_idx: int) -> bool:
        for prior_idx in range(stage_idx):
            if not _stage_completed(prior_idx):
                return False
        return True

    def _stage_eligible(stage_idx: int) -> bool:
        if not pending_by_stage[stage_idx]:
            return False
        if stage_idx == 0:
            return True
        if _all_prior_stages_completed(stage_idx):
            return True
        prev_idx = stage_idx - 1
        return (
            _all_prior_stages_completed(prev_idx)
            and (not pending_by_stage[prev_idx])
            and (running_counts[prev_idx] == 1)
        )

    def _pick_next_stage_idx() -> Optional[int]:
        for stage_idx in range(len(pending_by_stage)):
            if _stage_eligible(stage_idx):
                return stage_idx
        return None

    while any(pending_by_stage) or running:
        while True:
            gpu, proposed_next_idx = _pick_next_available_gpu()
            if gpu is None:
                break
            stage_idx = _pick_next_stage_idx()
            if stage_idx is None:
                break
            next_gpu_idx = proposed_next_idx

            job = pending_by_stage[stage_idx].pop(0)
            job.save_dir.mkdir(parents=True, exist_ok=True)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(job.log_path, "w", encoding="utf-8", buffering=1)
            log_handle.write(f"stage={job.stage_key}\n")
            log_handle.write(f"mode={job.mode}\n")
            log_handle.write(f"gpu={gpu}\n")
            log_handle.write(f"command={_command_to_str(job.command)}\n")
            log_handle.write(f"start_utc={datetime.utcnow().isoformat()}Z\n\n")
            log_handle.flush()

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                job.command,
                cwd=str(repo_root),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = {
                "job": job,
                "stage_idx": int(stage_idx),
                "proc": proc,
                "log_handle": log_handle,
                "start_ts": time.time(),
            }
            running_counts[stage_idx] += 1

        if not running:
            if any(pending_by_stage):
                raise RuntimeError(
                    "Scheduler deadlock: pending jobs remain but none are running or eligible to start."
                )
            break

        time.sleep(max(0.1, poll_interval_sec))
        completed_gpus: List[str] = []
        for gpu, info in list(running.items()):
            proc = info["proc"]
            rc = proc.poll()
            if rc is None:
                continue

            log_handle = info["log_handle"]
            elapsed = time.time() - float(info["start_ts"])
            log_handle.write(f"\nend_utc={datetime.utcnow().isoformat()}Z\n")
            log_handle.write(f"elapsed_sec={elapsed:.2f}\n")
            log_handle.write(f"return_code={rc}\n")
            log_handle.close()

            job = info["job"]
            stage_idx = int(info["stage_idx"])
            running_counts[stage_idx] = max(0, int(running_counts[stage_idx]) - 1)

            all_results.append(
                {
                    "stage_key": job.stage_key,
                    "mode": job.mode,
                    "gpu": gpu,
                    "command": job.command,
                    "save_dir": str(job.save_dir),
                    "log_path": str(job.log_path),
                    "elapsed_sec": elapsed,
                    "return_code": int(rc),
                }
            )
            if rc != 0:
                failed = True
            completed_gpus.append(gpu)

        for gpu in completed_gpus:
            running.pop(gpu, None)

        if failed and not continue_on_error:
            for info in running.values():
                proc = info["proc"]
                log_handle = info["log_handle"]
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                log_handle.write("\nterminated_due_to_prior_failure=1\n")
                log_handle.close()
            running.clear()
            break

    return all_results, failed, next_gpu_idx


def _run_analysis_stage(
    args: argparse.Namespace,
    repo_root: Path,
    run_root: Path,
    gpu_ids: Sequence[str],
    gpu_index_hint: int,
) -> Dict:
    analysis_script = Path(str(args.analysis_script))
    if not analysis_script.is_absolute():
        analysis_script = repo_root / analysis_script
    if not analysis_script.exists():
        raise FileNotFoundError(f"Analysis script not found: {analysis_script}")

    analysis_dir = run_root / str(args.analysis_output_dir_name)
    analysis_log = run_root / "analysis.log"
    cmd = [
        str(args.python_bin),
        str(analysis_script),
        "--experiment_run_root",
        str(run_root),
        "--output_dir",
        str(analysis_dir),
    ]

    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty for analysis stage.")
    chosen_gpu = str(gpu_ids[int(gpu_index_hint) % len(gpu_ids)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = chosen_gpu
    env["PYTHONUNBUFFERED"] = "1"
    start = time.time()
    with open(analysis_log, "w", encoding="utf-8", buffering=1) as fh:
        fh.write(f"command={_command_to_str(cmd)}\n")
        fh.write(f"start_utc={datetime.utcnow().isoformat()}Z\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        rc = int(proc.wait())
        elapsed = time.time() - start
        fh.write(f"end_utc={datetime.utcnow().isoformat()}Z\n")
        fh.write(f"elapsed_sec={elapsed:.2f}\n")
        fh.write(f"return_code={rc}\n")

    return {
        "stage_key": "analysis",
        "mode": "all",
        "gpu": chosen_gpu,
        "command": cmd,
        "save_dir": str(analysis_dir),
        "log_path": str(analysis_log),
        "elapsed_sec": float(elapsed),
        "return_code": rc,
    }


def _validate_qwen_preextracted_artifacts(vlm: str) -> None:
    if str(vlm) != "qwen3_vl_32b_instruct":
        return
    missing = [str(p) for p in QWEN_PREEXTRACTED_REQUIRED_PATHS if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Qwen full-run requires pre-extracted activation caches. Missing:\n"
            + "\n".join(missing)
            + "\nGenerate with: python scripts/data/activation_extraction/extract_qwen_activations.py"
        )


def _validate_glm_preextracted_artifacts(vlm: str) -> None:
    if str(vlm) != "glm_4_6v_flash":
        return
    missing = [str(p) for p in GLM_PREEXTRACTED_REQUIRED_PATHS if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "GLM full-run requires pre-extracted activation caches. Missing:\n"
            + "\n".join(missing)
            + "\nGenerate with: python scripts/data/activation_extraction/extract_glm_activations.py"
        )


def _requests_additional_preextract_cache(
    common_extra_args: Sequence[str],
    stage_extra_args: Dict[str, List[str]],
) -> bool:
    all_tokens = list(common_extra_args)
    for toks in stage_extra_args.values():
        all_tokens.extend([str(x) for x in toks])

    if "--use_additional_feature_preextract_cache" in all_tokens:
        return True
    if "--no_use_additional_feature_preextract_cache" in all_tokens:
        return False

    includes_additional = (
        ("--include_attention_probes" in all_tokens)
        or ("--include_mlp_probes" in all_tokens)
        or ("--include_additional_attention_mlp_probes" in all_tokens)
    )
    excludes_residual = ("--no_include_residual_probes" in all_tokens)
    return bool(includes_additional and excludes_residual)


def _validate_qwen_preextracted_additional_artifacts(vlm: str) -> None:
    if str(vlm) != "qwen3_vl_32b_instruct":
        return
    missing = [str(p) for p in QWEN_PREEXTRACTED_ADDITIONAL_REQUIRED_PATHS if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Qwen full-run in additional-feature-cache mode requires dedicated pre-extracted caches. Missing:\n"
            + "\n".join(missing)
            + "\nGenerate with: python scripts/data/activation_extraction/extract_qwen_activations.py --extract_additional_feature_caches"
        )


def _validate_glm_preextracted_additional_artifacts(vlm: str) -> None:
    if str(vlm) != "glm_4_6v_flash":
        return
    missing = [str(p) for p in GLM_PREEXTRACTED_ADDITIONAL_REQUIRED_PATHS if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "GLM full-run in additional-feature-cache mode requires dedicated pre-extracted caches. Missing:\n"
            + "\n".join(missing)
            + "\nGenerate with: python scripts/data/activation_extraction/extract_glm_activations.py --extract_additional_feature_caches"
        )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    experiment_root = Path(args.experiment_root).resolve()
    experiment_root.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = experiment_root / f"{args.run_name_prefix}_{args.vlm}_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)
    stage_extra_args = _load_stage_extra_args(args.stage_extra_args_json)
    common_extra_args = _parse_extra_args(args.extra_args)
    if bool(args.execute):
        use_additional_preextract_cache = _requests_additional_preextract_cache(
            common_extra_args=common_extra_args,
            stage_extra_args=stage_extra_args,
        )
        if use_additional_preextract_cache:
            _validate_qwen_preextracted_additional_artifacts(vlm=str(args.vlm))
            _validate_glm_preextracted_additional_artifacts(vlm=str(args.vlm))
        else:
            _validate_qwen_preextracted_artifacts(vlm=str(args.vlm))
            _validate_glm_preextracted_artifacts(vlm=str(args.vlm))

    gpu_ids = _parse_gpu_ids(args.gpus)
    modes = _parse_modes(spec=args.benchmark_modes, include_all_mode=bool(args.include_all_mode))
    if str(args.vlm) == "glm_4_6v_flash":
        filtered_modes = [m for m in modes if m not in GLM_SKIPPED_BENCHMARK_MODES]
        skipped = [m for m in modes if m in GLM_SKIPPED_BENCHMARK_MODES]
        if skipped:
            print(f"GLM mode filter (all stages): skipping unsupported benchmark-only modes: {', '.join(skipped)}")
        print("GLM mode filter (contrastive stages): forcing vqa_rad-only runs by policy.")
        modes = filtered_modes
        if not modes:
            raise ValueError(
                "No benchmark modes remain after filtering GLM-unsupported benchmark-only modes. "
                "Include at least one of: vqa_rad, mmmu_pro, all."
            )
    stages = _build_stage_configs(repo_root=repo_root, stage_extra_args=stage_extra_args, args=args)
    plan = _build_job_plan(stages=stages, modes=modes, args=args, run_root=run_root)
    plan_path = _write_plan_file(plan=plan, run_root=run_root, args=args)
    analysis_script = Path(str(args.analysis_script))
    if not analysis_script.is_absolute():
        analysis_script = repo_root / analysis_script
    analysis_cmd = [
        str(args.python_bin),
        str(analysis_script),
        "--experiment_run_root",
        str(run_root),
        "--output_dir",
        str(run_root / str(args.analysis_output_dir_name)),
    ]

    if not args.execute:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "run_root": str(run_root),
                    "plan_path": str(plan_path),
                    "analysis_command": analysis_cmd,
                },
                indent=2,
            )
        )
        for stage_jobs in plan:
            if not stage_jobs:
                continue
            print(f"\n[{stage_jobs[0].stage_key}]")
            for job in stage_jobs:
                print(_command_to_str(job.command))
        print("\n[analysis]")
        print(_command_to_str(analysis_cmd))
        return

    print("Running staged jobs with overlap scheduling enabled")
    all_results, any_failure, gpu_round_robin_idx = _run_plan_with_stage_overlap(
        plan=plan,
        repo_root=repo_root,
        gpu_ids=gpu_ids,
        poll_interval_sec=float(args.poll_interval_sec),
        continue_on_error=bool(args.continue_on_error),
        start_gpu_idx=0,
    )

    analysis_result = None
    if (not any_failure) or bool(args.continue_on_error):
        print("Running final analysis stage")
        analysis_result = _run_analysis_stage(
            args=args,
            repo_root=repo_root,
            run_root=run_root,
            gpu_ids=gpu_ids,
            gpu_index_hint=gpu_round_robin_idx,
        )
        all_results.append(analysis_result)
        if int(analysis_result.get("return_code", 1)) != 0:
            any_failure = True

    summary = {
        "run_root": str(run_root),
        "vlm": args.vlm,
        "execute": True,
        "plan_path": str(plan_path),
        "analysis_command": analysis_cmd,
        "analysis_result": analysis_result,
        "any_failure": any_failure,
        "results": all_results,
    }
    summary_path = run_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({"summary_path": str(summary_path), "any_failure": any_failure}, indent=2))
    if any_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
