#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from vlm_model_registry import resolve_vlm_config


def parse_args() -> argparse.Namespace:
    cfg = resolve_vlm_config("qwen3_vl_32b_instruct")
    parser = argparse.ArgumentParser(
        description="Download Qwen3-VL-32B-Instruct into a local models directory."
    )
    parser.add_argument(
        "--destination",
        type=str,
        default=str(cfg.default_local_path),
        help="Local directory for the snapshot.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="",
        help="Optional model revision (branch, tag, or commit).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default="",
        help="Optional Hugging Face token for gated/private access.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = resolve_vlm_config("qwen3_vl_32b_instruct")
    destination = Path(args.destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "repo_id": cfg.hf_repo_id,
        "local_dir": str(destination),
    }
    if str(args.revision).strip():
        kwargs["revision"] = str(args.revision).strip()
    if str(args.token).strip():
        kwargs["token"] = str(args.token).strip()

    snapshot_path = snapshot_download(**kwargs)
    print(
        json.dumps(
            {
                "vlm_key": cfg.key,
                "repo_id": cfg.hf_repo_id,
                "revision": str(args.revision).strip() or None,
                "destination": str(destination),
                "snapshot_path": str(snapshot_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
