#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class VLMConfig:
    key: str
    hf_repo_id: str
    default_vllm_model_name: str
    local_subdir: str

    @property
    def default_local_path(self) -> Path:
        return DEFAULT_MODELS_DIR / self.local_subdir


VLM_REGISTRY: Dict[str, VLMConfig] = {
    "ovis": VLMConfig(
        key="ovis",
        hf_repo_id="AIDC-AI/Ovis2.5-2B",
        default_vllm_model_name="ovis2_5_2b",
        local_subdir="ovis",
    ),
    "qwen3_vl_32b_instruct": VLMConfig(
        key="qwen3_vl_32b_instruct",
        hf_repo_id="Qwen/Qwen3-VL-32B-Instruct",
        default_vllm_model_name="qwen3_vl_32b_instruct",
        local_subdir="qwen3_vl_32b_instruct",
    ),
    "glm_4_6v_flash": VLMConfig(
        key="glm_4_6v_flash",
        hf_repo_id="zai-org/GLM-4.6V-Flash",
        default_vllm_model_name="glm_4_6v_flash",
        local_subdir="glm_4_6v_flash",
    ),
}

VLM_CHOICES = tuple(sorted(VLM_REGISTRY.keys()))
DEFAULT_VLM_KEY = "ovis"


def resolve_vlm_config(vlm_key: str) -> VLMConfig:
    key = str(vlm_key).strip().lower()
    if key not in VLM_REGISTRY:
        raise ValueError(f"Unknown --vlm '{vlm_key}'. Expected one of: {VLM_CHOICES}")
    return VLM_REGISTRY[key]


def resolve_vllm_model_name(
    vlm_key: str,
    override_model_name: Optional[str] = None,
) -> str:
    override = str(override_model_name or "").strip()
    if override:
        return override
    return resolve_vlm_config(vlm_key).default_vllm_model_name
