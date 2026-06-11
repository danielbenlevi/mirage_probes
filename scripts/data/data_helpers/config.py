"""Shared data-path configuration."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_ROOT = Path(os.getenv("DATA_ROOT", str(ROOT_DIR / "raw_data")))

DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_ROOT.mkdir(exist_ok=True)

ORIGINAL_DATA_SOURCES = {
    "vqa_rad": {
        "data_dir": str(RAW_DATA_ROOT / "vqa_rad"),
        "image_dir": str(RAW_DATA_ROOT / "vqa_rad" / "images"),
    },
    "microvqa": {
        "data_dir": str(RAW_DATA_ROOT / "MicroVQA"),
    },
    "medxpertqa_mm": {
        "data_dir": str(RAW_DATA_ROOT / "MedXpertQA-MM"),
        "image_dir": str(RAW_DATA_ROOT / "MedXpertQA-MM" / "images"),
    },
    "mmmu_pro": {
        "data_dir": str(RAW_DATA_ROOT / "MMMU-Pro"),
    },
}

PARQUET_SCHEMA = [
    "unique_id",
    "question_id",
    "category",
    "question",
    "options",
    "images",
    "ground_truth",
]
