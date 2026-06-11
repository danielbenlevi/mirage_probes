"""MedXpertQA-MM dataset converter."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES
from scripts.data.data_helpers.datasets.base import BaseDataset, DatasetRegistry


@DatasetRegistry.register("medxpertqa_mm")
class MedXpertQAMMConverter(BaseDataset):
    DATASET_NAME = "medxpertqa_mm"

    def __init__(self, source_config: Optional[Dict[str, Any]] = None):
        super().__init__(source_config)
        if not self.source_config:
            self.source_config = ORIGINAL_DATA_SOURCES.get("medxpertqa_mm", {})

    def convert(self) -> pd.DataFrame:
        data_dir = Path(self.source_config.get("data_dir", ""))
        image_dir = Path(self.source_config.get("image_dir", data_dir / "images"))
        data_file = data_dir / "test.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        items = []
        idx = 0
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                options_dict = item.get("options", {})
                if options_dict:
                    options_parts = []
                    for letter in sorted(options_dict.keys()):
                        opt_text = options_dict[letter]
                        if not opt_text.startswith(f"{letter}.") and not opt_text.startswith(f"{letter} "):
                            options_parts.append(f"{letter}. {opt_text}")
                        else:
                            options_parts.append(opt_text)
                    options = "\n".join(options_parts)
                else:
                    options = ""
                images = []
                for img_path in item.get("images", []) or []:
                    full_path = image_dir / img_path
                    if full_path.exists():
                        images.append(full_path.read_bytes())
                items.append(
                    {
                        "unique_id": f"medxpertqa_mm_{idx}",
                        "question_id": str(item.get("id", idx)),
                        "category": item.get("medical_task", item.get("body_system", "")),
                        "question": item.get("question", "").strip(),
                        "options": options,
                        "images": images,
                        "ground_truth": item.get("label", "").strip().upper(),
                    }
                )
                idx += 1
        return pd.DataFrame(items)
