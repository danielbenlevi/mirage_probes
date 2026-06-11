"""VQA-RAD dataset converter."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES
from scripts.data.data_helpers.datasets.base import BaseDataset, DatasetRegistry


@DatasetRegistry.register("vqa_rad")
class VQARadConverter(BaseDataset):
    DATASET_NAME = "vqa_rad"

    def __init__(self, source_config: Optional[Dict[str, Any]] = None):
        super().__init__(source_config)
        if not self.source_config:
            self.source_config = ORIGINAL_DATA_SOURCES.get("vqa_rad", {})

    def convert(self) -> pd.DataFrame:
        data_dir = Path(self.source_config.get("data_dir", ""))
        image_dir = Path(self.source_config.get("image_dir", data_dir / "images"))
        splits_file = data_dir / "vqa_rad_balanced_split_and_human_eval_inclusions.tsv"
        data_file = data_dir / "VQA_RAD Dataset Public.json"
        if not splits_file.exists():
            raise FileNotFoundError(f"Splits file not found: {splits_file}")
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        splits = pd.read_csv(splits_file, sep="\t")
        test_ids = set(splits[splits["SPLIT_BALANCED"] == "test"].QID_unique.tolist())
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        items = []
        idx = 0
        for item in raw_data:
            qid = item["qid"]
            if qid not in test_ids:
                continue
            if item.get("answer_type") != "CLOSED":
                continue
            images = []
            image_name = item.get("image_name", "")
            if image_name and image_dir.exists():
                image_path = image_dir / image_name
                if image_path.exists():
                    images.append(image_path.read_bytes())
            items.append(
                {
                    "unique_id": f"vqa_rad_{idx}",
                    "question_id": str(qid),
                    "category": item.get("question_type", item.get("answer_type", "")),
                    "question": item["question"].strip(),
                    "options": "",
                    "images": images,
                    "ground_truth": item["answer"].strip(),
                }
            )
            idx += 1
        return pd.DataFrame(items)
