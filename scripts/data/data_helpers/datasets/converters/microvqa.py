"""MicroVQA dataset converter."""

import base64
import glob
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow.parquet as pq

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES
from scripts.data.data_helpers.datasets.base import BaseDataset, DatasetRegistry


@DatasetRegistry.register("microvqa")
class MicroVQAConverter(BaseDataset):
    DATASET_NAME = "microvqa"

    def __init__(self, source_config: Optional[Dict[str, Any]] = None):
        super().__init__(source_config)
        if not self.source_config:
            self.source_config = ORIGINAL_DATA_SOURCES.get("microvqa", {})

    def convert(self) -> pd.DataFrame:
        data_dir = Path(self.source_config.get("data_dir", ""))
        test_files = sorted(glob.glob(str(data_dir / "test-*.parquet")))
        if not test_files:
            raise FileNotFoundError(f"No test parquet files found in {data_dir}")
        items = []
        idx = 0
        for file_path in test_files:
            table = pq.read_table(file_path)
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            for _, row in df.iterrows():
                choices = row["choices"]
                if hasattr(choices, "tolist"):
                    choices = choices.tolist()
                correct_index = int(row["correct_index"])
                choice_letters = "ABCDEFGHIJ"
                options = "\n".join(
                    f"{choice_letters[i]}. {choice}"
                    for i, choice in enumerate(choices)
                    if i < len(choice_letters)
                )
                ground_truth = (
                    choice_letters[correct_index]
                    if correct_index < len(choice_letters)
                    else str(correct_index)
                )
                images_bytes = []
                if "images_list" in row and row["images_list"] is not None:
                    images_list = row["images_list"]
                    if hasattr(images_list, "tolist"):
                        images_list = images_list.tolist()
                    for img_dict in images_list:
                        if isinstance(img_dict, dict) and img_dict.get("bytes"):
                            images_bytes.append(img_dict["bytes"])
                images = (
                    json.dumps([base64.b64encode(img).decode("ascii") for img in images_bytes])
                    if images_bytes
                    else ""
                )
                items.append(
                    {
                        "unique_id": f"microvqa_{idx}",
                        "question_id": str(row.get("key_question", idx)),
                        "category": row.get("task_str", ""),
                        "question": row["question"],
                        "options": options,
                        "images": images,
                        "ground_truth": ground_truth,
                    }
                )
                idx += 1
        return pd.DataFrame(items)
