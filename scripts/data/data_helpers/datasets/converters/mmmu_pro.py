"""MMMU-Pro dataset converter."""

import ast
import glob
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from scripts.data.data_helpers.config import ORIGINAL_DATA_SOURCES
from scripts.data.data_helpers.datasets.base import BaseDataset, DatasetRegistry


@DatasetRegistry.register("mmmu_pro")
class MMMUProConverter(BaseDataset):
    DATASET_NAME = "mmmu_pro"

    def __init__(self, source_config: Optional[Dict[str, Any]] = None, num_options: int = 4):
        super().__init__(source_config)
        if not self.source_config:
            self.source_config = ORIGINAL_DATA_SOURCES.get("mmmu_pro", {})
        self.num_options = num_options

    def convert(self) -> pd.DataFrame:
        data_dir = Path(self.source_config.get("data_dir", ""))
        subdir = data_dir / f"standard-{self.num_options}-options"
        test_files = sorted(glob.glob(str(subdir / "test-*.parquet")))
        if not test_files:
            raise FileNotFoundError(f"No test parquet files found in {subdir}")
        items = []
        idx = 0
        for file_path in test_files:
            df = pd.read_parquet(file_path)
            for _, row in df.iterrows():
                options_raw = row["options"]
                if isinstance(options_raw, str):
                    options_list = ast.literal_eval(options_raw)
                else:
                    options_list = list(options_raw)
                choice_letters = "ABCDEFGHIJ"
                options = "\n".join(
                    f"{choice_letters[i]}. {opt}"
                    for i, opt in enumerate(options_list)
                    if i < len(choice_letters)
                )
                images = []
                for i in range(1, 8):
                    img_col = f"image_{i}"
                    if img_col in row and row[img_col] is not None:
                        img = row[img_col]
                        if isinstance(img, dict) and img.get("bytes"):
                            images.append(img["bytes"])
                items.append(
                    {
                        "unique_id": f"mmmu_pro_{idx}",
                        "question_id": str(row.get("id", idx)),
                        "category": row.get("subject", ""),
                        "question": row["question"],
                        "options": options,
                        "images": images,
                        "ground_truth": row["answer"].strip().upper(),
                    }
                )
                idx += 1
        return pd.DataFrame(items)
