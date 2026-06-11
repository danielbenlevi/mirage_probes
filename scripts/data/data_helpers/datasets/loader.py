"""Dataset loader for unified access to local parquet datasets."""

import base64
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.data.data_helpers.config import DATA_DIR
from .base import DataItem


class DatasetLoader:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self._cache: Dict[str, pd.DataFrame] = {}

    def list_available(self) -> List[str]:
        return [p.stem for p in self.data_dir.glob("*.parquet")]

    def load(self, dataset_name: str, use_cache: bool = True) -> pd.DataFrame:
        if use_cache and dataset_name in self._cache:
            return self._cache[dataset_name]
        parquet_path = self.data_dir / f"{dataset_name}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {parquet_path}. Available: {self.list_available()}"
            )
        df = pd.read_parquet(parquet_path)
        if use_cache:
            self._cache[dataset_name] = df
        return df

    def get_items(self, dataset_name: str) -> List[DataItem]:
        df = self.load(dataset_name)
        items = []
        for _, row in df.iterrows():
            images = row["images"]
            if images is None or (isinstance(images, str) and images == ""):
                images = []
            elif isinstance(images, str):
                try:
                    b64_list = json.loads(images)
                    images = [base64.b64decode(b64) for b64 in b64_list]
                except (json.JSONDecodeError, ValueError):
                    images = []
            else:
                if hasattr(images, "tolist"):
                    images = images.tolist()
                elif not isinstance(images, list):
                    images = [images]
            items.append(
                DataItem(
                    unique_id=row["unique_id"],
                    question_id=row["question_id"],
                    category=row["category"] if pd.notna(row["category"]) else "",
                    question=row["question"],
                    options=row["options"] if pd.notna(row["options"]) else "",
                    images=images,
                    ground_truth=row["ground_truth"],
                )
            )
        return items
