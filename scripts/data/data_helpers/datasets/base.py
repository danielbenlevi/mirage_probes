"""Base classes for dataset handling."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from scripts.data.data_helpers.config import DATA_DIR, PARQUET_SCHEMA


@dataclass
class DataItem:
    unique_id: str
    question_id: str
    category: str
    question: str
    options: str
    images: List[bytes]
    ground_truth: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    DATASET_NAME: str = ""

    def __init__(self, source_config: Optional[Dict[str, Any]] = None):
        self.source_config = source_config or {}
        self._data: Optional[pd.DataFrame] = None

    @property
    def parquet_path(self) -> Path:
        return DATA_DIR / f"{self.DATASET_NAME}.parquet"

    @property
    def exists(self) -> bool:
        return self.parquet_path.exists()

    @abstractmethod
    def convert(self) -> pd.DataFrame:
        raise NotImplementedError

    def build(self, force: bool = False) -> Path:
        if self.exists and not force:
            print(f"  {self.DATASET_NAME} parquet already exists")
            return self.parquet_path

        print(f"  Building {self.DATASET_NAME}...")
        df = self.convert()
        for col in PARQUET_SCHEMA:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        df = df[PARQUET_SCHEMA]
        df.to_parquet(self.parquet_path, index=False)
        print(f"  Saved {len(df)} items to {self.parquet_path}")
        return self.parquet_path

    def load(self) -> pd.DataFrame:
        if not self.exists:
            raise FileNotFoundError(
                f"Parquet not found: {self.parquet_path}. Run build() first to create it."
            )
        if self._data is None:
            self._data = pd.read_parquet(self.parquet_path)
        return self._data

    def get_items(self) -> List[DataItem]:
        df = self.load()
        items = []
        for _, row in df.iterrows():
            items.append(
                DataItem(
                    unique_id=row["unique_id"],
                    question_id=row["question_id"],
                    category=row["category"],
                    question=row["question"],
                    options=row["options"],
                    images=row["images"] if row["images"] is not None else [],
                    ground_truth=row["ground_truth"],
                )
            )
        return items


class DatasetRegistry:
    _datasets: Dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dataset_cls: Type[BaseDataset]):
            cls._datasets[name] = dataset_cls
            return dataset_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseDataset]:
        if name not in cls._datasets:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {list(cls._datasets.keys())}"
            )
        return cls._datasets[name]

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls._datasets.keys())
