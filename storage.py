from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models import ImpactRecord


DEFAULT_STORAGE_PATH = Path(__file__).resolve().parent / "data" / "impact_records.jsonl"


class RecordStore:
    def __init__(self, file_path: Path | str | None = None) -> None:
        self.file_path = Path(file_path) if file_path is not None else DEFAULT_STORAGE_PATH
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ImpactRecord) -> None:
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json())
            handle.write("\n")

    def load_all(self) -> list[ImpactRecord]:
        if not self.file_path.exists():
            return []
        records: list[ImpactRecord] = []
        with self.file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                cleaned = line.strip()
                if not cleaned:
                    continue
                records.append(ImpactRecord.model_validate(json.loads(cleaned)))
        return records

    def to_dataframe(self) -> pd.DataFrame:
        records = self.load_all()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([record.model_dump(mode="json") for record in records])