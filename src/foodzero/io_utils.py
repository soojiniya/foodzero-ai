from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


KOREAN_ENCODINGS = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)


def detect_csv_encoding(path: Path) -> str:
    sample = path.read_bytes()[:200_000]
    for enc in KOREAN_ENCODINGS:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def sniff_csv_dialect(path: Path, encoding: str) -> csv.Dialect:
    text = path.read_text(encoding=encoding, errors="replace")[:50_000]
    try:
        return csv.Sniffer().sniff(text, delimiters=",\t|;")
    except csv.Error:
        return csv.get_dialect("excel")


def read_csv_safely(path: Path, nrows: int | None = None, dtype: Any = str) -> tuple[pd.DataFrame, str]:
    encoding = detect_csv_encoding(path)
    dialect = sniff_csv_dialect(path, encoding)
    df = pd.read_csv(path, encoding=encoding, sep=dialect.delimiter, nrows=nrows, dtype=dtype)
    df.columns = [normalize_header(c) for c in df.columns]
    return df, encoding


def read_table(path: Path, nrows: int | None = None, sheet_name: str | int | None = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df, encoding = read_csv_safely(path, nrows=nrows)
        return df, {"kind": "csv", "encoding": encoding}
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name, nrows=nrows, dtype=str)
        df.columns = [normalize_header(c) for c in df.columns]
        return df, {"kind": "excel", "sheet_name": sheet_name}
    raise ValueError(f"Unsupported table file: {path}")


def normalize_header(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text.replace("\ufeff", "")).strip()


def normalize_name(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False),
        errors="coerce",
    )

