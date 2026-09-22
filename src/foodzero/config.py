from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
DOCS_DIR = PROJECT_ROOT / "docs"
CONFIG_DIR = PROJECT_ROOT / "config"

FOOD_WASTE_RAW_DIR = RAW_DIR / "food_waste"
POPULATION_RAW_DIR = RAW_DIR / "population"
REFERENCE_RAW_DIR = RAW_DIR / "reference"
WEATHER_RAW_DIR = RAW_DIR / "weather"
WEATHER_INTERIM_DIR = INTERIM_DIR / "weather"

MODEL_DATASET_PATH = PROCESSED_DIR / "foodzero_model_dataset.csv"
ASOS_DAILY_WEATHER_PATH = PROCESSED_DIR / "asos_daily_weather.csv"
SOURCE_INVENTORY_PATH = INTERIM_DIR / "source_inventory.json"
AUDIT_DOC_PATH = DOCS_DIR / "data_audit.md"

KMA_BASE_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService"
KMA_ENDPOINT = "/getWthrDataList"
KMA_START_DATE = "20210101"
KMA_END_DATE = "20240131"
