from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ASOS_DAILY_WEATHER_PATH, CONFIG_DIR, KMA_BASE_URL, KMA_ENDPOINT, KMA_END_DATE, KMA_START_DATE, WEATHER_RAW_DIR
from .io_utils import ensure_parent


WEATHER_COLUMNS = ["tm", "stnId", "stnNm", "avgTa", "maxTa", "minTa", "sumRn", "avgRhm"]
MAPPING_COLUMNS = ["municipality", "station_id", "station_name", "mapping_status", "evidence", "source_url", "notes"]
VALID_MAPPING_STATUSES = {"validated", "confirmed", "proposed", "review_required"}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_station_mapping(path: Path | None = None) -> pd.DataFrame:
    mapping_path = path or CONFIG_DIR / "asos_station_mapping.csv"
    df = pd.read_csv(mapping_path, dtype=str, comment="#").fillna("")
    legacy_renames = {"stn_id": "station_id", "stn_name": "station_name", "sigungu_key": "municipality"}
    for old, new in legacy_renames.items():
        if new not in df.columns and old in df.columns:
            df[new] = df[old]
    missing = [col for col in MAPPING_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"ASOS mapping missing columns: {missing}")
    return df


def valid_station_ids(mapping: pd.DataFrame) -> list[str]:
    if "station_id" not in mapping.columns and "stn_id" in mapping.columns:
        mapping = mapping.assign(station_id=mapping["stn_id"])
    valid = mapping[
        mapping["station_id"].astype(str).str.strip().ne("")
        & mapping["mapping_status"].astype(str).str.lower().isin(VALID_MAPPING_STATUSES)
    ]
    return sorted(valid["station_id"].astype(str).str.strip().unique(), key=lambda x: int(x) if x.isdigit() else x)


def fetch_weather_for_station(
    stn_id: str,
    api_key: str,
    start_date: str = KMA_START_DATE,
    end_date: str = KMA_END_DATE,
    retries: int = 3,
    sleep_seconds: float = 0.5,
) -> dict[str, Any]:
    params = {
        "serviceKey": api_key,
        "pageNo": "1",
        "numOfRows": "999",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_date,
        "endDt": end_date,
        "stnIds": str(stn_id),
    }
    url = f"{KMA_BASE_URL}{KMA_ENDPOINT}?{urllib.parse.urlencode(params, safe='%')}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read().decode("utf-8")
            data = json.loads(payload)
            header = data.get("response", {}).get("header", {})
            result_code = str(header.get("resultCode", ""))
            if result_code not in {"00", "0"}:
                raise RuntimeError(f"KMA API error for stnIds={stn_id}: {header}")
            return data
        except Exception as exc:  # noqa: BLE001 - keep API recovery broad and logged by caller.
            last_error = exc
            if attempt < retries:
                time.sleep(sleep_seconds * attempt)
    raise RuntimeError(f"Failed KMA API call for stnIds={stn_id}") from last_error


def cached_weather_path(stn_id: str, start_date: str, end_date: str) -> Path:
    return WEATHER_RAW_DIR / f"asos_daily_{stn_id}_{start_date}_{end_date}.json"


def get_weather_json(stn_id: str, api_key: str, start_date: str, end_date: str, force: bool = False) -> dict[str, Any]:
    path = cached_weather_path(stn_id, start_date, end_date)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    data = fetch_weather_for_station(stn_id, api_key, start_date, end_date)
    ensure_parent(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def parse_weather_items(data: dict[str, Any]) -> pd.DataFrame:
    body = data.get("response", {}).get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    df = pd.DataFrame(items)
    if df.empty:
        return pd.DataFrame(columns=WEATHER_COLUMNS)
    keep = [col for col in WEATHER_COLUMNS if col in df.columns]
    df = df[keep].copy()
    for col in WEATHER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df["date"] = pd.to_datetime(df["tm"], errors="coerce")
    for col in ["avgTa", "maxTa", "minTa", "sumRn", "avgRhm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="coerce")
    return df


def api_test_call(stn_id: str, api_key: str) -> dict[str, Any]:
    data = get_weather_json(stn_id, api_key, "20210101", "20210103", force=True)
    df = parse_weather_items(data)
    return {
        "success": not df.empty,
        "station_id": stn_id,
        "rows": int(len(df)),
        "fields": list(df.columns),
        "date_min": df["date"].min().date().isoformat() if not df.empty else None,
        "date_max": df["date"].max().date().isoformat() if not df.empty else None,
    }


def date_chunks(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(pd.Timestamp(year=cursor.year, month=12, day=31), end)
        chunks.append((cursor.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        cursor = chunk_end + pd.Timedelta(days=1)
    return chunks


def collect_station_weather(stn_id: str, api_key: str, start_date: str, end_date: str, force: bool = False) -> pd.DataFrame:
    frames = []
    for chunk_start, chunk_end in date_chunks(start_date, end_date):
        data = get_weather_json(stn_id, api_key, chunk_start, chunk_end, force=force)
        frames.append(parse_weather_items(data))
    if not frames:
        return pd.DataFrame(columns=WEATHER_COLUMNS + ["date"])
    return pd.concat(frames, ignore_index=True)


def collect_weather(force: bool = False, start_date: str = KMA_START_DATE, end_date: str = KMA_END_DATE) -> pd.DataFrame:
    load_dotenv(Path.cwd() / ".env")
    api_key = os.getenv("KMA_API_KEY")
    if not api_key:
        raise RuntimeError("KMA_API_KEY is not set. Add it to .env; do not hard-code it.")
    mapping = read_station_mapping()
    station_ids = valid_station_ids(mapping)
    if not station_ids:
        raise RuntimeError("No ASOS stations with station_id found in config/asos_station_mapping.csv.")
    api_test = api_test_call(station_ids[0], api_key)
    if not api_test["success"]:
        raise RuntimeError(f"KMA API test call failed for station_id={station_ids[0]}")
    frames = []
    errors = []
    for stn_id in station_ids:
        try:
            frames.append(collect_station_weather(stn_id, api_key, start_date, end_date, force=force))
        except Exception as exc:  # noqa: BLE001
            errors.append({"stn_id": stn_id, "error": str(exc)})
    if errors:
        err_path = WEATHER_RAW_DIR / "weather_errors.json"
        ensure_parent(err_path)
        err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
    if not frames:
        raise RuntimeError("No weather data collected. See data/interim/weather/weather_errors.json.")
    weather = pd.concat(frames, ignore_index=True)
    weather = weather.drop_duplicates(subset=["date", "stnId"])
    ensure_parent(ASOS_DAILY_WEATHER_PATH)
    weather.to_csv(ASOS_DAILY_WEATHER_PATH, index=False, encoding="utf-8-sig")
    quality = weather_quality_report(weather, station_ids, errors)
    quality_path = WEATHER_RAW_DIR / "weather_quality_report.json"
    ensure_parent(quality_path)
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return weather


def weather_quality_report(weather: pd.DataFrame, station_ids: list[str], errors: list[dict[str, str]]) -> dict[str, Any]:
    expected_dates = pd.date_range("2021-01-01", "2024-01-31", freq="D")
    coverage = []
    for station_id in station_ids:
        subset = weather[weather["stnId"].astype(str).eq(str(station_id))]
        observed_dates = set(subset["date"].dropna().dt.normalize())
        missing_dates = [d.date().isoformat() for d in expected_dates if d not in observed_dates]
        coverage.append(
            {
                "station_id": station_id,
                "rows": int(len(subset)),
                "date_min": subset["date"].min().date().isoformat() if not subset.empty else None,
                "date_max": subset["date"].max().date().isoformat() if not subset.empty else None,
                "missing_date_count": len(missing_dates),
                "missing_dates_sample": missing_dates[:20],
            }
        )
    return {
        "date_min": weather["date"].min().date().isoformat() if not weather.empty else None,
        "date_max": weather["date"].max().date().isoformat() if not weather.empty else None,
        "rows": int(len(weather)),
        "unique_station_count": int(weather["stnId"].nunique()),
        "duplicate_station_date_rows": int(weather.duplicated(subset=["stnId", "date"]).sum()),
        "missing_by_variable": {col: int(weather[col].isna().sum()) for col in ["avgTa", "minTa", "maxTa", "sumRn", "avgRhm"] if col in weather.columns},
        "coverage": coverage,
        "coverage_issues": [item for item in coverage if item["missing_date_count"] > 0],
        "failed_stations": errors,
    }


if __name__ == "__main__":
    collect_weather()
