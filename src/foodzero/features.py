from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ASOS_DAILY_WEATHER_PATH, CONFIG_DIR, DOCS_DIR, FOOD_WASTE_RAW_DIR, MODEL_DATASET_PATH, POPULATION_RAW_DIR, PROCESSED_DIR
from .io_utils import normalize_name, numeric_series, read_table
from .weather import read_station_mapping


STANDARD_SIDO_RULES = {
    "강원특별자치도": "강원도",
    "전북특별자치도": "전라북도",
}


STANDARD_SIGUNGU_RULES = {
    ("경기도", "성남시분당구"): "성남시 분당구",
    ("경기도", "성남시수정구"): "성남시 수정구",
    ("경기도", "성남시중원구"): "성남시 중원구",
    ("경기도", "수원시권선구"): "수원시 권선구",
    ("경기도", "수원시영통구"): "수원시 영통구",
    ("경기도", "수원시장안구"): "수원시 장안구",
    ("경기도", "수원시팔달구"): "수원시 팔달구",
}


def standardize_authority_name(sido: str, sigungu: str) -> tuple[str, str, str]:
    standard_sido = STANDARD_SIDO_RULES.get(str(sido).strip(), str(sido).strip())
    raw_sigungu = str(sigungu).strip()
    standard_sigungu = STANDARD_SIGUNGU_RULES.get((standard_sido, raw_sigungu), raw_sigungu)
    return standard_sido, standard_sigungu, f"{standard_sido} {standard_sigungu}".strip()


def load_aliases(path: Path | None = None) -> dict[str, Any]:
    alias_path = path or CONFIG_DIR / "schema_aliases.json"
    return json.loads(alias_path.read_text(encoding="utf-8"))


def find_column(columns: list[str], aliases: list[str], required: bool = True) -> str | None:
    normalized = {clean_column(c): c for c in columns}
    matches = []
    for alias in aliases:
        key = clean_column(alias)
        if key in normalized:
            matches.append(normalized[key])
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous columns for aliases {aliases}: {matches}")
    if required:
        raise ValueError(f"Could not find required column from aliases {aliases}. Actual columns: {columns}")
    return None


def clean_column(value: str) -> str:
    return str(value).lower().replace(" ", "").replace("_", "").replace("-", "").replace("(", "").replace(")", "")


def source_files(raw_dir: Path) -> list[Path]:
    return sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".xls"})


def load_food_waste() -> pd.DataFrame:
    aliases = load_aliases()["food_waste"]
    frames = []
    for path in source_files(FOOD_WASTE_RAW_DIR):
        if path.suffix.lower() in {".xlsx", ".xls"}:
            xls = pd.ExcelFile(path)
            sheet_names = xls.sheet_names
        else:
            sheet_names = [0]
        for sheet in sheet_names:
            df, _meta = read_table(path, sheet_name=sheet)
            if df.empty:
                continue
            columns = list(df.columns)
            date_col = find_column(columns, aliases["date"], required=False)
            year_col = find_column(columns, aliases["year"], required=False)
            month_col = find_column(columns, aliases["month"], required=False)
            day_col = find_column(columns, aliases["day"], required=False)
            sido_col = find_column(columns, aliases["sido"], required=False)
            sigungu_col = find_column(columns, aliases["sigungu"])
            amount_col = find_column(columns, aliases["waste_amount"])
            count_col = find_column(columns, ["배출횟수", "배출 회수", "배출건수", "배출횟수(건)"], required=False)
            out = pd.DataFrame(
                {
                    "source_file": path.name,
                    "source_sheet": str(sheet),
                    "sido": normalize_series(df[sido_col]) if sido_col else "",
                    "sigungu": normalize_series(df[sigungu_col]),
                    "waste_amount": numeric_series(df[amount_col]),
                    "discharge_count": numeric_series(df[count_col]) if count_col else pd.NA,
                }
            )
            if date_col:
                out["date"] = pd.to_datetime(df[date_col], errors="coerce")
            elif year_col and month_col and day_col:
                out["date"] = pd.to_datetime(
                    df[year_col].astype(str).str.extract(r"(\d{4})", expand=False).str.zfill(4)
                    + "-"
                    + df[month_col].astype(str).str.extract(r"(\d{1,2})", expand=False).str.zfill(2)
                    + "-"
                    + df[day_col].astype(str).str.extract(r"(\d{1,2})", expand=False).str.zfill(2),
                    errors="coerce",
                )
            else:
                raise ValueError(f"No date columns found in {path.name} sheet {sheet}. Actual columns: {columns}")
            frames.append(out)
    if not frames:
        raise FileNotFoundError("No food waste raw files found in data/raw/food_waste.")
    waste = pd.concat(frames, ignore_index=True)
    std = waste.apply(lambda r: standardize_authority_name(r["sido"], r["sigungu"]), axis=1, result_type="expand")
    waste[["sido", "sigungu", "sigungu_key"]] = std
    waste = waste.dropna(subset=["date", "waste_amount"])
    waste = waste.groupby(["date", "sido", "sigungu", "sigungu_key"], as_index=False)[["waste_amount", "discharge_count"]].sum(min_count=1)
    return waste


def normalize_series(series: pd.Series) -> pd.Series:
    return series.map(normalize_name)


def make_sigungu_key(sido: pd.Series | str, sigungu: pd.Series | str) -> pd.Series:
    sido_s = pd.Series(sido) if not isinstance(sido, pd.Series) else sido.fillna("")
    sigungu_s = pd.Series(sigungu) if not isinstance(sigungu, pd.Series) else sigungu.fillna("")
    return (sido_s.astype(str).str.strip() + " " + sigungu_s.astype(str).str.strip()).str.strip()


def load_population() -> pd.DataFrame:
    aliases = load_aliases()["population"]
    frames = []
    for path in source_files(POPULATION_RAW_DIR):
        df, _meta = read_table(path)
        if df.empty:
            continue
        columns = list(df.columns)
        wide = parse_population_wide(df, path.name)
        if not wide.empty:
            frames.append(wide)
            continue
        period_col = find_column(columns, aliases["period"], required=False)
        sido_col = find_column(columns, aliases["sido"], required=False)
        sigungu_col = find_column(columns, aliases["sigungu"])
        pop_col = find_column(columns, aliases["total_population"])
        household_col = find_column(columns, aliases["households"])
        period = parse_population_period(df[period_col], path.name) if period_col else parse_population_period(pd.Series([path.name] * len(df)), path.name)
        frames.append(
            pd.DataFrame(
                {
                    "year_month": period,
                    "sido": normalize_series(df[sido_col]) if sido_col else "",
                    "sigungu": normalize_series(df[sigungu_col]),
                    "total_population": numeric_series(df[pop_col]),
                    "households": numeric_series(df[household_col]),
                }
            )
        )
    if not frames:
        raise FileNotFoundError("No population raw files found in data/raw/population.")
    pop = pd.concat(frames, ignore_index=True).dropna(subset=["year_month"])
    std = pop.apply(lambda r: standardize_authority_name(r["sido"], r["sigungu"]), axis=1, result_type="expand")
    pop[["sido", "sigungu", "sigungu_key"]] = std
    out = pop.groupby(["year_month", "sido", "sigungu", "sigungu_key"], as_index=False)[["total_population", "households"]].sum(min_count=1)
    out["population_per_household"] = np.where(out["households"].gt(0), out["total_population"] / out["households"], np.nan)
    return out


def parse_population_wide(df: pd.DataFrame, source_file: str = "") -> pd.DataFrame:
    admin_col = next((col for col in df.columns if clean_column(col) in {"행정구역", "행정기관", "행정기관코드"}), None)
    if not admin_col:
        return pd.DataFrame()
    rows = []
    for col in df.columns:
        match = re.match(r"((?:20)\d{2})년?\s?(\d{1,2})월?_?(총인구수|세대수)$", str(col).replace(" ", ""))
        if not match:
            continue
        year, month, metric = match.groups()
        year_month = f"{year}-{int(month):02d}"
        value_col = "total_population" if metric == "총인구수" else "households"
        temp = parse_admin_area(df[admin_col]).copy()
        temp["year_month"] = year_month
        temp[value_col] = numeric_series(df[col])
        temp["source_file"] = source_file
        rows.append(temp)
    if not rows:
        return pd.DataFrame()
    long = pd.concat(rows, ignore_index=True)
    value_cols = [col for col in ["total_population", "households"] if col in long.columns]
    out = (
        long.groupby(["year_month", "sido", "sigungu", "sigungu_key", "admin_code"], as_index=False)[value_cols]
        .first()
        .reset_index(drop=True)
    )
    for col in ["total_population", "households"]:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def parse_admin_area(series: pd.Series) -> pd.DataFrame:
    raw = series.map(normalize_name)
    code = raw.str.extract(r"\((\d+)\)", expand=False).fillna("")
    name = raw.str.replace(r"\s*\(\d+\)\s*$", "", regex=True).str.strip()
    parts = name.str.split()
    sido = parts.str[0].fillna("")
    sigungu = parts.map(lambda xs: " ".join(xs[1:]) if isinstance(xs, list) and len(xs) > 1 else "")
    return pd.DataFrame(
        {
            "admin_name": name,
            "admin_code": code,
            "sido": sido,
            "sigungu": sigungu,
            "sigungu_key": make_sigungu_key(sido, sigungu),
        }
    )


def parse_population_period(series: pd.Series, fallback: str) -> pd.Series:
    raw = series.astype(str).where(series.notna(), fallback)
    extracted = raw.str.extract(r"((?:20)\d{2})[.\-년_/ ]?\s*(\d{1,2})", expand=True)
    return pd.to_datetime(extracted[0] + "-" + extracted[1].str.zfill(2) + "-01", errors="coerce").dt.to_period("M").astype(str)


def load_weather() -> pd.DataFrame:
    path = ASOS_DAILY_WEATHER_PATH
    if not path.exists():
        raise FileNotFoundError("Weather cache not found. Run `python -m src.foodzero.weather` first.")
    weather = pd.read_csv(path, dtype={"stnId": str, "station_id": str})
    if "station_id" not in weather.columns and "stnId" in weather.columns:
        weather["station_id"] = weather["stnId"]
    if "station_name" not in weather.columns and "stnNm" in weather.columns:
        weather["station_name"] = weather["stnNm"]
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather["station_id"] = weather["station_id"].astype(str).str.strip()
    return weather


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month
    out["day"] = out["date"].dt.day
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["season"] = np.select(
        [
            out["month"].isin([3, 4, 5]),
            out["month"].isin([6, 7, 8]),
            out["month"].isin([9, 10, 11]),
            out["month"].isin([12, 1, 2]),
        ],
        ["spring", "summer", "autumn", "winter"],
        default="unknown",
    )
    out["year_month"] = out["date"].dt.to_period("M").astype(str)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["sigungu_key", "date"]).copy()
    group = out.groupby("sigungu_key", sort=False)["waste_amount"]
    out["lag_1"] = group.shift(1)
    out["lag_7"] = group.shift(7)
    shifted = group.shift(1)
    out["rolling_mean_7"] = shifted.groupby(out["sigungu_key"]).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    out["rolling_mean_14"] = shifted.groupby(out["sigungu_key"]).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
    return out


def deduplicate_daily_authority(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["date", "sigungu_key"], keep="first").copy()


def merge_population(waste: pd.DataFrame, population: pd.DataFrame) -> pd.DataFrame:
    cols = ["year_month", "sigungu_key", "total_population", "households", "population_per_household"]
    return waste.merge(population[cols], on=["year_month", "sigungu_key"], how="left")


def _mapping_join_key(mapping: pd.DataFrame) -> str:
    if "standard_sigungu_key" in mapping.columns:
        return "standard_sigungu_key"
    if "sigungu_key" in mapping.columns:
        return "sigungu_key"
    return "municipality"


def merge_weather_mapping(df: pd.DataFrame, mapping: pd.DataFrame | None = None) -> pd.DataFrame:
    mapping_df = read_station_mapping() if mapping is None else mapping.copy()
    join_key = _mapping_join_key(mapping_df)
    mapping_df[join_key] = mapping_df[join_key].astype(str).str.strip()
    keep = [join_key, "station_id", "station_name", "mapping_status", "mapping_method", "confidence"]
    for col in keep:
        if col not in mapping_df.columns:
            mapping_df[col] = pd.NA
    mapping_df = mapping_df[keep].drop_duplicates(subset=[join_key])
    mapping_df = mapping_df.rename(
        columns={
            join_key: "sigungu_key",
            "mapping_status": "weather_mapping_status",
            "mapping_method": "weather_mapping_method",
            "confidence": "weather_mapping_confidence",
        }
    )
    mapping_df["station_id"] = mapping_df["station_id"].astype(str).str.strip().replace({"": pd.NA, "nan": pd.NA})
    return df.merge(mapping_df, on="sigungu_key", how="left")


def merge_weather(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    weather_cols = ["date", "station_id", "avgTa", "minTa", "maxTa", "sumRn", "avgRhm"]
    weather_df = weather.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
    weather_df["station_id"] = weather_df["station_id"].astype(str).str.strip()
    weather_df = weather_df[weather_cols].drop_duplicates(subset=["date", "station_id"])
    out = df.copy()
    out["station_id"] = out["station_id"].astype("string").str.strip()
    return out.merge(weather_df, on=["date", "station_id"], how="left")


def missing_counts(df: pd.DataFrame) -> dict[str, int]:
    return {col: int(value) for col, value in df.isna().sum().items()}


def analyze_sumrn(weather: pd.DataFrame) -> dict[str, Any]:
    weather_df = weather.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
    weather_df["year_month"] = weather_df["date"].dt.to_period("M").astype(str)
    missing = weather_df["sumRn"].isna()
    by_station = (
        weather_df.assign(sumRn_missing=missing)
        .groupby(["station_id", "station_name"], dropna=False)["sumRn_missing"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )
    by_month = (
        weather_df.assign(sumRn_missing=missing)
        .groupby("year_month")["sumRn_missing"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
    )
    return {
        "weather_rows": int(len(weather_df)),
        "sumRn_missing": int(missing.sum()),
        "sumRn_missing_rate": float(missing.mean()) if len(weather_df) else 0.0,
        "top_missing_stations": {f"{idx[0]} {idx[1]}": int(value) for idx, value in by_station.items()},
        "top_missing_months": {str(idx): int(value) for idx, value in by_month.items()},
        "decision": "keep_nan",
        "reason": "ASOS API response contains blank sumRn values, but the collected daily response has no companion flag that separates no precipitation from observation missingness. No zero-imputation is applied in preprocessing.",
    }


def build_quality_report(dataset: pd.DataFrame, weather: pd.DataFrame, sumrn_analysis: dict[str, Any]) -> dict[str, Any]:
    weather_vars = ["avgTa", "minTa", "maxTa", "sumRn", "avgRhm"]
    station_present = dataset["station_id"].notna()
    weather_any_present = dataset[weather_vars].notna().any(axis=1)
    q1 = dataset["waste_amount"].quantile(0.25)
    q3 = dataset["waste_amount"].quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    initial_missing = {col: int(dataset[col].isna().sum()) for col in ["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"]}
    return {
        "rows": int(len(dataset)),
        "date_min": dataset["date"].min().date().isoformat(),
        "date_max": dataset["date"].max().date().isoformat(),
        "unique_municipalities": int(dataset["sigungu_key"].nunique()),
        "population_merge_success_rate": float(dataset[["total_population", "households"]].notna().all(axis=1).mean()),
        "population_missing_rows": int(dataset[["total_population", "households"]].isna().any(axis=1).sum()),
        "weather_merge_success_rate_overall": float(weather_any_present.mean()),
        "weather_merge_success_rate_with_station": float(weather_any_present[station_present].mean()) if station_present.any() else 0.0,
        "station_id_missing_rows": int(dataset["station_id"].isna().sum()),
        "station_id_missing_municipalities": sorted(dataset.loc[dataset["station_id"].isna(), "sigungu_key"].dropna().unique().tolist()),
        "weather_mapping_status_counts": {str(k): int(v) for k, v in dataset["weather_mapping_status"].fillna("missing").value_counts().items()},
        "missing_by_column": missing_counts(dataset),
        "duplicate_date_municipality_rows": int(dataset.duplicated(subset=["date", "sigungu_key"]).sum()),
        "lag_rolling_missing_counts": initial_missing,
        "target_negative_count": int(dataset["waste_amount"].lt(0).sum()),
        "target_zero_count": int(dataset["waste_amount"].eq(0).sum()),
        "target_min": float(dataset["waste_amount"].min()),
        "target_max": float(dataset["waste_amount"].max()),
        "target_q1": float(q1),
        "target_q3": float(q3),
        "target_iqr_upper_fence": float(upper_fence),
        "target_iqr_high_outlier_count": int(dataset["waste_amount"].gt(upper_fence).sum()),
        "weather_source_rows": int(len(weather)),
        "sumRn_analysis": sumrn_analysis,
    }


def write_preprocessing_decisions(report: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    sumrn = report["sumRn_analysis"]
    lines = [
        "# Preprocessing Decisions",
        "",
        "## Target Design",
        "",
        "이번 통합 데이터셋의 예측 target은 해당 날짜의 `waste_amount`(`배출량(g)`)입니다. 현재 파일은 일별 배출 실적과 일별 ASOS 관측값을 같은 날짜 기준으로 결합하는 설명/백테스트용 구조입니다.",
        "",
        "실제 서비스에서 미래 날짜를 예측할 때는 당일 ASOS 관측 날씨가 사전에 확정되어 있지 않습니다. 모델링 단계에서는 (A) 과거 관측 날씨를 이용한 설명/백테스트 모델과 (B) 예보 날씨 또는 lagged weather를 사용하는 서비스용 예측 모델을 분리해야 합니다.",
        "",
        "## Leakage Control",
        "",
        "`lag_1`, `lag_7`, `rolling_mean_7`, `rolling_mean_14`는 지자체별 날짜 정렬 뒤 과거 값만 사용하도록 생성했습니다. rolling 변수는 현재 날짜 target이 포함되지 않도록 `shift(1)` 후 계산합니다. `discharge_count`는 원본 정보로 보존하지만, 같은 날 배출량 예측에서 누수 가능성이 있어 현재 단계의 모델 feature로 사용하지 않습니다.",
        "",
        "## Rainfall (`sumRn`) Missing Values",
        "",
        f"- ASOS 원천 행 수: {sumrn['weather_rows']:,}",
        f"- `sumRn` 결측 행 수: {sumrn['sumRn_missing']:,}",
        f"- `sumRn` 결측률: {sumrn['sumRn_missing_rate']:.2%}",
        "",
        "KMA ASOS 일자료 API 응답에서 `sumRn`은 빈 문자열로 내려오는 경우가 많고, 현재 수집 응답 필드만으로는 강수 없음과 관측 결측을 명확히 구분하는 별도 플래그가 확인되지 않았습니다. 따라서 이번 전처리에서는 빈 `sumRn`을 0으로 대체하지 않고 `NaN`으로 유지합니다.",
        "",
        "상위 결측 관측소:",
    ]
    for station, count in sumrn["top_missing_stations"].items():
        lines.append(f"- {station}: {count:,}")
    lines.extend(["", "상위 결측 연월:"])
    for month, count in sumrn["top_missing_months"].items():
        lines.append(f"- {month}: {count:,}")
    (DOCS_DIR / "preprocessing_decisions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_model_dataset() -> pd.DataFrame:
    waste = deduplicate_daily_authority(add_lag_features(add_date_features(load_food_waste())))
    population = load_population()
    weather = load_weather()
    mapping = read_station_mapping()
    merged = merge_population(waste, population)
    merged = merge_weather_mapping(merged, mapping)
    merged = merge_weather(merged, weather)
    merged = deduplicate_daily_authority(merged)
    MODEL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MODEL_DATASET_PATH, index=False, encoding="utf-8-sig")
    sumrn_analysis = analyze_sumrn(weather)
    report = build_quality_report(merged, weather, sumrn_analysis)
    (PROCESSED_DIR / "foodzero_model_dataset_quality.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_preprocessing_decisions(report)
    return merged


if __name__ == "__main__":
    build_model_dataset()
