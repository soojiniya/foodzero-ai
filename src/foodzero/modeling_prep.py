from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import DOCS_DIR, MODEL_DATASET_PATH, PROCESSED_DIR


WEATHER_FEATURES = ["avgTa", "minTa", "maxTa", "avgRhm"]
DATE_FEATURES = ["month", "day_of_week", "is_weekend", "season"]
POPULATION_FEATURES = ["total_population", "households", "population_per_household"]
LAG_FEATURES = ["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_14"]
BASE_FEATURES = ["sigungu_key", *POPULATION_FEATURES, *DATE_FEATURES, *LAG_FEATURES]
WEATHER_VALIDATED_FEATURES = [*BASE_FEATURES, *WEATHER_FEATURES]
NATIONWIDE_FEATURES = BASE_FEATURES.copy()
TARGET = "waste_amount"


def load_integrated_dataset(path=MODEL_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"station_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def date_continuity_report(df: pd.DataFrame) -> dict[str, Any]:
    rows = []
    for key, group in df.groupby("sigungu_key"):
        dates = pd.DatetimeIndex(group["date"].dropna().sort_values().unique())
        if dates.empty:
            continue
        expected = pd.date_range(dates.min(), dates.max(), freq="D")
        missing = expected.difference(dates)
        rows.append(
            {
                "sigungu_key": key,
                "date_min": dates.min().date().isoformat(),
                "date_max": dates.max().date().isoformat(),
                "actual_days": int(len(dates)),
                "expected_days": int(len(expected)),
                "missing_days": int(len(missing)),
                "missing_sample": [d.date().isoformat() for d in missing[:10]],
            }
        )
    detail = pd.DataFrame(rows).sort_values(["missing_days", "sigungu_key"], ascending=[False, True])
    return {
        "municipality_count": int(detail["sigungu_key"].nunique()) if not detail.empty else 0,
        "municipalities_with_gaps": int(detail["missing_days"].gt(0).sum()) if not detail.empty else 0,
        "total_missing_calendar_days": int(detail["missing_days"].sum()) if not detail.empty else 0,
        "top_gap_municipalities": detail[detail["missing_days"].gt(0)].head(30).to_dict(orient="records"),
    }


def add_calendar_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop(columns=[col for col in LAG_FEATURES if col in df.columns]).copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    base = out[["sigungu_key", "date", "waste_amount"]].copy()
    for days, col in [(1, "lag_1"), (7, "lag_7")]:
        lagged = base.copy()
        lagged["date"] = lagged["date"] + pd.Timedelta(days=days)
        lagged = lagged.rename(columns={"waste_amount": col})
        out = out.merge(lagged[["sigungu_key", "date", col]], on=["sigungu_key", "date"], how="left")

    rolling_frames = []
    for key, group in base.groupby("sigungu_key", sort=False):
        series = group.sort_values("date").set_index("date")["waste_amount"]
        daily = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
        shifted = daily.shift(1)
        rolled = pd.DataFrame(
            {
                "date": daily.index,
                "sigungu_key": key,
                "rolling_mean_7": shifted.rolling(7, min_periods=7).mean().to_numpy(),
                "rolling_mean_14": shifted.rolling(14, min_periods=14).mean().to_numpy(),
            }
        )
        rolling_frames.append(rolled)
    rolling = pd.concat(rolling_frames, ignore_index=True)
    out = out.merge(rolling, on=["sigungu_key", "date"], how="left")
    return out.sort_values(["sigungu_key", "date"]).reset_index(drop=True)


def validate_lag_features(df: pd.DataFrame) -> dict[str, Any]:
    expected = add_calendar_lag_features(df.drop(columns=[col for col in LAG_FEATURES if col in df.columns]))
    merged = df[["sigungu_key", "date", *LAG_FEATURES]].merge(
        expected[["sigungu_key", "date", *LAG_FEATURES]],
        on=["sigungu_key", "date"],
        suffixes=("", "_expected"),
        how="left",
    )
    mismatches: dict[str, int] = {}
    for col in LAG_FEATURES:
        left = pd.to_numeric(merged[col], errors="coerce")
        right = pd.to_numeric(merged[f"{col}_expected"], errors="coerce")
        matches = np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True)
        mismatches[col] = int((~matches).sum())
    return {
        "mismatches_against_calendar_lag": mismatches,
        "missing_counts": {col: int(df[col].isna().sum()) for col in LAG_FEATURES},
    }


def add_time_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    conditions = [
        out["date"].between("2021-01-01", "2022-12-31"),
        out["date"].between("2023-01-01", "2023-06-30"),
        out["date"].between("2023-07-01", "2024-01-31"),
    ]
    out["split"] = np.select(conditions, ["train", "validation", "test"], default="out_of_range")
    return out


def split_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary = {}
    for split, group in df.groupby("split"):
        summary[str(split)] = {
            "rows": int(len(group)),
            "municipalities": int(group["sigungu_key"].nunique()),
            "date_min": group["date"].min().date().isoformat() if not group.empty else None,
            "date_max": group["date"].max().date().isoformat() if not group.empty else None,
        }
    return summary


def target_distribution(df: pd.DataFrame) -> dict[str, Any]:
    target = df[TARGET]
    quantiles = target.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    per_muni = df.groupby("sigungu_key")[TARGET].mean().sort_values(ascending=False)
    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)
    upper = q3 + 1.5 * (q3 - q1)
    high = df[df[TARGET] > upper].copy()
    high_by_muni = high.groupby("sigungu_key")[TARGET].agg(["count", "mean", "max"]).sort_values("count", ascending=False)
    municipal_q3 = df.groupby("sigungu_key")[TARGET].quantile(0.75)
    municipal_q1 = df.groupby("sigungu_key")[TARGET].quantile(0.25)
    municipal_upper = municipal_q3 + 1.5 * (municipal_q3 - municipal_q1)
    local_high = df.join(municipal_upper.rename("municipal_upper_fence"), on="sigungu_key")
    local_high_count = int(local_high[TARGET].gt(local_high["municipal_upper_fence"]).sum())
    return {
        "mean": float(target.mean()),
        "median": float(target.median()),
        "std": float(target.std()),
        "quantiles": {str(k): float(v) for k, v in quantiles.items()},
        "top_1_percent_threshold": float(target.quantile(0.99)),
        "top_1_percent_rows": int(target.ge(target.quantile(0.99)).sum()),
        "top_municipality_means": {str(k): float(v) for k, v in per_muni.head(20).items()},
        "global_iqr_upper_fence": float(upper),
        "global_iqr_high_rows": int(len(high)),
        "global_iqr_high_top_municipalities": high_by_muni.head(20).reset_index().to_dict(orient="records"),
        "municipality_iqr_high_rows": local_high_count,
    }


def required_feature_complete(df: pd.DataFrame, features: list[str]) -> pd.Series:
    return df[features + [TARGET]].notna().all(axis=1)


def write_modeling_plan(report: dict[str, Any]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target = report["target_distribution"]
    lines = [
        "# Modeling Plan",
        "",
        "## Lag And Rolling Validation",
        "",
        "기존 통합 데이터셋은 보존했습니다. 모델 실험용 데이터셋에서는 지자체별 실제 calendar day 기준으로 `lag_1`, `lag_7`, `rolling_mean_7`, `rolling_mean_14`를 재계산했습니다.",
        "",
        "이전 버전의 rolling 변수는 `shift(1)` 뒤 `min_periods=1`로 계산되어 각 지자체의 첫 관측일만 결측이었습니다. 그래서 `rolling_mean_7`과 `rolling_mean_14` 초기 결측치가 각각 지자체 수와 같은 181개였습니다.",
        "",
        "시계열 검증을 더 엄격하게 하기 위해 이번 실험 데이터셋은 정확히 이전 7일 또는 14일 calendar-day 값이 모두 존재할 때만 rolling 평균을 생성합니다. 현재 날짜의 `waste_amount`는 rolling 계산에 포함하지 않습니다.",
        "",
        "## Date Continuity",
        "",
        f"- 날짜 공백이 있는 지자체 수: {report['date_continuity']['municipalities_with_gaps']}",
        f"- 전체 누락 calendar day 수: {report['date_continuity']['total_missing_calendar_days']}",
        "",
        "## Experimental Datasets",
        "",
        f"- `weather_validated_dataset.csv`: {report['weather_validated']['rows']:,} rows, {report['weather_validated']['municipalities']:,} municipalities",
        f"- `nationwide_dataset.csv`: {report['nationwide']['rows']:,} rows, {report['nationwide']['municipalities']:,} municipalities",
        "",
        "`sumRn`은 원본 컬럼으로 보존하지만 1차 모델 feature에서는 제외합니다. 결측이 많고 강수 없음과 관측 결측을 명확히 구분하지 못했기 때문입니다. 같은 날 `discharge_count`도 target leakage 가능성이 있어 feature에서 제외합니다.",
        "",
        "## Time Split",
        "",
        "랜덤 분할은 사용하지 않습니다.",
    ]
    for split, info in report["recommended_split"].items():
        lines.append(f"- {split}: {info['date_min']} ~ {info['date_max']}, rows={info['rows']:,}, municipalities={info['municipalities']:,}")
    lines.extend(
        [
            "",
            "## Feature Candidates",
            "",
            "Weather validated model:",
            "- " + ", ".join(WEATHER_VALIDATED_FEATURES),
            "",
            "Nationwide model:",
            "- " + ", ".join(NATIONWIDE_FEATURES),
            "",
            f"Target: `{TARGET}`",
            "",
            "Excluded from 1st experiment:",
            "- `discharge_count`",
            "- `sumRn`",
            "- future or same-day post-outcome variables",
            "",
            "## Target Distribution",
            "",
            f"- Mean: {target['mean']:,.2f}",
            f"- Median: {target['median']:,.2f}",
            f"- Std: {target['std']:,.2f}",
            f"- Top 1% threshold: {target['top_1_percent_threshold']:,.2f}",
            f"- Global IQR high rows: {target['global_iqr_high_rows']:,}",
            f"- Municipality-level IQR high rows: {target['municipality_iqr_high_rows']:,}",
            "",
            "배출량 규모가 지자체별로 크게 다르므로 전역 IQR 초과 행을 자동 삭제하지 않습니다. 대형 지자체의 정상적인 규모 효과일 수 있어, 모델링 단계에서 원 target과 `log1p(waste_amount)` target 실험을 함께 비교하는 것을 권장합니다.",
        ]
    )
    (DOCS_DIR / "modeling_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_experiment_datasets() -> dict[str, Any]:
    original = load_integrated_dataset()
    continuity = date_continuity_report(original)
    previous_lag_validation = validate_lag_features(original)
    modeling = add_time_split(add_calendar_lag_features(original))

    weather_required = WEATHER_FEATURES + LAG_FEATURES + POPULATION_FEATURES + ["month", "day_of_week", "is_weekend", "season"]
    weather_validated = modeling[
        modeling["weather_mapping_status"].eq("validated")
        & required_feature_complete(modeling, weather_required)
    ].copy()
    nationwide_required = LAG_FEATURES + POPULATION_FEATURES + ["month", "day_of_week", "is_weekend", "season"]
    nationwide = modeling[required_feature_complete(modeling, nationwide_required)].copy()

    weather_validated_path = PROCESSED_DIR / "weather_validated_dataset.csv"
    nationwide_path = PROCESSED_DIR / "nationwide_dataset.csv"
    weather_validated.to_csv(weather_validated_path, index=False, encoding="utf-8-sig")
    nationwide.to_csv(nationwide_path, index=False, encoding="utf-8-sig")

    report = {
        "source_dataset": str(MODEL_DATASET_PATH),
        "calendar_lag_applied_to_experiment_datasets": True,
        "previous_lag_validation": previous_lag_validation,
        "experiment_lag_validation": validate_lag_features(modeling),
        "date_continuity": continuity,
        "weather_validated": {
            "path": str(weather_validated_path),
            "rows": int(len(weather_validated)),
            "municipalities": int(weather_validated["sigungu_key"].nunique()),
            "split": split_summary(weather_validated),
        },
        "nationwide": {
            "path": str(nationwide_path),
            "rows": int(len(nationwide)),
            "municipalities": int(nationwide["sigungu_key"].nunique()),
            "split": split_summary(nationwide),
        },
        "target_distribution": target_distribution(modeling),
        "recommended_split": split_summary(modeling),
        "weather_validated_features": WEATHER_VALIDATED_FEATURES,
        "nationwide_features": NATIONWIDE_FEATURES,
        "target": TARGET,
        "excluded_features": ["discharge_count", "sumRn"],
    }
    (PROCESSED_DIR / "modeling_prep_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_modeling_plan(report)
    return report


if __name__ == "__main__":
    build_experiment_datasets()
