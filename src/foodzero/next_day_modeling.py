from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DOCS_DIR, PROCESSED_DIR, PROJECT_ROOT
from .features import add_date_features
from .modeling import dataframe_to_markdown, metrics, rmse, sklearn_imports


RANDOM_STATE = 42
NEXT_DAY_DIR = PROJECT_ROOT / "evaluation" / "next_day"
NEXT_DAY_PLOTS_DIR = NEXT_DAY_DIR / "plots"
NEXT_DAY_MODEL_PATH = PROJECT_ROOT / "models" / "foodzero_next_day_model.joblib"
NEXT_DAY_DATASET_PATH = PROCESSED_DIR / "next_day_model_dataset.csv"
NEXT_DAY_SERVICE_FEATURES_PATH = PROCESSED_DIR / "next_day_service_features.csv"
MODEL_VERSION = "foodzero-next-day-rf-v1"

TRAIN_START = "2021-01-01"
TRAIN_END = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END = "2023-06-30"
TEST_START = "2023-07-01"
TEST_END = "2024-01-31"

TARGET = "target_next_day"
FORBIDDEN_FEATURES = {"target_next_day", "target_date", "discharge_count", "sumRn", "avgTa", "minTa", "maxTa", "avgRhm"}
FEATURES = [
    "sigungu_key",
    "total_population",
    "households",
    "population_per_household",
    "target_month",
    "target_day_of_week",
    "target_is_weekend",
    "target_season",
    "waste_amount_t",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_14",
]
CATEGORICAL_FEATURES = ["sigungu_key", "target_month", "target_day_of_week", "target_season"]


@dataclass(frozen=True)
class NextDayExperimentResult:
    comparison: pd.DataFrame
    test_metrics: dict[str, Any]
    municipality_metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    surge_thresholds: pd.DataFrame


def load_nationwide_dataset(path: Path = PROCESSED_DIR / "nationwide_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"station_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def season_from_month(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
                month.isin([12, 1, 2]),
            ],
            ["spring", "summer", "autumn", "winter"],
            default="unknown",
        ),
        index=month.index,
    )


def add_recent_7day_average(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    base = df[["sigungu_key", "date", "waste_amount"]].copy()
    for key, group in base.groupby("sigungu_key", sort=False):
        series = group.sort_values("date").set_index("date")["waste_amount"]
        daily = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="D"))
        recent = daily.rolling(7, min_periods=7).mean()
        frames.append(pd.DataFrame({"sigungu_key": key, "date": recent.index, "recent_7day_average_g": recent.to_numpy()}))
    recent_df = pd.concat(frames, ignore_index=True)
    return df.merge(recent_df, on=["sigungu_key", "date"], how="left")


def build_next_day_dataset(nationwide: pd.DataFrame | None = None, write_files: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = load_nationwide_dataset() if nationwide is None else nationwide.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = add_recent_7day_average(source)

    target_lookup = source[["sigungu_key", "date", "waste_amount"]].rename(
        columns={"date": "target_date", "waste_amount": TARGET}
    )
    dataset = source.copy()
    dataset["target_date"] = dataset["date"] + pd.Timedelta(days=1)
    dataset = dataset.merge(target_lookup, on=["sigungu_key", "target_date"], how="left")

    weekly_lookup = source[["sigungu_key", "date", "waste_amount"]].copy()
    weekly_lookup["date"] = weekly_lookup["date"] + pd.Timedelta(days=6)
    weekly_lookup = weekly_lookup.rename(columns={"waste_amount": "weekly_persistence"})
    dataset = dataset.merge(weekly_lookup[["sigungu_key", "date", "weekly_persistence"]], on=["sigungu_key", "date"], how="left")

    target_cal = add_date_features(pd.DataFrame({"date": dataset["target_date"]})).rename(
        columns={
            "month": "target_month",
            "day_of_week": "target_day_of_week",
            "is_weekend": "target_is_weekend",
            "season": "target_season",
        }
    )
    dataset[["target_month", "target_day_of_week", "target_is_weekend", "target_season"]] = target_cal[
        ["target_month", "target_day_of_week", "target_is_weekend", "target_season"]
    ]
    dataset = dataset.rename(columns={"waste_amount": "waste_amount_t"})
    dataset = dataset.dropna(subset=[TARGET, "recent_7day_average_g", "weekly_persistence", *FEATURES]).copy()
    dataset["surge_ratio_actual"] = dataset[TARGET] / dataset["recent_7day_average_g"] - 1
    dataset["surge_ratio_actual_pct"] = dataset["surge_ratio_actual"] * 100

    service_cols = [
        "date",
        "target_date",
        *FEATURES,
        "recent_7day_average_g",
        "weekly_persistence",
    ]
    service_features = dataset[service_cols].copy()
    if write_files:
        NEXT_DAY_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(NEXT_DAY_DATASET_PATH, index=False, encoding="utf-8-sig")
        service_features.to_csv(NEXT_DAY_SERVICE_FEATURES_PATH, index=False, encoding="utf-8-sig")
    return dataset, service_features


def split_by_target_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["target_date"].between(TRAIN_START, TRAIN_END)].copy(),
        "validation": df[df["target_date"].between(VALIDATION_START, VALIDATION_END)].copy(),
        "test": df[df["target_date"].between(TEST_START, TEST_END)].copy(),
    }


def split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        key: {
            "rows": int(len(value)),
            "municipalities": int(value["sigungu_key"].nunique()),
            "target_date_min": value["target_date"].min().date().isoformat() if not value.empty else None,
            "target_date_max": value["target_date"].max().date().isoformat() if not value.empty else None,
        }
        for key, value in splits.items()
    }


def assert_target_date_splits_do_not_overlap(splits: dict[str, pd.DataFrame]) -> None:
    if not splits["train"]["target_date"].between(TRAIN_START, TRAIN_END).all():
        raise ValueError("Train target_date out of range.")
    if not splits["validation"]["target_date"].between(VALIDATION_START, VALIDATION_END).all():
        raise ValueError("Validation target_date out of range.")
    if not splits["test"]["target_date"].between(TEST_START, TEST_END).all():
        raise ValueError("Test target_date out of range.")
    if splits["train"]["target_date"].max() >= splits["validation"]["target_date"].min():
        raise ValueError("Train and validation target_date overlap.")
    if splits["validation"]["target_date"].max() >= splits["test"]["target_date"].min():
        raise ValueError("Validation and test target_date overlap.")


def assert_no_forbidden_features(features: list[str]) -> None:
    forbidden = sorted(set(features) & FORBIDDEN_FEATURES)
    if forbidden:
        raise ValueError(f"Forbidden service-time features included: {forbidden}")


def make_preprocessor(features: list[str], model_family: str):
    sk = sklearn_imports()
    categorical = [col for col in CATEGORICAL_FEATURES if col in features]
    numeric = [col for col in features if col not in categorical]
    return sk["ColumnTransformer"](
        transformers=[
            ("categorical", sk["OneHotEncoder"](handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", sk["StandardScaler"]() if model_family == "linear" else "passthrough", numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_pipeline(features: list[str], train: pd.DataFrame, model_name: str):
    sk = sklearn_imports()
    estimators = {
        "ridge": ("linear", sk["Ridge"](alpha=10.0)),
        "random_forest": (
            "tree",
            sk["RandomForestRegressor"](
                n_estimators=90,
                max_depth=18,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        "hist_gradient_boosting": (
            "tree",
            sk["HistGradientBoostingRegressor"](
                max_iter=180,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=RANDOM_STATE,
            ),
        ),
    }
    family, estimator = estimators[model_name]
    pipe = sk["Pipeline"](
        steps=[
            ("preprocessor", make_preprocessor(features, family)),
            ("model", sk["clone"](estimator)),
        ]
    )
    pipe.fit(train[features], train[TARGET])
    return pipe


def predict(model: Any, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return np.clip(np.asarray(model.predict(frame[features]), dtype=float), 0, None)


def baseline_rows(split_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    baselines = [
        ("persistence_today", "waste_amount_t"),
        ("weekly_persistence", "weekly_persistence"),
        ("moving_average_7", "recent_7day_average_g"),
    ]
    return [
        {
            "model": name,
            "split": split_name,
            "rows": int(len(frame)),
            **metrics(frame[TARGET], frame[col].to_numpy()),
        }
        for name, col in baselines
    ]


def train_and_validate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    assert_no_forbidden_features(FEATURES)
    splits = split_by_target_date(df)
    assert_target_date_splits_do_not_overlap(splits)
    rows = baseline_rows("validation", splits["validation"])
    fitted: dict[str, Any] = {}
    for model_name in ["ridge", "random_forest", "hist_gradient_boosting"]:
        model = fit_pipeline(FEATURES, splits["train"], model_name)
        pred = predict(model, splits["validation"], FEATURES)
        rows.append(
            {
                "model": model_name,
                "split": "validation",
                "rows": int(len(splits["validation"])),
                **metrics(splits["validation"][TARGET], pred),
            }
        )
        fitted[model_name] = model
    comparison = pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)
    return comparison, fitted, split_summary(splits)


def select_final_model(comparison: pd.DataFrame) -> str:
    ml = comparison[comparison["model"].isin(["ridge", "random_forest", "hist_gradient_boosting"])].copy()
    return str(ml.sort_values(["MAE", "RMSE"]).iloc[0]["model"])


def municipality_metrics(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = test[["sigungu_key", TARGET]].copy()
    out["prediction"] = pred
    out["absolute_error"] = (out[TARGET] - out["prediction"]).abs()
    summary = (
        out.groupby("sigungu_key")
        .agg(
            rows=(TARGET, "size"),
            mean_target_next_day=(TARGET, "mean"),
            mae=("absolute_error", "mean"),
            rmse=("absolute_error", lambda x: rmse(pd.Series(np.zeros(len(x))), x.to_numpy())),
        )
        .reset_index()
    )
    summary["mae_to_mean_ratio"] = summary["mae"] / summary["mean_target_next_day"]
    return summary.sort_values("mae", ascending=False)


def size_group_metrics(municipality: pd.DataFrame) -> pd.DataFrame:
    out = municipality.copy()
    out["size_group"] = pd.qcut(out["mean_target_next_day"], q=3, labels=["small", "medium", "large"])
    return (
        out.groupby("size_group", observed=True)
        .agg(
            municipalities=("sigungu_key", "nunique"),
            mean_target_next_day=("mean_target_next_day", "mean"),
            mean_mae=("mae", "mean"),
            mean_mae_to_mean_ratio=("mae_to_mean_ratio", "mean"),
        )
        .reset_index()
    )


def permutation_feature_importance(model: Any, test: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    sample = test.sample(n=min(8000, len(test)), random_state=RANDOM_STATE).copy()
    baseline = metrics(sample[TARGET], predict(model, sample, FEATURES))["MAE"]
    rows = []
    for feature in FEATURES:
        losses = []
        for _ in range(5):
            shuffled = sample.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            losses.append(metrics(sample[TARGET], predict(model, shuffled, FEATURES))["MAE"])
        rows.append(
            {
                "feature": feature,
                "baseline_mae": baseline,
                "permuted_mae": float(np.mean(losses)),
                "importance_mae_increase": float(np.mean(losses) - baseline),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mae_increase", ascending=False)


def surge_threshold_analysis(train: pd.DataFrame) -> pd.DataFrame:
    ratios = train["surge_ratio_actual"]
    candidates = [
        ("q70_q90", ratios.quantile(0.70), ratios.quantile(0.90)),
        ("q75_q90", ratios.quantile(0.75), ratios.quantile(0.90)),
        ("q80_q95", ratios.quantile(0.80), ratios.quantile(0.95)),
    ]
    rows = []
    for name, attention, high in candidates:
        levels = classify_surge_ratio(ratios, attention, high)
        counts = levels.value_counts(normalize=True)
        rows.append(
            {
                "candidate": name,
                "attention_threshold_pct": float(attention * 100),
                "high_threshold_pct": float(high * 100),
                "normal_rate": float(counts.get("normal", 0.0)),
                "attention_rate": float(counts.get("attention", 0.0)),
                "high_rate": float(counts.get("high", 0.0)),
            }
        )
    return pd.DataFrame(rows)


def classify_surge_ratio(ratio: pd.Series | np.ndarray | float, attention_threshold: float, high_threshold: float):
    if np.isscalar(ratio):
        value = float(ratio)
        if value >= high_threshold:
            return "high"
        if value >= attention_threshold:
            return "attention"
        return "normal"
    series = pd.Series(ratio)
    return pd.Series(
        np.select(
            [series.ge(high_threshold), series.ge(attention_threshold)],
            ["high", "attention"],
            default="normal",
        ),
        index=series.index,
    )


def setup_korean_font() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
    ]
    for path in candidates:
        if path.exists():
            fm.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = fm.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def make_plots(test: pd.DataFrame, pred: np.ndarray, municipality: pd.DataFrame) -> None:
    setup_korean_font()
    import matplotlib.pyplot as plt

    NEXT_DAY_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = test.assign(prediction=pred).sample(n=min(12000, len(test)), random_state=RANDOM_STATE)
    plt.figure(figsize=(7, 7))
    plt.scatter(sample[TARGET], sample["prediction"], s=5, alpha=0.25)
    upper = max(sample[TARGET].max(), sample["prediction"].max())
    plt.plot([0, upper], [0, upper], color="red", linewidth=1)
    plt.xlabel("실제 다음날 배출량(g)")
    plt.ylabel("예측 다음날 배출량(g)")
    plt.tight_layout()
    plt.savefig(NEXT_DAY_PLOTS_DIR / "actual_vs_predicted.png", dpi=140)
    plt.close()

    residual = sample["prediction"] - sample[TARGET]
    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=80)
    plt.xlabel("잔차(g)")
    plt.ylabel("건수")
    plt.tight_layout()
    plt.savefig(NEXT_DAY_PLOTS_DIR / "residual_distribution.png", dpi=140)
    plt.close()

    top = municipality.head(25).sort_values("mae")
    plt.figure(figsize=(8, 8))
    plt.barh(top["sigungu_key"], top["mae"])
    plt.xlabel("Test MAE(g)")
    plt.tight_layout()
    plt.savefig(NEXT_DAY_PLOTS_DIR / "municipality_error.png", dpi=140)
    plt.close()


def save_model(model: Any, selected_model: str, thresholds: dict[str, float], test_metrics: dict[str, Any]) -> None:
    sk = sklearn_imports()
    NEXT_DAY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "model_name": selected_model,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "target": TARGET,
        "thresholds": thresholds,
        "test_metrics": test_metrics,
        "service_features_path": str(NEXT_DAY_SERVICE_FEATURES_PATH),
    }
    sk["dump"](payload, NEXT_DAY_MODEL_PATH)
    loaded = sk["load"](NEXT_DAY_MODEL_PATH)
    sample = pd.read_csv(NEXT_DAY_SERVICE_FEATURES_PATH, encoding="utf-8-sig").head(3)
    _ = loaded["model"].predict(sample[loaded["features"]])


def predict_next_day(
    municipality: str,
    reference_date: str,
    model_path: Path = NEXT_DAY_MODEL_PATH,
    service_features_path: Path = NEXT_DAY_SERVICE_FEATURES_PATH,
) -> dict[str, Any]:
    sk = sklearn_imports()
    payload = sk["load"](model_path)
    features = payload["features"]
    if service_features_path.suffix == ".parquet":
        service = pd.read_parquet(service_features_path)
    else:
        service = pd.read_csv(service_features_path, encoding="utf-8-sig")
    service["date"] = pd.to_datetime(service["date"], errors="coerce")
    service["target_date"] = pd.to_datetime(service["target_date"], errors="coerce")
    row = service[
        service["sigungu_key"].eq(municipality)
        & service["date"].eq(pd.Timestamp(reference_date))
    ]
    if row.empty:
        raise ValueError(f"No service feature row for municipality={municipality}, reference_date={reference_date}")
    row = row.iloc[[0]].copy()
    pred = float(np.clip(payload["model"].predict(row[features])[0], 0, None))
    recent = float(row["recent_7day_average_g"].iloc[0])
    change = pred / recent - 1 if recent > 0 else np.nan
    thresholds = payload["thresholds"]
    surge = classify_surge_ratio(change, thresholds["attention"], thresholds["high"])
    return {
        "municipality": municipality,
        "reference_date": pd.Timestamp(reference_date).date().isoformat(),
        "prediction_date": row["target_date"].iloc[0].date().isoformat(),
        "predicted_waste_g": pred,
        "predicted_waste_kg": pred / 1000,
        "recent_7day_average_g": recent,
        "change_vs_recent_average_pct": float(change * 100),
        "surge_level": surge,
        "model_version": payload["model_version"],
    }


def write_results_doc(
    comparison: pd.DataFrame,
    test_metrics: dict[str, Any],
    municipality: pd.DataFrame,
    size_groups: pd.DataFrame,
    feature_importance: pd.DataFrame,
    surge: pd.DataFrame,
    split_info: dict[str, Any],
    prediction_example: dict[str, Any],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Next-Day Modeling Results",
        "",
        "## Problem Definition",
        "",
        "날짜 `t`에 사용 가능한 지자체, 인구, 달력, 과거 배출 이력 feature를 이용해 실제 calendar day `t+1`의 `target_next_day` 배출량을 예측합니다.",
        "",
        "## Feature Availability And Leakage Control",
        "",
        "- `target_next_day`는 지자체별 `date + 1 day`가 실제 존재할 때만 생성했습니다.",
        "- 날짜 공백이 있는 경우 다음 관측일을 다음날로 연결하지 않습니다.",
        "- 서비스용 prediction 함수는 target이 없는 `next_day_service_features.csv`만 읽습니다.",
        "- 같은 날/다음날 `discharge_count`, 미래 target, 실제 관측 기상값은 사용하지 않습니다.",
        "",
        "## Time Split",
        "",
        "Split 기준은 `target_date`입니다.",
    ]
    for split, info in split_info.items():
        lines.append(f"- {split}: rows={info['rows']:,}, municipalities={info['municipalities']:,}, target_date={info['target_date_min']}~{info['target_date_max']}")
    lines.extend(
        [
            "",
            "## Features",
            "",
            "- " + ", ".join(FEATURES),
            "",
            "## Baselines And Validation Comparison",
            "",
            dataframe_to_markdown(comparison),
            "",
            "## Final Service Model",
            "",
            f"- Selected model: `{test_metrics['selected_model']}`",
            f"- Test MAE: {test_metrics['MAE']:,.2f}",
            f"- Test RMSE: {test_metrics['RMSE']:,.2f}",
            f"- Test R2: {test_metrics['R2']:.4f}",
            f"- Test mean target: {test_metrics['test_target_mean']:,.2f}",
            f"- MAE / mean target: {test_metrics['mae_to_mean_target_ratio']:.2%}",
            "",
            "## Municipality Performance",
            "",
            dataframe_to_markdown(municipality, max_rows=20),
            "",
            "## Size Group Performance",
            "",
            dataframe_to_markdown(size_groups),
            "",
            "## Surge Threshold Candidates",
            "",
            dataframe_to_markdown(surge),
            "",
            "## Feature Importance",
            "",
            dataframe_to_markdown(feature_importance, max_rows=20),
            "",
            "Feature importance is predictive diagnostic information, not causality.",
            "",
            "## Service Usage",
            "",
            "Example:",
            "",
            "```json",
            json.dumps(prediction_example, ensure_ascii=False, indent=2),
            "```",
            "",
            "## Limitations",
            "",
            "이번 서비스용 1차 모델은 기상변수를 제외했습니다. 향후 예보 데이터 또는 lagged weather feature를 추가해 service-time availability를 지키는 weather 모델을 별도로 비교할 수 있습니다.",
        ]
    )
    (DOCS_DIR / "next_day_modeling_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_next_day_modeling() -> dict[str, Any]:
    dataset, _service = build_next_day_dataset()
    splits = split_by_target_date(dataset)
    comparison, fitted, split_info = train_and_validate(dataset)
    selected_model = select_final_model(comparison)
    final_model = fitted[selected_model]
    test = splits["test"]
    test_pred = predict(final_model, test, FEATURES)
    test_metrics = metrics(test[TARGET], test_pred)
    test_metrics.update(
        {
            "selected_model": selected_model,
            "test_rows": int(len(test)),
            "test_municipalities": int(test["sigungu_key"].nunique()),
            "test_target_mean": float(test[TARGET].mean()),
            "mae_to_mean_target_ratio": float(test_metrics["MAE"] / test[TARGET].mean()),
            "features": FEATURES,
        }
    )
    municipality = municipality_metrics(test, test_pred)
    size_groups = size_group_metrics(municipality)
    importance = permutation_feature_importance(final_model, test)
    surge = surge_threshold_analysis(splits["train"])
    selected_threshold_row = surge[surge["candidate"].eq("q75_q90")].iloc[0]
    thresholds = {
        "attention": float(selected_threshold_row["attention_threshold_pct"] / 100),
        "high": float(selected_threshold_row["high_threshold_pct"] / 100),
    }

    NEXT_DAY_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(NEXT_DAY_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")
    (NEXT_DAY_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    municipality.to_csv(NEXT_DAY_DIR / "municipality_metrics.csv", index=False, encoding="utf-8-sig")
    surge.to_csv(NEXT_DAY_DIR / "surge_threshold_analysis.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(NEXT_DAY_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    size_groups.to_csv(NEXT_DAY_DIR / "size_group_metrics.csv", index=False, encoding="utf-8-sig")

    save_model(final_model, selected_model, thresholds, test_metrics)
    make_plots(test, test_pred, municipality)
    example_row = test.sort_values(["target_date", "sigungu_key"]).iloc[0]
    prediction_example = predict_next_day(str(example_row["sigungu_key"]), example_row["date"].date().isoformat())
    write_results_doc(comparison, test_metrics, municipality, size_groups, importance, surge, split_info, prediction_example)
    return {
        "dataset_rows": int(len(dataset)),
        "dataset_municipalities": int(dataset["sigungu_key"].nunique()),
        "split_info": split_info,
        "comparison": comparison.to_dict(orient="records"),
        "test_metrics": test_metrics,
        "top_municipality_errors": municipality.head(10).to_dict(orient="records"),
        "size_group_metrics": size_groups.to_dict(orient="records"),
        "feature_importance_top10": importance.head(10).to_dict(orient="records"),
        "surge_thresholds": surge.to_dict(orient="records"),
        "prediction_example": prediction_example,
    }


if __name__ == "__main__":
    print(json.dumps(run_next_day_modeling(), ensure_ascii=False, indent=2))
