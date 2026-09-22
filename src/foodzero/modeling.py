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
from .modeling_prep import NATIONWIDE_FEATURES, TARGET, WEATHER_FEATURES, WEATHER_VALIDATED_FEATURES


RANDOM_STATE = 42
TRAIN_START = "2021-01-01"
TRAIN_END = "2022-12-31"
VALIDATION_START = "2023-01-01"
VALIDATION_END = "2023-06-30"
TEST_START = "2023-07-01"
TEST_END = "2024-01-31"

MODELS_DIR = PROJECT_ROOT / "models"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
PLOTS_DIR = EVALUATION_DIR / "plots"

LEAKAGE_COLUMNS = {"discharge_count", "sumRn"}
CATEGORICAL_FEATURES = ["sigungu_key", "month", "day_of_week", "season"]


@dataclass(frozen=True)
class Experiment:
    name: str
    dataset_path: Path
    features: list[str]
    description: str


def sklearn_imports() -> dict[str, Any]:
    from joblib import dump, load
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return {
        "dump": dump,
        "load": load,
        "clone": clone,
        "ColumnTransformer": ColumnTransformer,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "RandomForestRegressor": RandomForestRegressor,
        "permutation_importance": permutation_importance,
        "Ridge": Ridge,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
    }


def experiments() -> list[Experiment]:
    weather_path = PROCESSED_DIR / "weather_validated_dataset.csv"
    nationwide_path = PROCESSED_DIR / "nationwide_dataset.csv"
    weather_without_weather = [feature for feature in WEATHER_VALIDATED_FEATURES if feature not in WEATHER_FEATURES]
    return [
        Experiment(
            name="weather_with_weather",
            dataset_path=weather_path,
            features=WEATHER_VALIDATED_FEATURES,
            description="Validated ASOS municipalities with weather features",
        ),
        Experiment(
            name="weather_without_weather",
            dataset_path=weather_path,
            features=weather_without_weather,
            description="Same rows as weather_validated_dataset, without weather features",
        ),
        Experiment(
            name="nationwide",
            dataset_path=nationwide_path,
            features=NATIONWIDE_FEATURES,
            description="Nationwide non-weather model dataset",
        ),
    ]


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={"station_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["date"].between(TRAIN_START, TRAIN_END)].copy(),
        "validation": df[df["date"].between(VALIDATION_START, VALIDATION_END)].copy(),
        "test": df[df["date"].between(TEST_START, TEST_END)].copy(),
    }


def split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, Any]:
    return {
        name: {
            "rows": int(len(frame)),
            "municipalities": int(frame["sigungu_key"].nunique()),
            "date_min": frame["date"].min().date().isoformat() if not frame.empty else None,
            "date_max": frame["date"].max().date().isoformat() if not frame.empty else None,
        }
        for name, frame in splits.items()
    }


def assert_no_split_overlap(splits: dict[str, pd.DataFrame]) -> None:
    ranges = {
        "train": (pd.Timestamp(TRAIN_START), pd.Timestamp(TRAIN_END)),
        "validation": (pd.Timestamp(VALIDATION_START), pd.Timestamp(VALIDATION_END)),
        "test": (pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)),
    }
    for left_name, (left_start, left_end) in ranges.items():
        for right_name, (right_start, right_end) in ranges.items():
            if left_name >= right_name:
                continue
            if max(left_start, right_start) <= min(left_end, right_end):
                raise ValueError(f"Split date ranges overlap: {left_name}, {right_name}")
    if not splits["train"]["date"].le(TRAIN_END).all():
        raise ValueError("Train split contains dates after train end.")
    if not splits["validation"]["date"].between(VALIDATION_START, VALIDATION_END).all():
        raise ValueError("Validation split has out-of-range dates.")
    if not splits["test"]["date"].ge(TEST_START).all():
        raise ValueError("Test split contains dates before test start.")


def assert_no_leakage_features(features: list[str]) -> None:
    forbidden = sorted(set(features) & LEAKAGE_COLUMNS)
    if forbidden:
        raise ValueError(f"Leakage or excluded features included: {forbidden}")


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> float:
    return float(math.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    sk = sklearn_imports()
    return {
        "MAE": float(sk["mean_absolute_error"](y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "R2": float(sk["r2_score"](y_true, y_pred)),
    }


def baseline_rows(experiment_name: str, split_name: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for baseline_name, pred_col in [("persistence_lag_1", "lag_1"), ("moving_average_7", "rolling_mean_7")]:
        pred = frame[pred_col].to_numpy()
        row = {
            "experiment": experiment_name,
            "model": baseline_name,
            "target_transform": "none",
            "split": split_name,
            "rows": int(len(frame)),
            **metrics(frame[TARGET], pred),
        }
        rows.append(row)
    return rows


def make_preprocessor(features: list[str], model_family: str):
    sk = sklearn_imports()
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    numeric = [feature for feature in features if feature not in categorical]
    encoder = sk["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)
    numeric_transformer = sk["StandardScaler"]() if model_family == "linear" else "passthrough"
    return sk["ColumnTransformer"](
        transformers=[
            ("categorical", encoder, categorical),
            ("numeric", numeric_transformer, numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_specs() -> list[dict[str, Any]]:
    sk = sklearn_imports()
    specs = [
        {
            "name": "ridge",
            "family": "linear",
            "estimator": sk["Ridge"](alpha=10.0),
        },
        {
            "name": "random_forest",
            "family": "tree",
            "estimator": sk["RandomForestRegressor"](
                n_estimators=80,
                max_depth=18,
                min_samples_leaf=3,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        },
        {
            "name": "hist_gradient_boosting",
            "family": "tree",
            "estimator": sk["HistGradientBoostingRegressor"](
                max_iter=180,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=RANDOM_STATE,
            ),
        },
    ]
    try:
        from xgboost import XGBRegressor  # type: ignore

        specs.append(
            {
                "name": "xgboost",
                "family": "tree",
                "estimator": XGBRegressor(
                    n_estimators=250,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            }
        )
    except Exception:
        pass
    return specs


def fit_pipeline(spec: dict[str, Any], features: list[str], x_train: pd.DataFrame, y_train: pd.Series, transform: str):
    sk = sklearn_imports()
    pipe = sk["Pipeline"](
        steps=[
            ("preprocessor", make_preprocessor(features, spec["family"])),
            ("model", sk["clone"](spec["estimator"])),
        ]
    )
    target = np.log1p(y_train) if transform == "log1p" else y_train
    pipe.fit(x_train[features], target)
    return pipe


def predict_original_units(model: Any, x: pd.DataFrame, features: list[str], transform: str) -> np.ndarray:
    pred = model.predict(x[features])
    if transform == "log1p":
        pred = np.expm1(pred)
    return np.clip(np.asarray(pred, dtype=float), 0, None)


def train_and_validate() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    split_info: dict[str, Any] = {}

    for experiment in experiments():
        assert_no_leakage_features(experiment.features)
        df = load_dataset(experiment.dataset_path)
        splits = split_data(df)
        assert_no_split_overlap(splits)
        split_info[experiment.name] = split_summary(splits)
        comparison_rows.extend(baseline_rows(experiment.name, "validation", splits["validation"]))

        for spec in model_specs():
            for transform in ["none", "log1p"]:
                model_key = f"{experiment.name}__{spec['name']}__{transform}"
                model = fit_pipeline(spec, experiment.features, splits["train"], splits["train"][TARGET], transform)
                pred = predict_original_units(model, splits["validation"], experiment.features, transform)
                comparison_rows.append(
                    {
                        "experiment": experiment.name,
                        "model": spec["name"],
                        "target_transform": transform,
                        "split": "validation",
                        "rows": int(len(splits["validation"])),
                        **metrics(splits["validation"][TARGET], pred),
                    }
                )
                fitted[model_key] = {
                    "model": model,
                    "experiment": experiment,
                    "spec": spec,
                    "target_transform": transform,
                    "validation_prediction": pred,
                }

    comparison = pd.DataFrame(comparison_rows).sort_values(["experiment", "MAE", "RMSE"]).reset_index(drop=True)
    return comparison, fitted, split_info


def select_final_model(comparison: pd.DataFrame) -> pd.Series:
    ml = comparison[~comparison["model"].isin(["persistence_lag_1", "moving_average_7"])].copy()
    return ml.sort_values(["MAE", "RMSE"], ascending=[True, True]).iloc[0]


def final_model_key(row: pd.Series) -> str:
    return f"{row['experiment']}__{row['model']}__{row['target_transform']}"


def municipality_metrics(test: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = test[["sigungu_key", TARGET]].copy()
    out["prediction"] = pred
    out["absolute_error"] = (out[TARGET] - out["prediction"]).abs()
    summary = (
        out.groupby("sigungu_key")
        .agg(
            rows=(TARGET, "size"),
            mean_waste_amount=(TARGET, "mean"),
            mae=("absolute_error", "mean"),
            rmse=("absolute_error", lambda x: math.sqrt(float(np.mean(np.square(x))))),
        )
        .reset_index()
    )
    summary["mae_to_mean_ratio"] = summary["mae"] / summary["mean_waste_amount"]
    return summary.sort_values("mae", ascending=False)


def permutation_feature_importance(model: Any, test: pd.DataFrame, features: list[str], transform: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    sample = test.sample(n=min(8000, len(test)), random_state=RANDOM_STATE).copy()
    y = sample[TARGET]
    baseline = metrics(y, predict_original_units(model, sample, features, transform))["MAE"]
    rows = []
    for feature in features:
        losses = []
        for _ in range(5):
            shuffled = sample.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            pred = predict_original_units(model, shuffled, features, transform)
            losses.append(metrics(y, pred)["MAE"])
        rows.append(
            {
                "feature": feature,
                "baseline_mae": baseline,
                "permuted_mae": float(np.mean(losses)),
                "importance_mae_increase": float(np.mean(losses) - baseline),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mae_increase", ascending=False)


def make_plots(test: pd.DataFrame, pred: np.ndarray, municipality: pd.DataFrame) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sample = test.assign(prediction=pred).sample(n=min(12000, len(test)), random_state=RANDOM_STATE)
    plt.figure(figsize=(7, 7))
    plt.scatter(sample[TARGET], sample["prediction"], s=5, alpha=0.25)
    upper = max(sample[TARGET].max(), sample["prediction"].max())
    plt.plot([0, upper], [0, upper], color="red", linewidth=1)
    plt.xlabel("Actual waste_amount")
    plt.ylabel("Predicted waste_amount")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=140)
    plt.close()

    residual = sample["prediction"] - sample[TARGET]
    plt.figure(figsize=(8, 5))
    plt.hist(residual, bins=80)
    plt.xlabel("Prediction residual")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_distribution.png", dpi=140)
    plt.close()

    top = municipality.head(25).sort_values("mae")
    plt.figure(figsize=(8, 8))
    plt.barh(top["sigungu_key"], top["mae"])
    plt.xlabel("Test MAE")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "municipality_mae_top25.png", dpi=140)
    plt.close()


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int | None = None) -> str:
    table = df.copy()
    if max_rows is not None:
        table = table.head(max_rows)
    if table.empty:
        return "_No rows_"
    formatted = table.copy()
    for col in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: f"{x:,.4f}" if pd.notna(x) else "")
        else:
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(formatted.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_results_doc(
    comparison: pd.DataFrame,
    test_metrics: dict[str, Any],
    municipality: pd.DataFrame,
    feature_importance: pd.DataFrame,
    split_info: dict[str, Any],
) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    best_validation = comparison[~comparison["model"].isin(["persistence_lag_1", "moving_average_7"])].sort_values("MAE").head(12)
    baseline = comparison[comparison["model"].isin(["persistence_lag_1", "moving_average_7"])]
    lines = [
        "# Modeling Results",
        "",
        "## Datasets",
        "",
        "- `weather_validated_dataset.csv`: validated ASOS weather experiment data",
        "- `nationwide_dataset.csv`: nationwide non-weather experiment data",
        "",
        "## Time Split",
        "",
        f"- Train: {TRAIN_START} ~ {TRAIN_END}",
        f"- Validation: {VALIDATION_START} ~ {VALIDATION_END}",
        f"- Test: {TEST_START} ~ {TEST_END}",
        "",
    ]
    for experiment, info in split_info.items():
        lines.append(f"### {experiment}")
        for split, values in info.items():
            lines.append(f"- {split}: rows={values['rows']:,}, municipalities={values['municipalities']:,}, dates={values['date_min']}~{values['date_max']}")
        lines.append("")
    lines.extend(
        [
            "## Features",
            "",
            "- Weather model features: " + ", ".join(WEATHER_VALIDATED_FEATURES),
            "- Nationwide model features: " + ", ".join(NATIONWIDE_FEATURES),
            "- Excluded: `discharge_count`, `sumRn`, future target-derived variables",
            "",
            "## Baselines",
            "",
            dataframe_to_markdown(baseline.sort_values(["experiment", "model"])),
            "",
            "## Validation Model Comparison",
            "",
            dataframe_to_markdown(best_validation),
            "",
            "## Final Model",
            "",
            f"- Selected by validation MAE: `{test_metrics['selected_experiment']} / {test_metrics['selected_model']} / {test_metrics['target_transform']}`",
            f"- Test MAE: {test_metrics['MAE']:,.2f}",
            f"- Test RMSE: {test_metrics['RMSE']:,.2f}",
            f"- Test R2: {test_metrics['R2']:.4f}",
            f"- Test mean target: {test_metrics['test_target_mean']:,.2f}",
            f"- MAE / mean target: {test_metrics['mae_to_mean_target_ratio']:.2%}",
            "",
            "## Weather Effect",
            "",
            "Weather effect is evaluated only inside the same `weather_validated_dataset.csv` rows by comparing `weather_with_weather` and `weather_without_weather`. Do not compare it directly with `nationwide`, because the municipality coverage differs.",
            "",
            "## Municipality Performance",
            "",
            dataframe_to_markdown(municipality, max_rows=20),
            "",
            "## Feature Importance",
            "",
            dataframe_to_markdown(feature_importance, max_rows=20),
            "",
            "Feature importance is model behavior diagnostics, not causal interpretation.",
            "",
            "## Limitations",
            "",
            "The weather model uses same-day ASOS observations. These are not known at a real future prediction time, so this experiment is a historical backtest. A production service should use weather forecasts or lagged weather variables.",
        ]
    )
    (DOCS_DIR / "modeling_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_modeling() -> dict[str, Any]:
    sk = sklearn_imports()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    comparison, fitted, split_info = train_and_validate()
    comparison.to_csv(EVALUATION_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    selected = select_final_model(comparison)
    key = final_model_key(selected)
    selected_info = fitted[key]
    experiment: Experiment = selected_info["experiment"]
    final_model = selected_info["model"]
    transform = selected_info["target_transform"]
    df = load_dataset(experiment.dataset_path)
    splits = split_data(df)
    test = splits["test"]
    test_pred = predict_original_units(final_model, test, experiment.features, transform)
    test_metric = metrics(test[TARGET], test_pred)
    test_metric.update(
        {
            "selected_experiment": experiment.name,
            "selected_model": selected_info["spec"]["name"],
            "target_transform": transform,
            "test_rows": int(len(test)),
            "test_municipalities": int(test["sigungu_key"].nunique()),
            "test_target_mean": float(test[TARGET].mean()),
            "mae_to_mean_target_ratio": float(test_metric["MAE"] / test[TARGET].mean()),
            "features": experiment.features,
        }
    )
    (EVALUATION_DIR / "test_metrics.json").write_text(json.dumps(test_metric, ensure_ascii=False, indent=2), encoding="utf-8")

    muni = municipality_metrics(test, test_pred)
    muni.to_csv(EVALUATION_DIR / "municipality_metrics.csv", index=False, encoding="utf-8-sig")

    importance = permutation_feature_importance(final_model, test, experiment.features, transform)
    importance.to_csv(EVALUATION_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")

    model_payload = {
        "model": final_model,
        "experiment": experiment.name,
        "features": experiment.features,
        "target_transform": transform,
        "selected_by": "validation_MAE",
        "test_metrics": test_metric,
    }
    sk["dump"](model_payload, MODELS_DIR / "foodzero_final_model.joblib")
    loaded = sk["load"](MODELS_DIR / "foodzero_final_model.joblib")
    loaded_pred = predict_original_units(loaded["model"], test.head(5), loaded["features"], loaded["target_transform"])
    if len(loaded_pred) != min(5, len(test)):
        raise RuntimeError("Loaded model prediction sanity check failed.")

    make_plots(test, test_pred, muni)
    write_results_doc(comparison, test_metric, muni, importance, split_info)
    return {
        "comparison_rows": int(len(comparison)),
        "selected": test_metric,
        "top_municipality_errors": muni.head(10).to_dict(orient="records"),
        "top_feature_importance": importance.head(10).to_dict(orient="records"),
        "split_info": split_info,
    }


if __name__ == "__main__":
    result = run_modeling()
    print(json.dumps(result, ensure_ascii=False, indent=2))
