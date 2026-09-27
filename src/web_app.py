from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from foodzero.next_day_modeling import predict_next_day, sklearn_imports


DEPLOY_DATA_DIR = PROJECT_ROOT / "data" / "deploy"
DEPLOY_MODEL_DIR = PROJECT_ROOT / "models" / "deploy"
NEXT_DAY_SERVICE_FEATURES_PATH = DEPLOY_DATA_DIR / "next_day_service_features.parquet"
NEXT_DAY_DATASET_PATH = DEPLOY_DATA_DIR / "next_day_model_dataset.parquet"
DATA_OVERVIEW_COUNTS_PATH = DEPLOY_DATA_DIR / "data_overview_counts.json"
NEXT_DAY_MODEL_PATH = DEPLOY_MODEL_DIR / "foodzero_next_day_model.joblib"
NEXT_DAY_EVAL_DIR = PROJECT_ROOT / "evaluation" / "deploy" / "next_day"
MODEL_VERSION = "foodzero-next-day-rf-v1"


STATUS_LABELS = {"normal": "평소 수준", "attention": "확인 필요", "high": "우선 확인"}
MANAGEMENT_STATUS_LABELS = {"normal": "안정", "attention": "주의", "high": "집중관리"}
STATUS_CLASS = {"normal": "status-normal", "attention": "status-attention", "high": "status-high"}
STATUS_ORDER = {"high": 0, "attention": 1, "normal": 2}
NAV_LABELS = {
    "Home": "FoodZero",
    "지역 현황": "지역 데이터",
    "배출량 예측": "배출량 예측",
    "수거·관리 지원": "수거·관리",
    "데이터 인사이트": "데이터 인사이트",
    "FoodZero 소개": "프로젝트 소개",
}
PLOT_FONT = "Pretendard, Pretendard Variable, SUIT, Inter, Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif"


def format_weight(value_g: float | int | None) -> str:
    if value_g is None or pd.isna(value_g):
        return "-"
    value = float(value_g)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f} t"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.1f} kg"
    return f"{value:,.0f} g"


def format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):+,.2f}%"


def format_abs_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):,.2f}%"


def split_admin_key(sigungu_key: str) -> tuple[str, str]:
    parts = str(sigungu_key).split(maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def normalize_status(level: str) -> str:
    return STATUS_LABELS.get(str(level), str(level))


def normalize_management_status(level: str) -> str:
    return MANAGEMENT_STATUS_LABELS.get(str(level), str(level))


def apply_plot_style(fig: go.Figure, height: int | None = None, showlegend: bool | None = None) -> go.Figure:
    layout: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": PLOT_FONT, "color": "#929891", "size": 17},
        "margin": {"l": 16, "r": 18, "t": 18, "b": 18},
        "xaxis": {
            "gridcolor": "rgba(255,255,255,.06)",
            "zerolinecolor": "rgba(255,255,255,.08)",
            "linecolor": "rgba(255,255,255,.10)",
            "tickfont": {"color": "#929891", "size": 16},
            "title": {"font": {"color": "#929891", "size": 16}},
        },
        "yaxis": {
            "gridcolor": "rgba(255,255,255,.06)",
            "zerolinecolor": "rgba(255,255,255,.08)",
            "linecolor": "rgba(255,255,255,.10)",
            "tickfont": {"color": "#929891", "size": 16},
            "title": {"font": {"color": "#929891", "size": 16}},
        },
        "legend": {"font": {"color": "#929891", "size": 16}, "orientation": "h", "y": 1.05},
    }
    if height:
        layout["height"] = height
    if showlegend is not None:
        layout["showlegend"] = showlegend
    fig.update_layout(**layout)
    return fig


def status_badge(level: str) -> str:
    label = normalize_status(level)
    css = STATUS_CLASS.get(level, "status-normal")
    if label == "확인 필요":
        content = "<span>확인</span><span>필요</span>"
        css = f"{css} status-two-line"
    elif label == "우선 확인":
        content = "<span>우선</span><span>확인</span>"
        css = f"{css} status-two-line"
    else:
        content = label
    return f'<span class="status-badge {css}" aria-label="{label}">{content}</span>'


def management_status_badge(level: str) -> str:
    label = normalize_management_status(level)
    css = STATUS_CLASS.get(level, "status-normal")
    return f'<span class="status-badge {css}">{label}</span>'


@st.cache_data(show_spinner=False)
def load_service_features() -> pd.DataFrame:
    df = pd.read_parquet(NEXT_DAY_SERVICE_FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
    sido_sigungu = df["sigungu_key"].map(split_admin_key)
    df["sido"] = sido_sigungu.map(lambda x: x[0])
    df["sigungu_name"] = sido_sigungu.map(lambda x: x[1])
    return df


@st.cache_data(show_spinner=False)
def load_next_day_dataset() -> pd.DataFrame:
    df = pd.read_parquet(NEXT_DAY_DATASET_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
    sido_sigungu = df["sigungu_key"].map(split_admin_key)
    df["sido"] = sido_sigungu.map(lambda x: x[0])
    df["sigungu_name"] = sido_sigungu.map(lambda x: x[1])
    return df


@st.cache_data(show_spinner=False)
def load_data_overview_counts() -> dict[str, int]:
    return json.loads(DATA_OVERVIEW_COUNTS_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def load_next_day_model() -> dict[str, Any]:
    sk = sklearn_imports()
    return sk["load"](NEXT_DAY_MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_evaluation_tables() -> dict[str, Any]:
    tables: dict[str, Any] = {
        "comparison": pd.read_csv(NEXT_DAY_EVAL_DIR / "model_comparison.csv", encoding="utf-8-sig"),
        "municipality": pd.read_csv(NEXT_DAY_EVAL_DIR / "municipality_metrics.csv", encoding="utf-8-sig"),
        "surge": pd.read_csv(NEXT_DAY_EVAL_DIR / "surge_threshold_analysis.csv", encoding="utf-8-sig"),
        "importance": pd.read_csv(NEXT_DAY_EVAL_DIR / "feature_importance.csv", encoding="utf-8-sig"),
        "size_group": pd.read_csv(NEXT_DAY_EVAL_DIR / "size_group_metrics.csv", encoding="utf-8-sig"),
    }
    tables["test_metrics"] = json.loads((NEXT_DAY_EVAL_DIR / "test_metrics.json").read_text(encoding="utf-8"))
    return tables


def classify_surge(change_pct: pd.Series | np.ndarray | float, attention_pct: float, high_pct: float):
    if np.isscalar(change_pct):
        value = float(change_pct)
        if value >= high_pct:
            return "high"
        if value >= attention_pct:
            return "attention"
        return "normal"
    series = pd.Series(change_pct)
    return pd.Series(np.select([series.ge(high_pct), series.ge(attention_pct)], ["high", "attention"], default="normal"), index=series.index)


def calculate_change_vs_recent_average_pct(predicted_g: pd.Series | np.ndarray | float, recent_average_g: pd.Series | np.ndarray | float):
    predicted = pd.Series(predicted_g, dtype="float64")
    recent = pd.Series(recent_average_g, dtype="float64")
    valid = recent.notna() & recent.gt(0)
    change = pd.Series(np.nan, index=predicted.index, dtype="float64")
    change.loc[valid] = (predicted.loc[valid] / recent.loc[valid] - 1) * 100
    if np.isscalar(predicted_g) and np.isscalar(recent_average_g):
        return float(change.iloc[0]) if not pd.isna(change.iloc[0]) else np.nan
    return change


def predict_for_reference_date(reference_date: pd.Timestamp) -> pd.DataFrame:
    service = load_service_features()
    payload = load_next_day_model()
    rows = service[service["date"].eq(reference_date)].copy()
    if rows.empty:
        return rows
    predictions = np.clip(payload["model"].predict(rows[payload["features"]]), 0, None)
    rows["predicted_waste_g"] = predictions
    rows["predicted_waste_kg"] = rows["predicted_waste_g"] / 1000
    rows["change_vs_recent_average_pct"] = calculate_change_vs_recent_average_pct(
        rows["predicted_waste_g"],
        rows["recent_7day_average_g"],
    )
    rows["surge_level"] = classify_surge(
        rows["change_vs_recent_average_pct"],
        payload["thresholds"]["attention"] * 100,
        payload["thresholds"]["high"] * 100,
    )
    rows["surge_label"] = rows["surge_level"].map(normalize_status)
    return rows.sort_values(["surge_level", "predicted_waste_g"], key=lambda s: s.map(STATUS_ORDER).fillna(s) if s.name == "surge_level" else s, ascending=[True, False])


def dashboard_kpis(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions.empty:
        return {
            "municipalities": 0,
            "predicted_total_g": np.nan,
            "surge_regions": 0,
            "priority_regions": 0,
        }
    return {
        "municipalities": int(predictions["sigungu_key"].nunique()),
        "predicted_total_g": float(predictions["predicted_waste_g"].sum()),
        "surge_regions": int(predictions["surge_level"].isin(["attention", "high"]).sum()),
        "priority_regions": int(predictions["surge_level"].eq("high").sum()),
    }


def management_kpis(predictions: pd.DataFrame) -> dict[str, int]:
    if predictions.empty:
        return {"municipalities": 0, "high": 0, "attention": 0, "normal": 0}
    return {
        "municipalities": int(predictions["sigungu_key"].nunique()),
        "high": int(predictions["surge_level"].eq("high").sum()),
        "attention": int(predictions["surge_level"].eq("attention").sum()),
        "normal": int(predictions["surge_level"].eq("normal").sum()),
    }


def prepare_management_predictions(reference_date: pd.Timestamp) -> pd.DataFrame:
    predictions = predict_for_reference_date(reference_date).copy()
    if predictions.empty:
        return predictions

    metrics = load_evaluation_tables()["municipality"][["sigungu_key", "mae_to_mean_ratio"]].copy()
    predictions = predictions.merge(metrics, on="sigungu_key", how="left")
    predictions["risk_label"] = predictions["surge_level"].map(normalize_management_status)
    predictions["risk_order"] = predictions["surge_level"].map(STATUS_ORDER).fillna(STATUS_ORDER["normal"])
    predictions = predictions.sort_values(
        ["risk_order", "change_vs_recent_average_pct", "predicted_waste_g"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    predictions["management_rank"] = np.arange(1, len(predictions) + 1)
    return predictions


def priority_management_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    return predictions[predictions["surge_level"].isin(["high", "attention"])].copy()


def make_display_table(predictions: pd.DataFrame) -> pd.DataFrame:
    cols = ["sigungu_key", "predicted_waste_g", "recent_7day_average_g", "change_vs_recent_average_pct", "surge_label"]
    table = predictions[cols].copy()
    table = table.rename(
        columns={
            "sigungu_key": "지자체",
            "predicted_waste_g": "예상 배출량",
            "recent_7day_average_g": "최근 7일 평균",
            "change_vs_recent_average_pct": "예상 증감률",
            "surge_label": "상태",
        }
    )
    table["예상 배출량"] = table["예상 배출량"].map(format_weight)
    table["최근 7일 평균"] = table["최근 7일 평균"].map(format_weight)
    table["예상 증감률"] = table["예상 증감률"].map(format_pct)
    return table


def get_available_sidos() -> list[str]:
    return sorted(load_service_features()["sido"].dropna().unique().tolist())


def get_municipalities(sido: str | None = None) -> list[str]:
    df = load_service_features()
    if sido and sido != "전체":
        df = df[df["sido"].eq(sido)]
    return sorted(df["sigungu_key"].dropna().unique().tolist())


def get_available_reference_dates(municipality: str | None = None) -> list[pd.Timestamp]:
    df = load_service_features()
    if municipality:
        df = df[df["sigungu_key"].eq(municipality)]
    return sorted(pd.to_datetime(df["date"].dropna().unique()).tolist())


def run_prediction(municipality: str, reference_date: pd.Timestamp) -> dict[str, Any]:
    return predict_next_day(
        municipality,
        reference_date.date().isoformat(),
        model_path=NEXT_DAY_MODEL_PATH,
        service_features_path=NEXT_DAY_SERVICE_FEATURES_PATH,
    )


def actual_outcome_for_prediction(municipality: str, prediction_date: pd.Timestamp) -> float | None:
    df = load_next_day_dataset()
    target_date = pd.Timestamp(prediction_date)
    match = df[df["sigungu_key"].eq(municipality) & df["target_date"].eq(target_date)]
    if match.empty:
        return None
    actual = match["target_next_day"].dropna()
    if actual.empty:
        return None
    return float(actual.iloc[0])


def prediction_error_summary(predicted_g: float, actual_g: float | None) -> dict[str, float | None]:
    if actual_g is None or pd.isna(actual_g):
        return {"actual_g": None, "absolute_error_g": None, "absolute_percentage_error": None}
    absolute_error = abs(float(actual_g) - float(predicted_g))
    absolute_percentage_error = absolute_error / float(actual_g) * 100 if float(actual_g) else None
    return {
        "actual_g": float(actual_g),
        "absolute_error_g": absolute_error,
        "absolute_percentage_error": absolute_percentage_error,
    }


def recent_trend(municipality: str, reference_date: pd.Timestamp, days: int = 30) -> pd.DataFrame:
    df = load_next_day_dataset()
    start = reference_date - pd.Timedelta(days=days - 1)
    return df[df["sigungu_key"].eq(municipality) & df["date"].between(start, reference_date)].sort_values("date")


def apply_css() -> None:
    st.html(
        """
        <style>
        :root {
          --fz-bg: #f7f8f4;
          --fz-bg-2: #eef2e9;
          --fz-surface: #ffffff;
          --fz-surface-soft: #fbfcf7;
          --fz-green: #0e5a3a;
          --fz-green-2: #1c7a50;
          --fz-sage: #e8efe8;
          --fz-sage-2: #f0f5ef;
          --fz-charcoal: #101914;
          --fz-text: #202a24;
          --fz-muted: #68716b;
          --fz-muted-2: #7b887f;
          --fz-border: #dce2d7;
          --fz-border-2: #ebeee7;
          --fz-amber: #b8792a;
          --fz-coral: #c85f4d;
          --fz-soft-green: #edf5ec;
          --fz-soft-amber: #fff5df;
          --fz-soft-red: #fff0ec;
          --fz-dark: #0d1210;
        }
        html, body, .stApp {
          background: var(--fz-bg);
          color: var(--fz-text);
          font-family: Pretendard, "SUIT", Inter, "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
        }
        [data-testid="stSidebar"] {
          display: block;
          background: #f1f4ed;
          border-right: 1px solid var(--fz-border);
          box-shadow: none;
        }
        [data-testid="stSidebar"] > div:first-child {
          background: #f1f4ed;
          width: 238px;
          padding: 0;
        }
        [data-testid="stSidebarContent"] {
          background: #f1f4ed;
          padding: 0;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        button[kind="header"] {
          display: none !important;
          visibility: hidden !important;
        }
        #MainMenu,
        footer,
        header,
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"] {
          display: none !important;
          visibility: hidden !important;
          height: 0 !important;
        }
        .stApp > header {
          display: none !important;
          height: 0 !important;
        }
        .block-container {
          max-width: 1360px;
          width: 100%;
          padding: 3rem 3.5rem 3.2rem;
        }
        div[data-testid="stVerticalBlock"] { gap: 0.9rem; }
        .fz-sidebar-brand {
          padding: 28px 22px 22px;
          border-bottom: 1px solid rgba(47, 107, 79, 0.13);
        }
        .fz-sidebar-title {
          color: var(--fz-charcoal);
          font-size: 27px;
          font-weight: 840;
          letter-spacing: 0;
          line-height: 1.05;
        }
        .fz-sidebar-subtitle {
          color: var(--fz-muted);
          font-size: 12px;
          text-transform: uppercase;
          letter-spacing: 0.04em;
          margin-top: 7px;
          line-height: 1.45;
        }
        .fz-sidebar-meta {
          margin: 46px 22px 22px;
          padding-top: 18px;
          border-top: 1px solid rgba(47, 107, 79, 0.13);
          color: var(--fz-muted);
          font-size: 12px;
          line-height: 1.65;
          text-transform: uppercase;
          letter-spacing: 0.04em;
        }
        .fz-sidebar-meta strong {
          display: block;
          color: var(--fz-text);
          font-size: 13px;
          margin-top: 2px;
          margin-bottom: 10px;
        }
        .fz-header {
          background: transparent;
          border-bottom: 1px solid rgba(47, 107, 79, 0.16);
          padding: 24px 0 18px;
          margin-bottom: 2px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 20px;
        }
        .fz-brand { display:flex; align-items:center; gap: 14px; }
        .fz-mark { display:none; }
        .fz-title { font-size: 30px; font-weight: 820; line-height: 1.08; color: var(--fz-charcoal); letter-spacing: 0; }
        .fz-subtitle { color: var(--fz-muted); font-size: 15px; margin-top: 7px; }
        .fz-meta { display:flex; align-items:center; gap: 10px; flex-wrap: wrap; justify-content:flex-end; }
        .meta-item {
          border: 0;
          background: transparent;
          padding: 0;
          min-width: auto;
        }
        .meta-label { display:none; }
        .meta-value { color: var(--fz-muted); font-size: 14px; font-weight: 620; }
        .ops-dot { display:none; }

        [data-testid="stSidebar"] div[role="radiogroup"] {
          background: transparent;
          border-bottom: 0;
          display: flex;
          flex-direction: column;
          gap: 4px;
          padding: 16px 14px 0;
          margin: 0;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label {
          padding: 0;
          margin: 0;
          min-height: 46px;
          border-bottom: 0;
          border-left: 3px solid transparent;
          border-radius: 9px;
          cursor: pointer;
          transition: background-color 0.16s ease, border-color 0.16s ease;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
          background: rgba(47, 107, 79, 0.06);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display: none !important; }
        [data-testid="stSidebar"] [data-testid="stRadio"] input,
        [data-testid="stSidebar"] div[role="radiogroup"] label::before,
        [data-testid="stSidebar"] div[role="radiogroup"] label::after,
        [data-testid="stSidebar"] div[role="radiogroup"] label p::before,
        [data-testid="stSidebar"] div[role="radiogroup"] label p::after,
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child::before,
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child::after,
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child > div,
        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child > div > div { display: none !important; content: none !important; }
        [data-testid="stSidebar"] div[role="radiogroup"] label > div {
          display: flex;
          align-items: center;
          min-height: 46px;
          padding: 0 14px;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label p {
          color: #44534b;
          font-weight: 670;
          font-size: 16px;
          line-height: 1.2;
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
          background: #e4eddf;
          border-left-color: var(--fz-green);
        }
        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
          color: var(--fz-green);
          font-weight: 760;
        }
        [data-testid="stSidebar"] .stButton {
          padding: 0 14px;
          margin: 0;
        }
        [data-testid="stSidebar"] .stButton > button {
          width: 100%;
          min-height: 46px;
          justify-content: flex-start;
          text-align: left;
          padding: 0 14px;
          border: 1px solid transparent;
          border-left: 3px solid transparent;
          border-radius: 9px;
          background: transparent !important;
          color: #44534b !important;
          font-size: 16px;
          font-weight: 670;
          box-shadow: none !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
          background: rgba(47, 107, 79, 0.06) !important;
          border-color: transparent !important;
          color: var(--fz-green) !important;
        }
        [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {
          background: #e4eddf !important;
          border-left-color: var(--fz-green) !important;
          color: var(--fz-green) !important;
          font-weight: 760;
        }
        [data-testid="stSidebar"] button[kind="primary"] {
          background: #e4eddf !important;
          border-left-color: var(--fz-green) !important;
          color: var(--fz-green) !important;
          font-weight: 760;
        }

        .page-hero {
          display:flex;
          justify-content:space-between;
          align-items:flex-end;
          gap:28px;
          margin: 28px 0 24px;
        }
        .fz-page-title { font-size: 38px; font-weight: 820; margin: 0 0 10px; color: var(--fz-charcoal); letter-spacing: 0; }
        .fz-page-desc { color: var(--fz-muted); font-size: 17px; line-height: 1.62; margin: 0; max-width: 840px; }
        .eyebrow {
          color: var(--fz-green);
          font-size: 12px;
          font-weight: 820;
          letter-spacing: 0.11em;
          text-transform: uppercase;
          margin-bottom: 16px;
        }
        .landing-hero {
          display:grid;
          grid-template-columns: minmax(0, 1.12fr) minmax(420px, 0.88fr);
          gap: clamp(34px, 5vw, 78px);
          align-items:center;
          min-height: 610px;
          padding: 42px 0 82px;
          position: relative;
        }
        .landing-title {
          color: var(--fz-charcoal);
          font-size: clamp(48px, 5.1vw, 68px);
          line-height: 1.08;
          font-weight: 730;
          letter-spacing: -0.045em;
          margin: 0 0 26px;
          max-width: 760px;
        }
        .landing-hero-copy {
          padding: 32px 0 40px;
          max-width: 720px;
        }
        .landing-copy {
          color: var(--fz-muted);
          font-size: 18px;
          line-height: 1.72;
          max-width: 600px;
          margin-bottom: 30px;
        }
        .sentence-line { display:block; white-space:nowrap; }
        .landing-copy:has(.sentence-line), .home-section-copy:has(.sentence-line), .editorial-copy:has(.sentence-line) { max-width:none; }
        .cta-row { display:flex; gap: 12px; flex-wrap:wrap; align-items:center; }
        .home-cta-row {
          display:flex;
          gap: 10px;
          flex-wrap: wrap;
          margin-top: 10px;
        }
        .home-cta-row [data-testid="stButton"] > button {
          min-height: 46px;
          border-radius: 7px;
          padding: 0 18px;
          font-size: 14px;
          font-weight: 760;
          box-shadow: none;
        }
        .home-cta-row [data-testid="stBaseButton-primary"] {
          background: var(--fz-green);
          border-color: var(--fz-green);
          color: #fff;
        }
        .home-cta-row [data-testid="stBaseButton-secondary"] {
          background: rgba(255, 255, 255, 0.52);
          border-color: var(--fz-border);
          color: var(--fz-charcoal);
        }
        .hero-visual-wrap {
          position: relative;
          min-height: 430px;
          display:flex;
          align-items:center;
        }
        .data-visual {
          width: 100%;
          background: rgba(255, 255, 255, 0.62);
          border: 1px solid rgba(20, 90, 60, 0.16);
          border-radius: 20px;
          padding: 28px 28px 24px;
          min-height: 430px;
          position: relative;
          overflow:hidden;
          box-shadow: 0 18px 55px rgba(23, 56, 38, 0.07);
        }
        .data-visual:before {
          content:"";
          position:absolute;
          inset: 0;
          background:
            radial-gradient(circle at 78% 16%, rgba(112, 158, 123, 0.20), transparent 38%),
            repeating-linear-gradient(90deg, rgba(14, 90, 58, 0.045) 0 1px, transparent 1px 42px),
            repeating-linear-gradient(0deg, rgba(14, 90, 58, 0.035) 0 1px, transparent 1px 42px);
          pointer-events:none;
        }
        .visual-title { position:relative; color: var(--fz-charcoal); font-size: 15px; font-weight: 800; letter-spacing: 0.01em; margin-bottom: 7px; }
        .visual-caption { position:relative; color: var(--fz-muted); font-size: 13px; line-height: 1.55; margin-bottom: 20px; max-width: 320px; }
        .visual-svg { position:relative; display:block; width:100%; height:auto; overflow:visible; }
        .visual-svg .flow-path { fill:none; stroke-linecap:round; stroke-width:2.2; opacity:0.82; }
        .visual-svg .flow-point { stroke:#fff; stroke-width:2; }
        .visual-svg .flow-label { fill:var(--fz-muted); font-size:10px; font-family:inherit; }
        .visual-svg .flow-number { fill:var(--fz-charcoal); font-size:11px; font-weight:700; font-family:inherit; }
        .visual-footer { position:relative; display:flex; justify-content:space-between; gap:12px; margin-top: 4px; padding-top: 14px; border-top:1px solid rgba(20, 90, 60, 0.13); color:var(--fz-muted); font-size:11px; letter-spacing:0.04em; text-transform:uppercase; }
        .editorial-section { padding: 74px 0 24px; border-top: 1px solid var(--fz-border); }
        .editorial-kicker { color: var(--fz-green); font-size: 12px; font-weight: 820; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px; }
        .editorial-title { color: var(--fz-charcoal); font-size: clamp(30px, 3.2vw, 44px); line-height: 1.16; font-weight: 720; letter-spacing:-0.035em; margin-bottom: 12px; }
        .editorial-copy { color: var(--fz-muted); font-size: 16px; line-height: 1.68; max-width: 760px; }
        .editorial-stats { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 24px; margin-top: 34px; padding: 26px 0 30px; border-top:1px solid var(--fz-border); border-bottom:1px solid var(--fz-border); }
        .editorial-stat-value { color: var(--fz-charcoal); font-size: clamp(36px, 4vw, 56px); font-weight: 700; letter-spacing:-0.04em; line-height:1; }
        .editorial-stat-label { color: var(--fz-muted); font-size: 14px; margin-top: 9px; }
        .home-section { padding: 76px 0 26px; border-top: 1px solid var(--fz-border); }
        .home-section-grid { display:grid; grid-template-columns: minmax(250px, 0.72fr) minmax(0, 1.28fr); gap: clamp(30px, 6vw, 92px); align-items:start; }
        .home-section-grid-wide { display:grid; grid-template-columns: minmax(300px, 0.88fr) minmax(0, 1.12fr); gap: clamp(30px, 6vw, 92px); align-items:start; }
        .home-section-kicker { color:var(--fz-green); font-size:11px; font-weight:800; letter-spacing:0.13em; text-transform:uppercase; margin-bottom:14px; }
        .home-section-title { color:var(--fz-charcoal); font-size:clamp(30px, 3.3vw, 46px); line-height:1.12; font-weight:720; letter-spacing:-0.04em; margin:0 0 14px; }
        .home-section-copy { color:var(--fz-muted); font-size:16px; line-height:1.72; max-width:430px; }
        .home-insight { margin-top:28px; padding-top:18px; border-top:1px solid var(--fz-border); color:var(--fz-charcoal); font-size:14px; line-height:1.65; }
        .home-insight strong { color:var(--fz-green); font-weight:800; }
        .home-ranking { min-width:0; }
        .home-ranking-head { display:flex; justify-content:space-between; align-items:flex-end; gap:14px; margin-bottom:12px; }
        .home-ranking-label { color:var(--fz-muted); font-size:12px; text-transform:uppercase; letter-spacing:0.1em; }
        .compact-list { display:grid; gap: 0; border-top: 1px solid var(--fz-border); }
        .compact-row { display:grid; grid-template-columns: minmax(150px, 1.35fr) 0.8fr 0.8fr 0.6fr 0.7fr; gap: 12px; align-items:center; background:transparent; padding: 16px 0; color: var(--fz-text); font-size: 14px; border-bottom:1px solid var(--fz-border); }
        .compact-row.head { padding: 10px 0; color: var(--fz-muted); font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; }
        .risk-distribution { display:grid; gap: 12px; margin: 16px 0 34px; padding: 18px 0 8px; border-top:1px solid var(--fz-border); }
        .risk-row { display:grid; grid-template-columns: 112px minmax(0, 1fr) 94px; gap: 16px; align-items:center; color:var(--fz-text); }
        .risk-label { color:var(--fz-charcoal); font-size:14px; font-weight:760; }
        .risk-track { height: 12px; border-radius:999px; background:rgba(47,107,79,.10); overflow:hidden; border:1px solid rgba(47,107,79,.10); }
        .risk-fill { height:100%; border-radius:999px; }
        .risk-fill-high { background:var(--fz-coral); }
        .risk-fill-attention { background:var(--fz-amber); }
        .risk-fill-normal { background:var(--fz-green); }
        .risk-value { color:var(--fz-muted); font-size:13px; text-align:right; }
        .risk-summary { display:flex; justify-content:space-between; gap:18px; align-items:flex-end; margin-bottom:4px; padding-bottom:14px; border-bottom:1px solid var(--fz-border-2); }
        .risk-summary-label { color:var(--fz-muted); font-size:12px; letter-spacing:0.09em; text-transform:uppercase; }
        .risk-summary-value { color:var(--fz-charcoal); font-size:24px; font-weight:780; line-height:1.1; margin-top:4px; }
        .risk-summary-note { color:var(--fz-muted); font-size:13px; text-align:right; line-height:1.5; }
        .management-table th,
        .management-table td,
        .management-table .num { text-align:left !important; }
        .management-table .status-badge { margin-left:0; }
        .management-table th:nth-child(2),
        .management-table td:nth-child(2) { min-width: 170px; }
        .home-forecast { padding: 80px 0 32px; border-top: 1px solid var(--fz-border); }
        .home-forecast-inner { display:grid; grid-template-columns:minmax(0, 1fr) auto; gap:30px; align-items:end; }
        .home-forecast .editorial-copy { max-width: 660px; }
        .home-forecast-note { color:var(--fz-muted); font-size:11px; letter-spacing:0.11em; text-transform:uppercase; margin-top:22px; }
        .dark-band { background: var(--fz-dark); color:#fff; border-radius: 18px; padding: clamp(38px, 6vw, 72px) clamp(28px, 6vw, 76px); margin: 54px 0 18px; position:relative; overflow:hidden; }
        .dark-band:after { content:""; position:absolute; width:420px; height:420px; right:-130px; top:-170px; border:1px solid rgba(157, 190, 163, 0.18); border-radius:50%; box-shadow:0 0 0 42px rgba(157, 190, 163, 0.04), 0 0 0 84px rgba(157, 190, 163, 0.025); pointer-events:none; }
        .dark-band h2 { color:#fff; font-size:clamp(36px, 5vw, 62px); line-height:1.08; letter-spacing:-0.045em; font-weight:700; margin:0 0 14px; position:relative; z-index:1; }
        .dark-band p { color:#aab5ad; font-size: 16px; line-height:1.7; max-width:660px; margin-bottom: 24px; position:relative; z-index:1; }
        .home-dark-kicker { color:#a8c5ac; font-size:11px; font-weight:800; letter-spacing:0.14em; text-transform:uppercase; margin-bottom:18px; position:relative; z-index:1; }
        .home-dark-cta { position:relative; z-index:1; }
        .home-dark-cta [data-testid="stBaseButton-secondary"] { background:transparent; border-color:rgba(176, 203, 181, 0.35); color:#e7f0e8; }
        .control-caption { color: var(--fz-muted); font-size: 13px; font-weight: 740; margin-bottom: 7px; }
        .filter-toolbar {
          display:flex;
          align-items:flex-end;
          gap:12px;
          flex-wrap:wrap;
          margin: 8px 0 18px;
        }

        div[data-baseweb="select"] > div {
          background: #fff !important;
          border: 1px solid var(--fz-border) !important;
          border-radius: 8px !important;
          min-height: 46px;
          box-shadow: none !important;
        }
        div[data-baseweb="select"] > div:focus-within {
          border-color: var(--fz-green) !important;
          box-shadow: 0 0 0 3px rgba(47, 107, 79, 0.10) !important;
        }
        div[data-baseweb="select"] span, div[data-baseweb="select"] div {
          color: var(--fz-text) !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] * {
          color: var(--fz-text) !important;
        }
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] [role="listbox"] {
          background: #fff !important;
          border: 1px solid var(--fz-border) !important;
          border-radius: 8px !important;
        }
        div[data-baseweb="popover"] [role="option"] {
          background: #fff !important;
          color: var(--fz-text) !important;
        }
        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] [aria-selected="true"] {
          background: #edf3e9 !important;
          color: var(--fz-green) !important;
        }
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stDateInput"] label p,
        div[data-testid="stMultiSelect"] label p {
          color: var(--fz-muted);
          font-size: 13px;
          font-weight: 750;
        }
        .stDateInput input {
          background: #fff !important;
          color: var(--fz-text) !important;
          border: 1px solid var(--fz-border) !important;
          border-radius: 8px !important;
          min-height: 46px;
        }
        .summary-band {
          background: linear-gradient(135deg, #ffffff 0%, #f0f5ec 100%);
          border: 1px solid var(--fz-border);
          border-radius: 14px;
          padding: 30px 34px;
          margin: 22px 0 34px;
          display: grid;
          grid-template-columns: minmax(320px, 1.55fr) repeat(3, minmax(120px, 0.55fr));
          gap: 28px;
          align-items:end;
        }
        .summary-main-label, .summary-label { color: var(--fz-muted); font-size: 14px; font-weight: 760; margin-bottom: 9px; }
        .summary-main-value { color: var(--fz-charcoal); font-size: 52px; font-weight: 830; line-height: 1.02; letter-spacing: 0; }
        .summary-main-note, .summary-note { color: var(--fz-muted); font-size: 13px; margin-top: 8px; }
        .summary-value { color: var(--fz-charcoal); font-size: 36px; font-weight: 820; line-height: 1.05; }
        .stat-row { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 22px; border-top:1px solid var(--fz-border); border-bottom:1px solid var(--fz-border); padding: 18px 0; margin: 18px 0 30px; }
        .stat-label { color: var(--fz-muted); font-size: 13px; font-weight: 760; margin-bottom: 7px; }
        .stat-value { color: var(--fz-charcoal); font-size: 30px; font-weight: 810; line-height:1.08; }
        .stat-note { color: var(--fz-muted); font-size: 13px; margin-top: 5px; }
        .section {
          background: transparent;
          border: 0;
          border-radius: 0;
          padding: 0;
          margin: 20px 0 24px;
        }
        .section-head {
          display:flex;
          align-items:flex-end;
          justify-content:space-between;
          gap: 16px;
          margin: 28px 0 14px;
        }
        .section-title { font-size: 24px; font-weight: 800; color: var(--fz-charcoal); margin-bottom: 5px; letter-spacing: 0; }
        .section-desc { color: var(--fz-muted); font-size: 15px; line-height: 1.55; margin-bottom: 0; }
        .status-badge {
          display:inline-flex;
          align-items:center;
          justify-content:center;
          min-width: 46px;
          padding: 4px 8px;
          border-radius: 999px;
          font-size: 12px;
          font-weight: 800;
          border: 1px solid transparent;
        }
        .status-badge.status-two-line {
          flex-direction:column;
          gap:0;
          line-height:1.05;
          white-space:normal;
        }
        .status-normal { color:#2f6b4f; background: var(--fz-soft-green); border-color:#cbdcc6; }
        .status-attention { color:#96651c; background: var(--fz-soft-amber); border-color:#efd39a; }
        .status-high { color:#b74e3d; background: var(--fz-soft-red); border-color:#edc0b8; }
        .notice {
          border-left: 4px solid var(--fz-green); background:#eef5ec; padding: 14px 16px;
          color:#31473b; margin: 14px 0 18px; font-size: 15px;
        }
        .fz-table {
          width:100%;
          border-collapse: collapse;
          font-size: 14px;
          background: #fff;
          border: 1px solid var(--fz-border);
          border-radius: 10px;
          overflow: hidden;
        }
        .fz-table thead th {
          background: #f2f5ef;
          color: var(--fz-text);
          font-weight: 800;
          text-align: left;
          padding: 13px 14px;
          border-bottom: 1px solid var(--fz-border);
        }
        .fz-table tbody td {
          background:#fff;
          color: var(--fz-text);
          padding: 13px 14px;
          border-bottom: 1px solid var(--fz-border-2);
          vertical-align: middle;
        }
        .fz-table tbody tr:last-child td { border-bottom: 0; }
        .fz-table .num { text-align:right; font-variant-numeric: tabular-nums; }
        div[data-testid="stDataFrame"] {
          border: 1px solid var(--fz-border);
          border-radius: 10px;
          overflow: hidden;
        }
        .footer {
          border-top: 1px solid var(--fz-border); margin-top: 42px; padding: 20px 0;
          color: var(--fz-muted); font-size: 14px; display:flex; justify-content:space-between; gap:12px;
        }
        div[data-testid="stMetric"] {
          background:#fff; border:1px solid var(--fz-border); border-radius:10px; padding: 17px;
        }
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
          color: var(--fz-charcoal) !important;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
          font-size: 32px;
          font-weight: 820;
          letter-spacing: 0;
          line-height: 1.12;
        }
        .stButton > button {
          background: var(--fz-green); color: white; border: 1px solid var(--fz-green);
          border-radius: 8px; font-weight: 800; min-height: 46px;
          padding-left: 18px; padding-right: 18px;
        }
        [data-testid="stBaseButton-secondary"] {
          background: rgba(255, 255, 255, 0.58) !important;
          color: var(--fz-charcoal) !important;
          border-color: var(--fz-border) !important;
        }
        [data-testid="stBaseButton-secondary"]:hover {
          background: var(--fz-sage-2) !important;
          color: var(--fz-green) !important;
          border-color: var(--fz-green) !important;
        }
        .stButton > button:hover { background: #285b43; border-color: #285b43; color: white; }
        div[data-testid="stPlotlyChart"] {
          background: transparent;
          border: 0;
          border-radius: 0;
          padding: 0;
        }
        .prediction-spotlight {
          background: linear-gradient(135deg, #ffffff 0%, #eef5ec 100%);
          border: 1px solid var(--fz-border);
          border-radius: 14px;
          padding: 28px 32px;
          display:grid;
          grid-template-columns: minmax(220px, 1.2fr) repeat(3, minmax(118px, 0.7fr));
          gap: 24px;
          align-items:end;
          margin: 16px 0 28px;
        }
        .prediction-date { color: var(--fz-muted); font-size: 17px; font-weight: 760; margin-bottom: 8px; }
        .prediction-label { color: var(--fz-muted); font-size: 14px; font-weight: 760; margin-bottom: 8px; }
        .prediction-main { color: var(--fz-charcoal); font-size: 50px; font-weight: 830; line-height:1.02; }
        .prediction-sub { color: var(--fz-charcoal); font-size: 30px; font-weight: 820; }
        .story-grid { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; margin: 24px 0 30px; }
        .story-step { border-top: 3px solid var(--fz-green); padding-top: 14px; }
        .story-num { color: var(--fz-green); font-size: 13px; font-weight: 820; margin-bottom: 7px; }
        .story-title { color: var(--fz-charcoal); font-size: 21px; font-weight: 800; margin-bottom: 8px; }
        .story-text { color: var(--fz-muted); font-size: 15px; line-height: 1.6; }
        .pipeline-steps { display:grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 18px 0 22px; }
        .pipeline-step { background:#fff; border:1px solid var(--fz-border); border-radius: 10px; padding: 16px; color: var(--fz-text); font-weight:760; text-align:center; }
        .tech-note { color: var(--fz-muted); font-size: 14px; line-height: 1.65; background: #fff; border: 1px solid var(--fz-border); border-radius: 10px; padding: 16px 18px; }
        .intro-quiet { color: var(--fz-muted); font-size: 15px; line-height: 1.65; max-width: 900px; }
        .light-list { color: var(--fz-text); font-size: 15px; line-height: 1.75; }
        .light-list li { margin-bottom: 5px; }
        .emphasis { color: var(--fz-green); font-weight: 820; }
        div[data-testid="stAlert"] {
          background: #eef5ec;
          color: var(--fz-text);
          border: 1px solid #cddfc8;
          border-radius: 10px;
        }
        @media (max-width: 1100px) {
          .landing-hero { grid-template-columns: 1fr; min-height: auto; }
          .hero-visual-wrap { min-height: 0; }
          .home-section-grid, .home-section-grid-wide { grid-template-columns: 1fr; gap: 34px; }
          .home-section-copy { max-width: 680px; }
          .home-forecast-inner { grid-template-columns: 1fr; align-items:start; }
          .summary-band, .prediction-spotlight { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .stat-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .story-grid, .pipeline-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .fz-header { flex-direction: column; align-items: flex-start; }
          .fz-meta { justify-content:flex-start; }
          .page-hero { flex-direction: column; align-items: stretch; }
          [data-testid="stSidebar"] > div:first-child { width: 220px; }
        }
        @media (max-width: 760px) {
          .block-container { padding-left: 1rem; padding-right: 1rem; }
          .landing-title { font-size: 40px; }
          .landing-hero { padding-top: 18px; padding-bottom: 52px; }
          .landing-hero-copy { padding: 20px 0 6px; }
          .data-visual { min-height: 360px; padding: 22px 18px 18px; }
          .compact-row { grid-template-columns: 1fr 0.75fr 0.75fr; gap: 8px; }
          .compact-row.head div:nth-child(3), .compact-row.head div:nth-child(4), .compact-row.head div:nth-child(5),
          .compact-row div:nth-child(3), .compact-row div:nth-child(4), .compact-row div:nth-child(5) { display:none; }
          .editorial-stats { grid-template-columns: 1fr; gap: 22px; }
          .home-section, .home-forecast { padding-top: 56px; }
          .summary-band, .prediction-spotlight, .stat-row, .story-grid, .pipeline-steps { grid-template-columns: 1fr; }
          .fz-page-title { font-size: 31px; }
          .summary-main-value, .prediction-main { font-size: 40px; }
        }
        /* FoodZero dark product system */
        :root {
          --fz-bg: #070808;
          --fz-bg-2: #121414;
          --fz-surface: #121414;
          --fz-surface-soft: #191c1b;
          --fz-green: #83d7a5;
          --fz-green-2: #6fcf97;
          --fz-sage: #152019;
          --fz-sage-2: #18251d;
          --fz-charcoal: #f3f4f1;
          --fz-text: #e7ebe7;
          --fz-muted: #929891;
          --fz-muted-2: #656b66;
          --fz-border: rgba(255,255,255,.10);
          --fz-border-2: rgba(255,255,255,.06);
          --fz-amber: #d5a866;
          --fz-coral: #e78370;
          --fz-soft-green: rgba(131,215,165,.10);
          --fz-soft-amber: rgba(213,168,102,.10);
          --fz-soft-red: rgba(231,131,112,.10);
          --fz-dark: #000000;
        }
        html, body { background:#070808 !important; }
        .stApp {
          background:
            linear-gradient(180deg,
              #070808 0%,
              #090a0a 16%,
              #0c0d0d 31%,
              #111313 49%,
              #151817 64%,
              #191c1b 80%,
              #202321 100%) !important;
          color:var(--fz-text) !important;
        }
        [data-testid="stAppViewContainer"], [data-testid="stMain"] { background:transparent !important; color:var(--fz-text) !important; }
        [data-testid="stMainBlockContainer"], .block-container { background:transparent !important; }
        [data-testid="stMain"], section.stMain, .stApp { overflow-x:hidden !important; }
        :root { --fz-font-sans:Pretendard, "Pretendard Variable", "SUIT", Inter, "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif; }
        html, body, .stApp, .stApp *, [data-testid="stSidebar"], [data-testid="stSidebar"] *, button, input, textarea, select {
          font-family:var(--fz-font-sans) !important;
          font-synthesis-weight:none;
        }
        .stApp, .landing-copy, .editorial-copy, .home-section-copy, .fz-page-desc, .section-desc, .story-text, .light-list, .tech-note, .footer {
          font-weight:400;
        }
        .block-container { max-width: 1320px; padding: 2.1rem clamp(1.5rem, 3vw, 3.5rem) 4rem; }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child, [data-testid="stSidebarContent"] { background:linear-gradient(180deg,#070808 0%,#0b0c0c 58%,#111313 100%) !important; border-color:var(--fz-border) !important; }
        [data-testid="stSidebar"] > div:first-child { width:242px; }
        .fz-sidebar-brand { padding:30px 24px 24px; border-color:var(--fz-border); }
        .fz-sidebar-title, .fz-title, .fz-page-title, .section-title, .landing-title, .editorial-title, .home-section-title, .stat-value, .summary-main-value, .summary-value, .prediction-main, .prediction-sub, .story-title { color:var(--fz-charcoal) !important; }
        .fz-sidebar-subtitle, .fz-sidebar-meta, .fz-page-desc, .fz-subtitle, .meta-value, .section-desc, .editorial-copy, .home-section-copy, .landing-copy, .visual-caption, .stat-label, .stat-note, .summary-label, .summary-note, .summary-main-label, .summary-main-note, .prediction-label, .prediction-date, .story-text, .intro-quiet, .light-list, .tech-note, .footer, .home-ranking-label, .home-forecast-note { color:var(--fz-muted) !important; }
        .fz-sidebar-meta { border-color:var(--fz-border); }
        .fz-sidebar-meta strong { color:var(--fz-text) !important; }
        [data-testid="stSidebar"] .stButton { padding:0 14px; }
        [data-testid="stSidebar"] .stButton > button { min-height:44px; padding:0 12px; border:0 !important; border-left:2px solid transparent !important; border-radius:0 !important; background:transparent !important; color:var(--fz-muted) !important; font-weight:560; box-shadow:none !important; }
        [data-testid="stSidebar"] .stButton { margin-top:2px; }
        [data-testid="stSidebar"] .stButton > button:hover { color:var(--fz-text) !important; background:rgba(255,255,255,.035) !important; }
        [data-testid="stSidebar"] button[kind="primary"], [data-testid="stSidebar"] [data-testid="stBaseButton-primary"] { background:transparent !important; color:var(--fz-text) !important; border-left-color:var(--fz-green) !important; font-weight:700; }
        .page-hero, .section-head { border-color:var(--fz-border); }
        .eyebrow, .editorial-kicker, .home-section-kicker, .home-insight strong, .emphasis { color:var(--fz-green) !important; }
        .landing-hero { min-height:72vh; padding:1.25rem 0 3.6rem; grid-template-columns:minmax(560px,1.42fr) minmax(330px,.82fr); gap:clamp(24px,3.2vw,54px); }
        div[data-testid="stHorizontalBlock"]:has(.landing-hero-copy):has(.hero-visual-wrap) { align-items:center; gap:clamp(24px,3vw,52px); }
        div[data-testid="column"]:has(.landing-hero-copy) { min-width:560px; flex:1.34 1 560px !important; }
        div[data-testid="column"]:has(.hero-visual-wrap) { flex:.86 1 340px !important; }
        .fz-sidebar-title, .fz-title, .landing-title { font-weight:780 !important; }
        .fz-page-title, .editorial-title, .dark-band h2 { font-weight:760 !important; }
        .section-title, .home-section-title, .story-title { font-weight:700 !important; }
        .summary-main-value, .summary-value, .stat-value, .prediction-main, .prediction-sub, .editorial-stat-value { font-weight:680 !important; }
        .forecast-preview .forecast-value { font-weight:650 !important; }
        .summary-main-label, .summary-label, .stat-label, .prediction-label, .prediction-date, .control-caption, .compact-row.head, .status-badge, .stButton > button { font-weight:600 !important; }
        .eyebrow, .editorial-kicker, .home-section-kicker, .home-ranking-label, .home-forecast-note, .home-dark-kicker, .forecast-preview .forecast-label, .data-meta, .fz-sidebar-subtitle, .fz-sidebar-meta { text-transform:uppercase; letter-spacing:.1em; font-weight:700 !important; }
        .landing-copy, .editorial-copy, .home-section-copy, .fz-page-desc, .section-desc, .story-text, .intro-quiet, .light-list, .tech-note, .footer, .visual-caption, .summary-note, .summary-main-note, .stat-note, .prediction-date { font-weight:400 !important; }
        .home-insight, .compact-row, .meta-value { font-weight:500 !important; }
        .landing-title { width:max-content; max-width:100%; font-size:clamp(40px,2.9vw,56px); font-weight:780 !important; line-height:1.1; letter-spacing:-.04em; word-break:keep-all; overflow-wrap:normal; }
        .landing-title .hero-line { display:block; white-space:nowrap; word-break:keep-all; overflow-wrap:normal; line-break:strict; }
        .fz-page-title, .section-title, .editorial-title, .home-section-title, .dark-band h2, .story-title { word-break:keep-all; overflow-wrap:normal; }
        .landing-hero-copy { padding:0; }
        .hero-visual-wrap { min-height:420px; }
        .data-visual { background:transparent; border:0; box-shadow:none; min-height:0; padding:0; overflow:visible; }
        .data-visual:before { background:radial-gradient(circle at 50% 48%, rgba(220,225,222,.09), transparent 48%), radial-gradient(circle at 50% 50%, rgba(255,255,255,.035), transparent 64%); }
        .visual-title, .visual-caption, .visual-footer { display:none; }
        .visual-svg { filter:drop-shadow(0 0 24px rgba(131,215,165,.12)); min-height:430px; }
        .visual-svg .flow-path { stroke-width:1.8; opacity:.68; }
        .visual-svg .flow-point { stroke:#050605; stroke-width:1.5; }
        .visual-svg .flow-label, .visual-svg .flow-number { display:none; }
        .orb-svg { min-height:520px; }
        .orb-svg { transform:scale(1.12); filter:drop-shadow(0 0 24px rgba(255,255,255,.14)) drop-shadow(0 0 34px rgba(104,211,145,.16)); }
        .orb-object { position:relative; width:min(100%,430px); height:430px; margin:0 auto; overflow:visible; background:radial-gradient(circle at 50% 50%,rgba(235,239,236,.13),rgba(175,182,178,.045) 30%,transparent 62%); filter:drop-shadow(0 0 24px rgba(230,235,232,.13)) drop-shadow(0 0 32px rgba(131,215,165,.08)); }
        .orb-halo { position:absolute; inset:8%; border-radius:50%; background:radial-gradient(circle,rgba(230,235,232,.09),rgba(180,187,183,.035) 38%,transparent 70%); filter:blur(18px); }
        .orb-rings, .orb-points { position:absolute; inset:0; }
        .orb-ring { position:absolute; left:50%; top:50%; display:block; border:1.65px solid rgba(244,248,245,.72); border-radius:50%; box-shadow:0 0 14px rgba(255,255,255,.075), inset 0 0 10px rgba(255,255,255,.035); }
        .orb-point { position:absolute; display:block; border-radius:50%; background:#78d39b; box-shadow:0 0 8px rgba(120,211,155,.58); transform:translate(-50%,-50%); }
        .orb-core { position:absolute; left:50%; top:50%; width:18px; height:18px; border-radius:50%; transform:translate(-50%,-50%); background:#f3f4f1; box-shadow:0 0 18px rgba(255,255,255,.42),0 0 24px rgba(104,211,145,.34); }
        .orb-line { fill:none; stroke:rgba(240,245,241,.55); stroke-width:1.25; transform-origin:center; }
        .orb-point { fill:#78d39b; filter:drop-shadow(0 0 8px rgba(120,211,155,.58)); }
        .orb-core { fill:#f3f4f1; filter:drop-shadow(0 0 18px rgba(255,255,255,.42)) drop-shadow(0 0 24px rgba(104,211,145,.34)); }
        .hero-notes {
          display:grid;
          grid-template-columns:repeat(3,minmax(0,1fr));
          gap:0;
          color:var(--fz-muted);
          margin:-.8rem 0 clamp(38px,4.4vw,64px);
          border-top:1px solid var(--fz-border);
          border-bottom:1px solid var(--fz-border);
        }
        .hero-note {
          padding:18px clamp(18px,2.6vw,34px) 20px;
          border-left:1px solid var(--fz-border);
        }
        .hero-note:first-child { border-left:0; padding-left:0; }
        .hero-note-title {
          color:var(--fz-text);
          font-size:12px;
          line-height:1.2;
          letter-spacing:.11em;
          text-transform:uppercase;
          font-weight:700;
          margin-bottom:7px;
        }
        .hero-note-copy {
          color:var(--fz-muted);
          font-size:14px;
          line-height:1.5;
        }
        .editorial-section, .home-section, .home-forecast { border-color:var(--fz-border); padding-top:clamp(54px,5.8vw,86px); }
        .home-section.data-overview { background:transparent; border-top:1px solid rgba(225,230,227,.09); border-bottom:1px solid rgba(225,230,227,.09); padding:clamp(48px,5.6vw,76px) 0; margin-left:0; margin-right:0; }
        .editorial-stats { border-color:var(--fz-border); }
        .editorial-stat-value { color:var(--fz-charcoal); }
        .compact-list, .compact-row, .stat-row, .footer { border-color:var(--fz-border); }
        .compact-row { color:var(--fz-text); }
        .regional-flow-section { color:var(--fz-text); }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) {
          background:linear-gradient(180deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
          color:var(--fz-text);
          width:100%;
          margin-left:auto;
          margin-right:auto;
          margin-top:clamp(22px,3vw,38px);
          margin-bottom:clamp(24px,3.4vw,44px);
          padding:clamp(46px,5vw,72px) clamp(22px,3.4vw,50px);
          border-top:1px solid rgba(255,255,255,.10);
          border-bottom:1px solid rgba(255,255,255,.08);
          box-shadow:none;
          border-radius:0;
        }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-section { padding-top:0; border-top:0; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-section-title { font-size:clamp(32px,2.8vw,38px); line-height:1.1; white-space:nowrap; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-section-kicker, div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-insight strong { color:var(--fz-green) !important; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-section-title { color:var(--fz-charcoal) !important; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-section-copy, div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-insight, div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-ranking-label { color:var(--fz-muted) !important; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .home-insight, div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .compact-list { border-color:var(--fz-border); }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .compact-row { color:var(--fz-text); border-color:var(--fz-border); }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) div[data-testid="stPlotlyChart"] { background:transparent; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) .stSelectbox label p { color:var(--fz-muted) !important; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) div[data-baseweb="select"] > div { background:#101210 !important; border-color:rgba(255,255,255,.14) !important; }
        div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) div[data-baseweb="select"] span, div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) div[data-baseweb="select"] div { color:var(--fz-text) !important; }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) { background:radial-gradient(circle at 84% 18%, rgba(220,225,222,.055), transparent 28%); color:var(--fz-text); margin-left:clamp(-50px,-3vw,-18px); margin-right:clamp(-50px,-3vw,-18px); padding:clamp(52px,5.4vw,78px) clamp(22px,3.4vw,50px); align-items:stretch !important; }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) [data-testid="column"] {
          display:flex;
          flex-direction:column;
        }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) [data-testid="column"] > div {
          display:flex;
          flex:1 1 auto;
          flex-direction:column;
        }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) .home-section { padding-top:0; border-top:0; }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) .change-detection-section {
          display:flex;
          flex-direction:column;
          flex:1 1 auto;
          height:100%;
          box-sizing:border-box;
        }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) .change-detection-section:after {
          margin-top:auto;
          background:var(--fz-border);
        }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) .compact-list { border-color:rgba(255,255,255,.12); }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) .compact-row { border-color:rgba(255,255,255,.12); }
        div[data-testid="stHorizontalBlock"]:has(.change-detection-section) .status-badge {
          width: 52px;
          height: 52px;
          padding: 0;
          text-align: center;
          line-height: 1.25;
          white-space: normal;
        }
        div[data-testid="stHorizontalBlock"]:has(.forecast-section) { background:radial-gradient(circle at 78% 28%, rgba(220,225,222,.075), transparent 31%); color:var(--fz-text); margin-left:clamp(-50px,-3vw,-18px); margin-right:clamp(-50px,-3vw,-18px); padding:clamp(52px,5.5vw,82px) clamp(22px,3.4vw,50px); }
        div[data-testid="stHorizontalBlock"]:has(.forecast-section) .home-forecast { padding-top:0; border-top:0; }
        div[data-testid="stHorizontalBlock"]:has(.forecast-section) .home-forecast-note { color:var(--fz-muted) !important; }
        .forecast-preview { border-left:1px solid rgba(255,255,255,.16); padding:12px 0 12px 28px; color:var(--fz-text); }
        .forecast-preview .forecast-label { color:var(--fz-muted); font-size:11px; letter-spacing:.14em; text-transform:uppercase; }
        .forecast-preview .forecast-date { font-size:20px; margin:7px 0 16px; }
        .forecast-preview .forecast-value { color:#f3f4f1; font-size:42px; letter-spacing:-.04em; font-weight:600; }
        .dark-band { margin-left:clamp(-50px,-3vw,-18px); margin-right:clamp(-50px,-3vw,-18px); }
        .dark-band { background:radial-gradient(circle at 80% 30%, rgba(220,225,222,.06), transparent 34%); border:1px solid var(--fz-border); border-radius:0; margin-top:72px; }
        .dark-band:after { border-color:rgba(200,207,203,.10); box-shadow:0 0 0 42px rgba(255,255,255,.035),0 0 0 84px rgba(210,216,212,.018); }
        .home-section, .home-forecast { position:relative; }
        .home-section:after, .home-forecast:after { content:""; display:block; height:1px; margin-top:clamp(34px,4.5vw,62px); background:linear-gradient(90deg,transparent,rgba(220,225,222,.18),transparent); }
        .summary-band, .prediction-spotlight, .section, .pipeline-step, .tech-note, div[data-testid="stMetric"] { background:var(--fz-surface) !important; border-color:var(--fz-border) !important; box-shadow:none !important; }
        .summary-band, .prediction-spotlight { background:transparent !important; border:0 !important; border-top:1px solid var(--fz-border) !important; border-bottom:1px solid var(--fz-border) !important; border-radius:0; }
        .editorial-stats { gap:0; padding:30px 0 32px; border-top:1px solid rgba(255,255,255,.12); border-bottom:1px solid rgba(255,255,255,.12); }
        .editorial-stats > div { padding:0 clamp(20px,3vw,42px); border-left:1px solid rgba(255,255,255,.10); }
        .editorial-stats > div:first-child { padding-left:0; border-left:0; }
        .editorial-stat-label { color:var(--fz-muted) !important; letter-spacing:.02em; }
        .summary-band { padding:24px 0 28px !important; margin:18px 0 30px; gap:0; }
        .summary-band > div { padding:0 clamp(18px,2.5vw,34px); border-left:1px solid var(--fz-border); }
        .summary-band > div:first-child { padding-left:0; border-left:0; }
        .stat-row { gap:0; padding:22px 0 26px; margin:18px 0 32px; border-top:1px solid var(--fz-border); border-bottom:1px solid var(--fz-border); }
        .stat-row > div { padding:0 clamp(18px,2.5vw,34px); border-left:1px solid var(--fz-border); }
        .stat-row > div:first-child { padding-left:0; border-left:0; }
        .stat-label, .summary-label, .summary-main-label, .prediction-label { text-transform:uppercase; letter-spacing:.08em; font-size:11px !important; }
        .stat-value { font-size:clamp(28px,3vw,42px); letter-spacing:-.035em; }
        .section-head { border-top:1px solid var(--fz-border); padding-top:24px; margin:38px 0 18px; }
        .section-head + div[data-testid="stPlotlyChart"], .section-head + .element-container div[data-testid="stPlotlyChart"] { margin-top:10px; }
        .prediction-spotlight { grid-template-columns:minmax(320px,1.35fr) repeat(3,minmax(120px,.65fr)); gap:0; padding:28px 0 32px; margin:22px 0 42px; align-items:end; }
        .prediction-spotlight > div { padding:0 clamp(18px,2.6vw,34px); border-left:1px solid var(--fz-border); }
        .prediction-spotlight > div:first-child { padding-left:0; border-left:0; grid-column:1 / -1; padding-bottom:24px; margin-bottom:24px; border-bottom:1px solid var(--fz-border); }
        .prediction-main { font-size:clamp(54px,7vw,92px); letter-spacing:-.055em; }
        .prediction-date { font-size:15px; margin:8px 0 0; }
        .prediction-sub { font-size:clamp(24px,2.8vw,38px); letter-spacing:-.04em; }
        .story-grid { gap:0; border-top:1px solid var(--fz-border); border-bottom:1px solid var(--fz-border); }
        .story-step { border-top:0; border-left:1px solid var(--fz-border); padding:24px clamp(18px,2.5vw,30px) 28px; }
        .story-step:first-child { border-left:0; padding-left:0; }
        .story-num { letter-spacing:.12em; }
        .pipeline-step, div[data-testid="stMetric"] { background:transparent !important; border:0 !important; border-radius:0 !important; padding:0 !important; }
        .tech-note { background:transparent !important; border:0 !important; border-top:1px solid var(--fz-border) !important; border-bottom:1px solid var(--fz-border) !important; border-radius:0 !important; padding:18px 0 !important; }
        .final-model-callout { border-top:1px solid var(--fz-border); border-bottom:1px solid var(--fz-border); padding:22px 0 24px; margin:18px 0 24px; }
        .final-model-kicker { color:var(--fz-green); font-size:11px; letter-spacing:.14em; text-transform:uppercase; font-weight:700; margin-bottom:8px; }
        .final-model-name { color:var(--fz-text); font-size:clamp(30px,3.6vw,52px); line-height:1; letter-spacing:-.045em; font-weight:700; margin-bottom:12px; }
        .final-model-copy { color:var(--fz-muted); font-size:15px; line-height:1.65; max-width:780px; }
        .notice, .editorial-note {
          background:transparent !important;
          border:0 !important;
          border-left:1px solid rgba(205,213,208,.32) !important;
          border-radius:0 !important;
          box-shadow:none !important;
          color:var(--fz-muted) !important;
          padding:4px 0 16px 18px !important;
          margin:16px 0 28px !important;
          max-width:780px;
        }
        .editorial-note-kicker {
          color:var(--fz-green);
          font-size:11px;
          letter-spacing:.13em;
          text-transform:uppercase;
          font-weight:700;
          margin-bottom:8px;
        }
        .editorial-note-copy {
          color:var(--fz-muted);
          font-size:15px;
          line-height:1.7;
        }
        .editorial-note-line {
          height:1px;
          width:min(420px,100%);
          background:var(--fz-border);
          margin-top:16px;
        }
        .fz-table { border-radius:0; min-width:920px; font-size:14.5px; }
        div[data-testid="stDataFrame"] { border-radius:0; }
        .table-scroll { border-radius:0; }
        .notice { background:transparent !important; color:var(--fz-muted) !important; }
        .fz-table { background:var(--fz-surface); border-color:var(--fz-border); }
        .fz-table thead th { background:#101210; color:var(--fz-text); border-color:var(--fz-border); padding:14px 16px; }
        .fz-table tbody td { background:#0c0e0d; color:var(--fz-text); border-color:var(--fz-border-2); padding:14px 16px; }
        .table-scroll { max-height:640px; overflow-y:auto; overflow-x:auto; border:1px solid var(--fz-border); width:100%; }
        .table-scroll .fz-table { border:0; border-radius:0; }
        .table-scroll .fz-table thead th { position:sticky; top:0; z-index:2; }
        .data-meta { color:var(--fz-muted); font-size:11px; letter-spacing:.08em; text-transform:uppercase; margin:-4px 0 12px; }
        .status-normal { color:var(--fz-green); background:var(--fz-soft-green); border-color:rgba(131,215,165,.25); }
        .status-attention { color:var(--fz-amber); background:var(--fz-soft-amber); border-color:rgba(213,168,102,.25); }
        .status-high { color:var(--fz-coral); background:var(--fz-soft-red); border-color:rgba(231,131,112,.25); }
        div[data-baseweb="select"] > div, .stDateInput input, div[data-baseweb="input"] > div { background:#101210 !important; border-color:rgba(255,255,255,.12) !important; color:var(--fz-text) !important; border-radius:7px !important; }
        div[data-baseweb="select"] span, div[data-baseweb="select"] div, .stDateInput input, div[data-baseweb="input"] input { color:var(--fz-text) !important; }
        div[data-baseweb="select"] > div:focus-within, .stDateInput input:focus { border-color:var(--fz-green) !important; box-shadow:0 0 0 2px rgba(131,215,165,.12) !important; }
        div[data-baseweb="popover"], div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [role="listbox"], div[data-baseweb="popover"] [role="option"] { background:#101210 !important; color:var(--fz-text) !important; border-color:var(--fz-border) !important; }
        div[data-baseweb="popover"] [role="option"]:hover, div[data-baseweb="popover"] [aria-selected="true"] { background:#18251d !important; color:var(--fz-text) !important; }
        div[data-testid="stSelectbox"] label p, div[data-testid="stDateInput"] label p, div[data-testid="stMultiSelect"] label p { color:var(--fz-muted) !important; }
        .stButton > button { background:#f0f2ef; color:#070907; border:1px solid #f0f2ef; border-radius:7px; min-height:44px; font-weight:700; box-shadow:none; }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stSelectbox"]):has(.stButton) .stButton { margin-top:28px; }
        div[data-testid="stHorizontalBlock"]:has(div[data-testid="stSelectbox"]):has(.stButton) .stButton > button { min-height:44px; height:44px; font-size:18px !important; }
        [data-testid="stBaseButton-secondary"] { background:transparent !important; color:var(--fz-text) !important; border-color:rgba(255,255,255,.24) !important; }
        .stButton > button:hover { background:#fff; color:#000; border-color:#fff; }
        [data-testid="stBaseButton-secondary"]:hover { background:rgba(255,255,255,.06) !important; color:var(--fz-text) !important; border-color:rgba(255,255,255,.55) !important; }
        div[data-testid="stPlotlyChart"] { background:transparent; border:0; width:100% !important; }
        div[data-testid="stPlotlyChart"] > div { width:100% !important; }
        div[data-testid="stVegaLiteChart"], div[data-testid="stDataFrame"] { width:100% !important; }
        div[data-testid="stAlert"] { background:#101210; color:var(--fz-text); border-color:var(--fz-border); }
        /* Homepage-style product shell */
        [data-testid="stSidebar"] {
          display:none !important;
          visibility:hidden !important;
          width:0 !important;
          min-width:0 !important;
        }
        .block-container {
          max-width:1320px;
          padding-top:.9rem;
        }
        .landing-title {
          width:auto;
          max-width:780px;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) {
          position:sticky;
          top:0;
          z-index:20;
          display:flex;
          width:100%;
          max-width:1500px;
          align-items:center;
          justify-content:space-between;
          gap:10px;
          box-sizing:border-box;
          height:72px;
          min-height:72px;
          margin:0 auto clamp(18px,2.2vw,30px);
          padding:0;
          background:rgba(7,8,8,.88);
          border-bottom:1px solid var(--fz-border);
          backdrop-filter:blur(14px);
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"] {
          flex:1 1 0;
          width:auto !important;
          display:flex;
          flex-direction:column;
          justify-content:center;
          min-width:0;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button {
          min-height:58px;
          height:58px;
          padding:0 18px;
          border-radius:7px;
          background:transparent !important;
          border:1px solid transparent !important;
          color:var(--fz-muted) !important;
          font-size:16px;
          font-weight:760 !important;
          box-shadow:none !important;
          white-space:nowrap;
          justify-content:center;
          text-align:center;
          transition:background-color .16s ease, border-color .16s ease, color .16s ease;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button *,
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button p,
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button span {
          font-size:16px !important;
          line-height:1 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button:hover {
          background:transparent !important;
          border-color:transparent !important;
          color:var(--fz-text) !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stBaseButton-primary"],
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) button[kind="primary"] {
          color:var(--fz-charcoal) !important;
          background:transparent !important;
          border-color:transparent !important;
          font-weight:900 !important;
          box-shadow:none !important;
          text-shadow:0 0 12px rgba(243,244,241,.34);
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child .stButton > button {
          color:var(--fz-muted) !important;
          min-height:58px;
          height:58px;
          font-size:16px;
          font-weight:820 !important;
          justify-content:center;
          padding:0 18px;
          letter-spacing:0;
          background:transparent !important;
          border-color:transparent !important;
          box-shadow:none !important;
          line-height:1;
          text-align:center;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child [data-testid="stBaseButton-primary"],
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child button[kind="primary"] {
          color:var(--fz-charcoal) !important;
          background:transparent !important;
          border-color:transparent !important;
          font-weight:900 !important;
          text-shadow:0 0 12px rgba(243,244,241,.34);
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child [data-testid="stVerticalBlock"] {
          gap:0 !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child .stButton {
          margin:0;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child .stButton > button:hover {
          color:#fff !important;
          background:transparent !important;
          border-color:transparent !important;
        }
        .top-nav-brand span,
        .top-nav-meta {
          display:none;
          color:var(--fz-muted);
          font-size:10px;
          letter-spacing:.12em;
          text-transform:uppercase;
          line-height:1.2;
          white-space:nowrap;
        }
        .top-nav-brand {
          display:none;
        }
        .top-nav-meta {
          display:none;
          text-align:right;
          margin:0 2px 0 0;
          font-size:10px;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:last-child {
          align-items:center;
          justify-content:center;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:nth-child(n+2):nth-child(-n+6) .stButton > button {
          width:100%;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:nth-child(n+2):nth-child(-n+6) button[data-testid="stBaseButton-primary"],
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:nth-child(n+2):nth-child(-n+6) button[kind="primary"] {
          color:var(--fz-charcoal) !important;
          background:transparent !important;
          border-color:transparent !important;
          font-weight:900 !important;
          box-shadow:none !important;
          text-shadow:0 0 12px rgba(243,244,241,.34);
        }
        .page-eyebrow {
          color:var(--fz-green);
          font-size:11px;
          font-weight:720;
          letter-spacing:.14em;
          text-transform:uppercase;
          margin-bottom:14px;
        }
        .page-hero {
          align-items:flex-start;
          margin:0 0 clamp(26px,3.4vw,44px);
          padding-bottom:clamp(24px,3vw,38px);
          border-bottom:1px solid var(--fz-border);
        }
        .page-hero > div:first-child {
          flex:1 1 auto;
          min-width:0;
          width:100%;
        }
        .fz-page-title {
          max-width:none;
          font-size:clamp(38px,4.2vw,60px);
          line-height:1.06;
          letter-spacing:-.045em;
          white-space:nowrap;
        }
        .fz-page-desc {
          max-width:760px;
          font-size:18px;
        }
        .why-foodzero {
          padding-top:clamp(44px,5vw,72px);
          margin-top:0;
        }
        .service-index {
          display:grid;
          grid-template-columns:repeat(3,minmax(0,1fr));
          gap:0;
          margin:18px 0 14px;
          border-top:1px solid var(--fz-border);
          border-bottom:1px solid var(--fz-border);
        }
        .service-item {
          display:grid;
          grid-template-columns:44px minmax(0,1fr);
          gap:18px;
          padding:28px clamp(18px,2.5vw,34px) 30px;
          border-left:1px solid var(--fz-border);
        }
        .service-item:first-child { border-left:0; padding-left:0; }
        .service-num {
          color:var(--fz-green);
          font-size:12px;
          letter-spacing:.14em;
          font-weight:720;
        }
        .service-title {
          color:var(--fz-text);
          font-size:21px;
          font-weight:700;
          line-height:1.25;
          margin-bottom:10px;
        }
        .service-copy {
          color:var(--fz-muted);
          font-size:15px;
          line-height:1.62;
        }
        .pipeline-flow {
          display:grid;
          grid-template-columns:repeat(4,minmax(0,1fr));
          gap:0;
          margin:18px 0 42px;
          border-top:1px solid var(--fz-border);
          border-bottom:1px solid var(--fz-border);
        }
        .pipeline-flow > div {
          position:relative;
          padding:26px clamp(16px,2.4vw,30px) 28px;
          border-left:1px solid var(--fz-border);
        }
        .pipeline-flow > div:first-child { border-left:0; padding-left:0; }
        .pipeline-flow span {
          display:block;
          color:var(--fz-green);
          font-size:11px;
          letter-spacing:.14em;
          font-weight:720;
          margin-bottom:11px;
        }
        .pipeline-flow strong {
          display:block;
          color:var(--fz-text);
          font-size:18px;
          letter-spacing:.02em;
          margin-bottom:8px;
        }
        .pipeline-flow p {
          margin:0;
          color:var(--fz-muted);
          font-size:14px;
          line-height:1.55;
        }
        .model-snapshot {
          display:grid;
          grid-template-columns:minmax(0,1.05fr) minmax(420px,.95fr);
          gap:clamp(28px,5vw,72px);
          align-items:end;
          border-top:1px solid var(--fz-border);
          border-bottom:1px solid var(--fz-border);
          padding:30px 0 34px;
          margin:18px 0 28px;
        }
        .model-kicker {
          color:var(--fz-green);
          font-size:11px;
          letter-spacing:.14em;
          text-transform:uppercase;
          font-weight:720;
          margin-bottom:12px;
        }
        .model-title {
          color:var(--fz-text);
          font-size:clamp(27px,3vw,42px);
          line-height:1.12;
          letter-spacing:-.035em;
          font-weight:700;
          max-width:680px;
        }
        .model-copy {
          color:var(--fz-muted);
          font-size:15px;
          line-height:1.68;
          margin-top:16px;
          max-width:720px;
        }
        .model-metrics {
          display:grid;
          grid-template-columns:repeat(3,minmax(0,1fr));
          gap:0;
          border-top:1px solid var(--fz-border);
        }
        .model-metrics div {
          padding:18px 20px 0;
          border-left:1px solid var(--fz-border);
        }
        .model-metrics div:first-child { border-left:0; padding-left:0; }
        .model-metrics span {
          display:block;
          color:var(--fz-muted);
          font-size:11px;
          letter-spacing:.09em;
          text-transform:uppercase;
          margin-bottom:8px;
        }
        .model-metrics strong {
          display:block;
          color:var(--fz-text);
          font-size:clamp(26px,3vw,40px);
          line-height:1;
          font-weight:680;
          letter-spacing:-.035em;
        }
        .usage-flow {
          display:grid;
          grid-template-columns:repeat(4,minmax(0,1fr));
          gap:0;
          border-top:1px solid var(--fz-border);
          border-bottom:1px solid var(--fz-border);
          margin:-18px 0 42px;
        }
        .usage-flow span {
          color:var(--fz-muted);
          font-size:13px;
          padding:16px 18px;
          border-left:1px solid var(--fz-border);
        }
        .usage-flow span:first-child { border-left:0; padding-left:0; }
        .footer {
          display:grid;
          grid-template-columns:1.25fr 1.5fr .65fr 1.25fr;
          gap:clamp(18px,3vw,46px);
          align-items:start;
          margin-top:clamp(82px,9vw,132px);
          padding:30px 0 34px;
        }
        .footer strong {
          display:block;
          color:var(--fz-text);
          font-weight:700;
          margin-bottom:5px;
        }
        .footer span {
          display:block;
          color:var(--fz-muted);
          font-size:13px;
          line-height:1.7;
        }
        .footer-links {
          display:grid;
          grid-template-columns:repeat(2,minmax(0,1fr));
          gap:2px 18px;
        }
        .management-table {
          min-width:1040px;
        }
        .management-table th,
        .management-table td {
          padding-left:18px !important;
          padding-right:18px !important;
          line-height:1.55;
        }
        .management-table-wrap {
          max-height:680px;
        }
        .landing-copy,
        .editorial-copy,
        .home-section-copy,
        .section-desc,
        .story-text,
        .tech-note,
        .light-list,
        .intro-quiet {
          font-size:16.5px;
        }
        .compact-row {
          font-size:14.5px;
          padding:15px 0;
        }
        .stApp {
          font-size:20px;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button,
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child .stButton > button {
          font-size:20px;
        }
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button *,
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button p,
        div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button span {
          font-size:20px !important;
        }
        .landing-copy,
        .editorial-copy,
        .home-section-copy,
        .fz-page-desc,
        .section-desc,
        .story-text,
        .tech-note,
        .light-list,
        .intro-quiet,
        .model-copy {
          font-size:21px;
        }
        .hero-note-copy,
        .service-copy,
        .pipeline-flow p,
        .footer span,
        .usage-flow span,
        .model-metrics span,
        .stat-note,
        .summary-note,
        .summary-main-note,
        .prediction-date,
        .data-meta {
          font-size:18px;
        }
        .eyebrow,
        .page-eyebrow,
        .editorial-kicker,
        .home-section-kicker,
        .home-ranking-label,
        .home-forecast-note,
        .home-dark-kicker,
        .forecast-preview .forecast-label,
        .service-num,
        .pipeline-flow span,
        .model-kicker,
        .summary-main-label,
        .summary-label,
        .stat-label,
        .prediction-label,
        .compact-row.head {
          font-size:15px !important;
        }
        .hero-note-title {
          font-size:16px;
        }
        .service-title {
          font-size:25px;
        }
        .pipeline-flow strong {
          font-size:22px;
        }
        .section-title {
          font-size:30px;
        }
        .compact-row {
          font-size:18px;
        }
        .status-badge {
          font-size:16px;
        }
        .management-table,
        .fz-table {
          font-size:18px;
        }
        .stButton > button,
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stDateInput"] label p,
        div[data-testid="stMultiSelect"] label p,
        input,
        textarea,
        select {
          font-size:18px !important;
        }
        .st-key-predict-date div[data-baseweb="select"] span,
        .st-key-predict-date div[data-baseweb="select"] div {
          font-size:18px !important;
        }
        .editorial-note-kicker {
          font-size:14px;
        }
        .editorial-note-copy {
          font-size:18px;
        }
        .home-insight {
          font-size:18px;
          line-height:1.6;
        }
        .home-insight strong {
          font-size:18px;
        }
        .editorial-stat-label {
          font-size:18px;
          line-height:1.4;
        }
        .st-key-home_regions_cta button,
        .st-key-home_prediction_cta button,
        .st-key-home_service_region button,
        .st-key-home_service_prediction button,
        .st-key-home_service_management button,
        .st-key-home_forecast_cta button,
        .st-key-home_about_cta button,
        .st-key-home_analysis_cta button,
        .st-key-home_regions_cta button p,
        .st-key-home_prediction_cta button p,
        .st-key-home_service_region button p,
        .st-key-home_service_prediction button p,
        .st-key-home_service_management button p,
        .st-key-home_forecast_cta button p,
        .st-key-home_about_cta button p,
        .st-key-home_analysis_cta button p {
          font-size:19px !important;
          line-height:1.2 !important;
        }
        div[data-testid="stDateInput"] input {
          font-size:19px !important;
        }
        div[data-testid="stHorizontalBlock"]:has(.dark-band) .stButton > button,
        div[data-testid="stHorizontalBlock"]:has(.dark-band) .stButton > button p,
        div[data-testid="stHorizontalBlock"]:has(.dark-band) .stButton > button span {
          font-size:19px !important;
          line-height:1.2 !important;
        }
        /* Home process flow: keep all four steps on one shared baseline. */
        .pipeline-flow {
          align-items:stretch;
        }
        .pipeline-flow > div {
          box-sizing:border-box;
          display:flex;
          min-width:0;
          flex-direction:column;
          padding:24px clamp(20px,2.1vw,32px) 24px;
        }
        .pipeline-flow > div:first-child {
          padding-left:clamp(20px,2.1vw,32px);
        }
        .pipeline-flow span {
          margin-bottom:10px;
          font-size:15px !important;
          line-height:1.2;
        }
        .pipeline-flow strong {
          margin-bottom:12px;
          font-size:22px;
          line-height:1.2;
          font-weight:800;
        }
        .pipeline-flow p {
          display:block;
          flex:0 0 135px;
          max-width:260px;
          margin:0;
          font-size:18px;
          line-height:1.5;
        }
        /* Internal pages share a wider reading frame; the Home shell stays unchanged. */
        .block-container:has(.page-hero) {
          max-width:1480px;
        }
        .block-container:has(.page-hero) .page-hero {
          margin-bottom:clamp(22px,2.8vw,36px);
          padding-bottom:clamp(20px,2.5vw,32px);
        }
        .block-container:has(.page-hero) .section-head {
          margin:30px 0 16px;
        }
        .block-container:has(.page-hero) .section {
          margin:16px 0 20px;
        }
        .block-container:has(.page-hero) .story-grid {
          gap:0;
          margin:16px 0 26px;
          align-items:stretch;
        }
        .block-container:has(.page-hero) .story-step {
          display:flex;
          min-width:0;
          min-height:154px;
          flex-direction:column;
          padding:22px clamp(22px,2.2vw,34px) 26px;
        }
        .block-container:has(.page-hero) .story-step:first-child {
          padding-left:clamp(22px,2.2vw,34px);
        }
        .block-container:has(.page-hero) .story-text {
          margin-top:0;
          min-height:96px;
        }
        .block-container:has(.page-hero) .fz-table {
          font-size:19px;
        }
        .block-container:has(.page-hero) .fz-table thead th {
          font-size:16px;
          padding:15px 18px;
        }
        .block-container:has(.page-hero) .fz-table tbody td {
          padding:15px 18px;
        }
        .block-container:has(.page-hero) .footer {
          margin-top:clamp(62px,7vw,104px);
        }
        @media (max-width:1240px) {
          div[data-testid="stHorizontalBlock"]:has(.regional-flow-section) {
            width:100%;
            padding-left:clamp(1.2rem,3.4vw,3.75rem);
            padding-right:clamp(1.2rem,3.4vw,3.75rem);
          }
        }
        @media (max-width:1100px) {
          .landing-hero { grid-template-columns:1fr 1fr; }
          div[data-testid="column"]:has(.landing-hero-copy) { min-width:0; flex:1 1 100% !important; }
          .hero-notes { margin-top:0; }
          .landing-title { font-size:50px; }
          .editorial-stats > div, .summary-band > div, .stat-row > div, .prediction-spotlight > div, .story-step { border-left:0; padding-left:0; }
          .editorial-stats > div, .summary-band > div, .stat-row > div, .prediction-spotlight > div, .story-step { border-top:1px solid var(--fz-border); padding-top:18px; }
          .editorial-stats > div:first-child, .summary-band > div:first-child, .stat-row > div:first-child, .prediction-spotlight > div:first-child, .story-step:first-child { border-top:0; padding-top:0; }
          .prediction-spotlight > div:first-child { grid-column:1 / -1; }
        }
        @media (max-width:760px) {
          .landing-hero { grid-template-columns:1fr; min-height:auto; padding-top:1rem; }
          .sentence-line { white-space:normal; }
          .hero-notes { grid-template-columns:1fr; margin:8px 0 48px; }
          .hero-note { border-left:0; border-top:1px solid var(--fz-border); padding-left:0; }
          .hero-note:first-child { border-top:0; }
          .hero-visual-wrap { min-height:350px; }
          .landing-title { font-size:42px; }
        }
        @media (max-width:1180px) {
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) {
            position:relative;
            top:auto;
            grid-template-columns:none;
          }
          .top-nav-meta { display:none; }
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button {
            font-size:17px;
            padding:0 6px;
          }
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button *,
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button p,
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) .stButton > button span {
            font-size:17px !important;
          }
          .service-index, .pipeline-flow, .model-snapshot, .footer {
            grid-template-columns:1fr;
          }
          .fz-page-title {
            white-space:normal;
          }
          .service-item, .pipeline-flow > div {
            border-left:0;
            border-top:1px solid var(--fz-border);
            padding-left:0;
          }
          .service-item:first-child, .pipeline-flow > div:first-child { border-top:0; }
          .model-metrics { grid-template-columns:1fr; }
          .model-metrics div {
            border-left:0;
            border-top:1px solid var(--fz-border);
            padding-left:0;
          }
          .model-metrics div:first-child { border-top:0; }
        }
        @media (max-width:820px) {
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) {
            display:block;
            margin-bottom:44px;
            height:auto;
            padding:12px 0;
          }
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"] {
            width:100% !important;
            margin-bottom:4px;
          }
          div[data-testid="stHorizontalBlock"]:has(.top-nav-brand) [data-testid="stColumn"]:first-child .stButton > button {
            justify-content:center;
          }
          .top-nav-brand { text-align:center; margin-bottom:8px; }
          .usage-flow {
            grid-template-columns:1fr 1fr;
          }
          .usage-flow span:nth-child(odd) { border-left:0; padding-left:0; }
          .footer-links { grid-template-columns:1fr; }
        }
        @media (max-width:760px) {
          .block-container:has(.page-hero) {
            max-width:100%;
          }
          .block-container:has(.page-hero) .story-step {
            min-height:0;
          }
        }
        </style>
        """,
    )


def render_markup(markup: str, **_: Any) -> None:
    """Render trusted UI fragments without Markdown interpreting indentation as code."""
    st.html(markup)


def navigation(latest_date: pd.Timestamp) -> str:
    pages = ["Home", "지역 현황", "배출량 예측", "수거·관리 지원", "데이터 인사이트", "FoodZero 소개"]
    page = st.session_state.get("page_navigation", "Home")
    nav_cols = st.columns([1.25, 1, 1.05, 1, 1.16, 1.12], gap="small", vertical_alignment="center")
    with nav_cols[0]:
        st.button(
            "FoodZero",
            key="top_nav_home",
            type="primary" if page == "Home" else "secondary",
            on_click=navigate_to_page,
            args=("Home",),
            width="stretch",
        )
        render_markup('<div class="top-nav-brand" aria-hidden="true"></div>', unsafe_allow_html=True)
    for page_index, nav_page in enumerate(pages[1:], start=1):
        with nav_cols[page_index]:
            st.button(
                NAV_LABELS[nav_page],
                key=f"top_nav_{page_index}",
                type="primary" if nav_page == page else "secondary",
                on_click=navigate_to_page,
                args=(nav_page,),
                width="stretch",
            )
    return page


def navigate_to_page(page: str) -> None:
    """Move the shared sidebar navigation from a home CTA."""
    st.session_state["page_navigation"] = page


def render_page_title(title: str, description: str, control_html: str = "", eyebrow: str = "") -> None:
    control = f"<div>{control_html}</div>" if control_html else ""
    eyebrow_html = f'<div class="page-eyebrow">{escape_text(eyebrow)}</div>' if eyebrow else ""
    render_markup(
        f"""
        <div class="page-hero">
          <div>
            {eyebrow_html}
            <div class="fz-page-title">{title}</div>
            <div class="fz-page-desc">{description}</div>
          </div>
          {control}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str = "") -> None:
    render_markup(
        f"""
        <div class="section-head">
          <div>
            <div class="section-title">{title}</div>
            <div class="section-desc">{description}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_editorial_note(kicker: str, copy: str, prefix: str = "") -> None:
    lead = f"{escape_text(prefix)} / " if prefix else ""
    render_markup(
        f"""
        <div class="editorial-note">
          <div class="editorial-note-kicker">{lead}{escape_text(kicker)}</div>
          <div class="editorial-note-copy">{escape_text(copy)}</div>
          <div class="editorial-note-line"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def escape_text(value: Any) -> str:
    return html.escape(str(value))


def render_status_table(predictions: pd.DataFrame, max_rows: int = 30) -> None:
    if predictions.empty:
        render_editorial_note("NOTE", "선택한 기준일에는 최근 평균보다 뚜렷하게 증가할 것으로 예상되는 지역이 없습니다.")
        return
    rows = []
    for _, row in predictions.head(max_rows).iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape_text(row['sigungu_key'])}</td>"
            f"<td class='num'>{format_weight(row['predicted_waste_g'])}</td>"
            f"<td class='num'>{format_weight(row['recent_7day_average_g'])}</td>"
            f"<td class='num'>{format_pct(row['change_vs_recent_average_pct'])}</td>"
            f"<td>{status_badge(row['surge_level'])}</td>"
            "</tr>"
        )
    render_markup(
        """
        <table class="fz-table">
          <thead>
            <tr>
              <th>지역</th>
              <th class="num">예상 배출량</th>
              <th class="num">최근 평균</th>
              <th class="num">변화</th>
              <th>상태</th>
            </tr>
          </thead>
          <tbody>
        """
        + "\n".join(rows)
        + """
          </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def render_html_table(
    df: pd.DataFrame,
    columns: list[tuple[str, str]],
    max_rows: int | None = None,
    numeric_columns: set[str] | None = None,
    auto_align_numbers: bool = True,
) -> None:
    shown = df.head(max_rows) if max_rows else df
    header = "".join(f"<th>{escape_text(label)}</th>" for _, label in columns)
    numeric_columns = numeric_columns or set()
    body_rows = []
    for _, row in shown.iterrows():
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            is_number = isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value)
            css = " class='num'" if key in numeric_columns or (auto_align_numbers and is_number) else ""
            cells.append(f"<td{css}>{escape_text(value)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    render_markup(
        f"""
        <div class="table-scroll">
        <table class="fz-table">
          <thead><tr>{header}</tr></thead>
          <tbody>{''.join(body_rows)}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_management_table(predictions: pd.DataFrame, max_rows: int | None = None, include_error_reference: bool = True) -> None:
    shown = predictions.head(max_rows) if max_rows else predictions
    if shown.empty:
        render_editorial_note("NOTE", "선택 조건에 해당하는 지역이 없습니다.")
        return

    body_rows = []
    for _, row in shown.iterrows():
        error_ratio = row.get("mae_to_mean_ratio")
        error_text = format_abs_pct(float(error_ratio) * 100) if error_ratio is not None and not pd.isna(error_ratio) else "-"
        error_cell = f"<td class='num'>{error_text}</td>" if include_error_reference else ""
        body_rows.append(
            "<tr>"
            f"<td class='num'>{int(row['management_rank'])}</td>"
            f"<td>{escape_text(row['sigungu_key'])}</td>"
            f"<td class='num'>{format_weight(row['predicted_waste_g'])}</td>"
            f"<td class='num'>{format_weight(row['recent_7day_average_g'])}</td>"
            f"<td class='num'>{format_pct(row['change_vs_recent_average_pct'])}</td>"
            f"<td>{management_status_badge(row['surge_level'])}</td>"
            f"{error_cell}"
            "</tr>"
        )
    error_header = '<th class="num">예측 오차 참고</th>' if include_error_reference else ""
    render_markup(
        f"""
        <div class="table-scroll management-table-wrap">
        <table class="fz-table management-table">
          <thead>
            <tr>
              <th class="num">관리 순위</th>
              <th>지역</th>
              <th class="num">예측 배출량</th>
              <th class="num">최근 7일 평균</th>
              <th class="num">변화율</th>
              <th>위험도</th>
              {error_header}
            </tr>
          </thead>
          <tbody>
        """
        + "".join(body_rows)
        + """
          </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_distribution(kpis: dict[str, int]) -> None:
    total = max(int(kpis.get("municipalities", 0)), 1)
    action_needed = int(kpis.get("high", 0)) + int(kpis.get("attention", 0))
    action_pct = action_needed / total * 100
    summary = f"""
        <div class="risk-summary">
          <div>
            <div class="risk-summary-label">관리 확인 필요</div>
            <div class="risk-summary-value">{action_needed:,}곳</div>
          </div>
          <div class="risk-summary-note">전체 {int(kpis.get("municipalities", 0)):,}개 지역 중 {action_pct:.1f}%</div>
        </div>
    """
    rows = [
        ("집중관리", "high", int(kpis.get("high", 0))),
        ("주의", "attention", int(kpis.get("attention", 0))),
        ("안정", "normal", int(kpis.get("normal", 0))),
    ]
    fragments = []
    for label, level, count in rows:
        pct = count / total * 100
        display_pct = max(pct, 1.4) if count else 0
        fragments.append(
            f"""
            <div class="risk-row">
              <div class="risk-label">{escape_text(label)}</div>
              <div class="risk-track" aria-label="{escape_text(label)} {count:,}곳">
                <div class="risk-fill risk-fill-{level}" style="width:{display_pct:.2f}%"></div>
              </div>
              <div class="risk-value">{count:,}곳 · {pct:.1f}%</div>
            </div>
            """
        )
    render_markup(f"<div class='risk-distribution'>{summary}{''.join(fragments)}</div>", unsafe_allow_html=True)


def status_text(level: str) -> str:
    return STATUS_LABELS.get(str(level), str(level))


def render_kpi_panel(kpis: dict[str, Any], target_date: pd.Timestamp) -> None:
    render_markup(
        f"""
        <div class="summary-band">
          <div>
            <div class="summary-main-label">전국 예상 배출량</div>
            <div class="summary-main-value">{format_weight(kpis['predicted_total_g'])}</div>
            <div class="summary-main-note">{target_date.strftime('%Y.%m.%d')} 예측</div>
          </div>
          <div>
            <div class="summary-label">분석 지역</div>
            <div class="summary-value">{kpis['municipalities']:,}곳</div>
            <div class="summary-note">예측 가능 지역</div>
          </div>
          <div>
            <div class="summary-label">증가 예상</div>
            <div class="summary-value">{kpis['surge_regions']:,}곳</div>
            <div class="summary-note">최근 평균 대비 증가</div>
          </div>
          <div>
            <div class="summary-label">우선 확인</div>
            <div class="summary-value">{kpis['priority_regions']:,}곳</div>
            <div class="summary-note">변화 폭이 큰 지역</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_home_visual(predictions: pd.DataFrame) -> None:
    top = predictions.sort_values("predicted_waste_g", ascending=False).head(9).copy().reset_index(drop=True)
    max_value = float(top["predicted_waste_g"].max()) if not top.empty else 1.0
    orb_paths = []
    for index, row in top.iterrows():
        ratio = max(0.15, min(1.0, float(row["predicted_waste_g"]) / max_value))
        drift = float(row.get("change_vs_recent_average_pct", 0.0) or 0.0)
        radius = 92 + ratio * 52
        tilt = -28 + (index * 13) + max(-10, min(10, drift * 0.12))
        opacity = 0.28 + ratio * 0.42
        size = radius * 2
        orbit_paths = (
            f'<span class="orb-ring" style="width:{size:.1f}px;height:{(radius * .84):.1f}px;'
            f'transform:translate(-50%,-50%) rotate({tilt:.1f}deg) scaleX(.72);opacity:{opacity:.2f};"></span>'
        )
        orb_paths.append(orbit_paths)
    points = []
    for index, row in top.head(6).iterrows():
        angle = -1.15 + index * 0.46
        ratio = max(0.15, min(1.0, float(row["predicted_waste_g"]) / max_value))
        x = 180 + (106 + ratio * 32) * np.cos(angle)
        y = 190 + (48 + ratio * 20) * np.sin(angle)
        points.append(
            f'<span class="orb-point" style="left:{x / 360 * 100:.1f}%;top:{y / 380 * 100:.1f}%;'
            f'width:{2.2 + ratio * 2:.1f}px;height:{2.2 + ratio * 2:.1f}px;"></span>'
        )
    render_markup(
        f"""
        <div class="hero-visual-wrap">
          <div class="data-visual">
            <div class="orb-object" role="img" aria-label="지역별 데이터 흐름을 추상화한 FoodZero Flow Orb">
              <div class="orb-halo"></div>
              <div class="orb-rings">{''.join(orb_paths)}</div>
              <div class="orb-points">{''.join(points)}</div>
              <div class="orb-core"></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_change_rows(predictions: pd.DataFrame, max_rows: int = 6) -> None:
    rows = [
        """
        <div class="compact-row head">
          <div>지역</div><div>예상 배출량</div><div>최근 평균</div><div>변화율</div><div>상태</div>
        </div>
        """
    ]
    if predictions.empty:
        rows.append('<div class="compact-row"><div>변화가 큰 지역이 없습니다.</div><div></div><div></div><div></div><div></div></div>')
    else:
        for _, row in predictions.head(max_rows).iterrows():
            rows.append(
                f"""
                <div class="compact-row">
                  <div>{escape_text(row['sigungu_key'])}</div>
                  <div>{format_weight(row['predicted_waste_g'])}</div>
                  <div>{format_weight(row['recent_7day_average_g'])}</div>
                  <div>{format_pct(row['change_vs_recent_average_pct'])}</div>
                  <div>{status_badge(row['surge_level'])}</div>
                </div>
                """
            )
    render_markup(f"<div class='compact-list'>{''.join(rows)}</div>", unsafe_allow_html=True)


def render_stat_row(items: list[tuple[str, str, str]]) -> None:
    cells = []
    for label, value, note in items:
        cells.append(
            f"""
            <div>
              <div class="stat-label">{escape_text(label)}</div>
              <div class="stat-value">{escape_text(value)}</div>
              <div class="stat-note">{escape_text(note)}</div>
            </div>
            """
        )
    render_markup(f"<div class='stat-row'>{''.join(cells)}</div>", unsafe_allow_html=True)


def render_service_index() -> None:
    render_markup(
        """
        <div class="service-index" aria-label="FoodZero 핵심 서비스">
          <div class="service-item">
            <div class="service-num">01</div>
            <div>
              <div class="service-title">지역 배출 흐름 분석</div>
              <div class="service-copy">지역별 실제 배출 기록과 변화 패턴을 확인합니다.</div>
            </div>
          </div>
          <div class="service-item">
            <div class="service-num">02</div>
            <div>
              <div class="service-title">다음 날 배출량 예측</div>
              <div class="service-copy">과거 배출 패턴을 기반으로 다음 시점의 배출량을 예측합니다.</div>
            </div>
          </div>
          <div class="service-item">
            <div class="service-num">03</div>
            <div>
              <div class="service-title">수거·관리 우선 지역 탐색</div>
              <div class="service-copy">평상시보다 배출 증가가 예상되는 지역을 우선 확인합니다.</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_flow() -> None:
    render_markup(
        """
        <div class="pipeline-flow" aria-label="FoodZero 데이터 파이프라인">
          <div><span>01</span><strong>COLLECT</strong><p>지자체별 일별 RFID 음식물쓰레기 배출 데이터를 모읍니다.</p></div>
          <div><span>02</span><strong>UNDERSTAND</strong><p>인구와 세대, 날짜 흐름을 함께 보며 지역별 패턴을 이해합니다.</p></div>
          <div><span>03</span><strong>FORECAST</strong><p>최근 배출 패턴을 바탕으로 다음 흐름을 예측합니다.</p></div>
          <div><span>04</span><strong>ACT</strong><p>배출량 증가가 예상되는 지역을 사전에 확인해 수거 일정 조정, 인력·차량 배치, 감축 대상 지역 선정 등의 의사결정을 지원합니다.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_snapshot(metrics_json: dict[str, Any]) -> None:
    render_markup(
        f"""
        <div class="model-snapshot">
          <div>
            <div class="model-kicker">Random Forest next-day model</div>
            <div class="model-title">과거 시점의 정보만 사용해 다음 날 배출량을 예측합니다.</div>
            <div class="model-copy">시간 순서 기반 테스트 데이터에서 성능을 평가했으며, R²는 정확도가 아닌 결정계수로 해석합니다.</div>
          </div>
          <div class="model-metrics">
            <div><span>Test MAE</span><strong>{format_weight(metrics_json["MAE"])}</strong></div>
            <div><span>평균 대비 MAE</span><strong>{metrics_json["mae_to_mean_target_ratio"]:.2%}</strong></div>
            <div><span>R²</span><strong>{metrics_json["R2"]:.4f}</strong></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_dashboard() -> None:
    dates = get_available_reference_dates()
    default_date = max(dates)
    predictions = predict_for_reference_date(pd.Timestamp(default_date))
    if predictions.empty:
        render_editorial_note("NOTE", "사용할 수 있는 예측 데이터가 없습니다.")
        return
    hero_text, hero_visual = st.columns([1.55, 0.78], gap="medium")
    with hero_text:
        render_markup(
            """
            <div class="landing-hero-copy">
              <div class="eyebrow">Environmental Data Intelligence</div>
              <h1 class="landing-title"><span class="hero-line">데이터로 예측하고,</span><span class="hero-line">더 적게 버립니다.</span></h1>
              <div class="landing-copy">
                <span class="sentence-line">지역별 음식물쓰레기 배출 데이터를 분석해 다음 날 배출량을 예측하고,</span>
                <span class="sentence-line">관리가 필요한 지역을 먼저 찾아냅니다.</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cta_left, cta_right = st.columns([1, 1], gap="small")
        with cta_left:
            st.button(
                "지역 데이터 살펴보기",
                key="home_regions_cta",
                type="secondary",
                on_click=navigate_to_page,
                args=("지역 현황",),
                width="stretch",
            )
        with cta_right:
            st.button(
                "배출량 예측",
                key="home_prediction_cta",
                type="secondary",
                on_click=navigate_to_page,
                args=("배출량 예측",),
                width="stretch",
            )
    with hero_visual:
        render_home_visual(predictions)
    render_markup(
        """
        <div class="hero-notes" aria-label="FoodZero capabilities">
          <div class="hero-note">
            <div class="hero-note-title">Regional waste data</div>
            <div class="hero-note-copy">지역별 실제 배출 기록</div>
          </div>
          <div class="hero-note">
            <div class="hero-note-title">AI forecast</div>
            <div class="hero-note-copy">다음 날 배출량 예측</div>
          </div>
          <div class="hero-note">
            <div class="hero-note-title">Decision support</div>
            <div class="hero-note-copy">관리 우선 지역 확인</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_markup(
        """
        <section class="editorial-section why-foodzero" aria-label="FoodZero가 필요한 이유">
          <div class="editorial-kicker">Why FoodZero</div>
          <div class="editorial-title">음식물쓰레기는 매일 같은 양으로 발생하지 않습니다.</div>
          <div class="editorial-copy">
            지역 규모, 요일, 최근 배출 흐름에 따라 발생량은 달라집니다. FoodZero는 사후 대응에 머무르지 않고,
            다음 흐름을 먼저 읽어 수거와 관리 판단에 필요한 신호를 정리합니다.
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_section_header("FoodZero가 하는 일", "분석, 예측, 관리 우선순위를 하나의 서비스 흐름으로 연결합니다.")
    render_service_index()
    service_cta_cols = st.columns([1, 1, 1, 2.4], gap="small")
    with service_cta_cols[0]:
        st.button("지역 데이터", key="home_service_region", type="secondary", on_click=navigate_to_page, args=("지역 현황",), width="stretch")
    with service_cta_cols[1]:
        st.button("배출량 예측", key="home_service_prediction", type="secondary", on_click=navigate_to_page, args=("배출량 예측",), width="stretch")
    with service_cta_cols[2]:
        st.button("수거·관리", key="home_service_management", type="secondary", on_click=navigate_to_page, args=("수거·관리 지원",), width="stretch")

    dataset = load_next_day_dataset()
    counts = load_data_overview_counts()
    source_regions = int(dataset["sigungu_key"].nunique())
    render_markup(
        f"""
        <section class="home-section data-overview" aria-label="데이터 개요">
          <div class="home-section-kicker">FoodZero in numbers</div>
          <h2 class="home-section-title">실제 프로젝트 데이터로 구성한 예측 기반</h2>
          <p class="home-section-copy"><span class="sentence-line">공공데이터에 기록된 지역별 배출량을 같은 시간축 위에서 정리하고,</span><span class="sentence-line">다음 날 예측이 가능한 관측치를 모델링에 활용했습니다.</span></p>
          <div class="editorial-stats">
            <div><div class="editorial-stat-value">{counts['modeling_rows']:,}</div><div class="editorial-stat-label">예측 모델 관측치</div></div>
            <div><div class="editorial-stat-value">{source_regions:,}</div><div class="editorial-stat-label">예측 가능 지역</div></div>
            <div><div class="editorial-stat-value">2021—2024</div><div class="editorial-stat-label">분석 데이터 기간</div></div>
          </div>
          <p class="home-section-copy"><span class="sentence-line">원천 RFID 배출 기록 {counts['source_rows']:,}건을 정제하여 다음 날 예측이 가능한 {counts['modeling_rows']:,}건을 모델링에 활용했습니다.</span></p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    regional_left, regional_right = st.columns([1, 1], gap="large")
    with regional_left:
        render_markup(
            """
        <section class="home-section regional-flow-section" aria-label="지역별 배출 흐름">
              <div class="home-section-kicker">Regional flow / 02</div>
              <h2 class="home-section-title">지역마다 다른 배출 흐름</h2>
              <p class="home-section-copy"><span class="sentence-line">같은 기간에도 지역별 배출량과 변화 패턴은 다르게 나타납니다.</span><span class="sentence-line">실제 기록을 기준으로 규모가 큰 흐름부터 비교합니다.</span></p>
              <div class="home-insight"><strong>읽는 법</strong><span class="sentence-line">막대의 길이는 선택한 데이터셋 기준 배출량을,</span><span class="sentence-line">색은 최근 평균 대비 변화 상태를 나타냅니다.</span></div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with regional_right:
        render_markup('<div class="home-ranking-head"><div class="home-ranking-label">Regional comparison</div></div>', unsafe_allow_html=True)
        sido = st.selectbox("지역", ["전체", *get_available_sidos()], key="dashboard_sido")
        filtered = predictions if sido == "전체" else predictions[predictions["sido"].eq(sido)]
        top_chart = filtered.sort_values("predicted_waste_g", ascending=False).head(14)
        if not top_chart.empty:
            fig = px.bar(
                top_chart.sort_values("predicted_waste_g"),
                x="predicted_waste_g",
                y="sigungu_key",
                orientation="h",
                color="surge_label",
                color_discrete_map={"평소 수준": "#17231D", "확인 필요": "#397A58", "우선 확인": "#397A58"},
                labels={"predicted_waste_g": "배출량(g)", "sigungu_key": "", "surge_label": ""},
                height=520,
            )
            fig.update_traces(marker_line_width=0, hovertemplate="%{y}<br>배출량 %{x:,.0f} g<extra></extra>")
            fig.update_layout(legend_title_text="", xaxis_title="", yaxis_title="", showlegend=False)
            fig.update_yaxes(automargin=True, tickfont={"size": 12})
            fig = apply_plot_style(fig, height=520, showlegend=False)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"family": PLOT_FONT, "color": "#C7D1C9", "size": 17},
                margin={"l": 10, "r": 12, "t": 8, "b": 8},
                xaxis={"gridcolor": "rgba(255,255,255,.08)", "zerolinecolor": "rgba(255,255,255,.12)", "linecolor": "rgba(255,255,255,.12)", "tickfont": {"color": "#AEB8B0", "size": 16}},
                yaxis={"gridcolor": "rgba(255,255,255,.04)", "zerolinecolor": "rgba(255,255,255,.08)", "linecolor": "rgba(255,255,255,.12)", "tickfont": {"color": "#C7D1C9", "size": 16}},
            )
            st.plotly_chart(fig, width="stretch")
        else:
            render_editorial_note("NOTE", "선택한 지역의 데이터가 없습니다.")

    surge_table = filtered[filtered["surge_level"].isin(["attention", "high"])].sort_values(
        ["surge_level", "change_vs_recent_average_pct"], ascending=[True, False]
    )
    change_left, change_right = st.columns([1.08, 0.92], gap="large")
    with change_left:
        render_markup(
            """
            <section class="home-section change-detection-section" aria-label="변화 감지">
              <div class="home-section-kicker">Signals / 03</div>
              <h2 class="home-section-title">변화가 큰 지역을 먼저 발견합니다.</h2>
              <p class="home-section-copy"><span class="sentence-line">최근 평균과 비교해 눈에 띄는 변화가 있는 지역만 간결하게 보여주며</span><span class="sentence-line">모든 지역을 같은 방식으로 읽지 않아도 됩니다.</span></p>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with change_right:
        render_compact_change_rows(surge_table, max_rows=6)

    forecast_left, forecast_right = st.columns([1.45, 0.55], gap="large")
    with forecast_left:
        render_markup(
            """
            <section class="home-forecast forecast-section" aria-label="예측 기능 소개">
              <div class="editorial-kicker">Forecast simulation</div>
              <div class="editorial-title">과거의 패턴으로<br/>다음 흐름을 살펴봅니다.</div>
              <div class="editorial-copy"><span class="sentence-line">선택한 과거 시점의 배출 기록을 바탕으로</span><span class="sentence-line">다음 날의 음식물쓰레기 발생량을 예측해볼 수 있습니다.</span></div>
              <div class="home-forecast-note">Historical prediction simulation · 선택한 기준일 기반</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
    with forecast_right:
        render_markup(
            f"""
            <div class="forecast-preview" aria-label="선택된 과거 기준일 예측 미리보기">
              <div class="forecast-label">Reference</div>
              <div class="forecast-date">{pd.Timestamp(default_date).strftime('%Y.%m.%d')} → {pd.Timestamp(default_date + pd.Timedelta(days=1)).strftime('%Y.%m.%d')}</div>
              <div class="forecast-label">Forecast</div>
              <div class="forecast-value">{format_weight(predictions['predicted_waste_g'].sum())}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "배출량 예측 살펴보기",
            key="home_forecast_cta",
            type="secondary",
            on_click=navigate_to_page,
            args=("배출량 예측",),
            width="stretch",
        )

    render_section_header("작동 방식", "공공데이터를 수집하고, 지역별 패턴을 이해한 뒤, 다음 날 예측과 관리 판단으로 연결합니다.")
    render_pipeline_flow()

    metrics_json = load_evaluation_tables()["test_metrics"]
    render_section_header("AI 예측 모델", "FoodZero는 단순 시각화가 아니라 Random Forest 기반 다음 날 예측 모델을 사용합니다.")
    render_model_snapshot(metrics_json)
    render_markup(
        """
        <section class="dark-band" aria-label="FoodZero 소개">
          <div class="home-dark-kicker">FoodZero</div>
          <h2>더 적게 버리기 위한 데이터.</h2>
          <p>FoodZero는 지역별 배출 기록을 이해 가능한 데이터로 바꾸고, 다음 날 배출량 예측과 관리 우선 지역 분석을 통해 수거·관리 의사결정을 지원합니다.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    dark_left, dark_right = st.columns([1, 1], gap="small")
    with dark_left:
        st.button(
            "FoodZero 알아보기",
            key="home_about_cta",
            type="secondary",
            on_click=navigate_to_page,
            args=("FoodZero 소개",),
            width="stretch",
        )
    with dark_right:
        st.button(
            "데이터 인사이트",
            key="home_analysis_cta",
            type="secondary",
            on_click=navigate_to_page,
            args=("데이터 인사이트",),
            width="stretch",
        )


def page_prediction() -> None:
    render_page_title(
        "다음 날 음식물쓰레기 배출량을 예측합니다.",
        "선택한 과거 기준일까지의 정보만 사용해 다음 날 배출량을 예측하고 실제 결과와 비교합니다.",
        eyebrow="AI Forecast",
    )
    render_editorial_note("Historical backtest", "선택한 기준일까지의 정보만 모델 입력에 사용합니다.")
    render_markup(
        """
        <div class="usage-flow" aria-label="배출량 예측 사용 흐름">
          <span>01 지역 선택</span><span>02 기준일 선택</span><span>03 예측 실행</span><span>04 결과 확인</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_section_header("지역과 기준 데이터 선택", "지역과 기준일을 선택한 뒤 예측을 실행하세요.")
    col1, col2, col3, col4 = st.columns([1, 1.5, 1.1, 0.8])
    with col1:
        sido = st.selectbox("시도", get_available_sidos(), key="predict_sido")
    municipalities = get_municipalities(sido)
    with col2:
        municipality = st.selectbox("기초지자체", municipalities, key="predict_muni")
    dates = get_available_reference_dates(municipality)
    with col3:
        date = st.selectbox("기준일", dates, index=len(dates) - 1, format_func=lambda x: x.strftime("%Y.%m.%d"), key="predict_date")
    with col4:
        run = st.button("예측 실행", width="stretch")

    if not run:
        return
    try:
        result = run_prediction(municipality, pd.Timestamp(date))
    except ValueError as exc:
        render_editorial_note("NOTE", str(exc))
        return
    actual_g = actual_outcome_for_prediction(municipality, pd.Timestamp(result["prediction_date"]))
    outcome = prediction_error_summary(float(result["predicted_waste_g"]), actual_g)

    render_section_header("예측 결과", "선택한 기준일 다음 날의 예측 배출량과 실제 결과를 함께 확인합니다.")
    render_markup(
        f"""
        <div class="prediction-spotlight">
          <div>
            <div class="prediction-label">예측 배출량</div>
            <div class="prediction-main">{format_weight(result["predicted_waste_g"])}</div>
            <div class="prediction-date">예측일 {pd.Timestamp(result["prediction_date"]).strftime("%Y.%m.%d")}</div>
          </div>
          <div>
            <div class="prediction-label">실제 배출량</div>
            <div class="prediction-sub">{format_weight(outcome["actual_g"])}</div>
          </div>
          <div>
            <div class="prediction-label">최근 7일 평균</div>
            <div class="prediction-sub">{format_weight(result["recent_7day_average_g"])}</div>
          </div>
          <div>
            <div class="prediction-label">평균 대비</div>
            <div class="prediction-sub">{format_pct(result["change_vs_recent_average_pct"])}</div>
          </div>
          <div>
            <div class="prediction-label">예측 오차</div>
            <div class="prediction-sub">{format_weight(outcome["absolute_error_g"])}</div>
          </div>
          <div>
            <div class="prediction-label">오차율</div>
            <div class="prediction-sub">{format_abs_pct(outcome["absolute_percentage_error"])}</div>
          </div>
          <div>
            <div class="prediction-label">상태</div>
            <div class="prediction-sub">{status_text(result["surge_level"])}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_section_header("최근 실제 흐름과 예측 지점", "최근 실제 배출량과 예측 대상일의 지점을 함께 표시합니다.")
    trend = recent_trend(municipality, pd.Timestamp(date), days=30)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trend["date"], y=trend["waste_amount_t"], mode="lines+markers", name="실제 배출량", line=dict(color="rgba(205,213,208,.45)", width=2.5), marker=dict(size=5, color="rgba(205,213,208,.60)")))
    fig.add_trace(go.Scatter(x=[pd.Timestamp(result["prediction_date"])], y=[result["predicted_waste_g"]], mode="markers+text", name="예측값", marker=dict(color="#69c88f", size=15, symbol="diamond"), text=[f"예측<br>{format_weight(result['predicted_waste_g'])}"], textposition="top center"))
    if outcome["actual_g"] is not None:
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(result["prediction_date"])],
                y=[outcome["actual_g"]],
                mode="markers+text",
                name="실제 결과",
                marker=dict(color="rgba(205,213,208,.82)", size=10, symbol="circle", line=dict(color="#f3f4f1", width=1)),
                text=[f"실제<br>{format_weight(outcome['actual_g'])}"],
                textposition="bottom center",
            )
        )
    fig = apply_plot_style(fig, height=520, showlegend=True)
    fig.update_layout(xaxis_title="날짜", yaxis_title="배출량(g)")
    st.plotly_chart(fig, width="stretch")

    render_section_header("예측에 영향을 준 주요 요소", "아래 항목은 예측에 상대적으로 크게 활용된 변수이며, 인과관계를 의미하지 않습니다.")
    importance = load_evaluation_tables()["importance"].head(8).copy()
    name_map = {
        "rolling_mean_7": "최근 7일 평균 배출량",
        "waste_amount_t": "기준일 배출량",
        "target_day_of_week": "예측일 요일",
        "lag_1": "전일 배출량",
        "rolling_mean_14": "최근 14일 평균",
        "lag_7": "7일 전 배출량",
        "households": "세대 수",
        "total_population": "인구 수",
    }
    importance["feature_label"] = importance["feature"].map(name_map).fillna(importance["feature"])
    fig_imp = px.bar(importance.sort_values("importance_mae_increase"), x="importance_mae_increase", y="feature_label", orientation="h", labels={"importance_mae_increase": "영향도", "feature_label": "요소"}, height=430)
    fig_imp.update_traces(marker_color="#2f6b4f", marker_line_width=0, hovertemplate="%{y}<br>영향도 %{x:,.0f}<extra></extra>")
    fig_imp = apply_plot_style(fig_imp, height=450, showlegend=False)
    st.plotly_chart(fig_imp, width="stretch")


def page_management() -> None:
    render_page_title(
        "배출 증가가 예상되는 지역을 먼저 확인합니다.",
        "다음 날 예측 결과를 평상시 배출 수준과 비교해 관리 우선 확인 지역을 제공합니다.",
        eyebrow="Decision Support",
    )
    render_editorial_note("과거 데이터 기반 예측 시뮬레이션", "HISTORICAL SIMULATION · 실시간 운영 데이터가 아닌 과거 데이터 기반 분석 결과입니다.")

    dates = get_available_reference_dates()
    default_date = max(dates)
    reference_date = st.selectbox(
        "분석 기준일",
        dates,
        index=len(dates) - 1,
        format_func=lambda x: x.strftime("%Y.%m.%d"),
        key="management_reference_date",
    )
    reference_date = pd.Timestamp(reference_date)
    predictions = prepare_management_predictions(reference_date)

    render_markup(
        f"<div class='data-meta'>분석 기준일: {reference_date.strftime('%Y.%m.%d')} &nbsp;·&nbsp; 예측 대상일: {(reference_date + pd.Timedelta(days=1)).strftime('%Y.%m.%d')}</div>",
        unsafe_allow_html=True,
    )

    if predictions.empty:
        render_editorial_note("NOTE", "선택한 기준일에 예측 가능한 지역이 없습니다.")
        return

    kpis = management_kpis(predictions)
    render_section_header("관리 브리핑", "동일 기준일에 비교 가능한 지역만 집계합니다.")
    render_stat_row(
        [
            ("예측 대상 지역", f"{kpis['municipalities']:,}곳", "해당 기준일 가용 지역"),
            ("집중관리 지역", f"{kpis['high']:,}곳", "과거 분포 기준 높은 증가 수준"),
            ("주의 지역", f"{kpis['attention']:,}곳", "평소보다 증가 예상"),
            ("안정 지역", f"{kpis['normal']:,}곳", "평소 수준"),
        ]
    )

    render_section_header("위험도 분포", "해당 기준일 예측 가능 지역의 위험도 구성을 보여줍니다.")
    render_risk_distribution(kpis)

    priority_predictions = priority_management_predictions(predictions)
    render_section_header("관리 우선 확인 지역", "평상시보다 배출량 증가가 예상되어 우선 확인이 필요한 지역입니다.")
    render_management_table(priority_predictions, max_rows=10, include_error_reference=False)

    render_section_header("전체 지역 관리 현황", "지역별 모델 예측 오차 참고 지표입니다. 테스트 데이터에서 측정한 평균 배출량 대비 MAE 비율이며, 관리 우선순위 산정에는 사용되지 않습니다.")
    risk_filter = st.selectbox("위험도 필터", ["전체", "집중관리", "주의", "안정"], key="management_risk_filter")
    risk_to_level = {value: key for key, value in MANAGEMENT_STATUS_LABELS.items()}
    shown = predictions if risk_filter == "전체" else predictions[predictions["surge_level"].eq(risk_to_level[risk_filter])]
    render_management_table(shown, include_error_reference=True)


def page_region() -> None:
    render_page_title("지역별 배출 흐름", "지역마다 다른 음식물쓰레기 배출 패턴을 살펴보세요.", eyebrow="Regional Data")
    df = load_next_day_dataset()
    render_section_header("지역 선택", "시도와 시군구를 선택하면 해당 지역의 흐름을 보여줍니다.")
    col1, col2, col3 = st.columns([1, 1.5, 1.4])
    with col1:
        sido = st.selectbox("시도", get_available_sidos(), key="region_sido")
    municipalities = get_municipalities(sido)
    with col2:
        municipality = st.selectbox("기초지자체", municipalities, key="region_muni")
    region_df = df[df["sigungu_key"].eq(municipality)].sort_values("date")
    min_date, max_date = region_df["date"].min(), region_df["date"].max()
    with col3:
        period = st.date_input("기간", value=(max_date - pd.Timedelta(days=90), max_date), min_value=min_date, max_value=max_date)
    if isinstance(period, tuple) and len(period) == 2:
        start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    else:
        start, end = min_date, max_date
    shown = region_df[region_df["date"].between(start, end)].copy()
    if shown.empty:
        render_editorial_note("NOTE", "선택한 기간에 데이터가 없습니다.")
        return
    render_stat_row(
        [
            ("평균 배출량", format_weight(shown["waste_amount_t"].mean()), "선택 기간 일평균"),
            ("최대 배출량", format_weight(shown["waste_amount_t"].max()), "선택 기간 최대값"),
            ("최근 변화", format_pct((shown["waste_amount_t"].iloc[-1] / shown["waste_amount_t"].iloc[0] - 1) * 100 if shown["waste_amount_t"].iloc[0] else np.nan), "기간 첫날 대비"),
            ("인구 / 세대", f"{shown['total_population'].iloc[-1]:,.0f} / {shown['households'].iloc[-1]:,.0f}", "최근 기준"),
        ]
    )

    render_section_header("기간별 배출 흐름", "선택 지역의 일별 음식물쓰레기 배출량입니다.")
    fig = px.line(shown, x="date", y="waste_amount_t", labels={"date": "날짜", "waste_amount_t": "배출량(g)"}, height=540)
    fig.update_traces(line_color="#72c995", opacity=0.92, line_width=2.8, hovertemplate="날짜 %{x|%Y.%m.%d}<br>배출량 %{y:,.0f} g<extra></extra>")
    fig = apply_plot_style(fig, height=560, showlegend=False)
    st.plotly_chart(fig, width="stretch")

    render_section_header("지역 비교", "같은 시도 내 최대 3개 지역과 배출 추이를 비교합니다.")
    compare = st.multiselect("비교 지역", [m for m in get_municipalities(sido) if m != municipality], max_selections=3)
    if compare:
        comp = df[df["sigungu_key"].isin([municipality, *compare]) & df["date"].between(start, end)]
        fig_comp = px.line(comp, x="date", y="waste_amount_t", color="sigungu_key", labels={"date": "날짜", "waste_amount_t": "배출량(g)", "sigungu_key": "지역"}, height=500)
        fig_comp.update_traces(line_width=2.3)
        fig_comp = apply_plot_style(fig_comp, height=520, showlegend=True)
        st.plotly_chart(fig_comp, width="stretch")
    else:
        render_editorial_note("비교 지역을 선택하면", "같은 기간의 흐름을 함께 표시합니다.", prefix="NOTE")

    render_section_header("일별 데이터", "선택 조건에 해당하는 실제 집계 데이터입니다.")
    render_markup(
        f"<div class='data-meta'>{start.strftime('%Y.%m.%d')} — {end.strftime('%Y.%m.%d')} &nbsp;·&nbsp; {len(shown):,} records</div>",
        unsafe_allow_html=True,
    )
    table = shown[["date", "waste_amount_t", "recent_7day_average_g", "total_population", "households", "population_per_household"]].copy()
    table["date"] = table["date"].dt.strftime("%Y.%m.%d")
    table["waste_amount_t"] = table["waste_amount_t"].map(format_weight)
    table["recent_7day_average_g"] = table["recent_7day_average_g"].map(format_weight)
    table["total_population"] = table["total_population"].map(lambda x: f"{x:,.0f}")
    table["households"] = table["households"].map(lambda x: f"{x:,.0f}")
    table["population_per_household"] = table["population_per_household"].map(lambda x: f"{x:,.2f}")
    render_html_table(
        table.tail(30),
        [
            ("date", "날짜"),
            ("waste_amount_t", "배출량"),
            ("recent_7day_average_g", "최근 7일 평균"),
            ("total_population", "인구 수"),
            ("households", "세대 수"),
            ("population_per_household", "세대당 인구"),
        ],
    )


def page_analysis() -> None:
    render_page_title("FoodZero의 예측 모델을 데이터로 검증합니다.", "모델 성능, 중요 변수, 비교 결과를 분석 리포트 흐름으로 살펴봅니다.", eyebrow="Model Insights")
    tables = load_evaluation_tables()
    metrics_json = tables["test_metrics"]
    render_section_header("예측은 어느 정도 차이가 날까요?", "시간 기준으로 나눈 테스트 데이터에서 확인한 실제 평가 결과입니다.")
    render_stat_row(
        [
            ("테스트 MAE", format_weight(metrics_json["MAE"]), f"{metrics_json['test_rows']:,}개 테스트 행"),
            ("평균 배출량 대비 MAE", f"{metrics_json['mae_to_mean_target_ratio']:.2%}", "낮을수록 좋음"),
            ("결정계수(R²)", f"{metrics_json['R2']:.4f}", "정확도가 아닌 설명력 지표"),
            ("평가 지역", f"{metrics_json['test_municipalities']:,}곳", "시간 분할 테스트"),
        ]
    )
    render_editorial_note("Model note", "R²는 정확도가 아닌 결정계수입니다. 모델 성능은 MAE, RMSE, R²를 함께 확인합니다.")

    render_section_header("01 어떤 정보가 예측에 중요했을까요?", "모델이 예측할 때 상대적으로 크게 활용한 요소입니다.")
    importance = tables["importance"].head(8).copy()
    name_map = {
        "rolling_mean_7": "최근 7일 평균",
        "waste_amount_t": "기준일 배출량",
        "target_day_of_week": "요일",
        "lag_1": "최근 1일 배출량",
        "rolling_mean_14": "최근 14일 평균",
        "lag_7": "최근 7일 배출량",
        "households": "세대 수",
        "total_population": "인구 수",
    }
    importance["feature_label"] = importance["feature"].map(name_map).fillna(importance["feature"])
    sorted_importance = importance.sort_values("importance_mae_increase")
    fig_importance = px.bar(sorted_importance, x="importance_mae_increase", y="feature_label", orientation="h", labels={"importance_mae_increase": "영향도", "feature_label": "요소"}, height=470)
    importance_colors = ["#6E756F"] * len(sorted_importance)
    if importance_colors:
        importance_colors[-1] = "#6FCF97"
    fig_importance.update_traces(marker_color=importance_colors, marker_line_width=0, hovertemplate="%{y}<br>영향도 %{x:,.0f}<extra></extra>")
    fig_importance = apply_plot_style(fig_importance, height=500, showlegend=False)
    st.plotly_chart(fig_importance, width="stretch")

    render_section_header("02 Baseline 및 모델 비교", "Validation MAE 기준으로 비교한 성능입니다.")
    comparison = tables["comparison"].copy()
    final_model = comparison.sort_values("MAE", ascending=True).iloc[0]
    final_model_name = "Random Forest" if final_model["model"] == "random_forest" else str(final_model["model"])
    render_markup(
        f"""
        <div class="final-model-callout">
          <div class="final-model-kicker">FINAL MODEL</div>
          <div class="final-model-name">{escape_text(final_model_name)}</div>
          <div class="final-model-copy">Validation MAE {format_weight(final_model["MAE"])}으로 비교 모델 중 가장 낮은 오차를 보여 FoodZero의 다음 날 배출량 예측 모델로 선정했습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    sorted_comparison = comparison.sort_values("MAE", ascending=False)
    fig = px.bar(sorted_comparison, x="MAE", y="model", orientation="h", labels={"MAE": "Validation MAE(g)", "model": "모델"}, height=430)
    comparison_colors = ["#70CF97" if model == "random_forest" else "#747B77" for model in sorted_comparison["model"]]
    fig.update_traces(marker_color=comparison_colors, marker_line_width=0, hovertemplate="%{y}<br>MAE %{x:,.0f} g<extra></extra>")
    fig = apply_plot_style(fig, height=480, showlegend=False)
    st.plotly_chart(fig, width="stretch")
    comparison_display = comparison.copy()
    comparison_display["MAE"] = comparison_display["MAE"].map(format_weight)
    comparison_display["RMSE"] = comparison_display["RMSE"].map(format_weight)
    comparison_display["R2"] = comparison_display["R2"].map(lambda x: f"{x:.4f}")
    comparison_display["rows"] = comparison_display["rows"].map(lambda x: f"{x:,.0f}")
    render_html_table(
        comparison_display,
        [("model", "모델"), ("split", "구분"), ("rows", "행 수"), ("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²")],
    )

    render_section_header("03 지역 규모별 성능", "소형·중형·대형 지자체 그룹별 평균 성능입니다.")
    size_display = tables["size_group"].copy()
    size_display["mean_target_next_day"] = size_display["mean_target_next_day"].map(format_weight)
    size_display["mean_mae"] = size_display["mean_mae"].map(format_weight)
    size_display["mean_mae_to_mean_ratio"] = size_display["mean_mae_to_mean_ratio"].map(lambda x: f"{x:.2%}")
    render_html_table(
        size_display,
        [
            ("size_group", "지역 규모"),
            ("municipalities", "지자체 수"),
            ("mean_target_next_day", "평균 배출량"),
            ("mean_mae", "평균 MAE"),
            ("mean_mae_to_mean_ratio", "평균 대비 MAE"),
        ],
        auto_align_numbers=False,
    )


def page_about() -> None:
    render_page_title("더 적게 버리기 위한 데이터.", "FoodZero는 지역별 음식물쓰레기 배출 기록을 분석해 배출 흐름을 이해하고 다음 시점의 변화를 예측하는 환경 데이터 프로젝트입니다.", eyebrow="About FoodZero")
    render_markup(
        """
        <div class="intro-quiet">
          음식물쓰레기는 매일 같은 양으로 발생하지 않습니다. 지역의 생활 규모, 요일,
          최근 배출 흐름이 함께 움직이기 때문에 FoodZero는 실제 기록을 제품처럼 읽기 쉬운
          데이터 경험으로 정리합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_markup(
        """
        <div class="story-grid">
          <div class="story-step">
            <div class="story-num">01</div>
            <div class="story-title">COLLECT</div>
            <div class="story-text">지자체별 일별 RFID 음식물쓰레기 배출 데이터를 모읍니다.</div>
          </div>
          <div class="story-step">
            <div class="story-num">02</div>
            <div class="story-title">UNDERSTAND</div>
            <div class="story-text">인구와 세대, 날짜 흐름을 함께 보며 지역별 패턴을 이해합니다.</div>
          </div>
          <div class="story-step">
            <div class="story-num">03</div>
            <div class="story-title">FORECAST</div>
            <div class="story-text">최근 배출 패턴을 바탕으로 다음 흐름을 예측합니다.</div>
          </div>
          <div class="story-step">
            <div class="story-num">04</div>
            <div class="story-title">ACT</div>
            <div class="story-text">배출량 증가가 예상되는 지역을 사전에 확인해 수거 일정 조정, 인력·차량 배치, 감축 대상 지역 선정 등의 의사결정을 지원합니다.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_section_header("Data sources", "FoodZero가 읽는 데이터의 출처와 범위입니다.")
    render_markup(
        """
        <div class="tech-note">
          한국환경공단 RFID 음식물쓰레기 배출 데이터와 주민등록 인구·세대 데이터를 결합해 지역별 배출 흐름을 분석합니다.
          기상청 ASOS 데이터는 연구 단계에서 추가 변수로 실험했으며,
          현재 서비스의 다음 날 예측 모델은 배출 이력·요일·인구·세대 정보를 중심으로 구성했습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_section_header("Model", "현재 서비스 화면의 다음 시점 예측 모델입니다.")
    render_markup(
        """
        <div class="tech-note">
          서비스 모델은 foodzero-next-day-rf-v1이며, 최근 배출량, 이동평균, 요일,
          인구와 세대 정보를 활용한 RandomForestRegressor 기반 다음 날 배출량 예측 모델입니다.
          기상청 ASOS 일자료는 연구용 weather 실험에 사용했으나, 현재 서비스용 1차 모델에는 포함하지 않았습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_section_header("Responsible AI & Limitations", "예측 결과는 의사결정을 돕는 참고 정보입니다.")
    render_markup(
        """
        <ul class="light-list">
          <li>데이터가 제공되는 지역만 분석할 수 있습니다.</li>
          <li>현재 서비스는 2021~2024년 공공데이터를 활용한 과거 시점 예측 시뮬레이션입니다.</li>
          <li>실제 수거 계획에는 현장 상황, 시설 여건, 지역 행사 등 운영 정보를 함께 고려해야 합니다.</li>
          <li>예측 결과는 의사결정을 돕는 참고 정보입니다.</li>
          <li>향후 최신 RFID 배출 데이터가 지속적으로 연결되면 동일한 예측 파이프라인을 운영형 서비스로 확장할 수 있습니다.</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    render_markup(
        """
        <div class="footer">
          <div>
            <strong>FoodZero</strong>
            <span>Environmental Data Intelligence</span>
          </div>
          <div class="footer-links">
            <span>지역 데이터</span>
            <span>배출량 예측</span>
            <span>수거·관리</span>
            <span>데이터 인사이트</span>
          </div>
          <div>
            <span>Data</span>
            <strong>2021—2024</strong>
          </div>
          <div>
            <span>Project</span>
            <strong>Food Waste Prediction & Decision Support</strong>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="FoodZero", page_icon="FZ", layout="wide", initial_sidebar_state="collapsed")
    apply_css()
    latest = max(get_available_reference_dates())
    page = navigation(pd.Timestamp(latest))
    if page == "Home":
        page_dashboard()
    elif page == "지역 현황":
        page_region()
    elif page == "배출량 예측":
        page_prediction()
    elif page == "수거·관리 지원":
        page_management()
    elif page == "데이터 인사이트":
        page_analysis()
    else:
        page_about()
    render_footer()


if __name__ == "__main__":
    main()
