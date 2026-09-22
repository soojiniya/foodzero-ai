from __future__ import annotations

from typing import Any

import pandas as pd

from .config import AUDIT_DOC_PATH, FOOD_WASTE_RAW_DIR, POPULATION_RAW_DIR, SOURCE_INVENTORY_PATH
from .features import (
    find_column,
    load_aliases,
    load_food_waste,
    load_population,
    make_sigungu_key,
    normalize_series,
    parse_population_wide,
)
from .io_utils import numeric_series, read_table, write_json


TABLE_SUFFIXES = {".csv", ".xlsx", ".xls"}


def list_source_files() -> dict[str, list[Path]]:
    return {
        "food_waste": sorted(p for p in FOOD_WASTE_RAW_DIR.glob("*") if p.suffix.lower() in TABLE_SUFFIXES),
        "population": sorted(p for p in POPULATION_RAW_DIR.glob("*") if p.suffix.lower() in TABLE_SUFFIXES),
    }


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum().to_dict()
    blank = (df.astype(str).apply(lambda col: col.str.strip().eq("").sum())).to_dict()
    sample_values = {
        col: [x for x in df[col].dropna().astype(str).drop_duplicates().head(10).tolist()]
        for col in df.columns[:50]
    }
    return {
        "rows_loaded": int(len(df)),
        "columns": list(df.columns),
        "missing_by_column": {k: int(v) for k, v in missing.items()},
        "blank_string_by_column": {k: int(v) for k, v in blank.items()},
        "sample_values": sample_values,
    }


def profile_file(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xls = pd.ExcelFile(path)
        sheets = {}
        for sheet in xls.sheet_names:
            df, meta = read_table(path, sheet_name=sheet)
            sheets[sheet] = {**meta, **profile_dataframe(df)}
        return {
            "path": str(path),
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
            "kind": "excel",
            "sheets": sheets,
        }
    df, meta = read_table(path)
    return {
        "path": str(path),
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        **meta,
        **profile_dataframe(df),
    }


def run_source_audit() -> dict[str, Any]:
    files = list_source_files()
    inventory = {"food_waste": [], "population": []}
    for group, paths in files.items():
        for path in paths:
            inventory[group].append(profile_file(path))
    inventory["analysis"] = build_analysis()
    write_json(SOURCE_INVENTORY_PATH, inventory)
    render_markdown(inventory, AUDIT_DOC_PATH)
    return inventory


def build_analysis() -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    try:
        food = load_food_waste()
        analysis["food_waste"] = summarize_food_waste(food)
    except Exception as exc:  # noqa: BLE001
        analysis["food_waste_error"] = str(exc)
        food = pd.DataFrame()
    try:
        population = load_population()
        analysis["population"] = summarize_population(population)
    except Exception as exc:  # noqa: BLE001
        analysis["population_error"] = str(exc)
        population = pd.DataFrame()
    if not food.empty and not population.empty:
        analysis["name_match"] = compare_authority_names(food, population)
    return analysis


def summarize_food_waste(food: pd.DataFrame) -> dict[str, Any]:
    raw = load_food_waste_raw()
    duplicate_key_count = int(raw.duplicated(subset=["date", "sido", "sigungu"]).sum()) if not raw.empty else 0
    exact_duplicate_count = int(raw.duplicated().sum()) if not raw.empty else 0
    year_counts = food.assign(year=food["date"].dt.year).groupby("year").size().to_dict()
    return {
        "total_rows": int(len(food)),
        "raw_rows": int(len(raw)),
        "date_min": food["date"].min().date().isoformat(),
        "date_max": food["date"].max().date().isoformat(),
        "year_counts": {str(k): int(v) for k, v in year_counts.items()},
        "unique_sido_count": int(food["sido"].nunique()),
        "unique_sigungu_count": int(food["sigungu_key"].nunique()),
        "sido": sorted(food["sido"].dropna().unique().tolist()),
        "sigungu": sorted(food["sigungu_key"].dropna().unique().tolist()),
        "missing_by_column": {k: int(v) for k, v in raw.isna().sum().to_dict().items()},
        "duplicate_exact_rows": exact_duplicate_count,
        "duplicate_date_authority_rows": duplicate_key_count,
    }


def load_food_waste_raw() -> pd.DataFrame:
    aliases = load_aliases()["food_waste"]
    frames = []
    for path in list_source_files()["food_waste"]:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            sheet_names = pd.ExcelFile(path).sheet_names
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
            out = pd.DataFrame(
                {
                    "source_file": path.name,
                    "source_sheet": str(sheet),
                    "sido": normalize_series(df[sido_col]) if sido_col else "",
                    "sigungu": normalize_series(df[sigungu_col]),
                    "waste_amount": numeric_series(df[amount_col]),
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
            out["sigungu_key"] = make_sigungu_key(out["sido"], out["sigungu"])
            frames.append(out)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_population(population: pd.DataFrame) -> dict[str, Any]:
    months = sorted(population["year_month"].dropna().unique().tolist())
    expected = pd.period_range("2021-01", "2024-01", freq="M").astype(str).tolist()
    sigungu_level = population[population["sigungu"].astype(str).str.strip().ne("")]
    usable = sigungu_level["total_population"].notna().all() and sigungu_level["households"].notna().all()
    return {
        "rows": int(len(population)),
        "months": months,
        "month_min": min(months) if months else None,
        "month_max": max(months) if months else None,
        "expected_months": expected,
        "missing_months": [m for m in expected if m not in months],
        "extra_months": [m for m in months if m not in expected],
        "unique_sigungu_count": int(sigungu_level["sigungu_key"].nunique()),
        "has_total_population": bool("total_population" in population.columns),
        "has_households": bool("households" in population.columns),
        "sigungu_population_households_usable": bool(usable),
    }


def compare_authority_names(food: pd.DataFrame, population: pd.DataFrame) -> dict[str, Any]:
    food_keys = sorted((set(food["sigungu_key"]) - {""}))
    population_keys = sorted((set(population["sigungu_key"]) - {""}))
    population_lookup = {compact_key(key): key for key in population_keys}
    exact = sorted(set(food_keys) & set(population_keys))
    correction = []
    unmatched = []
    for key in sorted(set(food_keys) - set(exact)):
        compact = compact_key(key)
        if compact in population_lookup:
            correction.append({"food_waste": key, "population": population_lookup[compact], "reason": "공백 차이"})
        else:
            unmatched.append(key)
    return {
        "food_sigungu_count": len(food_keys),
        "population_sigungu_count": len(population_keys),
        "exact_count": len(exact),
        "exact": exact,
        "correction_count": len(correction),
        "correction": correction,
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
    }


def compact_key(value: str) -> str:
    return str(value).replace(" ", "")


def render_markdown(inventory: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FoodZero AI 데이터 구조 감사",
        "",
        "이 문서는 파이프라인이 `data/raw`의 실제 파일을 읽어 생성합니다. 컬럼명, 인코딩, 행 수, 결측치, 샘플 값은 원본에서 직접 추출합니다.",
        "",
    ]
    lines.extend(render_analysis(inventory.get("analysis", {})))
    for group in ("food_waste", "population"):
        lines.append(f"## {group}")
        if not inventory[group]:
            lines.append("")
            lines.append(f"- `{group}` 원본 파일을 찾지 못했습니다.")
            lines.append("")
            continue
        for item in inventory[group]:
            lines.extend(render_item(item))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_analysis(analysis: dict[str, Any]) -> list[str]:
    lines = ["## 요약", ""]
    if "food_waste" in analysis:
        fw = analysis["food_waste"]
        lines.extend(
            [
                "### 음식물쓰레기 정규화 결과",
                "",
                f"- 총 행 수: {fw['total_rows']:,}",
                f"- 원본 행 수: {fw['raw_rows']:,}",
                f"- 데이터 기간: {fw['date_min']} ~ {fw['date_max']}",
                f"- 고유 광역시도 수: {fw['unique_sido_count']:,}",
                f"- 고유 기초지자체 수: {fw['unique_sigungu_count']:,}",
                f"- 완전 중복 행 수: {fw['duplicate_exact_rows']:,}",
                f"- 날짜+지자체 기준 중복 행 수: {fw['duplicate_date_authority_rows']:,}",
                "- 연도별 행 수:",
            ]
        )
        for year, count in fw["year_counts"].items():
            lines.append(f"  - {year}: {count:,}")
        lines.append("- 포함 광역시도:")
        for sido in fw["sido"]:
            lines.append(f"  - {sido}")
        lines.append("")
    elif "food_waste_error" in analysis:
        lines.extend(["### 음식물쓰레기 정규화 결과", "", f"- 오류: `{analysis['food_waste_error']}`", ""])
    if "population" in analysis:
        pop = analysis["population"]
        lines.extend(
            [
                "### 주민등록 인구 정규화 결과",
                "",
                f"- 정규화 행 수: {pop['rows']:,}",
                f"- 기간: {pop['month_min']} ~ {pop['month_max']}",
                f"- 기대 기간 누락: {', '.join(pop['missing_months']) if pop['missing_months'] else '없음'}",
                f"- 기대 기간 외 월: {', '.join(pop['extra_months']) if pop['extra_months'] else '없음'}",
                f"- 고유 시군구 키 수: {pop['unique_sigungu_count']:,}",
                f"- 총인구수 컬럼 사용 가능: {pop['has_total_population']}",
                f"- 세대수 컬럼 사용 가능: {pop['has_households']}",
                f"- 시군구 단위 총인구수/세대수 결측 없이 사용 가능: {pop['sigungu_population_households_usable']}",
                "",
            ]
        )
    elif "population_error" in analysis:
        lines.extend(["### 주민등록 인구 정규화 결과", "", f"- 오류: `{analysis['population_error']}`", ""])
    if "name_match" in analysis:
        match = analysis["name_match"]
        lines.extend(
            [
                "### 음식물쓰레기-인구 지자체명 비교",
                "",
                f"- 음식물쓰레기 고유 기초지자체 수: {match['food_sigungu_count']:,}",
                f"- 인구 데이터 고유 지자체 키 수: {match['population_sigungu_count']:,}",
                f"- 자동 정확 매칭: {match['exact_count']:,}",
                f"- 명칭 보정 후 매칭 가능: {match['correction_count']:,}",
                f"- 매칭되지 않음: {match['unmatched_count']:,}",
                "",
                "#### 명칭 보정 후보",
            ]
        )
        if match["correction"]:
            for item in match["correction"]:
                lines.append(f"- 음식물쓰레기 `{item['food_waste']}` -> 인구 `{item['population']}` ({item['reason']})")
        else:
            lines.append("- 없음")
        lines.extend(["", "#### 매칭되지 않는 지자체"])
        if match["unmatched"]:
            for key in match["unmatched"]:
                lines.append(f"- {key}")
        else:
            lines.append("- 없음")
        lines.append("")
    return lines


def render_item(item: dict[str, Any]) -> list[str]:
    lines = [f"### {item['file_name']}", "", f"- 경로: `{item['path']}`", f"- 파일 크기: {item['size_bytes']:,} bytes"]
    if item["kind"] == "excel":
        lines.append("- 형식: Excel")
        lines.append("")
        for sheet_name, sheet in item["sheets"].items():
            lines.extend(render_profile(sheet, heading=f"시트 `{sheet_name}`"))
        return lines
    lines.append(f"- 형식: CSV")
    lines.append(f"- 감지 인코딩: `{item.get('encoding')}`")
    lines.append("")
    lines.extend(render_profile(item, heading="프로파일"))
    return lines


def render_profile(profile: dict[str, Any], heading: str) -> list[str]:
    lines = [f"#### {heading}", "", f"- 행 수: {profile['rows_loaded']:,}", f"- 컬럼 수: {len(profile['columns'])}", "- 컬럼명:"]
    for col in profile["columns"]:
        lines.append(f"  - `{col}`")
    lines.append("- 컬럼별 결측/공백:")
    for col in profile["columns"]:
        miss = profile["missing_by_column"].get(col, 0)
        blank = profile["blank_string_by_column"].get(col, 0)
        lines.append(f"  - `{col}`: 결측 {miss:,}, 공백문자열 {blank:,}")
    lines.append("- 샘플 값:")
    for col, values in profile["sample_values"].items():
        preview = ", ".join(f"`{v}`" for v in values[:10])
        lines.append(f"  - `{col}`: {preview}")
    lines.append("")
    return lines


if __name__ == "__main__":
    run_source_audit()
