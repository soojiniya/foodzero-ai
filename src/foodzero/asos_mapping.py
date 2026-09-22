from __future__ import annotations

import math
import re
import time
import urllib.parse
import urllib.request
import json
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any

import pandas as pd

from .config import CONFIG_DIR, INTERIM_DIR
from .features import load_food_waste, normalize_series, standardize_authority_name
from .io_utils import ensure_parent
from .weather import MAPPING_COLUMNS


KMA_STATION_LIST_URL = "https://data.kma.go.kr/tmeta/stn/selectStnListDownload.do"
KMA_STATION_PAGE_URL = "https://data.kma.go.kr/tmeta/stn/selectStnList.do"
KMA_SOURCE_URL = "https://data.kma.go.kr/tmeta/stn/selectStnList.do"
STATION_CACHE_PATH = INTERIM_DIR / "reference" / "kma_station_list.csv"
GEOCODE_CACHE_PATH = INTERIM_DIR / "reference" / "municipality_geocodes.json"
REVIEW_DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "asos_mapping_review.md"

ASOS_MAPPING_COLUMNS = [
    "sido",
    "sigungu",
    "sigungu_key",
    "standard_sido",
    "standard_sigungu",
    "standard_sigungu_key",
    "proposed_station_id",
    "proposed_station_name",
    "station_id",
    "station_name",
    "station_latitude",
    "station_longitude",
    "latitude",
    "longitude",
    "station_start_date",
    "station_end_date",
    "municipality",
    "municipality_latitude",
    "municipality_longitude",
    "municipality_coordinate_source",
    "distance_km",
    "mapping_method",
    "mapping_status",
    "confidence",
    "match_type",
    "evidence",
    "source_url",
    "alternate_candidates",
    "notes",
    "validation_flags",
]

VALIDATION_START = pd.Timestamp("2021-01-01")
VALIDATION_END = pd.Timestamp("2024-01-31")
METRO_REPRESENTATIVE = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
}


def standardize_sigungu(sido: str, sigungu: str) -> tuple[str, str, str]:
    return standardize_authority_name(sido, sigungu)


def download_kma_station_list(force: bool = False) -> Path:
    ensure_parent(STATION_CACHE_PATH)
    if STATION_CACHE_PATH.exists() and not force:
        return STATION_CACHE_PATH
    payload = {
        "fileType": "csv",
        "pageIndex": "1",
        "schListCnt": "10",
        "mddlClssCd": "SFC01",
        "stnIds": "",
        "serviceSe": "F00101",
        "txtStnNm": "종관기상관측",
        "txtElementNm": "",
        "dTreeId": "",
        "gTreeId": "",
        "mddlClssCdDiff": "SFC01",
        "pgmNo": "",
    }
    payload["pgmNo"] = "82"
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.open(
        urllib.request.Request(KMA_STATION_PAGE_URL, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=60,
    ).read()
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        KMA_STATION_LIST_URL,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": KMA_STATION_PAGE_URL,
            "User-Agent": "Mozilla/5.0",
        },
    )
    with opener.open(request, timeout=60) as response:
        raw = response.read()
    if raw.lstrip().startswith(b"<!DOCTYPE") or raw.lstrip().startswith(b"<html"):
        raise RuntimeError("KMA station metadata download returned HTML instead of CSV.")
    STATION_CACHE_PATH.write_bytes(raw)
    return STATION_CACHE_PATH


def read_kma_station_list(path: Path | None = None) -> pd.DataFrame:
    station_path = path or download_kma_station_list()
    last_error: Exception | None = None
    for encoding in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            df = pd.read_csv(station_path, encoding=encoding, dtype=str)
            if len(df.columns) <= 1:
                continue
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    else:
        raise RuntimeError(f"Could not read KMA station list: {station_path}") from last_error
    df.columns = [str(c).strip() for c in df.columns]
    colmap = infer_station_columns(df.columns)
    station = pd.DataFrame(
        {
            "station_id": df[colmap["station_id"]].astype(str).str.strip(),
            "station_name": normalize_series(df[colmap["station_name"]]),
            "latitude": pd.to_numeric(df[colmap["latitude"]], errors="coerce"),
            "longitude": pd.to_numeric(df[colmap["longitude"]], errors="coerce"),
            "station_start_date": df[colmap["station_start_date"]].astype(str).str.strip() if "station_start_date" in colmap else "",
            "station_end_date": df[colmap["station_end_date"]].astype(str).str.strip() if "station_end_date" in colmap else "",
            "station_address": normalize_series(df[colmap["station_address"]]) if "station_address" in colmap else "",
        }
    )
    station = station.dropna(subset=["latitude", "longitude"]).drop_duplicates("station_id")
    station = station[station["station_id"].str.fullmatch(r"\d+")]
    station["station_start_dt"] = pd.to_datetime(station["station_start_date"], errors="coerce")
    station["station_end_dt"] = pd.to_datetime(station["station_end_date"].replace({"": pd.NA, "nan": pd.NA}), errors="coerce")
    active_start = pd.Timestamp("2021-01-01")
    active_end = pd.Timestamp("2024-01-31")
    station = station[
        station["station_start_dt"].fillna(pd.Timestamp("1900-01-01")).le(active_start)
        & station["station_end_dt"].fillna(pd.Timestamp("2099-12-31")).ge(active_end)
    ].copy()
    station = station[
        station["latitude"].between(33.0, 39.7)
        & station["longitude"].between(124.0, 132.2)
    ].copy()
    return station.reset_index(drop=True)


def infer_station_columns(columns: list[str] | pd.Index) -> dict[str, str]:
    def find(*needles: str, required: bool = True) -> str | None:
        for col in columns:
            compact = str(col).replace(" ", "")
            if all(needle in compact for needle in needles):
                return str(col)
        if required:
            raise ValueError(f"Could not find station column containing {needles}. Actual: {list(columns)}")
        return None

    colmap = {
        "station_id": find("지점"),
        "station_name": find("지점명"),
        "latitude": find("위도"),
        "longitude": find("경도"),
    }
    optional = {
        "station_start_date": find("시작일", required=False),
        "station_end_date": find("종료일", required=False),
        "station_address": find("지점주소", required=False),
    }
    colmap.update({k: v for k, v in optional.items() if v is not None})
    return colmap


def build_authorities() -> pd.DataFrame:
    waste = load_food_waste()
    authorities = waste[["sido", "sigungu", "sigungu_key"]].drop_duplicates().sort_values(["sido", "sigungu"])
    std = authorities.apply(lambda r: standardize_sigungu(r["sido"], r["sigungu"]), axis=1, result_type="expand")
    authorities[["standard_sido", "standard_sigungu", "standard_sigungu_key"]] = std
    return authorities.reset_index(drop=True)


def make_mapping(force_station_download: bool = False) -> pd.DataFrame:
    station_path = download_kma_station_list(force=force_station_download)
    stations = read_kma_station_list(station_path)
    authorities = build_authorities()
    # Official station metadata does not include municipality centroids. We therefore use exact/in-name
    # station matches as proposed, and nearest ASOS based on station-to-station proxy otherwise only
    # when no direct municipality station exists.
    rows: list[dict[str, Any]] = []
    geocodes = load_or_fetch_municipality_geocodes(authorities)
    for _, authority in authorities.iterrows():
        row = propose_station(authority, stations, geocodes)
        rows.append(row)
    mapping = validate_mapping(pd.DataFrame(rows), stations)
    ensure_parent(CONFIG_DIR / "asos_station_mapping.csv")
    mapping[ASOS_MAPPING_COLUMNS].to_csv(CONFIG_DIR / "asos_station_mapping.csv", index=False, encoding="utf-8-sig")
    render_validation_review_doc(mapping, stations)
    return mapping


def propose_station(authority: pd.Series, stations: pd.DataFrame, geocodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sido = authority["standard_sido"]
    sigungu = authority["standard_sigungu"]
    short_name = simplify_authority_name(sigungu)
    municipality_key = f"{sido} {sigungu}".strip()
    municipality_coord = geocodes.get(municipality_key, {})
    direct = stations[stations["station_name"].map(lambda x: station_name_matches(str(x), sigungu, short_name))]
    if direct.empty:
        direct = stations[stations["station_address"].map(lambda x: address_mentions(str(x), sido, sigungu))]
    if not direct.empty:
        candidates = add_distance(direct, municipality_coord).sort_values(["distance_km_sort", "station_id"]).head(5)
        chosen = candidates.iloc[0]
        match_type = "inside_municipality_name_or_address"
        confidence = "high"
        status = "proposed"
        distance_km = chosen.get("distance_km", pd.NA)
        notes = "공식 관측소 메타데이터의 지점명 또는 주소가 지자체명과 일치합니다. 경계 기반 내부 판정은 사람 검토가 필요합니다."
    else:
        match_type = "nearest_asos_by_municipality_coordinate"
        candidates = add_distance(stations, municipality_coord)
        if candidates["distance_km"].notna().any():
            candidates = candidates.sort_values(["distance_km_sort", "station_id"]).head(5)
            chosen = candidates.iloc[0]
        else:
            candidates = candidates.head(0)
            chosen = pd.Series(dtype=object)
        distance_km = chosen.get("distance_km", pd.NA) if not chosen.empty else pd.NA
        if pd.notna(distance_km) and float(distance_km) <= 50:
            confidence = "medium"
            status = "proposed"
            notes = "지자체명 직접 일치 ASOS가 없어 대표좌표 기준 최단거리 ASOS를 후보로 제안했습니다. 내부 관측소가 아니므로 검토가 필요합니다."
        else:
            confidence = "review_required"
            status = "review_required"
            notes = "지자체명 직접 일치 ASOS가 없고 대표좌표 기준 최단거리 후보도 멀거나 좌표 확인이 불충분합니다."
    station_id = str(chosen.get("station_id", "")) if not chosen.empty else ""
    station_name = str(chosen.get("station_name", "")) if not chosen.empty else ""
    evidence = (
        f"KMA 관측지점정보: 지점명/주소 기반 후보. station_id={station_id}, station_name={station_name}"
        if station_id
        else "KMA 관측지점정보에서 지자체명 직접 일치 관측소를 찾지 못함"
    )
    return {
        "sido": authority["sido"],
        "sigungu": authority["sigungu"],
        "sigungu_key": authority["sigungu_key"],
        "standard_sido": sido,
        "standard_sigungu": sigungu,
        "standard_sigungu_key": authority["standard_sigungu_key"],
        "proposed_station_id": station_id,
        "proposed_station_name": station_name,
        "station_id": station_id,
        "station_name": station_name,
        "station_latitude": chosen.get("latitude", pd.NA) if not chosen.empty else pd.NA,
        "station_longitude": chosen.get("longitude", pd.NA) if not chosen.empty else pd.NA,
        "latitude": chosen.get("latitude", pd.NA) if not chosen.empty else pd.NA,
        "longitude": chosen.get("longitude", pd.NA) if not chosen.empty else pd.NA,
        "station_start_date": chosen.get("station_start_date", "") if not chosen.empty else "",
        "station_end_date": chosen.get("station_end_date", "") if not chosen.empty else "",
        "municipality": municipality_key,
        "municipality_latitude": municipality_coord.get("latitude", pd.NA),
        "municipality_longitude": municipality_coord.get("longitude", pd.NA),
        "municipality_coordinate_source": municipality_coord.get("source", ""),
        "distance_km": distance_km,
        "mapping_method": match_type,
        "mapping_status": status,
        "confidence": confidence,
        "match_type": match_type,
        "evidence": evidence,
        "source_url": KMA_SOURCE_URL,
        "alternate_candidates": format_candidates(candidates.iloc[1:] if len(candidates) > 1 else candidates.head(0)),
        "notes": notes,
        "validation_flags": "",
    }


def validate_mapping(mapping: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    station_meta = stations.set_index("station_id").to_dict("index")
    rows = []
    for _, row in mapping.iterrows():
        updated = row.to_dict()
        station_id = str(updated.get("station_id", "")).strip()
        meta = station_meta.get(station_id, {})
        flags = []
        if not station_id:
            updated["mapping_status"] = "review_required"
            updated["confidence"] = "review_required"
            updated["mapping_method"] = "unresolved_no_verified_station"
            flags.extend(["station_unresolved", "distance_unavailable"])
            updated["notes"] = append_note(updated.get("notes", ""), "공식 내부 관측소 근거와 지자체 대표좌표가 부족하여 관측소 후보를 지정하지 않았습니다.")
            updated["evidence"] = "KMA 관측지점정보에서 검증 가능한 ASOS 후보를 특정하지 못함"
            updated["validation_flags"] = ";".join(flags)
            rows.append(updated)
            continue
        period_ok = station_period_covers(meta)
        if not period_ok:
            updated["mapping_status"] = "unusable"
            updated["confidence"] = "unusable"
            updated["mapping_method"] = "unusable_period"
            flags.append("period_not_covered")
            updated["notes"] = append_note(updated.get("notes", ""), "공식 관측지점정보 기준 2021-01-01~2024-01-31 전체 기간 사용 가능 여부를 충족하지 못합니다.")
        else:
            method, confidence, status, extra_flags, note = classify_mapping(updated, meta)
            updated["mapping_method"] = method
            updated["confidence"] = confidence
            updated["mapping_status"] = status
            flags.extend(extra_flags)
            updated["notes"] = append_note(updated.get("notes", ""), note)
        updated["evidence"] = build_validation_evidence(updated, meta, period_ok)
        updated["validation_flags"] = ";".join(flags)
        rows.append(updated)
    out = pd.DataFrame(rows)
    for col in ASOS_MAPPING_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out


def station_period_covers(meta: dict[str, Any]) -> bool:
    if not meta:
        return False
    start = pd.to_datetime(meta.get("station_start_date"), errors="coerce")
    end = pd.to_datetime(meta.get("station_end_date"), errors="coerce")
    if pd.isna(start):
        return False
    if pd.isna(end):
        end = pd.Timestamp("2099-12-31")
    return start <= VALIDATION_START and end >= VALIDATION_END


def classify_mapping(row: dict[str, Any], meta: dict[str, Any]) -> tuple[str, str, str, list[str], str]:
    flags: list[str] = []
    sido = str(row.get("standard_sido", ""))
    sigungu = str(row.get("standard_sigungu", ""))
    station_name = str(row.get("station_name", ""))
    address = str(meta.get("station_address", ""))
    distance = pd.to_numeric(pd.Series([row.get("distance_km")]), errors="coerce").iloc[0]
    direct_address = address_mentions(address, sido, sigungu)
    direct_name = station_name_matches(station_name, sigungu, simplify_authority_name(sigungu))
    if direct_address or direct_name:
        if pd.isna(distance):
            flags.append("distance_unavailable")
        return (
            "inside_municipality",
            "high",
            "validated",
            flags,
            "공식 관측지점정보의 지점명 또는 주소가 지자체와 일치하여 내부 관측소 후보로 검증했습니다.",
        )
    metro_station = METRO_REPRESENTATIVE.get(sido)
    if metro_station and station_name == metro_station:
        if pd.isna(distance):
            flags.append("distance_unavailable")
        elif float(distance) >= 30:
            flags.append("distance_ge_30km")
        return (
            "regional_representative",
            "medium",
            "validated",
            flags,
            f"{sido}의 대표 ASOS({station_name})를 같은 광역시 내부 기초지자체의 지역 대표 관측소로 사용합니다. 구 단위 내부 ASOS 부재는 검토 기록으로 남깁니다.",
        )
    if pd.notna(distance) and float(distance) >= 30:
        flags.append("distance_ge_30km")
    if pd.isna(distance):
        flags.append("distance_unavailable")
    if pd.notna(distance) and float(distance) <= 20 and sido_in_station_address(sido, address):
        return (
            "nearest_adjacent",
            "medium",
            "validated",
            flags,
            "공식 관측소 정보가 유효하고, 같은 광역 행정권 내 최단거리 ASOS입니다. 내부 관측소가 아니므로 지역 대표성 검토 메모를 유지합니다.",
        )
    return (
        "nearest_adjacent",
        "review_required",
        "review_required",
        flags,
        "내부 ASOS가 아니며 거리, 행정권, 지역 대표성 중 일부 근거가 충분하지 않아 사람이 검토해야 합니다.",
    )


def sido_in_station_address(sido: str, address: str) -> bool:
    compact_sido = re.sub(r"\s+", "", sido)
    compact_address = re.sub(r"\s+", "", str(address))
    if compact_sido in compact_address:
        return True
    aliases = {
        "강원도": "강원특별자치도",
        "전라북도": "전북특별자치도",
        "전라남도": "전라남도",
        "경상북도": "경상북도",
        "경상남도": "경상남도",
        "충청남도": "충청남도",
        "충청북도": "충청북도",
    }
    alias = aliases.get(sido)
    return bool(alias and re.sub(r"\s+", "", alias) in compact_address)


def append_note(existing: Any, note: str) -> str:
    existing_text = "" if pd.isna(existing) else str(existing).strip()
    if not existing_text:
        return note
    if note in existing_text:
        return existing_text
    return f"{existing_text} {note}"


def build_validation_evidence(row: dict[str, Any], meta: dict[str, Any], period_ok: bool) -> str:
    parts = [
        f"KMA 관측지점정보 station_id={row.get('station_id', '')}",
        f"station_name={row.get('station_name', '')}",
        f"lat={row.get('station_latitude', row.get('latitude', ''))}",
        f"lon={row.get('station_longitude', row.get('longitude', ''))}",
        f"start={meta.get('station_start_date', row.get('station_start_date', ''))}",
        f"end={meta.get('station_end_date', row.get('station_end_date', '')) or '운영중/종료일 미기재'}",
        f"period_2021_2024={'ok' if period_ok else 'not_ok'}",
        f"address={meta.get('station_address', '')}",
    ]
    return "; ".join(str(p) for p in parts)


def add_distance(stations: pd.DataFrame, municipality_coord: dict[str, Any]) -> pd.DataFrame:
    out = stations.copy()
    lat = municipality_coord.get("latitude")
    lon = municipality_coord.get("longitude")
    if lat is None or lon is None:
        out["distance_km"] = pd.NA
        out["distance_km_sort"] = float("inf")
        return out
    out["distance_km"] = out.apply(lambda r: round(haversine_km(float(lat), float(lon), r["latitude"], r["longitude"]), 2), axis=1)
    out["distance_km_sort"] = out["distance_km"].fillna(float("inf"))
    return out


def load_or_fetch_municipality_geocodes(authorities: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if GEOCODE_CACHE_PATH.exists():
        cache = json.loads(GEOCODE_CACHE_PATH.read_text(encoding="utf-8"))
    else:
        cache = {}
    changed = False
    for key in authorities["standard_sigungu_key"].drop_duplicates().sort_values():
        if key in cache:
            continue
        result = geocode_municipality(key)
        cache[key] = result
        changed = True
        ensure_parent(GEOCODE_CACHE_PATH)
        GEOCODE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(1.05)
    if changed:
        ensure_parent(GEOCODE_CACHE_PATH)
        GEOCODE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache


def geocode_municipality(query: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"q": f"{query}, South Korea", "format": "json", "limit": "1", "accept-language": "ko"})
    request = urllib.request.Request(
        f"https://nominatim.openstreetmap.org/search?{params}",
        headers={"User-Agent": "FoodZeroAI/0.1 contact: local-research"},
    )
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            items = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"latitude": None, "longitude": None, "source": "OpenStreetMap Nominatim", "error": str(exc)}
    if not items:
        return {"latitude": None, "longitude": None, "source": "OpenStreetMap Nominatim", "error": "no_result"}
    item = items[0]
    return {
        "latitude": float(item["lat"]),
        "longitude": float(item["lon"]),
        "source": "OpenStreetMap Nominatim",
        "display_name": item.get("display_name", ""),
        "licence": item.get("licence", ""),
    }


def simplify_authority_name(sigungu: str) -> str:
    tokens = str(sigungu).split()
    if not tokens:
        return ""
    return tokens[-1]


def station_name_matches(station_name: str, sigungu: str, short_name: str) -> bool:
    normalized_station = re.sub(r"\s+", "", station_name)
    normalized_sigungu = re.sub(r"\s+", "", sigungu)
    normalized_short = re.sub(r"\s+", "", short_name)
    if not normalized_station:
        return False
    return normalized_station == normalized_short or normalized_station == normalized_sigungu


def address_mentions(address: str, sido: str, sigungu: str) -> bool:
    if not address:
        return False
    compact_address = re.sub(r"\s+", "", address)
    return re.sub(r"\s+", "", sido) in compact_address and re.sub(r"\s+", "", sigungu) in compact_address


def format_candidates(candidates: pd.DataFrame) -> str:
    values = []
    for _, row in candidates.head(5).iterrows():
        values.append(f"{row['station_id']}:{row['station_name']}({row['latitude']},{row['longitude']})")
    return "; ".join(values)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def render_validation_review_doc(mapping: pd.DataFrame, stations: pd.DataFrame) -> None:
    ensure_parent(REVIEW_DOC_PATH)
    validated = mapping[mapping["mapping_status"].eq("validated")]
    review = mapping[mapping["mapping_status"].eq("review_required")]
    unusable = mapping[mapping["mapping_status"].eq("unusable")]
    distance = pd.to_numeric(mapping["distance_km"], errors="coerce")
    ge30 = mapping[distance.ge(30)]
    unresolved_distance = int(distance.isna().sum())
    shared = (
        mapping[mapping["station_id"].fillna("").ne("")]
        .groupby(["station_id", "station_name"], dropna=False)["standard_sigungu_key"]
        .apply(lambda s: sorted(s.dropna().astype(str).tolist()))
        .reset_index(name="municipalities")
    )
    shared["count"] = shared["municipalities"].map(len)
    shared = shared[shared["count"].gt(1)].sort_values("count", ascending=False)
    lines = [
        "# ASOS 관측소 매핑 검토",
        "",
        "이 문서는 공식 기상자료개방포털 관측지점정보를 기준으로 자동 검증한 결과입니다. `validated`는 모델링 입력 후보로 사용할 근거가 충분하다는 뜻이며, 사람이 최종 승인한 `confirmed`는 아닙니다.",
        "",
        "## 출처",
        "",
        f"- 기상자료개방포털 관측지점정보: {KMA_SOURCE_URL}",
        "- 공공데이터포털 지상기상관측 지점정보 조회서비스: https://www.data.go.kr/data/15139439/openapi.do",
        "",
        "## 요약",
        "",
        f"- 총 지자체 수: {len(mapping):,}",
        f"- validated: {len(validated):,}",
        f"- review_required: {len(review):,}",
        f"- unusable: {len(unusable):,}",
        f"- 사용 가능한 ASOS 관측소 메타데이터 수: {len(stations):,}",
        f"- 거리 미확인: {unresolved_distance:,}",
        "- 지자체 대표좌표 출처: OpenStreetMap Nominatim. 공식 지자체 경계/대표좌표가 아니므로 거리 기반 매핑은 검토 보조 자료입니다.",
        "",
        "## 30km 이상 거리 플래그",
        "",
    ]
    if ge30.empty:
        lines.append("- 없음")
    else:
        for _, row in ge30.iterrows():
            lines.append(f"- `{row['standard_sigungu_key']}` -> `{row['station_id']} {row['station_name']}`: {row['distance_km']}km, {row['mapping_status']}")
    lines.extend(["", "## ASOS 공유 현황", ""])
    if shared.empty:
        lines.append("- 없음")
    else:
        for _, row in shared.head(30).iterrows():
            joined = ", ".join(row["municipalities"][:15])
            suffix = " ..." if row["count"] > 15 else ""
            lines.append(f"- `{row['station_id']} {row['station_name']}`: {row['count']}개 지자체 사용. {joined}{suffix}")
    lines.extend(["", "## 우선 검토 대상", ""])
    if review.empty:
        lines.append("- 없음")
    else:
        for _, row in review.iterrows():
            station = f"{row['station_id']} {row['station_name']}".strip() or "미지정"
            lines.append(f"- `{row['standard_sigungu_key']}` -> `{station}`: {row['notes']} flags={row.get('validation_flags', '')}")
    lines.extend(["", "## validated 후보", ""])
    for _, row in validated.iterrows():
        lines.append(
            f"- `{row['standard_sigungu_key']}` -> `{row['station_id']} {row['station_name']}` "
            f"({row['confidence']}, {row['mapping_method']}, distance={row.get('distance_km', '')})"
        )
    if not unusable.empty:
        lines.extend(["", "## unusable", ""])
        for _, row in unusable.iterrows():
            lines.append(f"- `{row['standard_sigungu_key']}` -> `{row['station_id']} {row['station_name']}`: {row['notes']}")
    REVIEW_DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_review_doc(mapping: pd.DataFrame, stations: pd.DataFrame) -> None:
    ensure_parent(REVIEW_DOC_PATH)
    proposed = mapping[mapping["mapping_status"].eq("proposed")]
    review = mapping[mapping["mapping_status"].eq("review_required")]
    lines = [
        "# ASOS 관측소 매핑 검토",
        "",
        "이 문서는 공식 기상자료개방포털 관측지점정보를 기준으로 자동 생성한 후보입니다. 자동 결과는 `confirmed`가 아니며, 사람이 근거를 확인한 뒤 확정해야 합니다.",
        "",
        "## 출처",
        "",
        f"- 기상자료개방포털 관측지점정보: {KMA_SOURCE_URL}",
        "- 공공데이터포털 지상기상관측 지점정보 조회서비스: https://www.data.go.kr/data/15139439/openapi.do",
        "",
        "## 요약",
        "",
        f"- 총 지자체 수: {len(mapping):,}",
        f"- proposed: {len(proposed):,}",
        f"- review_required: {len(review):,}",
        f"- 사용 가능한 ASOS 관측소 메타데이터 수: {len(stations):,}",
        "- 지자체 대표좌표 출처: OpenStreetMap Nominatim. 공식 지자체 경계/대표좌표가 아니므로 거리 기반 매핑은 검토용입니다.",
        "",
        "## 우선 검토 대상",
        "",
    ]
    if review.empty:
        lines.append("- 없음")
    else:
        for _, row in review.iterrows():
            lines.append(f"- `{row['standard_sigungu_key']}`: {row['notes']}")
    lines.extend(["", "## proposed 후보", ""])
    for _, row in proposed.iterrows():
        lines.append(
            f"- `{row['standard_sigungu_key']}` -> `{row['proposed_station_id']} {row['proposed_station_name']}` "
            f"({row['confidence']}, {row['match_type']})"
        )
    REVIEW_DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    make_mapping()
