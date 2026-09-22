from __future__ import annotations

import pandas as pd

from .config import CONFIG_DIR
from .features import load_food_waste
from .io_utils import ensure_parent
from .weather import MAPPING_COLUMNS, read_station_mapping


def export_mapping_template() -> pd.DataFrame:
    waste = load_food_waste()
    authorities = (
        waste[["sido", "sigungu", "sigungu_key"]]
        .drop_duplicates()
        .sort_values(["sido", "sigungu", "sigungu_key"])
        .reset_index(drop=True)
    )
    existing = read_station_mapping()
    existing = existing[existing["sigungu_key"].astype(str).str.strip().ne("")]
    existing_by_key = existing.drop_duplicates("sigungu_key").set_index("sigungu_key") if not existing.empty else pd.DataFrame()
    rows = []
    for _, row in authorities.iterrows():
        if not existing_by_key.empty and row["sigungu_key"] in existing_by_key.index:
            merged = existing_by_key.loc[row["sigungu_key"]].to_dict()
            merged["sido"] = row["sido"]
            merged["sigungu"] = row["sigungu"]
            merged["sigungu_key"] = row["sigungu_key"]
        else:
            merged = row.to_dict()
            merged.update(
                {
                    "stn_id": "",
                    "stn_name": "",
                    "mapping_status": "pending",
                    "evidence": "",
                    "source_url": "https://data.kma.go.kr",
                    "notes": "ASOS 관측소 위치와 지자체 대표성 확인 필요",
                }
            )
        rows.append(merged)
    mapping = pd.DataFrame(rows)
    for col in MAPPING_COLUMNS:
        if col not in mapping.columns:
            mapping[col] = ""
    mapping["mapping_status"] = mapping["mapping_status"].replace({"needs_review": "pending"}).fillna("pending")
    output_path = CONFIG_DIR / "asos_station_mapping.csv"
    ensure_parent(output_path)
    mapping[MAPPING_COLUMNS].to_csv(output_path, index=False, encoding="utf-8-sig")
    return mapping[MAPPING_COLUMNS]


if __name__ == "__main__":
    export_mapping_template()
