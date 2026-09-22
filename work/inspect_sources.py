from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.foodzero.io_utils import read_table


for root in ["data/raw/food_waste", "data/raw/population"]:
    print(f"\n## {root}")
    for path in sorted(Path(root).glob("*")):
        if path.suffix.lower() not in {".csv", ".xlsx", ".xls"}:
            continue
        print(f"\nFILE {path.name}")
        if path.suffix.lower() in {".xlsx", ".xls"}:
            xls = pd.ExcelFile(path)
            print("sheets", xls.sheet_names)
            sheet_names = xls.sheet_names[:5]
        else:
            sheet_names = [0]
        for sheet in sheet_names:
            df, meta = read_table(path, nrows=5, sheet_name=sheet)
            print("sheet", sheet, meta, "shape", df.shape)
            print("columns", list(df.columns))
            print(df.head(3).to_string())
