"""Download the UCI Concrete Compressive Strength dataset and save a clean CSV.

Usage:  python data/download.py
Output: data/raw/Concrete_Data.xls, data/concrete.csv
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip"
XLS_SHA256 = "710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4"  # verified 2026-10-23
HERE = Path(__file__).resolve().parent

COLUMNS = ["cement", "slag", "fly_ash", "water", "superplasticizer",
           "coarse_agg", "fine_agg", "age_days", "strength_mpa"]


def main():
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    xls = raw / "Concrete_Data.xls"
    if not xls.exists():
        print(f"downloading {URL}")
        with urllib.request.urlopen(URL, timeout=60) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        xls.write_bytes(zf.read("Concrete_Data.xls"))
    digest = hashlib.sha256(xls.read_bytes()).hexdigest()
    print(f"sha256(Concrete_Data.xls) = {digest}")
    if XLS_SHA256 and digest != XLS_SHA256:
        raise SystemExit("checksum mismatch: the source file has changed")

    df = pd.read_excel(xls)
    assert df.shape == (1030, 9), df.shape
    df.columns = COLUMNS
    df.to_csv(HERE / "concrete.csv", index=False)
    print(f"saved data/concrete.csv  ({len(df)} rows, {df.duplicated().sum()} exact duplicates kept in the raw CSV)")


if __name__ == "__main__":
    main()
