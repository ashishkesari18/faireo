"""Build one clean table from the CropHarvest download.

Reads   data/raw/labels.geojson and data/raw/features/arrays/*.h5
        data/ne_50m_admin_0_countries.geojson (Natural Earth country borders)
Writes  data/cropharvest.npz   X [N, 12 months, 18 bands], y, row ids
        data/points.parquet    one row per point: lat, lon, source dataset,
                               country, continent, income group, year

Usage:  python scripts/prepare.py
"""
from pathlib import Path
import glob
import sys

import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BANDS = ["VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9",
         "B11", "B12", "temperature_2m", "total_precipitation", "elevation",
         "slope", "NDVI"]

INCOME_ORDER = {"1. High income: OECD": "High", "2. High income: nonOECD": "High",
                "3. Upper middle income": "Upper-middle",
                "4. Lower middle income": "Lower-middle", "5. Low income": "Low"}


def main():
    files = sorted(glob.glob(str(RAW / "features" / "arrays" / "*.h5")))
    if not files:
        sys.exit(f"No feature files in {RAW/'features/arrays'}. Run scripts/download.sh first.")

    X, rows = [], []
    for f in tqdm(files, desc="reading points"):
        with h5py.File(f, "r") as h:
            a = h["array"][:]
            if a.shape != (12, 18):
                continue
            idx = int(Path(f).stem.split("_", 1)[0])
            X.append(a.astype(np.float32))
            rows.append(dict(index=idx, dataset=h.attrs["dataset"],
                             lat=float(h.attrs["instance_lat"]),
                             lon=float(h.attrs["instance_lon"]),
                             y=int(h.attrs["is_crop"])))
    X = np.stack(X)
    pts = pd.DataFrame(rows)

    # attach collection year from labels.geojson
    lab = gpd.read_file(RAW / "labels.geojson", ignore_geometry=True)
    lab = lab[["index", "dataset", "export_end_date"]]
    lab["year"] = pd.to_datetime(lab["export_end_date"]).dt.year - 1
    pts = pts.merge(lab[["index", "dataset", "year"]], on=["index", "dataset"], how="left")

    # country, continent and World Bank income group from Natural Earth borders
    ne = gpd.read_file(ROOT / "data" / "ne_50m_admin_0_countries.geojson")
    ne = ne[["ADMIN", "ISO_A3_EH", "CONTINENT", "REGION_WB", "INCOME_GRP", "geometry"]]
    g = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs=ne.crs)
    j = gpd.sjoin(g, ne, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    # points just off the coast: snap to the nearest country
    miss = j["ADMIN"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(g[miss].to_crs(3857), ne.to_crs(3857), how="left")
        near = near[~near.index.duplicated()]
        for c in ["ADMIN", "ISO_A3_EH", "CONTINENT", "REGION_WB", "INCOME_GRP"]:
            j.loc[miss, c] = near[c].values
    pts["country"] = j["ADMIN"].values
    pts["iso3"] = j["ISO_A3_EH"].values
    pts["continent"] = j["CONTINENT"].values
    pts["wb_region"] = j["REGION_WB"].values
    pts["income"] = j["INCOME_GRP"].map(INCOME_ORDER).values

    out = ROOT / "data"
    np.savez_compressed(out / "cropharvest.npz", X=X, y=pts.y.values, bands=np.array(BANDS))
    pts.to_parquet(out / "points.parquet")
    print(f"saved {len(pts):,} points, {pts.country.nunique()} countries -> data/")


if __name__ == "__main__":
    main()
