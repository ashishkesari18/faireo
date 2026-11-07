"""Read ESA WorldCover (10 m global land cover) at every CropHarvest point.

ESA WorldCover is a widely used free global map (2020 v100 and 2021 v200).
Class 40 is cropland. Only the few kilobytes around each point are read from
the public cloud-optimised GeoTIFFs, so nothing large is downloaded.

Writes data/worldcover.parquet with columns wc2020, wc2021 (class codes).
"""
import sys, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
import numpy as np, pandas as pd, rasterio
from tqdm import tqdm
from faireo.data import DATA

URL = {2020: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v100/2020/map/ESA_WorldCover_10m_2020_v100_{t}_Map.tif",
       2021: "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{t}_Map.tif"}


def tile(lat, lon):
    la, lo = 3 * math.floor(lat / 3), 3 * math.floor(lon / 3)
    return f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"


def read(args):
    year, t, idx, lats, lons = args
    try:
        with rasterio.open(URL[year].format(t=t)) as src:
            vals = [v[0] for v in src.sample(zip(lons, lats))]
        return idx, np.array(vals, dtype=np.int16)
    except Exception:
        return idx, np.full(len(idx), -1, dtype=np.int16)   # no tile (ocean) or failed read


def main():
    pts = pd.read_parquet(DATA / "points.parquet")
    pts["tile"] = [tile(a, b) for a, b in zip(pts.lat, pts.lon)]
    out = pd.DataFrame(index=pts.index)
    for year in (2020, 2021):
        col = np.full(len(pts), -1, dtype=np.int16)
        jobs = [(year, t, g.index.to_numpy(), g.lat.to_numpy(), g.lon.to_numpy())
                for t, g in pts.groupby("tile")]
        with ThreadPoolExecutor(16) as ex:
            for idx, v in tqdm(ex.map(read, jobs), total=len(jobs), desc=f"WorldCover {year}"):
                col[idx] = v
        out[f"wc{year}"] = col
    out.to_parquet(DATA / "worldcover.parquet")
    print("missing:", (out == -1).mean().round(4).to_dict())


if __name__ == "__main__":
    main()
