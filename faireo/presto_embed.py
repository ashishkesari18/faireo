"""Frozen Presto embeddings for every CropHarvest point (CPU is fine: ~1 min).

Presto (Tseng et al. 2023, NASA Harvest) is a small pretrained transformer for
pixel time series of Sentinel-1/2, ERA5 and SRTM, the same inputs CropHarvest
stores, so no reformatting or new downloads are needed.
"""
import numpy as np
import torch

from .data import BANDS, ROOT
from .presto_model import Presto

# Presto's normalisation (presto/dataops/pipelines/s1_s2_era5_srtm.py)
_SHIFT = {"VV": 25.0, "VH": 25.0, "temperature_2m": -272.15}
_DIV = {"VV": 25.0, "VH": 25.0, "temperature_2m": 35.0, "total_precipitation": 0.03,
        "elevation": 2000.0, "slope": 50.0, "NDVI": 1.0}
PRESTO_BANDS = [b for b in BANDS if b != "B9"]          # 17 bands, B9 dropped
START_MONTH = 1   # CropHarvest series run February..January (0 = January)


def normalise(X):
    shift = np.array([_SHIFT.get(b, 0.0) for b in BANDS], dtype=np.float32)
    div = np.array([_DIV.get(b, 1e4) for b in BANDS], dtype=np.float32)
    x = (X + shift) / div
    keep = [BANDS.index(b) for b in PRESTO_BANDS]
    x = x[:, :, keep]
    b8, b4 = x[:, :, PRESTO_BANDS.index("B8")], x[:, :, PRESTO_BANDS.index("B4")]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(b8 + b4 > 0, (b8 - b4) / (b8 + b4), 0)
    x[:, :, PRESTO_BANDS.index("NDVI")] = ndvi
    return x.astype(np.float32)


def load_encoder():
    model = Presto.construct()
    model.load_state_dict(torch.load(ROOT / "weights" / "presto.pt", map_location="cpu"))
    return model.encoder.eval()


@torch.no_grad()
def embed(X, lat, lon, use_latlon=True, batch=2048):
    """Returns [N, 128] embeddings. use_latlon=False hides location from Presto."""
    enc = load_encoder()
    x = torch.from_numpy(normalise(X))
    dw = torch.full(x.shape[:2], 9, dtype=torch.long)     # Dynamic World: missing
    ll = torch.from_numpy(np.stack([lat, lon], 1).astype(np.float32))
    if not use_latlon:
        ll = torch.zeros_like(ll)
    out = []
    for i in range(0, len(x), batch):
        out.append(enc(x[i:i + batch], dynamic_world=dw[i:i + batch],
                       latlons=ll[i:i + batch], month=START_MONTH).numpy())
    return np.concatenate(out)
