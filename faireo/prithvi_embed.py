"""Frozen Prithvi-EO-2.0-tiny (NASA/IBM) embeddings for CropHarvest points.

Prithvi is an image model: it expects 6 HLS bands (blue, green, red, narrow
NIR, SWIR1, SWIR2) on 16x16-pixel patches for 4 time steps. CropHarvest
stores one pixel per point, so here each point becomes a 16x16 patch filled
with that pixel's values, and the 12 months are averaged into 4 seasons.
The model therefore sees the right bands and seasons but no spatial texture.
This is the 'pixel adaptation' of Prithvi; it is reported separately from
image-based results.
"""
import json
import numpy as np
import torch

from .data import BANDS, ROOT
from .prithvi_mae import PrithviMAE

HLS_FROM_S2 = ["B2", "B3", "B4", "B8A", "B11", "B12"]
SEASON_DOY = [74, 166, 258, 349]       # mid-March, June, September, December
WEIGHTS = ROOT / "weights" / "Prithvi_EO_V2_tiny_TL.pt"
URL = "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL/resolve/main/Prithvi_EO_V2_tiny_TL.pt"


def load_encoder():
    cfg = json.load(open(ROOT / "weights" / "prithvi_tiny_config.json"))["pretrained_cfg"]
    if not WEIGHTS.exists():
        torch.hub.download_url_to_file(URL, str(WEIGHTS))
    keep = ["img_size", "num_frames", "patch_size", "in_chans", "embed_dim", "depth", "num_heads",
            "decoder_embed_dim", "decoder_depth", "decoder_num_heads", "mlp_ratio",
            "coords_encoding", "coords_scale_learn"]
    model = PrithviMAE(**{k: cfg[k] for k in keep})
    sd = torch.load(WEIGHTS, map_location="cpu")
    sd = {k: v for k, v in sd.items() if "pos_embed" not in k}   # rebuilt for our grid
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [m for m in missing if "pos_embed" not in m], missing
    return model.encoder.eval(), cfg


def to_frames(X, cfg):
    idx = [BANDS.index(b) for b in HLS_FROM_S2]
    x = X[:, :, idx]                                   # [N, 12, 6]
    x = x.reshape(len(X), 4, 3, 6).mean(2)             # [N, 4 seasons, 6]
    x = (x - np.array(cfg["mean"])) / np.array(cfg["std"])
    x = np.transpose(x, (0, 2, 1)).astype(np.float32)  # [N, 6, 4]
    return x


@torch.no_grad()
def embed(X, lat, lon, year, batch=512):
    enc, cfg = load_encoder()
    frames = torch.from_numpy(to_frames(X, cfg))
    out = []
    for i in range(0, len(X), batch):
        f = frames[i:i + batch]
        img = f[:, :, :, None, None].expand(-1, -1, -1, 16, 16).contiguous()   # [B, 6, 4, 16, 16]
        yr = torch.as_tensor(year[i:i + batch], dtype=torch.float32)
        tc = torch.stack([yr[:, None].expand(-1, 4),
                          torch.tensor(SEASON_DOY, dtype=torch.float32).expand(len(f), -1)], -1)
        lc = torch.as_tensor(np.stack([lat[i:i + batch], lon[i:i + batch]], 1), dtype=torch.float32)
        h = enc.forward_features(img, temporal_coords=tc, location_coords=lc)[-1]   # [B, 1+4, 192]
        out.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy())
    return np.concatenate(out)
