"""Build the data file for the interactive trust map (docs/trustmap/data.js)."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, geopandas as gpd
from faireo.data import DATA, RESULTS, ROOT

RUNS = {"trees": "lgbm_country_s0", "presto": "presto_lgbm_country_s0"}
ne = gpd.read_file(DATA / "ne_110m.geojson")[["ADMIN", "ISO_A3_EH", "INCOME_GRP", "geometry"]]
ne = ne[ne.ADMIN != "Antarctica"]
inc = {"1. High income: OECD": "High", "2. High income: nonOECD": "High", "3. Upper middle income": "Upper-middle",
       "4. Lower middle income": "Lower-middle", "5. Low income": "Low"}
out = {}
for key, run in RUNS.items():
    ct = pd.read_csv(RESULTS / f"countries_{run}.csv").set_index("country")
    tr = pd.read_csv(RESULTS / f"trust_{run}.csv").set_index("country")
    d = {}
    for c in tr.index:
        # Show leave-one-country-out scores for measured countries.
        # Full-fit estimates are reserved for countries excluded from fitting.
        prediction = (tr.loc[c, "expected_auc_loo"] if c in ct.index
                      else tr.loc[c, "expected_auc"])
        if pd.isna(prediction):
            raise ValueError(f"Missing trust prediction for {c} in {run}")
        r = {"exp": round(float(prediction), 3), "n": int(tr.loc[c, "n"])}
        if c in ct.index:
            x = ct.loc[c]
            r.update(auc=round(x.auc, 3), lo=round(x.auc_lo, 3), hi=round(x.auc_hi, 3),
                     rec=round(x.recall, 3), share=round(100 * x.prevalence, 1),
                     pshare=round(100 * x.pred_prevalence, 1))
        d[c] = r
    out[key] = d
feats = []
for _, r in ne.iterrows():
    g = r.geometry.simplify(0.15)
    gj = json.loads(gpd.GeoSeries([g]).to_json())["features"][0]["geometry"]
    def rnd(o):
        return [rnd(v) for v in o] if isinstance(o, list) and not isinstance(o[0], (int, float)) else [round(o[0], 2), round(o[1], 2)]
    gj["coordinates"] = rnd(gj["coordinates"])
    feats.append({"type": "Feature", "properties": {"name": r.ADMIN, "income": inc.get(r.INCOME_GRP, "")}, "geometry": gj})
js = "window.FAIREO=" + json.dumps({"world": {"type": "FeatureCollection", "features": feats}, "runs": out},
                                   separators=(",", ":")) + ";"
(ROOT / "docs" / "trustmap" / "data.js").write_text(js)
print(f"data.js {len(js)/1024:.0f} KB")
