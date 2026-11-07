"""All README figures -> docs/figures/*.png
  python scripts/figures.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from faireo.data import DATA, RESULTS, ROOT
from faireo.metrics import group_summary

OUT = ROOT / "docs" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INCOME = ["Low", "Lower-middle", "Upper-middle", "High"]
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
NAMES = {"lgbm": "Gradient-boosted trees", "rf": "Random forest", "presto_lgbm": "Presto + trees",
         "presto_lr": "Presto + linear", "prithvi_lgbm": "Prithvi + trees", "prithvi_lr": "Prithvi + linear"}


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", name)


def headline():
    models = [m for m in NAMES if (RESULTS / f"countries_{m}_country_s0.csv").exists()]
    nc = 3; nr = int(np.ceil(len(models) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(10.5, 3.7 * nr), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(models):]:
        ax.set_visible(False)
    for ax, m in zip(axes, models):
        for split, col, off, lab in [("random", ORANGE, 0.12, "country seen in training"),
                                     ("country", BLUE, -0.12, "country held out")]:
            f = RESULTS / f"countries_{m}_{split}_s0.csv"
            if not f.exists():
                continue
            gs = group_summary(pd.read_csv(f), "auc")
            x = np.arange(4) + off
            ax.errorbar(x, gs["mean"], yerr=[gs["mean"] - gs["lo"], gs["hi"] - gs["mean"]], fmt="o",
                        color=col, ms=7, lw=2, capsize=0, label=lab)
        ax.set_xticks(range(4)); ax.set_xticklabels(["Low", "Lower-\nmiddle", "Upper-\nmiddle", "High"])
        ax.set_title(NAMES[m], color=INK, fontsize=10, loc="left"); ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_xlabel("World Bank income group")
    for ax in axes[::nc]:
        ax.set_ylabel("AUC per country (mean, 95% CI)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Crop-vs-non-crop accuracy is lower in low-income countries, most of all when they are unseen",
                 x=0.01, ha="left", color=INK, fontsize=11)
    save(fig, "01_gap_by_income.png")


def world_map(run="lgbm_country_s0"):
    ne = gpd.read_file(DATA / "ne_50m_admin_0_countries.geojson")
    ct = pd.read_csv(RESULTS / f"countries_{run}.csv")
    w = ne.merge(ct[["country", "auc"]], left_on="ADMIN", right_on="country", how="left")
    cmap = LinearSegmentedColormap.from_list("seq", SEQ[::-1])   # dark = low AUC = worse
    fig, ax = plt.subplots(figsize=(11, 5.2))
    w[w.ADMIN != "Antarctica"].plot(ax=ax, color="#eeede9", edgecolor="white", lw=0.3)
    w[w.auc.notna()].plot(ax=ax, column="auc", cmap=cmap, vmin=0.5, vmax=0.95, edgecolor="white", lw=0.3,
                          legend=True, legend_kwds={"shrink": 0.55, "label": "AUC when the country is held out"})
    ax.set_axis_off()
    ax.set_title("Where the model fails: darker = worse on a country it never saw (gradient-boosted trees, 66 countries)",
                 loc="left", color=INK, fontsize=11)
    save(fig, "02_world_map_auc.png")


def undercount(run="lgbm_country_s0"):
    ct = pd.read_csv(RESULTS / f"countries_{run}.csv")
    ct["income"] = pd.Categorical(ct.income, INCOME, ordered=True)
    g = ct.groupby("income", observed=True)[["prevalence", "pred_prevalence", "recall"]].mean()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(4)
    ax.bar(x - 0.18, 100 * g.prevalence, 0.34, color=SEQ[2], label="true cropland share")
    ax.bar(x + 0.18, 100 * g.pred_prevalence, 0.34, color=SEQ[5], label="model's estimate")
    for i, (t, p_) in enumerate(zip(g.prevalence, g.pred_prevalence)):
        ax.text(i + 0.18, 100 * p_ + 1.5, f"{100*(p_-t):+.0f} pts", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(INCOME); ax.set_ylabel("% of labelled points that are crop")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_title("Unseen low-income countries: cropland is under-counted; rich ones over-counted",
                 loc="left", color=INK, fontsize=10.5)
    save(fig, "03_cropland_undercount.png")


def trust(run="lgbm_country_s0"):
    t = pd.read_csv(RESULTS / f"trust_{run}.csv").dropna(subset=["measured_auc"])
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    for inc, col in zip(INCOME, [SEQ[6], SEQ[4], SEQ[3], SEQ[1]]):
        d = t[t.income == inc]
        ax.scatter(d.expected_auc_loo, d.measured_auc, s=40, color=col, edgecolor=SURF, lw=1.5, label=inc)
    lim = [0.45, 1.0]; ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    for _, r in t.nsmallest(5, "measured_auc").iterrows():
        ax.annotate(r.country, (r.expected_auc_loo, r.measured_auc), fontsize=7.5, color=INK2,
                    xytext=(4, -3), textcoords="offset points")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(color=GRID, lw=0.8)
    ax.set_xlabel("expected AUC, predicted with no labels from the country"); ax.set_ylabel("measured AUC")
    ax.legend(frameon=False, fontsize=8, title="income", title_fontsize=8)
    ax.set_title("The trust score predicts where the model will fail", loc="left", color=INK, fontsize=10.5)
    save(fig, "04_trust_validation.png")


def budget():
    f = RESULTS / "label_budget.csv"
    if not f.exists():
        return
    b = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)
    pol = {"status_quo": ("more labels where there already are many", "#a4a29c"),
           "uniform": ("same number everywhere", ORANGE), "novelty": ("guided by novelty (label-free)", BLUE),
           "oracle_worst_first": ("oracle (upper bound)", AQUA)}
    for ax, m, t in [(axes[0], "low_income_auc", "Low-income countries, mean AUC"),
                     (axes[1], "worst10_auc", "Worst 10% of countries, mean AUC")]:
        for k, (lab, col) in pol.items():
            d = b[b.policy == k]
            ax.plot(d.budget, d[m], "-o", color=col, lw=2, ms=5, label=lab)
        ax.set_title(t, loc="left", color=INK, fontsize=10); ax.grid(color=GRID, lw=0.8)
        ax.set_xlabel("new labels collected (total, all countries)")
    axes[0].legend(frameon=False, fontsize=7.5)
    save(fig, "05_label_budget.png")


if __name__ == "__main__":
    headline(); world_map(); undercount(); trust(); budget()
