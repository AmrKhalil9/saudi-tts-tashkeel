"""Figures for the paper, generated from results/clips.csv (same data and capping as make_tables.py).
fig_delta.pdf : mean paired change in CER vs Raw, per model and diacritization condition, 95% bootstrap CI (Cohere, WB)
fig_entity.pdf: exact identifier recovery, digits vs spelled out, Raw vs Full (Cohere, WB)
Usage: python make_figures.py --root . --out paper_tables
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = [("habibi_sau", "Habibi-SAU"), ("habibi_unified", "Habibi-Unified"), ("omnivoice", "OmniVoice"),
          ("lahgtna", "Lahgtna"), ("chatterbox", "Chatterbox-ML"), ("xtts", "XTTS-v2")]
CONDS = [("catt_full", "Full"), ("catt_nocase", "No-case"), ("shadda_only", "Shadda"), ("ft_full", "FT-full")]
COLORS = ["#b2182b", "#ef8a62", "#999999", "#2166ac"]  # colour-blind-safe diverging set
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "pdf.fonttype": 42})


def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def fig_delta(df, out):
    g = df[(df.asr == "cohere") & (df.channel == "wb")]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.2), gridspec_kw={"width_ratios": [1, 2.2], "height_ratios": [1]})
    for ax, keep in zip(axes, [["habibi_sau", "habibi_unified"], ["omnivoice", "lahgtna", "chatterbox", "xtts"]]):
        names = [n for m, n in MODELS if m in keep]
        ys = np.arange(len(keep))
        for k, ((c, cname), col) in enumerate(zip(CONDS, COLORS)):
            for y, m in zip(ys, keep):
                b = g[(g.model == m) & (g.cond == "raw")].set_index(["id", "voice"]).cer
                o = g[(g.model == m) & (g.cond == c)].set_index(["id", "voice"]).cer
                j = pd.concat([b, o], axis=1, keys=["b", "c"]).dropna()
                d = (j.c - j.b).values * 100
                lo, hi = boot(d)
                yy = y + (k - 1.5) * 0.17
                ax.errorbar(d.mean(), yy, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", ms=3, color=col,
                            elinewidth=0.9, capsize=1.5, label=cname if (y == 0 and ax is axes[1]) else None)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(ys); ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.grid(axis="x", lw=0.3, alpha=0.5)
        ax.set_xlabel("$\\Delta$CER vs Raw (pp)")
    axes[0].set_title("Trained without diacritics", fontsize=8)
    axes[1].set_title("Other zero-shot models", fontsize=8)
    axes[1].legend(ncol=4, fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.32))
    fig.tight_layout()
    fig.savefig(out / "fig_delta.pdf", bbox_inches="tight")


def fig_entity(df, out):
    e = df[df.ent_ok.notna() & (df.asr == "cohere") & (df.channel == "wb")
           & df.entity_types.fillna("").str.contains("id|phone|order")].copy()
    e["ok"] = e.ent_ok.astype(str).str.lower().isin(["true", "1", "1.0"])
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    x = np.arange(len(MODELS)); w = 0.2
    bars = [("digits", "raw", "Digits, Raw", "#d1e5f0"), ("digits", "catt_full", "Digits, Full", "#67a9cf"),
            ("spelled", "raw", "Spelled, Raw", "#fddbc7"), ("spelled", "catt_full", "Spelled, Full", "#ef8a62")]
    for k, (nf, c, lab, col) in enumerate(bars):
        v = [e[(e.model == m) & (e.numform == nf) & (e.cond == c)].ok.mean() * 100 for m, _ in MODELS]
        ax.bar(x + (k - 1.5) * w, v, w, label=lab, color=col, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in MODELS], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Exact identifier recovery (%)"); ax.set_ylim(0, 100)
    fig.set_size_inches(3.1, 2.3)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    ax.legend(ncol=2, fontsize=6.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out / "fig_entity.pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(a.root) / "results/clips.csv")
    df["cer"] = df["cer"].clip(upper=1.0)
    fig_delta(df, out); fig_entity(df, out)
    print("figures written")


if __name__ == "__main__":
    main()
