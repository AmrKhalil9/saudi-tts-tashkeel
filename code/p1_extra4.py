"""Fourth set for Paper 1: full paired-test tables over all 749 sentences (appendix) and CER vs sentence length.
Usage: python p1_extra4.py --root . --out paper_tables"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
R, OUT = Path(a.root), Path(a.out)
MODELS = [("habibi_sau", "Habibi-SAU"), ("habibi_unified", "Habibi-Unified"), ("omnivoice", "OmniVoice"), ("lahgtna", "Lahgtna"),
          ("chatterbox", "Chatterbox-ML"), ("xtts", "XTTS-v2")]
CONDS = [("catt_full", "Full"), ("catt_nocase", "No-case"), ("shadda_only", "Shadda"), ("ft_full", "FT-full")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
df = pd.read_csv(R / "results/clips.csv"); df["cer"] = df["cer"].clip(upper=1.0)
rng = np.random.default_rng(0)


def holm(ps):
    o = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - k) * ps[i])); adj[i] = run
    return adj


def tab(asr, ch, label, title):
    q = df[(df.asr == asr) & (df.channel == ch)]; rows = []
    for m, n in MODELS:
        s = q[q.model == m]; b = s[s.cond == "raw"].set_index(["id", "voice"]).cer; res = []
        for c, cn in CONDS:
            o = s[s.cond == c].set_index(["id", "voice"]).cer; j = pd.concat([b, o], axis=1, keys=["b", "c"]).dropna()
            d = (j.c - j.b).values; B = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
            p = wilcoxon(j.c, j.b).pvalue if np.any(d != 0) else 1.0
            res.append([cn, d.mean(), np.percentile(B, 2.5), np.percentile(B, 97.5), p, (d > 0).mean(), (d < 0).mean()])
        adj = holm(np.array([r[4] for r in res]))
        for k, (cn, dm, lo, hi, _, wo, be) in enumerate(res):
            ps = r"$<$0.001" if adj[k] < 0.001 else f"{adj[k]:.3f}"
            rows.append(f"{n if k == 0 else ''} & {cn} & {100 * dm:+.1f} & [{100 * lo:+.1f}, {100 * hi:+.1f}] & {ps} & {100 * wo:.0f} & {100 * be:.0f} \\\\")
    head = (r"\begin{table*}[t]\centering\small\caption{" + title + r" Mean paired change in CER relative to Raw (percentage points) "
            r"with 95\% bootstrap interval, Holm-corrected Wilcoxon $p$-value, and share of sentence--voice pairs made worse or better "
            r"(\%).}\label{tab:tests" + label + "}\n" + r"\begin{tabular}{llrcrrr}\toprule" + "\n"
            + r"TTS model & Condition & $\Delta$CER & 95\% CI & $p_{\mathrm{Holm}}$ & Worse & Better \\\midrule" + "\n")
    open(OUT / f"x_tab_tests_{label}.tex", "w").write(head + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}\end{table*}" + "\n")


tab("cohere", "wb", "cw", "Paired tests, Cohere ASR, wideband, all 749 sentences.")
tab("whisper", "wb", "ww", "Paired tests, Whisper large-v3, wideband, all 749 sentences.")

T = {json.loads(l)["id"]: len(json.loads(l)["text"].split()) for l in open(R / "testset/diac.jsonl" if (R / "testset/diac.jsonl").exists() else R / "testset/diac_catt.jsonl")}
main = df[(df.asr == "cohere") & (df.channel == "wb")].copy(); main["len"] = main.id.map(T)
labels = [r"$\leq$5", "6-7", "8-9", "10-11", "12+"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
for ax, c, t in ((axes[0], "raw", "Raw"), (axes[1], "catt_full", "Full")):
    for m, n in MODELS:
        s = main[(main.model == m) & (main.cond == c)]
        s = s.assign(b=pd.cut(s.len, [0, 5, 7, 9, 11, 40], labels=labels))
        g = s.groupby("b", observed=True).cer.mean() * 100; ax.plot(range(len(g)), g.values, marker="o", ms=3, lw=1, label=n)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels); ax.set_xlabel("Sentence length (words)")
    ax.set_title(t, fontsize=8); ax.set_ylabel("Mean CER (%)"); ax.grid(alpha=.3, lw=.4)
axes[1].legend(fontsize=6, frameon=False, ncol=2); fig.tight_layout(); fig.savefig(OUT / "x_fig_len.pdf")
print("ok")
