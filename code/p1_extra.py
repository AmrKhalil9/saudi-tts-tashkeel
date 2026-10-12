"""Additional analyses for the journal-length version of Paper 1, from results/clips.csv and the test set.
Writes x_* tables/figures and x_numbers.tex (\\R{} definitions) into <out_dir>.
Usage: python p1_extra.py --root . --out paper_tables"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
R, OUT = Path(a.root), Path(a.out); OUT.mkdir(exist_ok=True, parents=True)
MODELS = [("habibi_sau", "Habibi-SAU"), ("habibi_unified", "Habibi-Unified"), ("omnivoice", "OmniVoice"), ("lahgtna", "Lahgtna"),
          ("chatterbox", "Chatterbox-ML"), ("xtts", "XTTS-v2")]
CONDS = [("raw", "Raw"), ("catt_full", "Full"), ("catt_nocase", "No-case"), ("shadda_only", "Shadda"), ("ft_full", "FT-full")]
MN, CN = dict(MODELS), dict(CONDS)
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v); return M[k]
df = pd.read_csv(R / "results/clips.csv"); df["cer"] = df["cer"].clip(upper=1.0)
main = df[(df.asr == "cohere") & (df.channel == "wb")]

# ---- A: CER distributions per model and condition (box plots)
fig, ax = plt.subplots(figsize=(7.0, 2.4))
pos, data, cols = [], [], []
colors = {"raw": "#4d4d4d", "catt_full": "#b2182b", "catt_nocase": "#ef8a62", "shadda_only": "#999999", "ft_full": "#2166ac"}
for i, (m, _) in enumerate(MODELS):
    for j, (c, _) in enumerate(CONDS):
        v = main[(main.model == m) & (main.cond == c)].cer.values * 100
        pos.append(i * 6 + j); data.append(v); cols.append(colors[c])
bp = ax.boxplot(data, positions=pos, widths=0.75, showfliers=False, patch_artist=True, medianprops={"color": "black", "lw": 0.8})
for b, c in zip(bp["boxes"], cols): b.set_facecolor(c); b.set_alpha(0.75); b.set_linewidth(0.5)
ax.set_xticks([i * 6 + 2 for i in range(len(MODELS))]); ax.set_xticklabels([n for _, n in MODELS])
ax.set_ylabel("CER (%)"); ax.grid(axis="y", alpha=.3, lw=.4)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=colors[c], alpha=.75) for c, _ in CONDS], [n for _, n in CONDS], ncol=5, fontsize=6, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "x_fig_box.pdf")

# ---- B: register x condition heatmap of paired delta CER (Cohere, WB)
def delta(g, m, c, reg=None):
    s = g[(g.model == m)]; s = s if reg is None else s[s.register == reg]
    b = s[s.cond == "raw"].set_index(["id", "voice"]).cer; o = s[s.cond == c].set_index(["id", "voice"]).cer
    j = pd.concat([b, o], axis=1, keys=["b", "c"]).dropna(); return (j.c - j.b).mean() * 100
regs = [("msa", "MSA"), ("dialect", "Dialect"), ("entity", "Entity")]
H = np.array([[delta(main, m, c, r) for r, _ in regs for c, _ in CONDS[1:]] for m, _ in MODELS])
fig, ax = plt.subplots(figsize=(7.0, 2.4))
lim = 10; im = ax.imshow(np.clip(H, -lim, lim), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([n for _, n in MODELS])
ax.set_xticks(range(H.shape[1])); ax.set_xticklabels([f"{rn}\n{cn}" for _, rn in regs for _, cn in CONDS[1:]], fontsize=6)
for i in range(H.shape[0]):
    for j in range(H.shape[1]): ax.text(j, i, f"{H[i, j]:+.1f}", ha="center", va="center", fontsize=5.5, color="white" if abs(H[i, j]) > 6 else "black")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="$\\Delta$CER vs Raw (pp, clipped at $\\pm$10)"); fig.tight_layout(); fig.savefig(OUT / "x_fig_regheat.pdf")

# ---- C: agreement between recognizers on the paired effects
pts = []
for m, _ in MODELS:
    for c, _ in CONDS[1:]:
        dc = delta(df[(df.asr == "cohere") & (df.channel == "wb")], m, c); dw = delta(df[(df.asr == "whisper") & (df.channel == "wb")], m, c)
        dg = delta(df[(df.asr == "cohere") & (df.channel == "g711")], m, c); pts.append((m, c, dc, dw, dg))
P = np.array([[p[2], p[3], p[4]] for p in pts])
put("x.asr.r", np.corrcoef(P[:, 0], P[:, 1])[0, 1], "{:.3f}"); put("x.chan.r", np.corrcoef(P[:, 0], P[:, 2])[0, 1], "{:.3f}")
small = [p for p in pts if abs(p[2]) < 15]
put("x.asr.r_small", np.corrcoef([p[2] for p in small], [p[3] for p in small])[0, 1], "{:.2f}")
put("x.chan.r_small", np.corrcoef([p[2] for p in small], [p[4] for p in small])[0, 1], "{:.2f}")
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
mk = dict(zip([c for c, _ in CONDS[1:]], "osD^"))
for ax, k, lab in ((axes[0], 3, "Whisper large-v3, wideband"), (axes[1], 4, "Cohere, G.711 telephone")):
    for p in pts: ax.scatter(p[2], p[k], marker=mk[p[1]], s=14, color=plt.cm.tab10(MODELS.index((p[0], MN[p[0]]))), edgecolor="none")
    lo, hi = P.min() - 2, P.max() + 2; ax.plot([lo, hi], [lo, hi], color="gray", lw=0.6, ls=":")
    ax.set_xlabel("$\\Delta$CER, Cohere wideband (pp)"); ax.set_ylabel(f"$\\Delta$CER, {lab} (pp)"); ax.grid(alpha=.3, lw=.4)
h = [plt.Line2D([], [], marker="o", ls="", color=plt.cm.tab10(i), ms=4) for i in range(len(MODELS))] + \
    [plt.Line2D([], [], marker=mk[c], ls="", color="gray", ms=4) for c, _ in CONDS[1:]]
fig.legend(h, [n for _, n in MODELS] + [n for _, n in CONDS[1:]], ncol=5, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "x_fig_agree.pdf")

# ---- E: harm vs number of case endings added (words where Full and No-case differ)
T = {json.loads(l)["id"]: json.loads(l) for l in open(R / "testset/diac_catt.jsonl")}
def n_case(r):
    fw, nw = r["cond_catt_full"].split(), r["cond_catt_nocase"].split() if "cond_catt_nocase" in r else []
    return sum(x != y for x, y in zip(fw, nw)) if nw else np.nan
ncase = {k: n_case(r) for k, r in T.items()}
rows, fig = [], None
fig, ax = plt.subplots(figsize=(3.4, 2.3))
for m, n in MODELS:
    s = main[main.model == m]; b = s[s.cond == "raw"].set_index(["id", "voice"]).cer; o = s[s.cond == "catt_full"].set_index(["id", "voice"]).cer
    j = pd.concat([b, o], axis=1, keys=["b", "c"]).dropna().reset_index(); j["d"] = (j.c - j.b) * 100; j["k"] = j.id.map(ncase)
    j = j.dropna(subset=["k"]); j["bin"] = pd.cut(j.k, [-1, 2, 4, 6, 100], labels=["0-2", "3-4", "5-6", "7+"])
    g = j.groupby("bin", observed=True).d.mean(); ax.plot(range(len(g)), g.values, marker="o", ms=3, lw=1, label=n)
    rho = j[["k", "d"]].corr(method="spearman").iloc[0, 1]; put(f"x.case.{m}.rho", rho, "{:.2f}")
ax.set_xticks(range(4)); ax.set_xticklabels(["0-2", "3-4", "5-6", "7+"]); ax.set_xlabel("Words with a case ending added by CATT")
ax.set_ylabel("$\\Delta$CER Full vs Raw (pp)"); ax.legend(fontsize=5.5, frameon=False, ncol=2); ax.grid(alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(OUT / "x_fig_case.pdf")
put("x.case.median_words", float(np.nanmedian(list(ncase.values()))), "{:.0f}")

# ---- F: duration ratio Full/Raw per model
fig, ax = plt.subplots(figsize=(3.4, 2.3)); ratios = []
for m, n in MODELS:
    s = main[main.model == m]; b = s[s.cond == "raw"].set_index(["id", "voice"]).dur; o = s[s.cond == "catt_full"].set_index(["id", "voice"]).dur
    j = pd.concat([b, o], axis=1, keys=["b", "c"]).dropna(); r = (j.c / j.b).values; ratios.append(r)
    put(f"x.dur.{m}.median", np.median(r), "{:.2f}")
ax.boxplot(ratios, showfliers=False, widths=0.6, medianprops={"color": "black", "lw": 0.8})
ax.axhline(1, color="gray", lw=0.6, ls=":"); ax.set_xticks(range(1, len(MODELS) + 1)); ax.set_xticklabels([n for _, n in MODELS], rotation=30, ha="right")
ax.set_ylabel("Duration ratio, Full / Raw"); ax.grid(axis="y", alpha=.3, lw=.4); fig.tight_layout(); fig.savefig(OUT / "x_fig_dur.pdf")

# ---- G: failure rate per model and condition
fig, ax = plt.subplots(figsize=(7.0, 2.2)); w = 0.16
for j, (c, cn) in enumerate(CONDS):
    v = [100 * (main[(main.model == m) & (main.cond == c)].cer >= 0.5).mean() for m, _ in MODELS]
    ax.bar(np.arange(len(MODELS)) + (j - 2) * w, v, w, label=cn, color=colors[c], alpha=.8, edgecolor="black", lw=0.3)
    for m, vv in zip(MODELS, v): put(f"x.fail.{m[0]}.{c}", vv)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([n for _, n in MODELS]); ax.set_ylabel("Clips with CER $\\geq$ 50% (%)")
ax.legend(ncol=5, fontsize=6, frameon=False); ax.grid(axis="y", alpha=.3, lw=.4); fig.tight_layout(); fig.savefig(OUT / "x_fig_fail.pdf")

# ---- I: entity accuracy by entity type (digits vs spelled, Raw), Cohere WB
e = main[main.ent_ok.notna() & (main.cond == "raw")].copy(); e["ok"] = e.ent_ok.astype(str).str.lower().isin(["true", "1", "1.0"])
e = e[e.entity_types.fillna("").str.contains("phone|id|order")]
e["etype"] = e.entity_types.fillna("").map(lambda t: "phone" if "phone" in t else ("ID" if t.split(",")[0] == "id" else "order"))
types = ["phone", "ID", "order"]
rows = []
for t in types:
    cells = []
    for m, _ in MODELS:
        for nf in ("digits", "spelled"):
            s = e[(e.etype == t) & (e.model == m) & (e.numform == nf)]; cells.append(f"{100 * s.ok.mean():.0f}" if len(s) else "--")
    n = len(e[(e.etype == t) & (e.model == MODELS[0][0]) & (e.numform == "digits")])
    rows.append(f"{t} & {n} & " + " & ".join(cells) + " \\\\")
head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{n}}}" for _, n in MODELS)
open(OUT / "x_tab_etype.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\\caption{Exact recovery (\\%) of phone, ID and order numbers by identifier type, written with "
    "digits (D) or spelled out (S), without diacritization (Raw; Cohere ASR, wideband). $n$: sentences per cell.}\\label{tab:etype}\n"
    "\\begin{tabular}{lr" + "rr" * len(MODELS) + "}\\toprule\n& & " + head + "\\\\\nType & $n$ & " + " & ".join(["D & S"] * len(MODELS))
    + " \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

# ---- K: transcript examples (Arabic) for one dialect sentence
sid, voice = None, None
s = main[(main.register == "dialect") & (main.model == "habibi_sau")]
piv = s.pivot_table(index=["id", "voice"], columns="cond", values="cer")
cand = piv[(piv["raw"] < 0.1) & (piv["catt_full"] > 0.5) & (piv["catt_nocase"] < 0.3)]
if len(cand):
    sid, voice = cand.index[0]
esc = lambda t: str(t).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
lines = []
if sid:
    ref = T[sid]["text"]
    lines.append(f"Reference & \\textarabic{{{esc(ref)}}} & -- \\\\\\midrule")
    for m, n in (("habibi_sau", "Habibi-SAU"), ("xtts", "XTTS-v2")):
        for c, cn in (("raw", "Raw"), ("catt_full", "Full"), ("catt_nocase", "No-case")):
            r = main[(main.model == m) & (main.cond == c) & (main.id == sid) & (main.voice == voice)]
            if len(r): lines.append(f"{n}, {cn} & \\textarabic{{{esc(r.hyp.iloc[0])[:140]}}} & {100 * r.cer.iloc[0]:.0f} \\\\")
    M["x.ex.id"] = sid
open(OUT / "x_examples_src.tex", "w").write(
    "\\documentclass[border=2pt]{standalone}\n\\usepackage{fontspec}\n\\usepackage{polyglossia}\n\\setmainlanguage{english}\n"
    "\\setotherlanguage{arabic}\n\\newfontfamily\\arabicfont[Script=Arabic,Scale=1.05]{Amiri}\n\\usepackage{booktabs}\n"
    "\\begin{document}\\small\n\\begin{tabular}{@{}l r r@{}}\\toprule\nSystem, condition & Cohere ASR transcript & CER (\\%)\\\\\\midrule\n"
    + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\n\\end{document}\n")

lines = ["\\makeatletter"] + [f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())] + ["\\makeatother"]
(OUT / "x_numbers.tex").write_text("\n".join(lines) + "\n")
nj = json.loads((OUT / "numbers.json").read_text()) if (OUT / "numbers.json").exists() else {}
nj.update(M); (OUT / "numbers.json").write_text(json.dumps(nj, indent=1, sort_keys=True, ensure_ascii=False))
print(len(M), "numbers"); [print(k, v) for k, v in sorted(M.items()) if not k.startswith("x.fail")]
