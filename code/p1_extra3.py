"""Third set for Paper 1: diacritizer comparison (CATT vs Fine-Tashkeel), naturalness/similarity distributions, clip-level
agreement between recognizers. Appends to x_numbers.tex. Usage: python p1_extra3.py --root . --out paper_tables"""
import argparse, json, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
R, OUT = Path(a.root), Path(a.out)
MODELS = [("habibi_sau", "Habibi-SAU"), ("habibi_unified", "Habibi-Unified"), ("omnivoice", "OmniVoice"), ("lahgtna", "Lahgtna"),
          ("chatterbox", "Chatterbox-ML"), ("xtts", "XTTS-v2")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v); return M[k]
DIAC = re.compile(r"[\u064B-\u0652\u0670]"); AR = re.compile(r"[\u0621-\u064A\u0640\u064B-\u0652\u0670]+")

# ---- diacritizer comparison: find the file that holds the Fine-Tashkeel condition
src = None
for f in sorted((R / "testset").glob("*.jsonl")):
    r = json.loads(open(f).readline())
    if "cond_ft_full" in r and "cond_catt_full" in r: src = f; break
rows_out = []
if src:
    D = pd.DataFrame([json.loads(l) for l in open(src)])
    def words(t): return AR.findall(t)
    def marks_per_word(t):
        w = words(t); return np.mean([len(DIAC.findall(x)) for x in w]) if w else np.nan
    def final_mark(w):
        m = re.search(r"([\u064B-\u0652]+)$", w); return m.group(1).replace("\u0651", "") if m else ""
    def strip_final(w): return re.sub(r"[\u064B-\u0650\u0652]+$", "", w)
    for reg, name in (("msa", "MSA"), ("dialect", "Dialect"), ("entity", "Entity")):
        d = D[D.register == reg]; agree, agree_nc, case_cov_c, case_cov_f, n = 0, 0, 0, 0, 0
        for _, r in d.iterrows():
            wc, wf = words(r.cond_catt_full), words(r.cond_ft_full)
            for x, y in zip(wc, wf):
                if DIAC.sub("", x) != DIAC.sub("", y): continue
                n += 1; agree += x == y; agree_nc += strip_final(x) == strip_final(y)
                case_cov_c += final_mark(x) != ""; case_cov_f += final_mark(y) != ""
        mc = d.cond_catt_full.map(marks_per_word).mean(); mf = d.cond_ft_full.map(marks_per_word).mean()
        rows_out.append(f"{name} & {mc:.2f} & {mf:.2f} & {100*case_cov_c/n:.0f} & {100*case_cov_f/n:.0f} & {100*agree/n:.1f} & {100*agree_nc/n:.1f} \\\\")
        put(f"x.diac.{reg}.agree", 100 * agree / n); put(f"x.diac.{reg}.agree_nc", 100 * agree_nc / n)
        put(f"x.diac.{reg}.case_catt", 100 * case_cov_c / n, "{:.0f}"); put(f"x.diac.{reg}.mpw_catt", mc, "{:.2f}")
    open(OUT / "x_tab_diac.tex", "w").write(
        "\\begin{table*}[t]\\centering\\small\\caption{Behaviour of the two diacritizers on the test set: mean diacritic marks per Arabic "
        "word, share of words that receive a case ending (a mark on the last letter other than shadda), and word-level agreement "
        "between CATT and Fine-Tashkeel, exactly and ignoring case endings.}\\label{tab:diac}\n"
        "\\begin{tabular}{lrrrrrr}\\toprule\n& \\multicolumn{2}{c}{Marks per word} & \\multicolumn{2}{c}{Case ending (\\%)} & "
        "\\multicolumn{2}{c}{Agreement (\\%)}\\\\\nRegister & CATT & FT & CATT & FT & exact & w/o case \\\\\\midrule\n" + "\n".join(rows_out)
        + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
M["x.diac.src"] = src.name if src else "none"

# ---- naturalness and similarity distributions
df = pd.read_csv(R / "results/clips.csv"); main = df[(df.asr == "cohere") & (df.channel == "wb") & df.model.isin([m for m, _ in MODELS])]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
for ax, col, lab in ((axes[0], "utmos", "UTMOSv2"), (axes[1], "sim", "Speaker similarity")):
    data, pos = [], []
    for i, (m, _) in enumerate(MODELS):
        for j, c in enumerate(["raw", "catt_full"]):
            data.append(main[(main.model == m) & (main.cond == c)][col].dropna().values); pos.append(i * 3 + j)
    bp = ax.boxplot(data, positions=pos, widths=0.8, showfliers=False, patch_artist=True, medianprops={"color": "black", "lw": 0.8})
    for k, b in enumerate(bp["boxes"]): b.set_facecolor("#bdbdbd" if k % 2 == 0 else "#b2182b"); b.set_alpha(0.75); b.set_linewidth(0.5)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(MODELS))]); ax.set_xticklabels([n for _, n in MODELS], rotation=30, ha="right")
    ax.set_ylabel(lab); ax.grid(axis="y", alpha=.3, lw=.4)
axes[0].legend([plt.Rectangle((0, 0), 1, 1, fc="#bdbdbd"), plt.Rectangle((0, 0), 1, 1, fc="#b2182b", alpha=.75)], ["Raw", "Full"], fontsize=6, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "x_fig_quality.pdf")

# ---- clip-level agreement between recognizers (Raw and Full, wideband)
c = df[(df.asr == "cohere") & (df.channel == "wb")].set_index(["model", "cond", "id", "voice"]).cer
w = df[(df.asr == "whisper") & (df.channel == "wb")].set_index(["model", "cond", "id", "voice"]).cer
j = pd.concat([c, w], axis=1, keys=["c", "w"]).dropna()
put("x.clip.r", j.corr().iloc[0, 1], "{:.2f}"); put("x.clip.rho", j.corr(method="spearman").iloc[0, 1], "{:.2f}"); put("x.clip.n", len(j), "{:,}")
fig, ax = plt.subplots(figsize=(3.4, 2.6))
hb = ax.hexbin(100 * j.c, 100 * j.w, gridsize=40, bins="log", cmap="Greys", mincnt=1)
ax.plot([0, 100], [0, 100], color="#b2182b", lw=0.6, ls=":"); ax.set_xlabel("CER, Cohere ASR (%)"); ax.set_ylabel("CER, Whisper large-v3 (%)")
fig.colorbar(hb, ax=ax, label="clips (log)"); fig.tight_layout(); fig.savefig(OUT / "x_fig_clipagree.pdf")

lines = (OUT / "x_numbers.tex").read_text().replace("\\makeatother\n", "")
(OUT / "x_numbers.tex").write_text(lines + "\n".join(f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())) + "\n\\makeatother\n")
nj = json.loads((OUT / "numbers.json").read_text()); nj.update(M); (OUT / "numbers.json").write_text(json.dumps(nj, indent=1, sort_keys=True, ensure_ascii=False))
print(M); print("\n".join(rows_out))
