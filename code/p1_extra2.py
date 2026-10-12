"""Second set of journal-length analyses for Paper 1: test-set table, per-voice and per-domain effects, full CER tables for
every recognizer/channel, WER and DNSMOS tables, diacritizer latency table. Appends to x_numbers.tex.
Usage: python p1_extra2.py --root . --out paper_tables"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("--out"); a = ap.parse_args()
R, OUT = Path(a.root), Path(a.out)
MODELS = [("habibi_sau", "Habibi-SAU"), ("habibi_unified", "Habibi-Unified"), ("omnivoice", "OmniVoice"), ("lahgtna", "Lahgtna"),
          ("chatterbox", "Chatterbox-ML"), ("xtts", "XTTS-v2")]
CONDS = [("raw", "Raw"), ("catt_full", "Full"), ("catt_nocase", "No-case"), ("shadda_only", "Shadda"), ("ft_full", "FT-full")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "pdf.fonttype": 42})
M = {}
def put(k, v, fmt="{:.1f}"): M[k] = fmt.format(v); return M[k]
df = pd.read_csv(R / "results/clips.csv"); df["cer"] = df["cer"].clip(upper=1.0); df["wer"] = df["wer"].clip(upper=1.0)
T = pd.DataFrame([json.loads(l) for l in open(R / "testset/diac_catt.jsonl")])

# ---- test set description
rows = []
for reg, name in (("msa", "MSA"), ("dialect", "Dialect"), ("entity", "Entity")):
    t = T[T.register == reg]; w = t.text.str.split().str.len()
    VN = {"msa": "MSA", "najdi": "Najdi", "hijazi": "Hijazi"}
    subs = ", ".join(f"{VN.get(k, k)} {v}" for k, v in t.subdialect.value_counts().items())
    doms = str(t.domain.nunique()) if reg != "entity" else "--"
    rows.append(f"{name} & {len(t)} & {doms} & {w.mean():.1f} & {subs} \\\\")
named = T[T.register != "entity"].domain.nunique()
open(OUT / "x_tab_testset.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Composition of the test set: sentences, service domains covered (the entity "
    "register uses mixed templates), mean length in words, and sentences per variety.}\\label{tab:testset}\n"
    "\\begin{tabular}{lrrrl}\\toprule\nRegister & Sentences & Domains & Words & "
    "Varieties \\\\\\midrule\n" + "\n".join(rows) + f"\n\\midrule\nTotal & {len(T)} & {named} & {T.text.str.split().str.len().mean():.1f} & \\\\\n"
    "\\bottomrule\\end{tabular}\\end{table}\n")
put("x.ts.domains", named, "{:d}")

def delta(g, m, c, by=None):
    s = g[g.model == m]
    b = s[s.cond == "raw"].set_index(["id", "voice"]).cer; o = s[s.cond == c].set_index(["id", "voice"]).cer
    j = pd.concat([b, o], axis=1, keys=["b", "c"]).dropna().reset_index(); j["d"] = (j.c - j.b) * 100
    return j if by is None else j
main = df[(df.asr == "cohere") & (df.channel == "wb")]

# ---- per-voice effect of Full
voices = sorted(main.voice.unique())
fig, ax = plt.subplots(figsize=(3.4, 2.3)); w = 0.2
for k, v in enumerate(voices):
    vals = []
    for m, _ in MODELS:
        j = delta(main, m, "catt_full"); vals.append(j[j.voice == v].d.mean())
    ax.bar(np.arange(len(MODELS)) + (k - 1.5) * w, vals, w, label=v.replace("_", " "), edgecolor="black", lw=0.3)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([n for _, n in MODELS], rotation=30, ha="right"); ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("$\\Delta$CER Full vs Raw (pp)"); ax.legend(fontsize=6, frameon=False); ax.grid(axis="y", alpha=.3, lw=.4)
fig.tight_layout(); fig.savefig(OUT / "x_fig_voice.pdf")
for m, _ in MODELS:
    j = delta(main, m, "catt_full"); g = j.groupby("voice").d.mean(); put(f"x.voice.{m}.range", g.max() - g.min())

# ---- per-domain effect (MSA register)
msa = main.merge(T[["id", "domain"]], on="id")
doms = sorted(T[T.register == "msa"].domain.unique())
H = np.array([[ (lambda j: j[j.id.isin(T[(T.domain == d) & (T.register == "msa")].id)].d.mean())(delta(msa[msa.register == "msa"], m, "catt_full")) for d in doms] for m, _ in MODELS])
fig, ax = plt.subplots(figsize=(3.4, 2.3)); im = ax.imshow(np.clip(H, -10, 10), cmap="RdBu_r", vmin=-10, vmax=10, aspect="auto")
ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([n for _, n in MODELS]); ax.set_xticks(range(len(doms))); ax.set_xticklabels(doms, rotation=35, ha="right")
for i in range(H.shape[0]):
    for j in range(H.shape[1]): ax.text(j, i, f"{H[i, j]:+.0f}", ha="center", va="center", fontsize=5.5, color="white" if abs(H[i, j]) > 6 else "black")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="$\\Delta$CER (pp, clipped)"); fig.tight_layout(); fig.savefig(OUT / "x_fig_domain.pdf")

# ---- full CER tables for every recognizer and channel (appendix), and WER
def mean_table(sub, metric):
    return {(m, c): 100 * sub[(sub.model == m) & (sub.cond == c)][metric].mean() for m, _ in MODELS for c, _ in CONDS}
blocks = []
for asr, an in (("cohere", "Cohere"), ("whisper", "Whisper large-v3")):
    for ch, chn in (("wb", "wideband"), ("g711", "G.711")):
        sub = df[(df.asr == asr) & (df.channel == ch)]; t = mean_table(sub, "cer")
        blocks.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{an}, {chn}}}}}\\\\\n" + "\n".join(
            f"{n} & " + " & ".join(f"{t[(m, c)]:.1f}" for c, _ in CONDS) + " \\\\" for m, n in MODELS))
om = df[df.asr == "omniasr"]
if len(om):
    t = mean_table(om, "cer")
    blocks.append("\\multicolumn{6}{l}{\\textit{omniASR-LLM-7B, wideband subsample}}\\\\\n" + "\n".join(
        f"{n} & " + " & ".join((f"{t[(m, c)]:.1f}" if not np.isnan(t[(m, c)]) else "--") for c, _ in CONDS) + " \\\\" for m, n in MODELS))
open(OUT / "x_tab_allcer.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\caption{Mean CER (\\%) of every model and text condition for every recognizer and channel "
    "(capped per clip at 100\\%).}\\label{tab:allcer}\n\\begin{tabular}{lrrrrr}\\toprule\nTTS model & "
    + " & ".join(n for _, n in CONDS) + " \\\\\\midrule\n" + "\n\\midrule\n".join(blocks) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")
t = mean_table(main, "wer")
open(OUT / "x_tab_wer.tex", "w").write(
    "\\begin{table}[t]\\centering\\small\\caption{Mean word error rate (\\%, capped per clip at 100\\%; Cohere ASR, wideband).}\\label{tab:wer}\n"
    "\\begin{tabular}{lrrrrr}\\toprule\nTTS model & " + " & ".join(n for _, n in CONDS) + " \\\\\\midrule\n" + "\n".join(
        f"{n} & " + " & ".join(f"{t[(m, c)]:.1f}" for c, _ in CONDS) + " \\\\" for m, n in MODELS) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
for m, _ in MODELS: put(f"x.wer.{m}.raw", t[(m, "raw")]); put(f"x.wer.{m}.catt_full", t[(m, "catt_full")])
rows = []
for m, n in MODELS:
    cells = []
    for c, _ in CONDS[:3]:
        s = main[(main.model == m) & (main.cond == c)]
        cells += [f"{s.dns_sig.mean():.2f}", f"{s.dns_bak.mean():.2f}", f"{s.dns_ovrl.mean():.2f}"]
    rows.append(f"{n} & " + " & ".join(cells) + " \\\\")
open(OUT / "x_tab_dnsmos.tex", "w").write(
    "\\begin{table*}[t]\\centering\\small\\caption{DNSMOS P.835 signal (SIG), background (BAK) and overall (OVRL) quality for the Raw, "
    "Full and No-case conditions (Cohere ASR set, wideband).}\\label{tab:dnsmos}\n\\begin{tabular}{l" + "rrr" * 3 + "}\\toprule\n& "
    + " & ".join(f"\\multicolumn{{3}}{{c}}{{{n}}}" for _, n in CONDS[:3]) + "\\\\\nTTS model & " + " & ".join(["SIG & BAK & OVRL"] * 3)
    + " \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table*}\n")

# ---- diacritizer latency table
lat = pd.read_csv(R / "results/latency.csv")
open(OUT / "x_lat_cols.txt", "w").write(",".join(lat.columns) + "\n" + lat.head(8).to_string())

lines = (OUT / "x_numbers.tex").read_text().replace("\\makeatother\n", "")
(OUT / "x_numbers.tex").write_text(lines + "\n".join(f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(M.items())) + "\n\\makeatother\n")
nj = json.loads((OUT / "numbers.json").read_text()); nj.update(M); (OUT / "numbers.json").write_text(json.dumps(nj, indent=1, sort_keys=True, ensure_ascii=False))
print(M)
