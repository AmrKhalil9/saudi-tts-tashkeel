"""Generate every number in the paper from results/clips.csv (+ diacritizer timing files).

Writes <out>/tab_*.tex and <out>/numbers.tex. numbers.tex defines \\R{key} values used in the prose, so no figure
in the paper is typed by hand. All CERs are per-clip CER capped at 1.0 (see s07), reported in percent.
Paired tests: per (model, ASR, channel), every diacritized condition vs raw on the same sentence and voice;
mean difference with a 10,000-sample paired bootstrap 95% CI, two-sided Wilcoxon signed-rank, Holm-corrected
over the conditions of that family.
Usage: python make_tables.py --root . --out paper_tables
"""
import argparse, json, statistics
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

MODELS = [("habibi_sau", "Habibi-SAU"), ("habibi_unified", "Habibi-Unified"), ("omnivoice", "OmniVoice"),
          ("lahgtna", "Lahgtna"), ("chatterbox", "Chatterbox-ML"), ("xtts", "XTTS-v2")]
CONDS = [("raw", "Raw"), ("catt_full", "Full"), ("catt_nocase", "No-case"), ("shadda_only", "Shadda"),
         ("ft_full", "FT-full")]
REGS = [("msa", "MSA"), ("dialect", "Dialect"), ("entity", "Entity")]
NUM = {}


def put(key, val, fmt="{:.1f}"):
    NUM[key] = fmt.format(val) if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
    return NUM[key]


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


def boot(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(df, model, asr, ch, conds, reg=None, base_model=None, base_cond="raw"):
    """Diacritized condition vs base, same sentence+voice. Returns {cond: dict}."""
    g = df[(df.asr == asr) & (df.channel == ch)]
    if reg:
        g = g[g.register == reg]
    b = g[(g.model == (base_model or model)) & (g.cond == base_cond)].set_index(["id", "voice"]).cer
    out = {}
    for c in conds:
        o = g[(g.model == model) & (g.cond == c)].set_index(["id", "voice"]).cer
        j = pd.concat([b, o], axis=1, keys=["b", "c"]).dropna()
        if len(j) < 10:
            continue
        d = (j.c - j.b).values * 100
        lo, hi = boot(d)
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        out[c] = dict(n=len(j), base=j.b.mean() * 100, val=j.c.mean() * 100, d=d.mean(), lo=lo, hi=hi, p=p)
    if out:
        for c, ph in zip(out, holm([v["p"] for v in out.values()])):
            out[c]["ph"] = ph
    return out


def mark(t):
    if t["ph"] >= 0.05:
        return ""
    return "$^{\\uparrow}$" if t["d"] > 0 else "$^{\\downarrow}$"


def table_main(df, out):
    lines = []
    for asr, ch, title in [("cohere", "wb", "Cohere ASR, wideband"), ("whisper", "wb", "Whisper, wideband"),
                           ("cohere", "g711", "Cohere ASR, G.711 telephone")]:
        lines.append(f"\\midrule\\multicolumn{{6}}{{l}}{{\\textit{{{title}}}}}\\\\")
        for m, name in MODELS:
            t = paired(df, m, asr, ch, [c for c, _ in CONDS[1:]])
            base = next(iter(t.values()))["base"]
            put(f"{m}.raw.{asr}.{ch}", base)
            cells = [f"{base:.1f}"]
            for c, _ in CONDS[1:]:
                put(f"{m}.{c}.{asr}.{ch}", t[c]["val"]); put(f"{m}.{c}.{asr}.{ch}.d", t[c]["d"])
                cells.append(f"{t[c]['val']:.1f}{mark(t[c])}")
            lines.append(f"{name} & " + " & ".join(cells) + "\\\\")
    body = "\n".join(lines)
    (out / "tab_main.tex").write_text(
        "\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrr}\\toprule\n"
        "TTS model & Raw & Full & No-case & Shadda & FT-full\\\\\n" + body + "\n\\bottomrule\\end{tabular}\n"
        "\\caption{Mean CER (\\%) over all 749 sentences, per text condition. $^{\\uparrow}$/$^{\\downarrow}$: "
        "significantly worse/better than Raw (paired Wilcoxon, Holm-corrected over the four conditions, "
        "$p<0.05$). Full, No-case and Shadda come from CATT, and FT-full comes from Fine-Tashkeel.}\\label{tab:main}\n"
        "\\end{table}\n")


def table_register(df, out):
    rows = []
    for m, name in MODELS:
        cells = []
        for c in ["catt_full", "catt_nocase"]:
            for r, _ in REGS:
                t = paired(df, m, "cohere", "wb", [c], reg=r)[c]
                put(f"{m}.{c}.{r}.d", t["d"]); put(f"{m}.raw.{r}", t["base"]); put(f"{m}.{c}.{r}", t["val"])
                cells.append(f"{t['d']:+.1f} {{\\tiny[{t['lo']:.1f}, {t['hi']:.1f}]}}")
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    (out / "tab_register.tex").write_text(
        "\\begin{table*}[t]\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrr|rrr}\\toprule\n"
        "& \\multicolumn{3}{c|}{Full (CATT)} & \\multicolumn{3}{c}{No-case (CATT)}\\\\\n"
        "TTS model & MSA & Dialect & Entity & MSA & Dialect & Entity\\\\\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n\\caption{Change in CER (percentage points) relative to Raw, by register, "
        "with paired-bootstrap 95\\% CIs (Cohere ASR, wideband; $n=200$ MSA, 200 dialect, 349 entity sentences).}"
        "\\label{tab:register}\n\\end{table*}\n")


def table_robust(df, out):
    settings = [("cohere", "wb", "Co-WB"), ("cohere", "g711", "Co-711"), ("whisper", "wb", "Wh-WB"),
                ("whisper", "g711", "Wh-711"), ("omniasr", "wb", "Omni$^\\dagger$")]
    rows = []
    for m, name in MODELS:
        cells = []
        for c in ["catt_full", "catt_nocase"]:
            for asr, ch, _ in settings:
                t = paired(df, m, asr, ch, [c]).get(c)
                cells.append("--" if t is None else f"{t['d']:+.1f}{mark(t)}")
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    hdr = " & ".join(s[2] for s in settings)
    (out / "tab_robust.tex").write_text(
        "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrr|rrrrr}\\toprule\n"
        "& \\multicolumn{5}{c|}{Full (CATT) vs Raw} & \\multicolumn{5}{c}{No-case (CATT) vs Raw}\\\\\n"
        f"TTS model & {hdr} & {hdr}\\\\\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n\\caption{$\\Delta$CER (pp) under three ASR systems and two channels. "
        "Co = Cohere ASR, Wh = Whisper large-v3, Omni = omniASR-LLM-7B; WB = wideband, 711 = G.711 $\\mu$-law. "
        "$^\\dagger$Paired subsample of 56 sentences (balanced over registers, fixed seed). Arrows as in "
        "Table~\\ref{tab:main}.}\\label{tab:robust}\n\\end{table*}\n")


def table_entity(df, out):
    # identifier sentences only (phone, ID, order number), which exist as paired digit/spelled twins
    e = df[df.ent_ok.notna() & (df.asr == "cohere") & (df.channel == "wb")
           & df.entity_types.fillna("").str.contains("id|phone|order")].copy()
    e["ent_ok"] = e.ent_ok.astype(str).str.lower().isin(["true", "1", "1.0"])
    rows = []
    for m, name in MODELS:
        cells = []
        for nf in ["digits", "spelled"]:
            for c in ["raw", "catt_full", "catt_nocase"]:
                v = e[(e.model == m) & (e.numform == nf) & (e.cond == c)].ent_ok.mean() * 100
                put(f"{m}.ent.{nf}.{c}", v, "{:.0f}")
                cells.append(f"{v:.0f}")
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    n = e[(e.model == "xtts") & (e.cond == "raw")].groupby("numform").size().to_dict()
    put("ent.n.digits", n.get("digits", 0), "{}"); put("ent.n.spelled", n.get("spelled", 0), "{}")
    (out / "tab_entity.tex").write_text(
        "\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrr|rrr}\\toprule\n"
        "& \\multicolumn{3}{c|}{Digits in text} & \\multicolumn{3}{c}{Spelled out}\\\\\n"
        "TTS model & Raw & Full & No-c. & Raw & Full & No-c.\\\\\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n\\caption{Entity accuracy (\\%): share of identifier sentences (phone, ID "
        f"and order numbers) whose exact digit sequence is recovered by the ASR (Cohere, wideband). The same "
        f"{NUM['ent.n.digits']} sentences appear with digits and with the digits spelled out as words."
        f"}}\\label{{tab:entity}}\n\\end{{table}}\n")


def table_ablation(df, out):
    rows = []
    for m, name in [("habibi_sau", "Habibi-SAU"), ("omnivoice", "OmniVoice")]:
        cells = []
        for asr in ["cohere", "whisper"]:
            full = paired(df, m, asr, "wb", ["catt_full"])["catt_full"]
            rr = paired(df, f"{m}_refraw", asr, "wb", ["catt_full"], base_model=m)["catt_full"]
            put(f"{m}.abl.{asr}.raw", full["base"]); put(f"{m}.abl.{asr}.full", full["val"])
            put(f"{m}.abl.{asr}.refraw", rr["val"]); put(f"{m}.abl.{asr}.refraw.p", rr["p"], "{:.2f}")
            cells += [f"{full['base']:.1f}", f"{full['val']:.1f}{mark(full)}", f"{rr['val']:.1f}{mark(rr)}"]
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    q = df[(df.asr == "cohere") & (df.channel == "wb")]
    for m in ["habibi_sau", "omnivoice"]:
        r = q[(q.model == m) & (q.cond == "raw")].dur.median()
        f = q[(q.model == m) & (q.cond == "catt_full")].dur.median()
        rr = q[(q.model == f"{m}_refraw") & (q.cond == "catt_full")].dur.median()
        put(f"{m}.durratio.full", f / r, "{:.2f}"); put(f"{m}.durratio.refraw", rr / r, "{:.2f}")
    (out / "tab_ablation.tex").write_text(
        "\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.2pt}\n\\begin{tabular}{lrrr|rrr}\\toprule\n"
        "& \\multicolumn{3}{c|}{Cohere ASR} & \\multicolumn{3}{c}{Whisper}\\\\\n"
        "TTS model & Raw & Full & Plain ref & Raw & Full & Plain ref\\\\\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n\\caption{Reference-transcript ablation (mean CER \\%, wideband, 749 "
        "sentences). Full: target \\emph{and} voice-prompt transcript diacritized (main setting). Plain ref: "
        "target diacritized, prompt transcript left undiacritized. Arrows vs Raw as in Table~\\ref{tab:main}.}"
        "\\label{tab:ablation}\n\\end{table}\n")


def table_quality(df, out):
    q = df[(df.asr == "cohere") & (df.channel == "wb") & df.model.isin([m for m, _ in MODELS])]
    rows = []
    for m, name in MODELS:
        x = q[q.model == m]
        r = x[x.cond == "raw"]
        cells = []
        for c in ["raw", "catt_full", "catt_nocase"]:
            y = x[x.cond == c]
            put(f"{m}.utmos.{c}", y.utmos.mean(), "{:.2f}"); put(f"{m}.sim.{c}", y.sim.mean(), "{:.2f}")
            cells.append(f"{y.utmos.mean():.2f}")
        for c in ["raw", "catt_full", "catt_nocase"]:
            cells.append(f"{x[x.cond == c].sim.mean():.2f}")
        dr = x[x.cond == "catt_full"].dur.median() / r.dur.median()
        put(f"{m}.durratio", dr, "{:.2f}")
        cells.append(f"{dr:.2f}")
        for c in ["raw", "catt_full"]:
            fr = x[x.cond == c].failure.astype(str).str.lower().isin(["true", "1"]).mean() * 100
            put(f"{m}.fail.{c}", fr)
            cells.append(f"{fr:.1f}")
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    (out / "tab_quality.tex").write_text(
        "\\begin{table*}[t]\\centering\\small\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrr|rrr|r|rr}\\toprule\n"
        "& \\multicolumn{3}{c|}{UTMOSv2 $\\uparrow$} & \\multicolumn{3}{c|}{Speaker SIM $\\uparrow$} & Dur. & "
        "\\multicolumn{2}{c}{Failure \\% $\\downarrow$}\\\\\n"
        "TTS model & Raw & Full & No-case & Raw & Full & No-case & Full/Raw & Raw & Full\\\\\\midrule\n" +
        "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\caption{Naturalness (UTMOSv2), speaker similarity "
        "(ECAPA cosine to the voice prompt), median duration ratio Full/Raw, and failure rate (share of clips "
        "with CER $\\geq$ 50\\%, Cohere ASR). Wideband, 749 sentences per cell.}\\label{tab:quality}\n\\end{table*}\n")


def table_subdialect(df, out):
    """Dialect register split by sub-dialect (Najdi vs Hijazi), Cohere wideband."""
    g = df[df.register == "dialect"]
    rows = []
    for m, name in MODELS:
        cells = []
        for sd in ["najdi", "hijazi"]:
            h = g[g.subdialect == sd]
            ts = paired(h, m, "cohere", "wb", ["catt_full", "catt_nocase"])
            cells += [f"{ts['catt_full']['base']:.1f}", f"{ts['catt_full']['val']:.1f}{mark(ts['catt_full'])}",
                      f"{ts['catt_nocase']['val']:.1f}{mark(ts['catt_nocase'])}"]
        rows.append(f"{name} & " + " & ".join(cells) + "\\\\")
    (out / "tab_subdialect.tex").write_text(
        "\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{2.5pt}\n\\begin{tabular}{lrrr|rrr}\\toprule\n"
        "& \\multicolumn{3}{c|}{Najdi ($n=100$)} & \\multicolumn{3}{c}{Hijazi ($n=100$)}\\\\\n"
        "TTS model & Raw & Full & No-c. & Raw & Full & No-c.\\\\\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\\end{tabular}\n\\caption{Dialect sentences by sub-dialect (mean CER \\%, Cohere ASR, "
        "wideband). Arrows as in Table~\\ref{tab:main}.}\\label{tab:subdialect}\n\\end{table}\n")


def latency(root, out):
    def med(path, key):
        v = [json.loads(l)[key] for l in open(path)]
        return statistics.median(v), statistics.quantiles(v, n=20)[18]
    g = med(root / "testset/diac.jsonl", "catt_ms")
    c8 = med(root / "bench/latency/catt_cpu8.jsonl", "catt_ms")
    c1 = med(root / "bench/latency/catt_cpu1.jsonl", "catt_ms")
    ft = med(root / "testset/diac.jsonl", "ft_ms")
    for k, (m, p) in {"catt.gpu": g, "catt.cpu8": c8, "catt.cpu1": c1, "ft.gpu": ft}.items():
        put(f"lat.{k}", m); put(f"lat.{k}.p95", p)
    put("lat.ft.ratio", ft[0] / g[0], "{:.0f}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="paper_tables")
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(root / "results/clips.csv")
    df["cer"] = df["cer"].clip(upper=1.0)
    table_main(df, out); table_register(df, out); table_robust(df, out); table_entity(df, out)
    table_ablation(df, out); table_quality(df, out); table_subdialect(df, out); latency(root, out)
    o = df[df.asr == "omniasr"]
    put("omni.n", len(o), "{}"); put("n.clips", len(df[(df.asr == "cohere") & (df.channel == "wb") &
                                                      df.model.isin([m for m, _ in MODELS])]), "{:,}")
    lines = ["% generated by make_tables.py -- do not edit", "\\makeatletter"]
    lines += [f"\\expandafter\\def\\csname r@{k}\\endcsname{{{v}}}" for k, v in sorted(NUM.items())]
    lines += ["\\makeatother",
              "\\newcommand{\\R}[1]{\\ifcsname r@#1\\endcsname\\csname r@#1\\endcsname\\else\\textbf{??#1}\\fi}"]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    (out / "numbers.json").write_text(json.dumps(NUM, indent=1, sort_keys=True))
    print(f"{len(NUM)} numbers; tables:", sorted(p.name for p in out.glob("tab_*.tex")))


if __name__ == "__main__":
    main()
