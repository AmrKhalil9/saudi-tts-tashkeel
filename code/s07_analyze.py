"""Stage 7: join everything into one per-clip table and run the paired statistics.

Inputs (under --root):
  testset/diac.jsonl                  sentence metadata (register, subdialect, numform, raw text)
  out/<model>/timing.jsonl            synthesis time / audio length
  asr/<backend>_<channel>.jsonl       transcripts (channel = wb | g711)
  quality/quality.jsonl               sim / utmos / dnsmos (wideband)
Outputs (under --root/results): clips.csv, summary.csv, paired_tests.csv, latency.csv
Paired tests: for each model x register x channel x ASR, each diacritized condition vs raw on the SAME sentence and
voice: mean delta CER with a 10,000-sample paired bootstrap 95% CI and a two-sided Wilcoxon signed-rank p-value,
Holm-corrected within each (model, ASR, channel) family. Per-clip CER is capped at 1.0 for the means and tests
(a TTS that babbles past the sentence end otherwise yields CER >> 1 and one clip dominates a cell); the uncapped
value is kept in clips.csv, and `failure` flags clips with capped CER >= 0.5 for a separate failure-rate table.
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from textnorm import cer, digits_of, wer  # noqa: E402

PAT_T = (r".*/{out}/(?P<model>[^/]+)/(?P<cond>[^/]+)/(?P<voice>[^/]+)/(?P<id>[^/]+?)\.(?P<ch>wb|g711)\.wav$")


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (n - k) * p[i])); adj[i] = run
    return adj


def boot_ci(d, B=10000, seed=0):
    rng = np.random.default_rng(seed); d = np.asarray(d)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=".")
    ap.add_argument("--out-name", default="out")
    a = ap.parse_args()
    R = Path(a.root)
    PAT = re.compile(PAT_T.replace("{out}", re.escape(a.out_name)))
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(R / "testset/diac.jsonl")}
    rows = []
    for f in sorted((R / "asr").glob("*_*.jsonl")):
        backend = f.stem.split("_")[0]
        for l in open(f):
            j = json.loads(l)
            m = PAT.match(j["path"])
            if not m or j["hyp"].startswith("<<ERROR"):
                continue
            g = m.groupdict(); s = meta[g["id"]]
            c, n = cer(s["text"], j["hyp"]); w, _ = wer(s["text"], j["hyp"])
            rd = digits_of(s["text"])  # entity accuracy: exact digit sequence, only where the reference has an ID/phone
            ent_ok = (digits_of(j["hyp"]) == rd) if len(rd) >= 5 else None
            rows.append(dict(model=g["model"], cond=g["cond"], voice=g["voice"], id=g["id"], channel=g["ch"],
                             asr=backend, register=s["register"], subdialect=s["subdialect"], numform=s["numform"],
                             entity_types=s["entity_types"], cer=c, wer=w, nchar=n, ent_ok=ent_ok, hyp=j["hyp"], path=j["path"]))
    df = pd.DataFrame(rows)
    df["cer_raw"] = df["cer"]
    df["cer"] = df["cer"].clip(upper=1.0)
    df["failure"] = df["cer"] >= 0.5
    qf = R / "quality/quality.jsonl"
    if qf.exists() and qf.stat().st_size:
        q = pd.DataFrame([json.loads(l) for l in open(qf)])
        q["path"] = q["path"].str.replace(".wb.wav", "", regex=False)
        df["key"] = df["path"].str.replace(r"\.(wb|g711)\.wav$", "", regex=True)
        df = df.merge(q.rename(columns={"path": "key"}), on="key", how="left")
    out = R / "results"; out.mkdir(exist_ok=True)
    df.to_csv(out / "clips.csv", index=False)

    keys = ["model", "asr", "channel", "register", "cond"]
    summ = df.groupby(keys).agg(n=("cer", "size"), cer=("cer", "mean"), cer_median=("cer", "median"),
                                failure_rate=("failure", "mean"), wer=("wer", "mean"),
                                sim=("sim", "mean") if "sim" in df else ("cer", "size"),
                                utmos=("utmos", "mean") if "utmos" in df else ("cer", "size")).reset_index()
    summ.to_csv(out / "summary.csv", index=False)

    tests = []
    for (model, asr, ch, reg), g in df.groupby(["model", "asr", "channel", "register"]):
        base = g[g["cond"] == "raw"].set_index(["id", "voice"])["cer"]
        for cond in sorted(set(g["cond"]) - {"raw"}):
            other = g[g["cond"] == cond].set_index(["id", "voice"])["cer"]
            j = pd.concat([base, other], axis=1, keys=["raw", "c"]).dropna()
            if len(j) < 10:
                continue
            d = (j["c"] - j["raw"]).values
            lo, hi = boot_ci(d)
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            tests.append(dict(model=model, asr=asr, channel=ch, register=reg, cond=cond, n=len(d),
                              raw_cer=j["raw"].mean(), cond_cer=j["c"].mean(), delta=d.mean(), ci_lo=lo, ci_hi=hi,
                              p=p, worse=(d > 0).mean(), better=(d < 0).mean()))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = np.nan
        for _, idx in t.groupby(["model", "asr", "channel"]).groups.items():
            t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"].values)
    t.to_csv(out / "paired_tests.csv", index=False)

    lat = []
    for f in (R / a.out_name).glob("*/timing.jsonl"):
        tj = pd.DataFrame([json.loads(l) for l in open(f)])
        if "ms" in tj:
            tj = tj.dropna(subset=["ms"])
            lat.append(dict(model=f.parent.name, n=len(tj), ms_median=tj["ms"].median(), rtf_median=tj["rtf"].median(),
                            audio_s_median=tj["audio_s"].median(), errors=int(tj.get("error", pd.Series()).notna().sum())))
    pd.DataFrame(lat).to_csv(out / "latency.csv", index=False)
    print(summ.head(40).to_string()); print(t.head(40).to_string() if len(t) else "no tests")


if __name__ == "__main__":
    main()
