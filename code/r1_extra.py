"""Revision analyses (reviewers R1.1-R1.2): failure-threshold sensitivity and the eight omniASR clips that could not be
transcribed (Chatterbox-ML, Full). Outputs: results_seeds/failure_threshold_sensitivity.json, results_seeds/omni_fix.json
Usage: python r1_extra.py --root .
"""
import argparse, json
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); R = Path(ap.parse_args().root)
df = pd.read_csv(R / "results/clips.csv")
CONDS = ["raw", "catt_full", "catt_nocase", "shadda_only", "ft_full"]

# share of clips with uncapped CER >= t (Cohere ASR, wideband)
d = df[(df.asr == "cohere") & (df.channel == "wb")]
out = {}
for t in (0.3, 0.5, 0.7, 1.0):
    g = d.assign(f=(d.cer_raw >= t)).groupby(["model", "cond"]).f.mean().mul(100).round(1).unstack()
    out[str(t)] = g[CONDS].to_dict("index")
json.dump(out, open(R / "results_seeds/failure_threshold_sensitivity.json", "w"), indent=1)

# Chatterbox-ML under omniASR: Full has 48 of 56 sentences; compare paired on the same 48
o = df[(df.asr == "omniasr") & (df.model == "chatterbox")].pivot_table(index="id", columns="cond", values="cer")
full_ids = o["catt_full"].dropna().index
pr = o.loc[full_ids]
fix = dict(n_full=int(len(full_ids)), old_full=round(100 * o["catt_full"].mean(), 1), raw48=round(100 * pr["raw"].mean(), 1),
           full48=round(100 * pr["catt_full"].mean(), 1), paired48=round(100 * (pr["catt_full"] - pr["raw"]).mean(), 1))
json.dump(fix, open(R / "results_seeds/omni_fix.json", "w"), indent=1)

# the eight missing clips under Cohere ASR (wideband): duration, CER, and the Full-vs-Raw change with and without them
miss = set(o.index[o["catt_full"].isna()])
c = d[(d.model == "chatterbox") & d.id.isin(o.index)]
cf = c[c.cond == "catt_full"].set_index("id")
pc = c.pivot_table(index="id", columns="cond", values="cer")
print("missing clips:", len(miss))
print("mean duration (s) missing %.1f, others %.1f" % (cf.loc[list(miss), "dur"].mean(), cf.drop(list(miss)).dur.mean()))
print("mean CER Full missing %.2f, others %.2f" % (cf.loc[list(miss), "cer"].mean(), cf.drop(list(miss)).cer.mean()))
print("Cohere Full-Raw (pp): all 56 %+.1f, same 48 %+.1f" % (100 * (pc.catt_full - pc.raw).mean(),
                                                             100 * (pc.catt_full - pc.raw).drop(list(miss)).mean()))
print(json.dumps(fix))
