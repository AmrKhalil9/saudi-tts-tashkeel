"""Reviewer R1.1 (revision): generation variance across synthesis seeds on the 56-sentence paired subsample (Cohere ASR, wideband).
Seed 1234 = main run (results/clips.csv); seeds 1235/1236 = out_seeds/ (asr_seeds/). Output: results_seeds/seed_summary.json
Usage: python r1_seed_analysis.py --root .
"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from textnorm import cer  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); R = Path(ap.parse_args().root)
ids = set((R / "out_seeds/ids56.txt").read_text().split())
meta = {json.loads(l)["id"]: json.loads(l) for l in open(R / "testset/diac.jsonl")}
CONDS = ["raw", "catt_full", "catt_nocase"]
PAT = re.compile(r"(?:.*/)?s\d+/(?P<model>[^/]+)/(?P<cond>[^/]+)/(?P<voice>[^/]+)/(?P<id>[^/]+?)\.wb\.wav$")

c = defaultdict(dict)  # (model, cond, id) -> {seed: capped cer}
for r in csv.DictReader(open(R / "results/clips.csv")):
    if r["asr"] == "cohere" and r["channel"] == "wb" and r["id"] in ids and r["cond"] in CONDS:
        c[(r["model"], r["cond"], r["id"])][1234] = min(float(r["cer"]), 1.0)
for seed in (1235, 1236):
  for f in (R / f"asr_seeds/cohere_wb_s{seed}.jsonl", R / f"asr_seeds/cohere_wb_s{seed}_b.jsonl"):
    if not f.exists():
        continue
    for l in open(f):
        j = json.loads(l); m = PAT.match(j["path"])
        if not m or j["hyp"].startswith("<<ERROR"):
            continue
        g = m.groupdict()
        c[(g["model"], g["cond"], g["id"])][seed] = min(cer(meta[g["id"]]["text"], j["hyp"])[0], 1.0)

out = {}
models = sorted({k[0] for k in c})
for mdl in models:
    seeds = sorted({s for (m, _, _), v in c.items() if m == mdl for s in v})
    if len(seeds) < 2:
        continue
    keep = [i for i in ids if all(len(c.get((mdl, cd, i), {})) == len(seeds) for cd in CONDS)]
    res = {"seeds": seeds, "n_sentences": len(keep)}
    for cd in CONDS:
        per_seed = [100 * np.mean([c[(mdl, cd, i)][s] for i in keep]) for s in seeds]
        fail = [100 * np.mean([c[(mdl, cd, i)][s] >= 0.5 for i in keep]) for s in seeds]
        clip_sd = 100 * np.mean([np.std([c[(mdl, cd, i)][s] for s in seeds], ddof=1) for i in keep])
        res[cd] = {"mean_cer_per_seed": [round(x, 1) for x in per_seed], "failure_pct_per_seed": [round(x, 1) for x in fail],
                   "mean_within_clip_sd_pp": round(clip_sd, 1)}
    for cd in ("catt_full", "catt_nocase"):
        eff = [100 * np.mean([c[(mdl, cd, i)][s] - c[(mdl, "raw", i)][s] for i in keep]) for s in seeds]
        # effect using the per-sentence mean over seeds (averages out generation noise)
        avg = 100 * np.mean([np.mean([c[(mdl, cd, i)][s] for s in seeds]) - np.mean([c[(mdl, "raw", i)][s] for s in seeds])
                             for i in keep])
        res[f"effect_{cd}_vs_raw_pp"] = {"per_seed": [round(x, 1) for x in eff], "seed_averaged": round(avg, 1)}
    out[mdl] = res

(R / "results_seeds").mkdir(exist_ok=True)
(R / "results_seeds/seed_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
