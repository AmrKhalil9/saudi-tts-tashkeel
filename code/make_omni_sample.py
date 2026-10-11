"""Paired omniASR subsample: 56 sentence ids balanced across registers (seeded), every model, the three key
conditions (raw, catt_full, catt_nocase), wideband only -> 56 x 6 x 3 = 1008 clips."""
import json, random, sys
from pathlib import Path
R = Path(sys.argv[1])
rows = [json.loads(l) for l in open(R / "testset/diac.jsonl")]
rng = random.Random(20261011)
ids = []
for reg, k in [("msa", 19), ("dialect", 19), ("entity", 18)]:
    pool = sorted(r["id"] for r in rows if r["register"] == reg)
    ids += rng.sample(pool, k)
voice = {}
for p in (R / "lists/out_wb.txt").read_text().split():
    parts = Path(p).parts  # .../out/<model>/<cond>/<voice>/<id>.wb.wav
    voice[(parts[-4], parts[-3], parts[-1].split(".")[0])] = p
out = [voice[(m, c, i)] for m in ["habibi_sau", "habibi_unified", "lahgtna", "omnivoice", "xtts", "chatterbox"]
       for c in ["raw", "catt_full", "catt_nocase"] for i in ids if (m, c, i) in voice]
(R / "lists/omni_sample.txt").write_text("\n".join(out) + "\n")
print(len(out), "clips in omniASR sample")
