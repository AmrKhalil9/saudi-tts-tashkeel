"""Rank reference-voice candidates from the public Saudi benchmark: 5-10 s, Najdi/Hijazi, by DNSMOS OVRL; F0 for gender."""
import json, numpy as np, soundfile as sf, librosa, onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
M = json.load(open("bench/sau_meta.json"))
C = [m for m in M if 5 <= m["duration"] <= 10 and m["dialect"] in ("SAU-Najdi", "SAU-Hijazi")]
S = ort.InferenceSession("models/DNSMOS/sig_bak_ovr.onnx", providers=["CPUExecutionProvider"])
P = [np.poly1d([-0.08397278, 1.22083953, 0.0052439]), np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
     np.poly1d([-0.06766283, 1.11546468, 0.04602535])]
def score(m):
    a, sr = sf.read(m["wav"]); a = a if a.ndim == 1 else a.mean(1)
    x = a.copy()
    while len(x) < int(9.01 * 16000): x = np.append(x, a)
    sig, bak, ovr = S.run(None, {S.get_inputs()[0].name: x[: int(9.01 * 16000)].astype("float32")[None]})[0][0]
    f0, vf, _ = librosa.pyin(a.astype(float), fmin=60, fmax=400, sr=sr)
    return dict(m, sig=float(P[0](sig)), bak=float(P[1](bak)), ovrl=float(P[2](ovr)), f0=float(np.nanmedian(f0)))
with ThreadPoolExecutor(16) as ex:
    R = list(ex.map(score, C))
R.sort(key=lambda r: -r["ovrl"])
json.dump(R, open("bench/ref_candidates.json", "w"), ensure_ascii=False)
best = {}
for r in R:
    g = "F" if r["f0"] > 165 else "M"
    k = (r["dialect"], g)
    best.setdefault(k, [])
    if len(best[k]) < 3 and r["speaker"] not in [x["speaker"] for x in best[k]]:
        best[k].append(r)
print(len(C), "candidates")
for k, v in best.items():
    for r in v:
        print(k, r["id"], r["speaker"], round(r["duration"], 1), "OVRL %.2f SIG %.2f BAK %.2f f0 %.0f" % (r["ovrl"], r["sig"], r["bak"], r["f0"]), r["text"][:70])
