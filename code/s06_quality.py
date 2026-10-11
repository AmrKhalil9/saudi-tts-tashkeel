"""Stage 6: per-clip quality metrics on the wideband (16 kHz) audio.
  sim      cosine similarity of ECAPA-TDNN speaker embeddings (speechbrain/spkrec-ecapa-voxceleb) vs the voice reference
  utmos    UTMOSv2 (fold 0) predicted MOS
  dnsmos   DNSMOS P.835 SIG / BAK / OVRL
Resumable. Usage: python s06_quality.py --list files.txt --voices voices.json --out quality.jsonl
Each path must look like .../<model>/<cond>/<voice>/<id>.wb.wav (voice name taken from the parent folder).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

DNS = "models/DNSMOS/sig_bak_ovr.onnx"
POLY = [np.poly1d([-0.08397278, 1.22083953, 0.0052439]), np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
        np.poly1d([-0.06766283, 1.11546468, 0.04602535])]


def load16(p):
    a, sr = sf.read(p, dtype="float32")
    a = a if a.ndim == 1 else a.mean(1)
    if sr != 16000:
        import librosa
        a = librosa.resample(a, orig_sr=sr, target_sr=16000)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True); ap.add_argument("--voices", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import fcntl  # one writer per output file: a parallel run waits, then resumes past what is done
    lock = open(a.out + ".lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
    paths = [l.strip() for l in open(a.list) if l.strip()]
    done = {json.loads(l)["path"] for l in open(a.out)} if Path(a.out).exists() else set()
    todo = [p for p in paths if p not in done]
    print(f"{len(todo)} to score", flush=True)
    if not todo:
        return
    from speechbrain.inference.speaker import EncoderClassifier
    import onnxruntime as ort
    import utmosv2
    enc = EncoderClassifier.from_hparams(source="models/ecapa", run_opts={"device": "cuda"})
    utm = utmosv2.create_model(pretrained=True, fold=0,
                               checkpoint_path="models/UTMOSv2/fold0_s42_best_model.pth", device="cuda")
    dns = ort.InferenceSession(DNS, providers=["CPUExecutionProvider"])

    def emb(x):
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(x)[None].to("cuda")).squeeze()
        return torch.nn.functional.normalize(e, dim=0)

    refs = {v["name"]: emb(load16(v["wav"])) for v in json.load(open(a.voices))}
    t0 = time.time()
    with open(a.out, "a") as f:
        for i, p in enumerate(todo):
            x = load16(p)
            voice = Path(p).parent.name
            sim = float((emb(x) * refs[voice]).sum()) if voice in refs else None
            mos = float(np.asarray(utm.predict(data=x, sr=16000, device="cuda", verbose=False)).ravel()[0])
            y = x.copy()
            while len(y) < int(9.01 * 16000):
                y = np.append(y, x)
            s, b, o = dns.run(None, {dns.get_inputs()[0].name: y[: int(9.01 * 16000)][None].astype("float32")})[0][0]
            f.write(json.dumps({"path": p, "sim": round(sim, 4) if sim is not None else None, "utmos": round(mos, 3),
                                "dns_sig": round(float(POLY[0](s)), 3), "dns_bak": round(float(POLY[1](b)), 3),
                                "dns_ovrl": round(float(POLY[2](o)), 3), "dur": round(len(x) / 16000, 3)}) + "\n")
            if i % 200 == 0:
                f.flush(); print(f"{i}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
