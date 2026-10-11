"""Stage 3: synthesis driver. Runs inside the model's own venv and imports adapters/<name>.py, which defines
    load(device) -> model
    synth(model, text, ref_wav, ref_text, seed) -> (np.ndarray float32 mono, sample_rate)
For every (sentence, condition, voice) it writes out/<model>/<cond>/<voice>/<id>.wav and appends a timing row to
out/<model>/timing.jsonl: synthesis ms, audio seconds, real-time factor. Existing wavs are skipped (resumable).
The reference transcript is given in the SAME condition as the target text (diacritized when the target is), as
Habibi's ablation shows reference-text conditioning matters; --ref-raw forces a plain reference (sub-condition).

Usage: python s03_synth.py --adapter habibi_sau --diac diac_gpu.jsonl --voices voices.json --out out \
          --conds raw,catt_full,catt_nocase,shadda_only [--limit N] [--ids-file f]
"""
import argparse, importlib, json, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from s02_diacritize import strip_case, shadda_only, AR_WORD  # noqa: E402


def ref_text_for(cond, v):
    ref_raw, ref_diac = v["text_raw"], v["text_catt"]
    if cond == "raw":
        return ref_raw
    if cond == "ft_full":
        return v["text_ft"]
    if cond == "catt_full":
        return ref_diac
    if cond == "catt_nocase":
        return " ".join(strip_case(w) for w in ref_diac.split())
    if cond == "shadda_only":
        return " ".join(shadda_only(w) for w in ref_diac.split())
    return ref_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True); ap.add_argument("--diac", required=True)
    ap.add_argument("--voices", required=True); ap.add_argument("--out", default="out")
    ap.add_argument("--conds", default="raw,catt_full,catt_nocase,shadda_only,ft_full")
    ap.add_argument("--all-voices", action="store_true", help="every sentence with every voice (default: rotate)")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--ids-file")
    ap.add_argument("--ref-raw", action="store_true")
    a = ap.parse_args()

    ad = importlib.import_module(f"adapters.{a.adapter}")
    rows = [json.loads(l) for l in open(a.diac)]
    if a.ids_file:
        keep = set(Path(a.ids_file).read_text().split())
        rows = [r for r in rows if r["id"] in keep]
    if a.limit:
        rows = rows[: a.limit]
    voices = json.load(open(a.voices))  # [{name, wav, text_raw, text_catt, text_ft}]
    # balanced rotation: each sentence gets ONE voice (paired across conditions); digit/spelled twins share it
    import re
    order = {}
    for r in rows:
        base = re.sub(r"_[dsn]$", "", r["id"])
        order.setdefault(r["register"], {}).setdefault(base, len(order.get(r["register"], {})))
    def voice_of(r):
        base = re.sub(r"_[dsn]$", "", r["id"])
        return voices[order[r["register"]][base] % len(voices)]["name"]
    tag = a.adapter + ("_refraw" if a.ref_raw else "")
    root = Path(a.out) / tag
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = ad.load(a.device)
    print(f"[{tag}] loaded in {time.time()-t0:.1f}s; {len(rows)} sentences x {len(voices)} voices", flush=True)
    log = open(root / "timing.jsonl", "a")
    n = 0
    for cond in a.conds.split(","):
        for v in voices:
            d = root / cond / v["name"]
            d.mkdir(parents=True, exist_ok=True)
            rt = v["text_raw"] if a.ref_raw else ref_text_for(cond, v)
            for r in rows:
                if not a.all_voices and voice_of(r) != v["name"]:
                    continue
                out = d / f"{r['id']}.wav"
                if out.exists():
                    continue
                text = r[f"cond_{cond}"]
                ts = time.perf_counter()
                try:
                    wav, sr = ad.synth(model, text, v["wav"], rt, a.seed)
                except Exception as e:  # log and continue; failures are reported as such
                    log.write(json.dumps({"id": r["id"], "cond": cond, "voice": v["name"], "error": str(e)[:300]}) + "\n")
                    continue
                ms = (time.perf_counter() - ts) * 1000
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                sf.write(out, wav, sr)
                dur = len(wav) / sr
                log.write(json.dumps({"id": r["id"], "cond": cond, "voice": v["name"], "ms": round(ms, 1),
                                      "audio_s": round(dur, 3), "rtf": round(ms / 1000 / max(dur, 1e-3), 4)}) + "\n")
                n += 1
                if n % 100 == 0:
                    log.flush(); print(f"[{tag}] {n} clips, {time.time()-t0:.0f}s", flush=True)
    log.close()
    print(f"[{tag}] done: {n} new clips in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
