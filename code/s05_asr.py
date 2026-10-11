"""Stage 5: transcribe audio with one ASR backend. Resumable: skips paths already in the output jsonl.

Backends (each run in its own venv):
  whisper  openai/whisper-large-v3 (transformers, fp16, language=ar, task=transcribe, greedy)
  cohere   CohereLabs/cohere-transcribe-arabic-07-2026 (public base checkpoint, bf16, greedy, max 256 tokens)
  omniasr  facebook/omniASR-LLM-7B (fairseq2, bf16), lang arb_Arab
Usage: python s05_asr.py --backend whisper --list files.txt --out asr_whisper.jsonl [--batch 16]
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import soundfile as sf


def load_16k(p):
    a, sr = sf.read(p, dtype="float32")
    a = a if a.ndim == 1 else a.mean(1)
    if sr != 16000:
        import soxr
        a = soxr.resample(a, sr, 16000).astype("float32")
    return a


def whisper_backend(batch):
    import torch
    from transformers import pipeline
    pipe = pipeline("automatic-speech-recognition", model="models/whisper-large-v3",
                    torch_dtype=torch.float16, device="cuda:0")
    kw = dict(generate_kwargs={"language": "ar", "task": "transcribe", "num_beams": 1}, batch_size=batch,
              chunk_length_s=30)  # some TTS outputs run past 30 s (babble after the sentence)
    return lambda paths: [o["text"].strip() for o in pipe([load_16k(p) for p in paths], **kw)]


def cohere_backend(batch):
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    mid = os.environ.get("COHERE_DIR", "models/cohere-transcribe-arabic-07-2026")
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(mid, torch_dtype=torch.bfloat16).to("cuda").eval()

    def run(paths):
        outs = []
        for p in paths:
            inp = proc(load_16k(p), sampling_rate=16000, return_tensors="pt", language="ar").to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=256, num_beams=1, repetition_penalty=1.1)
            outs.append(proc.batch_decode(g, skip_special_tokens=True)[0].strip())
        return outs
    return run


def omniasr_backend(batch):
    import torch
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    pipe = ASRInferencePipeline(model_card="omniASR_LLM_7B_local", device="cuda", dtype=torch.bfloat16)
    lang = os.environ.get("OMNI_LANG", "arb_Arab")
    return lambda paths: [t.strip() for t in pipe.transcribe(paths, lang=[lang] * len(paths), batch_size=batch)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["whisper", "cohere", "omniasr"])
    ap.add_argument("--list", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    import fcntl  # one writer per output file: a parallel run waits, then resumes past what is done
    lock = open(a.out + ".lock", "w"); fcntl.flock(lock, fcntl.LOCK_EX)
    paths = [l.strip() for l in open(a.list) if l.strip()]
    done = set()
    if Path(a.out).exists():
        done = {j["path"] for j in map(json.loads, open(a.out)) if not j["hyp"].startswith("<<ERROR")}
    todo = [p for p in paths if p not in done]
    print(f"[{a.backend}] {len(todo)} to do ({len(done)} done)", flush=True)
    if not todo:
        return
    t0 = time.time()
    run = {"whisper": whisper_backend, "cohere": cohere_backend, "omniasr": omniasr_backend}[a.backend](a.batch)
    print(f"[{a.backend}] loaded in {time.time()-t0:.0f}s", flush=True)
    with open(a.out, "a") as f:
        for i in range(0, len(todo), a.batch):
            chunk = todo[i:i + a.batch]
            try:
                hyps = run(chunk)
            except Exception as e:
                hyps = [f"<<ERROR {str(e)[:120]}>>"] * len(chunk)
            for p, h in zip(chunk, hyps):
                f.write(json.dumps({"path": p, "hyp": h}, ensure_ascii=False) + "\n")
            f.flush()
            if (i // a.batch) % 20 == 0:
                print(f"[{a.backend}] {i+len(chunk)}/{len(todo)} {time.time()-t0:.0f}s", flush=True)
    print(f"[{a.backend}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
