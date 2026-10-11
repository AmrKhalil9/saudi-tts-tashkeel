"""Stage 2b: add the Fine-Tashkeel (ByT5, MIT) condition `cond_ft_full` to an existing diacritized jsonl.

Same protocol as CATT: only the Arabic words are diacritized and spliced back; per-sentence latency at batch 1.
Usage: python s02b_finetashkeel.py in.jsonl out.jsonl --model models/Fine-Tashkeel --device cuda
"""
import argparse, json, statistics, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from s02_diacritize import AR_WORD, splice  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--model", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    torch.set_num_threads(a.threads)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = T5ForConditionalGeneration.from_pretrained(a.model).to(a.device).eval()

    def diac(text):
        ids = tok(text, return_tensors="pt").input_ids.to(a.device)
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=min(1024, ids.shape[1] * 3 + 16), num_beams=1)
        return tok.decode(out[0], skip_special_tokens=True)

    for _ in range(3):
        diac("مرحبا بكم في خدمة العملاء")
    rows = [json.loads(l) for l in open(a.inp)]
    lat = []
    with open(a.out, "w") as f:
        for r in rows:
            words = AR_WORD.findall(r["text"])
            t0 = time.perf_counter()
            d = diac(" ".join(words)) if words else ""
            if a.device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            lat.append(ms)
            r["cond_ft_full"] = splice(r["text"], d.split())
            r["ft_ms"] = round(ms, 2)
            r["ft_device"] = a.device
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    q = statistics.quantiles(lat, n=20)
    print(f"{len(rows)} rows on {a.device}: median {statistics.median(lat):.1f} ms, p95 {q[18]:.1f} ms; "
          f"alignment mismatches (word kept plain) {splice.mismatch}")


if __name__ == "__main__":
    main()
