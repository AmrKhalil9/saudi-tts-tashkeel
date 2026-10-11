"""Stage 2: produce the text conditions and time the diacritizer.

Conditions (column = what goes into TTS):
  raw          undiacritized text
  catt_full    CATT (encoder-only, Apache-2.0) full diacritization
  catt_nocase  CATT with the word-final short vowel / tanween / sukun removed (shadda kept)
  shadda_only  CATT's shadda marks only

CATT only sees Arabic letters, so we diacritize the Arabic words and splice them back into the original
sentence; digits, Latin words and punctuation are kept verbatim. Latency is measured per sentence on the
device given (cuda or cpu), after a warm-up, batch size 1 (the real-time agent case).

Usage (CATT repo on sys.path):  python s02_diacritize.py testset.jsonl out.jsonl --catt /path/to/catt --device cuda
"""
import argparse, json, re, statistics, sys, time

AR_WORD = re.compile(r"[ء-يـً-ْٰ]+")
HARAKAT = "ًٌٍَُِْ"  # tanween x3, fatha, damma, kasra, sukun
SHADDA = "ّ"
DIAC = re.compile(r"[ً-ْٰ]")


def strip_case(word):
    """Remove the final short vowel/tanween/sukun on the last letter, keep shadda."""
    i = len(word)
    while i > 0 and word[i - 1] in HARAKAT + SHADDA:
        i -= 1
    tail = word[i:]
    return word[:i] + "".join(c for c in tail if c == SHADDA)


def shadda_only(word):
    return "".join(c for c in word if not DIAC.match(c) or c == SHADDA)


def splice(original, diac_words):
    """Replace the Arabic words of `original` in order with `diac_words` (undiacritized match required)."""
    out, k = [], 0
    pos = 0
    for m in AR_WORD.finditer(original):
        out.append(original[pos:m.start()])
        w = diac_words[k] if k < len(diac_words) else m.group(0)
        if DIAC.sub("", w).replace("ـ", "") != DIAC.sub("", m.group(0)).replace("ـ", ""):
            w = m.group(0)  # alignment mismatch: keep the original word undiacritized (counted below)
            splice.mismatch += 1
        out.append(w)
        k += 1
        pos = m.end()
    out.append(original[pos:])
    return "".join(out)


splice.mismatch = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--catt", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    sys.path.insert(0, a.catt)
    import torch
    torch.set_num_threads(a.threads)
    from eo_pl import TashkeelModel
    from tashkeel_tokenizer import TashkeelTokenizer
    tok = TashkeelTokenizer()
    model = TashkeelModel(tok, max_seq_len=1024, n_layers=6, learnable_pos_emb=False)
    model.load_state_dict(torch.load(f"{a.catt}/models/best_eo_mlm_ns_epoch_193.pt", map_location=a.device))
    model.eval().to(a.device)

    rows = [json.loads(l) for l in open(a.inp)]
    for _ in range(5):
        model.do_tashkeel_batch(["مرحبا بكم في خدمة العملاء"], 1, False)
    lat = []
    with open(a.out, "w") as f:
        for r in rows:
            words = AR_WORD.findall(r["text"])
            t0 = time.perf_counter()
            with torch.inference_mode():
                d = model.do_tashkeel_batch([" ".join(words)], 1, False)[0] if words else ""
            if a.device == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            lat.append(ms)
            dw = d.split()
            full = splice(r["text"], dw)
            r.update(cond_raw=r["text"], cond_catt_full=full,
                     cond_catt_nocase=splice(r["text"], [strip_case(w) for w in dw]),
                     cond_shadda_only=splice(r["text"], [shadda_only(w) for w in dw]),
                     catt_ms=round(ms, 2), catt_device=a.device)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    q = statistics.quantiles(lat, n=20)
    print(f"{len(rows)} rows on {a.device}: median {statistics.median(lat):.1f} ms, p95 {q[18]:.1f} ms; "
          f"alignment mismatches {splice.mismatch}")


if __name__ == "__main__":
    main()
