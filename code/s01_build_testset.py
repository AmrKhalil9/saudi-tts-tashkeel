"""Stage 1: merge the three registers into testset.jsonl (one row per synthesis input).

Rows: id, register (msa|dialect|entity), subdialect (msa|najdi|hijazi), domain, numform (na|digits|spelled),
entity_types, text (undiacritized, normalized whitespace). Entity sentences appear twice (digits and spelled-out
forms) so the digits-vs-spelled sub-condition is paired on the same content.
"""
import json, re, sys
from pathlib import Path

DIAC = re.compile(r"[ً-ْٰ]")
root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")


def clean(t):
    return re.sub(r"\s+", " ", DIAC.sub("", t)).strip()


rows = []
for i, line in enumerate((root / "msa.txt").read_text().splitlines(), 1):
    domain, text = line.split("|", 1)
    rows.append(dict(id=f"msa{i:03d}", register="msa", subdialect="msa", domain=domain, numform="na",
                     entity_types="", text=clean(text)))
for i, line in enumerate((root / "dialect.txt").read_text().splitlines(), 1):
    sub, domain, text = line.split("|", 2)
    rows.append(dict(id=f"dia{i:03d}", register="dialect", subdialect=sub, domain=domain, numform="na",
                     entity_types="", text=clean(text)))
for line in (root / "entities.jsonl").read_text().splitlines():
    e = json.loads(line)
    forms = ("digits", "spelled") if e["text_digits"] != e["text_spelled"] else ("na",)
    for form in forms:
        src = e["text_digits"] if form == "na" else e[f"text_{form}"]
        rows.append(dict(id=f"{e['id']}_{form[0]}", register="entity", subdialect=e["frame"], domain="mixed",
                         numform=form, entity_types=e["entity_types"], text=clean(src)))

ids = [r["id"] for r in rows]
assert len(ids) == len(set(ids)), "duplicate ids"
with open(root / "testset.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(rows), "rows", Counter((r["register"], r["subdialect"], r["numform"]) for r in rows))
