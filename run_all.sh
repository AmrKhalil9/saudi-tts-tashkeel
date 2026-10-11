#!/bin/bash
# Full pipeline. Each TTS/ASR system has its own Python environment; point these variables at them.
# PY_TEXT (pandas/num2words/scipy), PY_CATT, PY_FT, PY_F5, PY_OMNI, PY_XTTS, PY_CB, PY_WHISPER, PY_COHERE, PY_OMNIASR, PY_QUALITY
set -eu
cd "$(dirname "$0")"
$PY_TEXT code/s01_build_testset.py                                   # -> testset/testset.jsonl (already included)
$PY_CATT code/s02_diacritize.py testset/testset.jsonl testset/diac.jsonl --catt third_party/catt --device cuda
$PY_FT   code/s02b_finetashkeel.py testset/diac.jsonl testset/diac.jsonl      # adds the FT-full condition
for m in habibi_sau:F5 habibi_unified:F5 lahgtna:OMNI omnivoice:OMNI xtts:XTTS chatterbox:CB; do
  py=PY_${m#*:}; ${!py} code/s03_synth.py --adapter ${m%%:*} --diac testset/diac.jsonl --voices voices/voices.json --out out
done
$PY_TEXT code/s04_channel.py out --jobs 16
for ch in wb g711; do
  mkdir -p lists; find out -name "*.$ch.wav" | sort > lists/out_$ch.txt
  $PY_WHISPER code/s05_asr.py --backend whisper --list lists/out_$ch.txt --out asr/whisper_$ch.jsonl --batch 8
  $PY_COHERE  code/s05_asr.py --backend cohere  --list lists/out_$ch.txt --out asr/cohere_$ch.jsonl
done
$PY_TEXT code/make_omni_sample.py .
$PY_OMNIASR code/s05_asr.py --backend omniasr --list lists/omni_sample.txt --out asr/omniasr_wb.jsonl --batch 8
$PY_QUALITY code/s06_quality.py --list lists/out_wb.txt --voices voices/voices.json --out quality/quality.jsonl
$PY_TEXT code/s07_analyze.py --root . --out-name out
$PY_TEXT code/make_tables.py --root . --out paper_tables
$PY_TEXT code/make_figures.py --root . --out paper_tables
