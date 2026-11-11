#!/bin/bash
# Revision (reviewer R1.1): repeat synthesis with seeds 1235 and 1236 on the 56-sentence paired subsample
# (Raw, Full, No-final = cond catt_nocase), all six models; seed 1234 is the main run in out/. Uses the PY_* variables of run_all.sh.
set -eu
cd "$(dirname "$0")"
for SEED in 1235 1236; do
  O=out_seeds/s$SEED
  for m in habibi_sau:F5 habibi_unified:F5 lahgtna:OMNI omnivoice:OMNI xtts:XTTS chatterbox:CB; do
    py=PY_${m#*:}; ${!py} code/s03_synth.py --adapter ${m%%:*} --diac testset/diac.jsonl --voices voices/voices.json --out $O \
      --conds raw,catt_full,catt_nocase --ids-file out_seeds/ids56.txt --seed $SEED
  done
  $PY_TEXT code/s04_channel.py $O --jobs 16
  find $O -name "*.wb.wav" | sort > out_seeds/list_s${SEED}_wb.txt
  $PY_COHERE code/s05_asr.py --backend cohere --list out_seeds/list_s${SEED}_wb.txt --out asr_seeds/cohere_wb_s$SEED.jsonl
done
$PY_TEXT code/r1_seed_analysis.py --root .      # -> results_seeds/seed_summary.json
$PY_TEXT code/r1_extra.py --root .              # -> results_seeds/failure_threshold_sensitivity.json, omni_fix.json
