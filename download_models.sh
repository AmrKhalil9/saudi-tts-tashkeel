#!/bin/bash
# Fetch every model at the exact revision used in the paper into ./models (needs `huggingface-cli`).
set -e
dl() { huggingface-cli download "$1" --revision "$2" --local-dir "models/$3" ${4:+--include "$4"}; }
dl SWivid/Habibi-TTS 3ad11a1 habibi
dl k2-fsa/OmniVoice c5fdb5c OmniVoice
dl oddadmix/lahgtna-omnivoice-v2 55c38a6 lahgtna-omnivoice-v2
dl ResembleAI/chatterbox 5bb1f6e chatterbox
dl coqui/XTTS-v2 6c2b0d7 XTTS-v2            # CPML licence: non-commercial research use only
dl basharalrfooh/Fine-Tashkeel 5807fb0 Fine-Tashkeel
dl openai/whisper-large-v3 06f233f whisper-large-v3
dl CohereLabs/cohere-transcribe-arabic-07-2026 c3e911b cohere-transcribe-arabic-07-2026   # gated: accept terms first
dl facebook/omniASR-LLM-7B 4b0bada omniASR-LLM-7B
dl sarulab-speech/UTMOSv2 506474f UTMOSv2
dl speechbrain/spkrec-ecapa-voxceleb main ecapa
mkdir -p third_party && git clone https://github.com/abjadai/catt third_party/catt && git -C third_party/catt checkout 8d53304
git clone https://github.com/microsoft/DNS-Challenge /tmp/dns && git -C /tmp/dns checkout 591184a && mkdir -p models/DNSMOS \
  && cp /tmp/dns/DNSMOS/DNSMOS/sig_bak_ovr.onnx models/DNSMOS/
echo "Note: the OmniVoice audio tokenizer must be reachable from models/lahgtna-omnivoice-v2 (symlink models/OmniVoice/audio_tokenizer)."
