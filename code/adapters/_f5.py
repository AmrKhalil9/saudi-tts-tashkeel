"""Shared F5-TTS (Habibi) adapter: cross-sentence prompting with the reference audio and its transcript."""
import torch
from f5_tts.api import F5TTS


def make(ckpt_dir, device):
    return F5TTS(model="F5TTS_v1_Base", ckpt_file=f"{ckpt_dir}/model_200000.safetensors",
                 vocab_file=f"{ckpt_dir}/vocab.txt", device=device)


def synth(model, text, ref_wav, ref_text, seed):
    wav, sr, _ = model.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=text, seed=seed,
                             show_info=lambda *a: None, progress=None, remove_silence=False)
    return wav, sr
