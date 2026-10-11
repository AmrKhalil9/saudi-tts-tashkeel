"""Shared OmniVoice adapter (base OmniVoice and the Lahgtna-OmniVoice-v2 fine-tune share the architecture)."""
import torch
from omnivoice import OmniVoice


def make(path, device):
    return OmniVoice.from_pretrained(path, device_map=device, dtype=torch.float16 if device == "cuda" else torch.float32)


def synth_lang(model, text, ref_wav, ref_text, seed, language):
    torch.manual_seed(seed)
    w = model.generate(text=text, language=language, ref_text=ref_text, ref_audio=ref_wav)[0]
    return w.squeeze(), model.sampling_rate
