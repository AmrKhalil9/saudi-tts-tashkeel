"""Chatterbox Multilingual (MIT). Conditions on reference audio only (no reference transcript). Outputs carry
Resemble's Perth watermark (disclosed in the paper; inaudible, applied to every condition alike)."""
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
CK = "models/chatterbox"
def load(device):
    return ChatterboxMultilingualTTS.from_local(CK, device)
def synth(model, text, ref_wav, ref_text, seed):
    torch.manual_seed(seed)
    wav = model.generate(text, language_id="ar", audio_prompt_path=ref_wav)
    return wav.squeeze().cpu().numpy(), model.sr
