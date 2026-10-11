"""XTTS-v2 (CPML, non-commercial; used for research under the licence). Conditions on reference audio only."""
import os, torch
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
D = "models/XTTS-v2"
_lat = {}
def load(device):
    c = XttsConfig(); c.load_json(D + "/config.json")
    m = Xtts.init_from_config(c); m.load_checkpoint(c, checkpoint_dir=D, use_deepspeed=False)
    m.to(device); m.eval()
    return m
def synth(model, text, ref_wav, ref_text, seed):
    if ref_wav not in _lat:
        _lat[ref_wav] = model.get_conditioning_latents(audio_path=[ref_wav])
    g, s = _lat[ref_wav]
    torch.manual_seed(seed)
    o = model.inference(text, "ar", g, s)
    return o["wav"], 24000
