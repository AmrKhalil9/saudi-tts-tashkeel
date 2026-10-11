from adapters._omni import make, synth_lang
def load(device):
    return make("models/OmniVoice", device)
def synth(model, text, ref_wav, ref_text, seed):
    return synth_lang(model, text, ref_wav, ref_text, seed, "ar")
