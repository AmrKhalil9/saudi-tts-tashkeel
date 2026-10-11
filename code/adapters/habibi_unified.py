from adapters._f5 import make, synth  # noqa: F401
def load(device):
    return make("models/habibi/Unified", device)
