"""Stage 4: channel simulation. For every synthesized wav, write three versions next to it:
   <name>.wb.wav    16 kHz mono PCM (wideband reference condition, also the ASR input rate)
   <name>.g711.wav  8 kHz G.711 mu-law encode/decode round trip, resampled back to 16 kHz for ASR
   <name>.amr.wav   AMR-NB 12.2 kbit/s round trip (only if ffmpeg has libopencore_amrnb), back to 16 kHz
Usage: python s04_channel.py <dir-with-wavs> [--jobs 16]
"""
import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def ff(*args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def has_amr():
    out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "libopencore_amrnb" in out


def one(wav, amr):
    base = wav.with_suffix("")
    if Path(f"{base}.wb.wav").exists() and Path(f"{base}.g711.wav").exists() and not amr:
        return  # already done (resumable; also keeps files stable while ASR may be reading them)
    ff("-i", str(wav), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", f"{base}.wb.wav")
    ff("-i", str(wav), "-ac", "1", "-ar", "8000", "-c:a", "pcm_mulaw", f"{base}.g711.tmp.wav")
    ff("-i", f"{base}.g711.tmp.wav", "-ar", "16000", "-c:a", "pcm_s16le", f"{base}.g711.wav")
    Path(f"{base}.g711.tmp.wav").unlink()
    if amr:
        ff("-i", str(wav), "-ac", "1", "-ar", "8000", "-c:a", "libopencore_amrnb", "-b:a", "12.2k", f"{base}.amr.tmp.amr")
        ff("-i", f"{base}.amr.tmp.amr", "-ar", "16000", "-c:a", "pcm_s16le", f"{base}.amr.wav")
        Path(f"{base}.amr.tmp.amr").unlink()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    amr = has_amr()
    wavs = [p for p in Path(a.dir).rglob("*.wav") if p.suffixes[-2:-1] not in ([".wb"], [".g711"], [".amr"])]
    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(lambda w: one(w, amr), wavs))
    print(f"{len(wavs)} files; AMR-NB {'on' if amr else 'unavailable'}")


if __name__ == "__main__":
    main()
