"""Re-create the four reference voices from the public Habibi benchmark (SWivid/Habibi, config SAU, test split).
The clip id sauNNNN is the row index NNNN of the test parquet. Writes voices/<name>.wav (16 kHz mono).
Usage: python code/get_voices.py  (needs huggingface_hub, pandas, pyarrow, soundfile)"""
import io, json
import pandas as pd, soundfile as sf
from huggingface_hub import hf_hub_download

f = hf_hub_download("SWivid/Habibi", "SAU/test-00000-of-00001.parquet", repo_type="dataset", revision="1d247ff")
df = pd.read_parquet(f)
for v in json.load(open("voices/voices.json")):
    row = df.iloc[int(v["source"].split("sau")[-1])]
    audio = row["audio"]
    data, sr = sf.read(io.BytesIO(audio["bytes"] if isinstance(audio, dict) else audio))
    sf.write(v["wav"], data, sr)
    print(v["name"], v["wav"], sr, round(len(data) / sr, 2), "s")
