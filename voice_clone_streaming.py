import os
import time
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download

print("Downloading model...")
checkpoint_path = snapshot_download("coqui/XTTS-v2")
config_path = f"{checkpoint_path}/config.json"

print("Loading model...")
config = XttsConfig()
config.load_json(config_path )
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=checkpoint_path , use_deepspeed=True)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["samples/sample_3.wav"])

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
"It took me quite a long time to develop a voice and now that I have it I am not going to be silent. ",
"en",
gpt_cond_latent,
speaker_embedding,
enable_text_splitting=True,
stream_chunk_size=120
# speed=2.0
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
        wav_chuncks.append(chunk)
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    # wav_chuncks.append(chunk)
# print("Total chunks {}".format(wav_chuncks))
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)
# torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)