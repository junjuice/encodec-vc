from encodec.model import EncodecModel
from encodec.lora.lora import EncodecLoRA
import torchaudio
import torch

torch.hub.set_dir("./cache")
model = EncodecModel.encodec_model_24khz()
lora = EncodecLoRA(model, 8, 5, strength=0.5, trainable_zero=True)

wave, sr = torchaudio.load("test.wav")
out = model(wave.unsqueeze(1))
torchaudio.save("test_out.wav", src=out[0].detach(), sample_rate=sr)