import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram

class Audio2Mel(MelSpectrogram):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_length = (self.n_fft - self.hop_length) // 2
    
    def forward(self, wav):
        # 1. パディング
        wav = F.pad(wav, [self.pad_length, self.pad_length], mode='reflect')
        # 2. STFT
        spec = self.spectrogram(wav)
        # 3. メルスペクトログラムへ
        mel = self.mel_scale(spec)
        # 4. Log-Melへ
        mel = torch.log(torch.clamp_min(mel, min=1e-5))
        return mel