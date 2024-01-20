import torch
from torch.utils.data import Dataset

import torchaudio

import glob
import random


class CustomDataset(Dataset):
    def __init__(self, dir: str, length=3, sr=24000):
        dir = dir.removesuffix("/")
        self.length = length
        self.sr = sr
        self.datas = glob.glob(dir+"/*.wav")

    def transform(self, x: torch.Tensor, sr: int):
        if sr != self.sr:
            x = torchaudio.transforms.Resample(sr, self.sr)(x)
        if len(x.shape) == 2:
            x = x[0]
        steps = int(self.length * self.sr)
        if x.shape[0] < steps:
            x = torch.nn.functional.pad(x, (0, steps - x.shape[-1]), value=0.)
        L = int(x.shape[0])
        if (L - steps) == 0:
            return x
        start = random.randint(0, L - steps - 1)
        return x[start:start+steps]
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        x, sr = torchaudio.load(self.datas[index])
        return self.transform(x, sr)