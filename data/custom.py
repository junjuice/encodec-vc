import torch
from torch.utils.data import Dataset

import torchaudio

import glob
import random
import tqdm
import requests


def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm.tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


class CustomDataset(Dataset):
    def __init__(self, dir: str, length=3, sr=24000, size=None):
        dir = dir.removesuffix("/")
        self.length = length
        self.sr = sr
        self.datas = glob.glob(dir+"/*.wav")
        self.size = size if size else len(self.datas)
        print("Custom dataset: {} files found".format(len(self.datas)))

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
        return self.size
    
    def __getitem__(self, index):
        index = index % len(self.datas)
        x, sr = torchaudio.load(self.datas[index])
        return self.transform(x, sr)