import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio import transforms
import glob
import random
import requests
import tqdm
import os
import shutil
import csv


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


def read_csv(path):
    x = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            x[row[0]] = [float(sim) for sim in row[1:]]
    return x


class JVS(Dataset):
    def __init__(self, batch_size: int=None, dir="./datasets", size=1e5, length=3, sr=24000, uttr_per_batch=None, random_length=False, mode="normal"):
        dir = dir.removesuffix("/")
        assert mode in ["normal", "speaker_encoder", "denoise"], NotImplementedError
        self.mode = mode
        self.batch_size = batch_size
        self.size = int(size)
        self.length = length
        self.max_length = length
        self.min_length = 1
        self.sr = sr
        self.resamp = None
        self.uttr_per_batch = uttr_per_batch if uttr_per_batch else batch_size
        self.random_length = random_length
        self.get_history = [None]
        if not os.path.isfile(dir+"/JVS/jvs_ver1/README.txt"):
            os.makedirs(dir+"/JVS", exist_ok=True)
            download("https://huggingface.co/junjuice0/test/resolve/main/jvs_ver1.zip", fname=dir+"/JVS/data.zip")
            shutil.unpack_archive(dir+"/JVS/data.zip", extract_dir=dir+"/JVS/")
        self.path = path = dir+"/JVS/jvs_ver1/"
        self.datas = []
        for i in range(1, 101):
            self.datas.append(glob.glob(self.path+"jvs{:0>3}/parallel100/wav24kHz16bit/*.wav".format(i))
                              +glob.glob(self.path+"jvs{:0>3}/nonpara30/wav24kHz16bit/*.wav".format(i)))
        if self.mode == "denoise":
            effect = ",".join(
                [
                    "aecho=in_gain=0.8:out_gain=0.9:delays=200:decays=0.3|delays=400:decays=0.3"
                ],
            )
            self.noise = torchaudio.io.AudioEffector(effect=effect)
        
    def transform(self, x: torch.Tensor, sr: int):
        if sr != self.sr:
            if self.resamp is None:
                self.resamp = torchaudio.transforms.Resample(sr, self.sr)
            x = self.resamp(x)
        if len(x.shape) == 2:
            x = x[0]
        steps = int(self.length * self.sr)
        if x.shape[-1] < steps:
            x = torch.nn.functional.pad(x, (0, steps - x.shape[-1]), value=0.)
            return x
        L = int(x.shape[0])
        start = random.randint(0, L - steps - 1)
        return x[start:start+steps]
    
    def get_speaker_and_idx(self, idx):
        speaker = 0
        new_idx = idx
        while True:
           new_idx -= len(self.datas[speaker])
           if new_idx < 0:
               break
           speaker += 1
           idx = new_idx
           if speaker == len(self.datas):
               speaker = 0
        return speaker, idx

    def __len__(self):
        return self.size
    
    def get_wave(self, speaker, idx):
        wave, sr = torchaudio.load(self.datas[speaker][idx])
        wave = self.transform(wave, sr)
        return wave

    def get_speaker_encoder(self, idx):
        x = []
        speaker = None
        while speaker in self.get_history:
            speaker = random.randint(0, len(self.datas)-1)
        for i in range(self.uttr_per_batch):
            index = random.randint(0, len(self.datas[speaker])-1)
            x.append(self.get_wave(speaker, index))
        x = torch.stack(x)
        self.get_history.append(speaker)
        if self.batch_size < len(self.get_history):
            self.get_history = [None]
            if self.random_length:
                self.length = random.random() * (self.max_length - self.min_length) + self.min_length
        return x
    
    def get_denoise(self, idx):
        x = self.get_normal(idx).squeeze().unsqueeze(-1)
        y = self.noise.apply(x, sample_rate=self.sr)
        x, y = x.transpose(0, 1), y.transpose(0, 1)
        x = torch.nn.functional.pad(x, (0, (y.shape[-1]-x.shape[-1])), value=0)
        return x, y

    def get_normal(self, idx):
        x = self.get_wave(*self.get_speaker_and_idx(idx))
        return x
    
    def __getitem__(self, idx):
        if self.mode == "normal":
            x = self.get_normal(idx)
        elif self.mode == "speaker_encoder":
            x = self.get_speaker_encoder(idx)
        elif self.mode == "denoise":
            x = self.get_denoise(idx)
        return x