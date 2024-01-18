import torch
from torch import nn

from encodec.modules.seanet import SEANetResnetBlock
from encodec.modules.conv import SConv1d
from encodec.model import EncodecModel

from m2d.m2d.runtime_audio import RuntimeM2D

from enum import Enum

import requests
import tqdm
import os


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

class ZeroModule(nn.Module):

    def __init__(self, trainable=False, eps=1e-2):
        super().__init__()
        if trainable:
            self.bias = nn.Parameter(torch.Tensor([eps]), requires_grad=True)
        else:
            self.bias = 0

    def forward(self, x):
        return self.bias


class EncodecLoRA(nn.Module):

    class OnlyMode(Enum):
        none = 0
        encoder = 1
        decoder = 2
        zero = 3

    def __init__(self, original: EncodecModel, rank: int, kernel_size: int, strength: float=1.0, only: OnlyMode=OnlyMode.none, trainable_zero=False):
        super().__init__()
        self.strength = strength
        encoder_only = only == self.OnlyMode.encoder
        decoder_only = only == self.OnlyMode.decoder
        zero_only = only == self.OnlyMode.zero

        lora_encoder = []
        for module in original.encoder.model:
            if isinstance(module, SEANetResnetBlock) and not decoder_only and not zero_only:
                dim = module.dim
                print(f"Making lora layer in encoder (dim={dim})")
                lora_encoder.append(nn.Sequential(
                    SConv1d(dim, rank, kernel_size=kernel_size), SConv1d(rank, dim, kernel_size=kernel_size)
                ))
            else:
                lora_encoder.append(ZeroModule(trainable=(trainable_zero and not decoder_only)))
        self.lora_encoder = nn.ModuleList(lora_encoder)

        lora_decoder = []
        for module in original.decoder.model:
            if isinstance(module, SEANetResnetBlock) and not encoder_only and not zero_only:
                dim = module.dim
                print(f"Making lora layer in decoder (dim={dim})")
                lora_decoder.append(nn.Sequential(
                    SConv1d(dim, rank, kernel_size=kernel_size), SConv1d(rank, dim, kernel_size=kernel_size)
                ))
            else:
                lora_decoder.append(ZeroModule(trainable=(trainable_zero and not encoder_only)))
        self.lora_decoder = nn.ModuleList(lora_decoder)

        self.modify_forward(original)
        self.summarize()
        original.eval()

    def summarize(self):
        trainable = 0
        frozen = 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        print(" |=========================")
        print(" | Parameters in LoRA")
        print(" | Trainable", trainable)
        print(" | Frozen", frozen)
        print(" |=========================")
        
    def modify_forward(self, original: EncodecModel):
        def modified_encode(x: torch.Tensor):
            for i, (orig, lora) in enumerate(zip(original.encoder.model, self.lora_encoder)):
                x = orig(x) + lora(x) * self.strength
            return x
            
        def modified_decode(x: torch.Tensor):
            for i, (orig, lora) in enumerate(zip(original.decoder.model, self.lora_decoder)):
                x = orig(x) + lora(x) * self.strength
            return x
            
        original.encoder.forward = modified_encode
        original.decoder.forward = modified_decode

    def undo_forward(self, model: EncodecModel):
        encoder = []
        for module in model.encoder.model:
            encoder.append(module)
        decoder = []
        for module in model.decoder.model:
            decoder.append(module)
        model.encoder = nn.Sequential(*encoder)
        model.decoder = nn.Sequential(*decoder)


class SpeakerEncoder(nn.Module):
    def __init__(self, m2d_path="m2d_vit_base-80x608p16x16-221006-mr6/pruned.pth"):
        super().__init__()
        if not os.path.isfile(m2d_path):
            download("https://huggingface.co/junjuice0/test/resolve/main/pruned.pth", m2d_path)
        self.m2d = RuntimeM2D(weight_file=m2d_path, encoder_only=True).eval().requires_grad_(False)
        self.lstm = nn.LSTM(input_size=self.m2d.cfg.feature_d, hidden_size=1024, num_layers=4, batch_first=True)
        self.state_dict = self.lstm.state_dict
        self.summarize()

    def summarize(self):
        trainable = 0
        frozen = 0
        for p in self.m2d.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        for p in self.lstm.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        print(" |=========================")
        print(" | Parameters in speaker encoder")
        print(" | Trainable", trainable)
        print(" | Frozen", frozen)
        print(" |=========================")
    
    def forward(self, x):
        x = self.m2d.encode(x)
        x, _ = self.lstm(x)
        return x[:, -1, :]
    
    def train(self, mode: bool = True):
        self.lstm = self.lstm.train(mode)
        self.m2d = self.m2d.train(False)
        return self
    
    def parameters(self, recurse: bool = True):
        return self.lstm.parameters(recurse=recurse)
    
    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.lstm = self.lstm.load_state_dict(state_dict=state_dict, strict=strict, assign=assign)
        return self
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        self.lstm = self.lstm._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        return self