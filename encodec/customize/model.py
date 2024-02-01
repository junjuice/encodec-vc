from encodec.modules.seanet import SEANetDecoder, SEANetEncoder
from encodec.customize.hubert import Hubert

import torch
from torch import nn

import time

class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SEANetEncoder(dimension=512, activation="GELU", activation_params={})
        self.hubert = Hubert(mask=False)
        self.hubert.feature_extractor = self.encoder
        self.decoder = SEANetDecoder(dimension=768)
        
    def forward(self, x):
        x, _ = self.hubert.encode(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        return x