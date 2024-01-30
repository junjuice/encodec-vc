from encodec.modules.seanet import SEANetDecoder
from encodec.customize.hubert import Hubert

import torch
from torch import nn


class TestModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.hubert = Hubert(mask=False)
        self.decoder = SEANetDecoder(dimension=768)

    def forward(self, x):
        x, _ = self.hubert.encode(x)
        x = x.transpose(1, 2)
        x = self.decoder(x)
        return x