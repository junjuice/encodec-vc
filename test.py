
import torch
from torch.autograd.grad_mode import no_grad
from torch.utils.data import DataLoader
import tqdm
from data.custom import CustomDataset
from encodec.lora.lora import SpeakerEncoder

with no_grad():
    embedder = SpeakerEncoder(pretrained_path="./speaker_encoder.pth")
    ds = CustomDataset("./datasets/JVS/jvs_ver1/jvs001/parallel100/wav24kHz16bit")
    dl = DataLoader(ds, 4)
    embs = []
    for data in tqdm.tqdm(dl):
        embs.append(embedder(data))
    embs = torch.cat(embs, dim=0)
    print(embs.std(dim=-1))
    torch.save(embs, "test_emb.pt")