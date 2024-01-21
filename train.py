from math import sqrt
from os import cpu_count, makedirs
import os
import sys
from torch import nn, optim
import torch
from torch.nn.modules.loss import MSELoss
from torch.utils.data.dataloader import DataLoader
import torchaudio
import tqdm
from data.custom import CustomDataset
from data.jvs import JVS
from encodec.lora.lora import EncodecLoRA, SpeakerEncoder
from encodec.lora.loss import Discriminator, GE2ELoss, SpeakerEncoderLoss
from encodec.model import EncodecModel
from encodec.msstftd import MultiScaleSTFTDiscriminator

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.hub.set_dir("./cache")
n_workers = cpu_count()


def train_speaker_encoder():
    b = 4
    batch_size = int(sqrt(b))
    epoch = 10
    savepoint = 1000

    dataset = JVS(batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size, num_workers=n_workers)

    model = SpeakerEncoder().to(device).train()
    criterion = GE2ELoss().to(device)
    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=1e-5)
    
    global_step = 1

    for ep in range(epoch):
        bar = tqdm.tqdm(dataloader)
        losses = []
        for data in bar:
            data = data.to(device)
            if global_step % savepoint == 0:
                test = [data[0, 0, :], data[0, 1, :], data[1, 0, :], data[1, 1, :]]
                test = torch.stack(test)
                with torch.no_grad():
                    out = model(test)
                same_cos_sim = (torch.cosine_similarity(out[0], out[1], dim=0) + torch.cosine_similarity(out[2], out[3], dim=0))/2
                diff_cos_sim = (torch.cosine_similarity(out[0], out[2], dim=0) + torch.cosine_similarity(out[1], out[3], dim=0))/2
                bar.write("===========================================================================")
                bar.write("STEP "+ str(global_step))
                bar.write("LOSS "+ str(float(torch.stack(losses).mean())))
                bar.write("SIMS (SAME " + str(float(same_cos_sim)) + ") (DIFF " + str(float(diff_cos_sim)) + ")")
                losses = []
                torch.save(model.state_dict(), f"models/speaker_encoder_step{str(global_step)}.pth")
                
            model.zero_grad()
            data = torch.reshape(data, (batch_size**2, data.shape[-1]))
            output = model(data)
            output = torch.reshape(output, (batch_size, batch_size, output.shape[-1]))
            loss = criterion(output)
            loss.backward()
            criterion.w.grad *= 0.01
            criterion.b.grad *= 0.01
            optimizer.step()
            losses.append(loss)
            bar.set_postfix({"loss": float(loss)})
            global_step += 1

@torch.no_grad()
def get_embedding():
    batch_size = 8
    epoch = 1
    path = "./embeddings/ange.pt"
    path_all = "./embeddings/ange_all.pt"
    target_dir = "C:/Users/Administrator/Downloads/ange"

    speaker_encoder = SpeakerEncoder(pretrained_path="./models/speaker_encoder/speaker_encoder.pth")
    target_dataset = CustomDataset(target_dir)
    target_dataloader = DataLoader(target_dataset, batch_size=batch_size, num_workers=n_workers)
    embeddings = []

    bar_epoch = tqdm.tqdm(range(epoch), position=0)
    bar_batch = tqdm.tqdm(target_dataloader, position=1)
    for i in bar_epoch:
        for data in bar_batch:
            embeddings.append(speaker_encoder(data))
    embeddings = torch.cat(embeddings, dim=0)
    print("STD is", float(embeddings.std(0).mean()))
    torch.save(embeddings, path_all)
    torch.save(embeddings.mean(0), path)
    print("Saved", path, "and", path_all)


def train_lora():
    rank = 16
    kernel_size = 11
    batch_size = 4
    epoch = 10
    checkpoint_step = 100
    target_embedding_path = "./embeddings/ange.pt"
    target_dataset_path = "./datasets/ange"

    dataset = JVS()
    dataloader = DataLoader(dataset, batch_size//2, num_workers=n_workers//2, shuffle=True)
    
    target_dataset = CustomDataset(target_dataset_path, size=len(dataset))
    target_dataloader = DataLoader(target_dataset, batch_size//2, num_workers=n_workers//2, shuffle=True)

    embedding = torch.load(target_embedding_path)
    speaker_encoder = SpeakerEncoder(pretrained_path="./models/speaker_encoder/speaker_encoder.pth", m2d_grad=True).to(device)
    criterion1 = SpeakerEncoderLoss(speaker_encoder, embedding).to(device)
    criterion2 = MSELoss()
    model = EncodecModel.encodec_model_24khz().to(device)
    lora = EncodecLoRA(model, rank=rank, kernel_size=kernel_size).to(device)
    optimizer = optim.AdamW(lora.parameters(), lr=1e-4)

    global_step = 0
    losses = []

    for i in range(epoch):
        criterion1.eval()
        model.eval()
        lora.train()

        bar = tqdm.tqdm(zip(dataloader, target_dataloader), desc="EPOCH {}".format(i), total=len(dataloader))
        for data1, data2 in bar:
            
            data1 = data1.to(device)
            if len(data1.shape) == 2:
                data1 = data1[:, None, :]
                
            data2 = data2.to(device)
            if len(data2.shape) == 2:
                data2 = data2[:, None, :]

            output1 = model(data1)
            output2 = model(data2)
            encoder_loss = criterion1(output1)
            mse_loss = criterion2(output2, data2) * 100
            loss = encoder_loss + mse_loss
            lora.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss)
            bar.set_postfix({"total_loss": float(loss), "encoder_loss": float(encoder_loss), "mse_loss": float(mse_loss)})
            global_step += 1
            if global_step % checkpoint_step == 0:
                makedirs("./logs/{:0>8}/wav/".format(global_step), exist_ok=True)
                makedirs("./logs/{:0>8}/checkpoints/".format(global_step), exist_ok=True)
                torch.save(criterion2.state_dict(), "./logs/{:0>8}/checkpoints/discriminator.pth".format(global_step))
                torch.save(lora.state_dict(), "./logs/{:0>8}/checkpoints/lora.pth".format(global_step))
                for j in range(batch_size):
                    torchaudio.save("./logs/{:0>8}/wav/{}_original.wav".format(global_step, j), data1[j].detach(), dataset.sr)
                    torchaudio.save("./logs/{:0>8}/wav/{}_output.wav".format(global_step, j), output1[j].detach(), dataset.sr)
                bar.write("===========================================================================")
                bar.write("STEP "+ str(global_step))
                bar.write("LOSS "+ str(float(torch.stack(losses).mean())))
                losses = []

        
def main():
    mode = sys.argv[1]
    if mode == "se":
        train_speaker_encoder()
    elif mode == "emb":
        get_embedding()
    elif mode == "lora":
        train_lora()
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    main()