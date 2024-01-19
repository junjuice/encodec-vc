from math import sqrt
from os import cpu_count
from torch import nn, optim
import torch
from torch.utils.data.dataloader import DataLoader
import tqdm
from data.jvs import JVS
from encodec.lora.lora import SpeakerEncoder
from encodec.lora.loss import GE2ELoss


def train_speaker_encoder():
    b = 4
    batch_size = int(sqrt(b))
    epoch = 10
    savepoint = 10
    n_workers = 2 #cpu_count()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SpeakerEncoder().to(device).train()
    dataset = JVS(batch_size=batch_size)
    criterion = GE2ELoss().to(device)
    dataloader = DataLoader(dataset, batch_size, num_workers=n_workers)
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


def main():
    train_speaker_encoder()

if __name__ == "__main__":
    main()