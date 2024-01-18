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
    batch_size = sqrt(64)
    epoch = 10
    savepoint = 1000

    model = SpeakerEncoder().train()
    dataset = JVS(batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size, num_workers=cpu_count())
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = GE2ELoss()
    global_step = 0

    for ep in range(epoch):
        bar = tqdm.tqdm(dataloader)
        losses = []
        for data in bar:
            global_step += 1
            model.zero_grad()
            output = []
            for i in range(batch_size):
                output.append(model(data[i]))
            output = torch.stack(output)
            loss = criterion(output)
            loss.backward()
            optimizer.step()
            losses.append(loss)
            bar.set_postfix({"loss": float(loss)})
            if global_step % savepoint == 0:
                bar.write("STEP "+ str(global_step))
                bar.write("LOSS "+ str(float(torch.stack(losses).mean())))
                torch.save(model.state_dict(), f"speaker_encoder_step{str(global_step)}.pth")

def main():
    train_speaker_encoder()

if __name__ == "__main__":
    main()