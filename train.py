from math import sqrt
from os import cpu_count, makedirs
import os
import sys
import logging
from venv import create
import torio
from torch import nn, optim
import torch
from torch.nn.modules.loss import MSELoss
from torch.utils.data.dataloader import DataLoader
import torchaudio
import tqdm
from data.custom import CustomDataset
from data.jvs import JVS
from encodec.model import EncodecModel
from encodec.msstftd import MultiScaleSTFTDiscriminator
from audio_to_mel import Audio2Mel

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.hub.set_dir("./cache")
n_workers = cpu_count()

def total_loss(fmap_real, logits_fake, fmap_fake, input_wav, output_wav, sample_rate=24000):
    """This function is used to compute the total loss of the encodec generator.
        Loss = \lambda_t * L_t + \lambda_f * L_f + \lambda_g * L_g + \lambda_feat * L_feat
        L_t: time domain loss | L_f: frequency domain loss | L_g: generator loss | L_feat: feature loss
        \lambda_t = 0.1       | \lambda_f = 1              | \lambda_g = 3       | \lambda_feat = 3
    Args:
        fmap_real (list): fmap_real is the output of the discriminator when the input is the real audio. 
            len(fmap_real) = len(fmap_fake) = disc.num_discriminators = 3
        logits_fake (_type_): logits_fake is the list of every sub discriminator output of the Multi discriminator 
            logits_fake, _ = disc_model(model(input_wav)[0].detach())
        fmap_fake (_type_): fmap_fake is the output of the discriminator when the input is the fake audio.
            fmap_fake = disc_model(model(input_wav)[0]) = disc_model(reconstructed_audio)
        input_wav (tensor): input_wav is the input audio of the generator (GT audio)
        output_wav (tensor): output_wav is the output of the generator (output = model(input_wav)[0])
        sample_rate (int, optional): Defaults to 24000.

    Returns:
        loss: total loss
    """
    relu = torch.nn.ReLU()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    l2Loss = torch.nn.MSELoss(reduction='mean')
    loss = torch.tensor([0.0], device=device, requires_grad=True)
    l_t = torch.tensor([0.0], device=device, requires_grad=True)
    l_f = torch.tensor([0.0], device=device, requires_grad=True)
    l_g = torch.tensor([0.0], device=device, requires_grad=True)
    l_feat = torch.tensor([0.0], device=device, requires_grad=True)

    #time domain loss, output_wav is the output of the generator
    l_t = l1Loss(input_wav, output_wav) 

    #frequency domain loss, window length is 2^i, hop length is 2^i/4, i \in [5,11]. combine l1 and l2 loss
    for i in range(5, 12): #e=5,...,11
        fft = Audio2Mel(n_fft=2 ** i,win_length=2 ** i, hop_length=(2 ** i) // 4, n_mel_channels=64, sampling_rate=sample_rate, device=device)
        l_f = l_f + l1Loss(fft(input_wav), fft(output_wav)) + l2Loss(fft(input_wav), fft(output_wav))
    
    #generator loss and feat loss, D_k(\hat x) = logits_fake[k], D_k^l(x) = fmap_real[k][l], D_k^l(\hat x) = fmap_fake[k][l]
    # l_g = \sum max(0, 1 - D_k(\hat x)) / K, K = disc.num_discriminators = len(fmap_real) = len(fmap_fake) = len(logits_fake) = 3
    # l_feat = \sum |D_k^l(x) - D_k^l(\hat x)| / |D_k^l(x)| / KL, KL = len(fmap_real[0])*len(fmap_real)=3 * 5
    for tt1 in range(len(fmap_real)): # len(fmap_real) = 3
        l_g = l_g + torch.mean(relu(1 - logits_fake[tt1])) / len(logits_fake)
        for tt2 in range(len(fmap_real[tt1])): # len(fmap_real[tt1]) = 5
            # l_feat = l_feat + l1Loss(fmap_real[tt1][tt2].detach(), fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2].detach()))
            l_feat = l_feat + l1Loss(fmap_real[tt1][tt2], fmap_fake[tt1][tt2]) / torch.mean(torch.abs(fmap_real[tt1][tt2]))
    
    KL_scale = len(fmap_real)*len(fmap_real[0]) # len(fmap_real) == len(fmap_fake) == len(logits_real) == len(logits_fake) == disc.num_discriminators == K
    K_scale = len(fmap_real) # len(fmap_real[0]) = len(fmap_fake[0]) == L
    
    loss = 3*l_g/K_scale + 3*l_feat/KL_scale + (l_t / 10) + l_f
    
    return loss

def disc_loss(logits_real, logits_fake):
    cx = torch.nn.ReLU()
    lossd = torch.tensor([0.0], device=device, requires_grad=True)
    for tt1 in range(len(logits_real)):
        lossd = lossd + torch.mean(cx(1-logits_real[tt1])) + torch.mean(cx(1+logits_fake[tt1]))
    lossd = lossd / len(logits_real)
    return lossd

'''
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
'''

def train_lora():
    rank = 16
    kernel_size = 23
    batch_size = 4
    epoch = 10
    warmup = 50
    checkpoint_step = 100
    target_embedding_path = "./embeddings/ange.pt"
    target_dataset_path = "./datasets/ange"
    logdir = "./logs"

    dataset = JVS(sr=16000, mode="denoise")
    dataloader = DataLoader(dataset, batch_size, num_workers=n_workers, shuffle=True)

    discriminator1 = MultiScaleSTFTDiscriminator(filters=32).to(device)
    model = EncodecModel.encodec_model_24khz().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    optimizer_d_1 = optim.AdamW(discriminator1.parameters(), lr=1e-4)

    global_step = 0
    last_d_loss_1 = 0
    losses = []

    for i in range(epoch):
        model.train()
        discriminator1.train()
        train_d = False

        bar = tqdm.tqdm(dataloader, desc="EPOCH {}".format(i), total=len(dataloader))
        for target, data1 in bar:
            train_d = not train_d

            optimizer.zero_grad()
            optimizer_d_1.zero_grad()
            discriminator1.zero_grad()
            
            data1 = data1.to(device)
            if len(data1.shape) == 2:
                data1 = data1[:, None, :]

            target = target.to(device)
            if len(target.shape) == 2:
                target = target[:, None, :]

            logits_real_1, fmap_real_1 = discriminator1(target)
            if train_d:
                logits_fake_1, _ = discriminator1(model(data1).detach())
                loss = disc_loss(logits_real_1, logits_fake_1)
                if loss > last_d_loss_1/2:
                    loss.backward()
                    optimizer_d_1.step()
                last_d_loss_1 = 0

            output1 = model(data1)
            logits_real_1, fmap_real_1 = discriminator1(target)
            logits_fake_1, fmap_fake_1 = discriminator1(output1)
            loss: torch.Tensor = total_loss(fmap_real_1, logits_fake_1, fmap_fake_1, data1, output1, sample_rate=16000)

            loss.backward()
            optimizer.step()

            losses.append(loss)
            bar.set_postfix({"total_loss": float(loss)})
            global_step += 1

            if global_step % checkpoint_step == 0:
                makedirs(logdir+"/logs/{:0>8}/wav/".format(global_step), exist_ok=True)
                makedirs(logdir+"/logs/{:0>8}/checkpoints/".format(global_step), exist_ok=True)
                torch.save(model.state_dict(), logdir+"/logs/{:0>8}/checkpoints/model.pth".format(global_step))
                for j in range(batch_size//2):
                    torchaudio.save(logdir+"/logs/{:0>8}/wav/vc_{}_original.wav".format(global_step, j), target[j].detach().cpu(), dataset.sr)
                    torchaudio.save(logdir+"/logs/{:0>8}/wav/vc_{}_noise.wav".format(global_step, j), data1[j].detach().cpu(), dataset.sr)
                    torchaudio.save(logdir+"/logs/{:0>8}/wav/vc_{}_output.wav".format(global_step, j), output1[j].detach().cpu(), dataset.sr)

                bar.write("===========================================================================")
                bar.write("STEP "+ str(global_step))
                bar.write("LOSS "+ str(float(torch.stack(losses).mean())))
                losses = []

        
def main():
    #mode = sys.argv[1]
    #if mode == "se":
    #    train_speaker_encoder()
    #elif mode == "emb":
    #    get_embedding()
    #elif mode == "lora":
    train_lora()
    #else:
    #    raise NotImplementedError
    

if __name__ == "__main__":
    main()