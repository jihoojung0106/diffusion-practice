
import os
from typing import Dict

import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import torch.nn as nn

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from accelerate import Accelerator
import wandb

def train(modelConfig: Dict,useWandb=False):
    device = torch.device(modelConfig["device"])
    flag=True
    
    if useWandb:
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(
            project_name="simple-ddpm-cifar10", 
            config={"dropout": modelConfig["dropout"], "learning_rate": modelConfig["lr"]}
            # init_kwargs={"wandb": {"entity": "my-wandb-team"}}
        )

    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], #1000 
                     ch=modelConfig["channel"],  #128
                     ch_mult=modelConfig["channel_mult"], #[1, 2, 3, 4],
                     attn=modelConfig["attn"], #[2]
                     num_res_blocks=modelConfig["num_res_blocks"], #2
                     dropout=modelConfig["dropout"] #
                     ).to(device)
    number_int=0
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["training_load_weight"]), map_location=device))
        number_str = modelConfig["training_load_weight"].split('_')[-2]  # Splitting by '_' and taking the second last element which should be '5'
        number_int = int(number_str)  # Converting the extracted string to an integer

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        net_model = nn.DataParallel(net_model)
    
    
    optimizer = torch.optim.AdamW(
        net_model.parameters(), 
        lr=modelConfig["lr"], 
        weight_decay=1e-4)
    
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, 
        T_max=modelConfig["epoch"], 
        eta_min=0, 
        last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, 
        multiplier=modelConfig["multiplier"],
        warm_epoch=modelConfig["epoch"] // 10, 
        after_scheduler=cosineScheduler)
    
    trainer = GaussianDiffusionTrainer(
        net_model, 
        modelConfig["beta_1"], #1e-4,
        modelConfig["beta_T"], #0.02
        modelConfig["T"]).to(device) #1000

    # start training
    for e in range(modelConfig["epoch"]):
        
        total_loss = 0.0
        n_batches = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device) #(batch_size,3,32,32)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                total_loss += loss.item()
                n_batches += 1
        avg_loss = total_loss / n_batches
        img=inference_in_the_middle(net_model,modelConfig,device,e)
        if useWandb:
            accelerator.log({"train_loss": avg_loss,"Image": [wandb.Image(img)]},step=e)
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(number_int+e) + "_.pt"))
        print("saved pt : ",os.path.join(modelConfig["save_weight_dir"], 'ckpt_' + str(number_int+e) + "_.pt"))
    if useWandb:
        accelerator.end_training()

def inference_in_the_middle(model,modelConfig,device,e):
    with torch.no_grad():
        model.eval()
        sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
            # Sampled from standard normal distribution
            #평균이 0이고 분산이 1인 정규분포에서 뽑은 값들 대부분 [-1,1]사이에 있음
        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        image_dir=os.path.join(modelConfig["sampled_dir"], f"{str(e)}_wandb_img.png")
        save_image(sampledImgs, image_dir, nrow=modelConfig["nrow"])
        print("this is dir",image_dir)
        return image_dir

def eval(modelConfig: Dict):
    # load model and evaluate
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(
            os.path.join(modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        
        
        sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        # Sampled from standard normal distribution
        #평균이 0이고 분산이 1인 정규분포에서 뽑은 값들 대부분 [-1,1]사이에 있음
        noisyImage = torch.randn(size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        #[-1,1]->[-1*0.5+0.5=0,1*0.5+0.5=1]사이로 옮기고 그 외의 값은 0,1로 clamp해버림.
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        
        save_image(saveNoisy, os.path.join(
            modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
        
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, os.path.join(
            modelConfig["sampled_dir"],  modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])