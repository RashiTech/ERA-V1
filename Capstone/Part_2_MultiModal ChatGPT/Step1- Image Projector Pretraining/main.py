import glob2
import torch
from network import CLIPPhi2Model, train_model, frange_cycle_linear
from dataset import collate_fn, llavadataset
from torch.utils.data import random_split, DataLoader
import wandb
import torch.nn as nn
from transformers import AutoTokenizer
import pickle
import bitsandbytes as bnb
import gc


def main():
    with open("coco_dataset_pickle", "rb") as fp:   # Unpickling
        coco_unpickle = pickle.load(fp)

    clip_model_name = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
    phi_model_name  = "microsoft/phi-2"
    train_batch_size = 2
    val_batch_size   = 2
    device     = 'cuda'
    tokenizer  = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)

    # model
    MModalGPT        = CLIPPhi2Model().to(device)
    max_steps        = 50000
    model_save_step  = 100
    model_val_step   = 1000
    log_step         = 1000
    max_token_filter = 35

    # data loaders
    train_dataloader = DataLoader(llavadataset(coco_unpickle, phi_model_name,clip_model_name,'train',tokenizer),
                      collate_fn=collate_fn, batch_size=train_batch_size, num_workers = 10, shuffle=True, pin_memory=True)

    val_dataloader   = DataLoader(llavadataset(coco_unpickle, phi_model_name,clip_model_name,'val',tokenizer),
                      collate_fn=collate_fn, batch_size=val_batch_size, num_workers = 10, shuffle=True, pin_memory=True)
   
    optimizer = bnb.optim.Adam8bit(filter(lambda p: p.requires_grad, MModalGPT.parameters()), lr=1e-4)
    train_model(MModalGPT, train_dataloader, val_dataloader, optimizer, device, max_steps,model_save_step,model_val_step,log_step,max_token_filter,tokenizer)

if __name__ == "__main__":
    wandb.init(project="tsai_clip_phi2_project", name="step1_coco_pretrain") #,log_model='all')
    torch.cuda.amp.autocast(enabled=True)
    torch.cuda.empty_cache()
    gc.collect()
    #torch.set_float32_matmul_precision('medium')
    main()