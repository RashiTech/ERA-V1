import torch
import wandb
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM
from torch.nn import functional as F
import random
import gc
import numpy as np
import os

# teacher forcing simulated annealing scheduler
def frange_cycle_linear(n_iter, start=0.0001, stop=0.9999,  n_cycle=1, ratio=0.8):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return (1 - L)

# define models
phi_model_name  = "microsoft/phi-2"
clip_model_name = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M"
device = 'cuda'
max_steps = 100000

annealing_teacher_forcing_scheduler = frange_cycle_linear(max_steps)

class SimpleResBlock(nn.Module):
    def __init__(self, phi_embed):
        super().__init__()
        self.pre_norm = nn.LayerNorm(phi_embed)
        self.proj = nn.Sequential(
            nn.Linear(phi_embed, phi_embed),
            nn.GELU(),
            nn.Linear(phi_embed, phi_embed)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class CLIPPhi2Model(torch.nn.Module):
    def __init__(self, clip_embed=640, phi_embed=2560):
        super().__init__()

        self.EOS_TOKEN_ID    = 50256
        self.IMAGE_TOKEN_ID  = 23893 # token for comment

        # pretrained models
        self.phi_model = AutoModelForCausalLM.from_pretrained(phi_model_name,
                                            torch_dtype=torch.float16,
                                            trust_remote_code=True)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)

        # projection layers
        self.projection = torch.nn.Linear(clip_embed, phi_embed)
        #self.resblock = SimpleResBlock(phi_embed)

        # Freeze Weights
        for network in [self.phi_model, self.clip_model]:
            for param in network.parameters():
                param.requires_grad_(False)

        # load checkpoint weights
        if os.path.isfile('model_chkpt/clipphi_proj.pth'):
            self.projection.load_state_dict(torch.load('model_chkpt/clipphi_proj.pth'))
            #self.resblock.load_state_dict(torch.load('model_chkpt/clipphi_resblock.pth'))


    def generate(self,images,max_length,tokenizer):
        
        # clip model output for image
        clip_outputs = self.clip_model(**images)
        # remove cls token
        images = clip_outputs.last_hidden_state[:,1:,:]
        image_embeds = self.projection(images).to(torch.float16)

        # add bos_token embedding from phi2
        batch_size = images.size(0)
        predicted_caption = torch.full((batch_size,max_length),50256)
        img_token_tensor = torch.tensor(self.IMAGE_TOKEN_ID).repeat(batch_size, 1)
        bos_token_embeds = self.phi_model.model.embed_tokens(img_token_tensor.to(image_embeds.device))
        combined_embeds  = torch.cat([image_embeds, bos_token_embeds], dim=1) # 4,9,2560

        for pos in range(max_length - 1):
            # pass through the model
            model_output_logits = self.phi_model.forward(inputs_embeds = combined_embeds)['logits'] # 4,49,51200
            predicted_word_token_logits = model_output_logits[:, -1, :].unsqueeze(1) # 4,1,51200
            predicted_word_token = torch.argmax(predicted_word_token_logits, dim = -1) # 4,1
            predicted_caption[:,pos] = predicted_word_token.view(1,-1).to('cpu')
            next_token_embeds = self.phi_model.model.embed_tokens(predicted_word_token) # 4,1,2560
            combined_embeds   = torch.cat([combined_embeds, next_token_embeds], dim=1)
        return predicted_caption

    def forward(self, images, target_captions,step,max_steps):

        batch_size    = target_captions.size(0)
        target_length = target_captions.shape[1]
         #print(f"GPU memory {torch.cuda.max_memory_allocated()/ (1024 ** 3):.2f} GB")

        # clip model output for image
        clip_outputs = self.clip_model(**images)
        images = clip_outputs.last_hidden_state[:,1:,:] # remove cls token

        # projection layer
        image_embeds = self.projection(images).to(torch.float16)
        #image_embeds = self.resblock(image_embeds).to(torch.float16)

        # add comment token from phi2
        img_token_tensor = torch.tensor(self.IMAGE_TOKEN_ID).repeat(batch_size, 1)
        img_token_embeds = self.phi_model.model.embed_tokens(img_token_tensor.to(image_embeds.device))
        combined_embeds  = torch.cat([image_embeds, img_token_embeds], dim=1) # 4,49,2560
        del clip_outputs
        del image_embeds

        # for loss
        loss = 0
        for pos in range(target_length - 1):
           
            # pass through the model
            model_output_logits = self.phi_model.forward(inputs_embeds = combined_embeds)['logits'] # 4,49,51200
            predicted_word_token_logits = model_output_logits[:, -1, :].unsqueeze(1) # 4,1,51200
            pos_loss = F.cross_entropy(predicted_word_token_logits.view(-1,predicted_word_token_logits.size(-1)), target_captions[:, pos].contiguous().view(-1), ignore_index=self.EOS_TOKEN_ID,label_smoothing=0.1)
            #print(f"pos {pos} loss {pos_loss}")
            loss += pos_loss

            predicted_word_token = torch.argmax(predicted_word_token_logits, dim=-1) # 4,1
            #print(f"predicted_word_token {predicted_word_token} and target_captions {target_captions[:,pos]}")
            # do teacher forcing or model output based on annealing scheduler probability
            if pos <= 5 and step <= int(0.6 * max_steps): # teacher forcing
                next_token_embeds = self.phi_model.model.embed_tokens(target_captions[:,pos].unsqueeze(1)) # 4,1,2560
            else:
                next_token_embeds = self.phi_model.model.embed_tokens(predicted_word_token) # 4,1,2560
            
            combined_embeds   = torch.cat([combined_embeds, next_token_embeds], dim=1)

        #average_loss
        loss = loss / target_length

        del combined_embeds
        del model_output_logits
        torch.cuda.empty_cache()

        return loss
        

def model_validate_one_batch(model,device,val_dataloader,max_length,tokenizer):
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, target_captions) in enumerate(val_dataloader):
            images = {'pixel_values': images.to(device)}
            target_captions = target_captions.to(device)
            target_captions_decoded = tokenizer.batch_decode(target_captions,ignore_index = 50256)
            predicted_captions = model.generate(images,max_length,tokenizer)
            predicted_captions_decoded = tokenizer.batch_decode(predicted_captions,ignore_index = 50256)

            for pc_idx,pc in enumerate(predicted_captions_decoded):
                print(f"{pc_idx} - Target captions:\n {target_captions_decoded[pc_idx]}  \n{pc_idx} - predicted_captions:\n {pc} ")
            return # validate only 1 batch
    
def train_model(model, train_loader, val_dataloader,optimizer, device,max_steps,model_save_step,model_val_step,log_step,max_token_filter,tokenizer):
    print(f"Training started.")
    
    max_step_reached = 0
    step = 0
    max_length = 20
    running_loss = 0.
    model.train()

    for epoch in range(100000):
        for batch_idx, (images, target_captions) in enumerate(train_loader):

            # manage OOM issue, skip batch for long captions
            if target_captions.shape[1] >= max_token_filter:
                print(f"Batch skipped as captions too long.")
                continue 
            images = {'pixel_values': images.to(device)}
            target_captions = target_captions.to(device)

            # techer forcing or not
            # if round(random.uniform(0.0, 1.0),2) <= annealing_teacher_forcing_scheduler[step]:
            #     teacher_forcing = True
            # else:
            #     teacher_forcing = False

            optimizer.zero_grad()
            loss = model(images, target_captions,step,max_steps)
            #print(f"teacher {teacher_forcing} and loss {loss}")
            running_loss += loss.item()

            # log step
            if (step % log_step == 0):
                if step == 0:
                    print(f"Step {step}/{max_steps}: Avg Running Loss = {running_loss}")
                else:
                    print(f"Step {step}/{max_steps}: Avg Running Loss = {running_loss /log_step}")
                running_loss = 0.
            wandb.log({"step": step, "train_loss": loss.item()})

            # increment step
            step += 1
            teacher_forcing = False

            # loss backprop
            loss.backward()
            optimizer.step()
            
            # save model
            if step % model_save_step == 0 or (step == max_steps):
                print("Saving Checkpoint for step : ", step)
                torch.save(model.projection.state_dict(),'./model_chkpt/clipphi_proj.pth')
                #torch.save(model.resblock.state_dict(),'./model_chkpt/clipphi_resblock.pth')
               
            # check random validation of images
            if step % model_val_step == 0 or (step == max_steps):
                model_validate_one_batch(model,device,val_dataloader,max_length,tokenizer)
                model.train()

            # global max steps reached
            if step >= max_steps:
                max_step_reached = 1
                break

        if max_step_reached == 1:
            break
    print(f"Reached the max steps. Training stopped.")