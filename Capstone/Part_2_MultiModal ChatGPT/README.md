# Capstone Part 2 - The School of AI - ERA v1

## Objective: To create MultiModal Large Language Model (Frozen Clip + Frozen Phi-2) 
## The image projection layer to be pre-trained with image-caption(COCO 2017) dataset and then the entire model to be finetuned on Visual Q&A (instruct150K) datset

### Image Processor : wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M
   
### Language Model : microsoft/phi-2

## Step 1 : Added and Trained the Projection layer from the CLIP embeddings to the Phi Model using single A100 40GB GPU

#### Image features are sent as Input and captions are passed as Target

#### This is a linear FC layer with input dimension = 640 (Tiny CLIP output embedding size) and output dimension = 2560 to match phi-2's embedding size

Tiny CLIP's config

 <img width="421" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/04fb4c84-6933-446d-bf0a-83a7f737a495">

#### Dataset used : COCO 2017

#### Memory Management for training on a single GPU (40GB)

8bit BNB quantized optimizer used - bitsandbytes.optim.Adam8bit (4 times less memory usage when compared to AdamW/Adam)

torch.cuda.amp.autocast(enabled=True)

torch.cuda.empty_cache() and gc.collect()

batch_size = 2

#### Training Loss dropped from 8.237 to 3.266 in 11K steps

<img width="587" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/b81c3b34-b0cd-4e8c-9c01-a4549f05a58e">

<img width="604" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/f84ce3d1-12ba-4479-b6c7-81ec8ac289ae">

<img width="882" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/cfbc0c89-4af5-462b-a002-c4a20644f593">


<img width="938" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/cfaf8c50-17ac-40bb-a742-88ead4750452">

## Step 2 : Fine Tuning the model with Instruct150K dataset

### WhisperX used for Automatic Speech Recognition to enable the audio part of the Multimodal GPT - Done
#### Accepts only the starting 10 seconds speech before transcribing

### For Visual Part we have the pretrained Projection Layer from Step 1- Done - to be further finetuned with instruct150k dataset

## This part is yet to be done
### The Input to the phi-2 model is the concatenation of the output from Image projection layer with the embeddings of the tokenized Audio output and instruction part from the dataset. The Answers are passed as Target.

### Further Action Planned:
#### The Text output of the model can be further converted to Visual/ Audio output using Generative AI







