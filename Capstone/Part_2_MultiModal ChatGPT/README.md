# Capstone Part 2 - The School of AI - ERA v1

## MultiModal Large Language Model (Clip + Phi-2) partially pre-trained and finetuned on instruct150K datset

### Image Processor : wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M
   
### Language Model : microsoft/phi-2

### Step 1 : Added and Trained the Projection layer from the CLIP embeddings to the Phi Model using single A100 40GB GPU

#### This is a linear FC layer with input dimension = 640 (tiny clip embeddings) and output dimension = 2560 to match phi-2's embedding size

 <img width="421" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/04fb4c84-6933-446d-bf0a-83a7f737a495">

#### Memory Management for training on a single GPU (40GB)

