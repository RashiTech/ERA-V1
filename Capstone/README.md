# Capstone - ERAv1 - The School of AI 
# MultiModal ChatGPT (Text, Image, Audio -->> Text)

## This is a complex and an exciting Project that involves the best of natural language processing (NLP) and computer vision (CV).

**This project entails the creation and integration of sophisticated language and vision models, showcasing a thorough approach in machine learning. The process involves training a customized language model, Phi 2, from the ground up, and further augmenting its capabilities through the implementation of techniques that align vision, audio and text for a robust understanding across multiple modalities.**

***Hugging Face deployment***

https://huggingface.co/spaces/RashiAgarwal/MultimodalChatGPT_TSAI

<img width="667" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/a05f7a6e-61b3-4fb4-bf0c-2c092cb62acd">

<img width="638" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/2f767ca3-8c2c-4035-bb5d-24169b4cd70a">


## Part - 1 : Training a Large Language Model from Scratch

### Pretraining of Language Model Microsoft/Phi-2

#### Project Keypoints

1. Large Language Model Microsoft-phi-2 (2.7 Billion Parameters) pre-trained from scratch on NVIDIA-A100 40GB GPU.

2. Model trained on a cleaned 100MB data (zipped) - A small subset of **RedPajama dataset**- only cc and c4

3. Memory Management for pretraining on a single 40GB GPU

   **8bit BNB quantized optimizer used - bitsandbytes.optim.Adam8bit (4 times less memory usage when compared to AdamW/Adam)**

    torch.cuda.amp.autocast(enabled=True)

    torch.cuda.empty_cache() and gc.collect()

    batch_size = 1

    gradient_accumulation_steps = 4

    gradient_checkpointing not supported by phi-2

#### 4. Training logs- Training loss dropped from 11.357 to 6.237 in 4000 iterations

Minimum loss observed = 5.442

<img width="583" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/234eedd0-57b4-40f1-ba73-768e062a779f">

<img width="603" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/501e3178-a999-4470-90af-a4c69833b2c7">

## Part - 2 : To create A Multimodal ChatGPT that accepts textual, Visual and Audio inputs and generate Textual response.

Models 
**Visual Processor : wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M**
   
**Language Model : microsoft/phi-2**

## Step 1 : Visual Text Alignment

**Training of the Custom Projection layer from the frozen pretrained CLIP embeddings to the frozen pretrained Phi-2 Model using single A100 40GB GPU**

Image features from Clip Model are sent as Input and captions are passed as Target to Phi-2

This is a linear FC layer with input dimension = 640 (Tiny CLIP output embedding size) and output dimension = 2560 to match phi-2's embedding size

**Tiny CLIP's config**

 <img width="421" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/04fb4c84-6933-446d-bf0a-83a7f737a495">

**Dataset used : COCO 2017**

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

## Step 2 : Instruction Following Fine Tuning the model with Instruct150K dataset

### Visual_projector

**Instruct150k** dataset used to finetune the pretrained Projection Layer from Step 1 and the added quantized adaptor for the Query, Key & Value projection layers and dense fully connected layers of Phi-2. Finetuning done using **QLoRA (Quantized Low Ranking Adaptation) Strategy**. Methods for dataset preparation comprise Autoregressive Token Prediction and the Standard Language Model Training Method which helped in achieving model efficiency along with reasonable computational resources.

### Audio_projector (No training involved)

**WhisperX** Model used for Automatic Speech Recognition to enable the audio part of the Multimodal GPT 
 
Accepts only the starting 15 seconds speech before transcribing

Tokenizing and processing the generated transcript for making it input ready to the model

### Finetuning done for 10000 iterations on Google-Colab Pro A100 40GB GPU 

Training loss dropped from 5.142 to 2.5 

<img width="560" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/9930d195-ff5b-480d-bf79-d51fe51801d5">

<img width="656" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/6fb5731e-593a-4928-b5d0-7e299e968b05">

**Logs from wandb**

https://wandb.ai/rashiagarwal/tsai_clip_phi2_project?workspace=user-rashiagarwal

<img width="603" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/fb8f6e01-ff5e-4e19-8167-d11dc2c5ab36">

### Some of the predictions made by the finetuned model

<img width="339" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/7a651d5c-3b37-43c8-85b6-0ea8f3372a8c">
Question: Is there any other notable building next to the clock tower?
Answer:   Yes, there is a skyscraper located next to the clock tower.
Model Predicted Ans: Yes, there is a large building next to the clock tower.The building next to the clock tower is a large, modern-looking building with a glass facade.It is a skyscraper, which is a tall, multi-story building.The skyscraper is located in a city or urban area, as it is surrounded by other buildings and streets.The skyscraper is visible from a distance, as it is tall and stands out in the cityscape.The skyscraper is a prominent feature of

<img width="375" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/40d14872-ec4f-4772-bc81-88aefbc7bb84">
Question: What color is the frisbee the man is holding?
Answer:   The frisbee the man is holding is blue.
Model Predicted Ans: The man is holding a blue frisbee.The man is standing in a field, which is visible in the image.The man is wearing a blue shirt and black pants.


### Further Action Planned:
 
The Output of the model to be further enhanced with Visual/ Audio content using Generative AI


