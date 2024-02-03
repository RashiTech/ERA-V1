# Capstone Part 1 - The School of AI - ERA v1 

## Pretraining of Language Model Microsoft/Phi-2

### Project Keypoints

#### 1. Large Language Model Microsoft-phi-2 (2.7 Billion Parameters) pre-trained from scratch on NVIDIA-A100 40GB GPU.

#### 2. Model trained on a cleaned 100MB data (zipped) - A small subset of RedPajama dataset- only cc and c4

#### 3. Memory Management for pretraining on a single 40GB GPU

8bit BNB quantized optimizer used - bitsandbytes.optim.Adam8bit (4 times less memory usage when compared to AdamW/Adam)

torch.cuda.amp.autocast(enabled=True)

torch.cuda.empty_cache() and gc.collect()

batch_size = 1

gradient_accumulation_steps = 4

gradient_checkpointing not supported by phi-2

#### 4. Training logs- Training loss dropped from 11.357 to 6.237 in 4000 iterations

Minimum loss observed = 5.442

**With more compute resources and longer training, a fairly low training loss could be achieved to develop a good LLM.**

<img width="583" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/234eedd0-57b4-40f1-ba73-768e062a779f">

<img width="603" alt="image" src="https://github.com/RashiTech/ERA-V1/assets/90626052/501e3178-a999-4470-90af-a4c69833b2c7">


