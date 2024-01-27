# Capstone- The School of AI - ERA v1

## Part 1- Pretraining of Language Model Microsoft/Phi-2

### Project Details

#### 1. Large Language Model Microsoft-phi-2 (2.7 Billion Parameters) pre-trained from scratch on NVIDIA-A100 40GB GPU.

#### 2. Model trained on a cleaned 100MB data (zipped) - A small subset of RedPajama dataset- only cc and c4

#### 3. Memory Management for pretraining os single 40GB GPU

Optimizer used - bnb.optim.Adam8bit (from bitsandbytes)

torch.cuda.amp.autocast(enabled=True)

torch.cuda.empty_cache() and gc.collect()

batch_size = 1

gradient_accumulation_steps = 4

gradient_checkpointing not supported by phi-2

#### 4. Training logs- Training loss dropped from 11.357 to 6.237 in 4000 iterations

Minimum loss observed = 5.442

![Untitled](https://github.com/RashiTech/ERA-V1/assets/90626052/a74d43a0-cb34-44db-96a0-31e9a43520e5)

![Untitled-1](https://github.com/RashiTech/ERA-V1/assets/90626052/58f8b992-c311-4b37-b05f-e627916a78ff)

