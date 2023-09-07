# Speed up of Transformer Model (encoder-decoder) for Language Translation English to French (Opus Book dataset)
## ERAv1 Partner Madhur Prakash Garg

## Final Result : Training Loss 1.71 at 33rd Epoch

## Strategies

### Data truncated to max 150 tokens in source language

### Data removed where len(fench_sentences) > len(english_sentrnce) + 10

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/be7eec45-e13a-4e9d-87d2-6b57f1a498fe)

### Mixed precision

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/788b019f-2e4f-4d25-be96-997696199bdb)

### One Cycle Policy for Learning Rate

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/9e2c4e04-bef4-424c-aa43-8e4fefd8d14c)

### Parameter Sharing

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/90371878-bad6-469a-bcb8-426261c45d58)

### Dynamic Padding - sos and eos token padding done in collate_fn

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/0aacedc0-a9c3-4779-b9ef-32000945195a)

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/335ce99a-15e3-4427-8b99-0b5b9f44a05a)

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/c78c6469-4987-49f4-b5a3-babd3fa14d51)

## Results :

#### Training time for one epoch : 3 mins and Loss achieved 1.84 in 25 epochs

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/fb56c89a-7a31-4be7-9cea-3872eb64f899)

### Further trained from 25 to 40 epochs without LR scheduler

#### Loss achieved 1.72 at 33rd epoch

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/9c3745bf-0429-4a24-89b7-2053d17e029a)

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/1c5be71c-3a64-4d95-9b5e-90d93eb0e371)

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/fca0f348-2a75-4ae9-9e5c-8bb461bda531)


