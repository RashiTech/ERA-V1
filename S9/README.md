# CIFAR 10 Classification using Dilated and depthwise separable convolutional layers

## Constraints
Total RF must be more than 44

One of the layers must use Depthwise Separable Convolution

One of the layers must use Dilated Convolution

Use GAP (compulsory)

Use albumentation library and apply:

horizontal flip

shiftScaleRotate

coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)

Achieve 85% accuracy

Total Params to be less than 200k

No constraint on epochs

## Model Summary

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/bc9e83a0-2a5c-4f17-b041-a7514f7cb722)

## 85% Test Accuracy achieved in 50 epochs

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/7b30fcbc-d40b-43dc-934f-673fbb7ced68)

