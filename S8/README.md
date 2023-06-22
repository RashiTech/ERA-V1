# Classification of CIFAR-10 dataset

## Objective to achieve 70% Test accuracy and study the impact of different normalization techniques applied to the convolution layers

### Constraints

1. Less than 50,000 parameters

2. Less than or equal to 20 epochs

## Network Summary   

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/eb832c2e-e7c3-4021-a24c-e4baa8fd3e84)

Batch Size = 128

Parameters= 35,504

Image Augmentation implemented- ColorJitter, RandomHorizontalFlip(p=0.3), RandomRotation((-10., 10.)

Dropout regularization (0.02)

Target accuracy of 70% achieved at around 10th epoch in all case.

### Graphs - Batch Normalization

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/439b8c20-00fc-4a2f-9186-2f37e182a4f2)

### Mispredicted Images _ Batch Normalization

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/c8e8aae3-8ece-44ee-bb91-b60c1a2c5818)

### Graphs - Group Normalization (Group Size = 2)

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/63bf54d2-172e-4062-833f-da3fe5560372)

### Mispredicted Images _ Group Normalization

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/189b9ed6-c28f-48b0-b70e-6003c36b75af)

### Graphs - Layer Normalization (Group Norm with Group Size = 1)

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/a57c46fc-c56b-4d14-8748-8bdc503c1051)

### Mispredicted Images _ Layer Normalization

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/969adb92-833a-4838-a740-d32d160f7fd7)

## Result : Train and Test Accuracies 

### Batch Normalization achieves best Train and Test accuracy followed by Group Normalization.

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/57399723-a431-4e6b-9034-d05371219684)
