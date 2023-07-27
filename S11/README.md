# Assignment 11

# Grad-CAM(Gradient-weighted Class Activation Mapping) implementation on CIFAR10 dataset for diagnosing the model (RESNET18) for mis-classified images

## Resnet18 Architecture

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/cbc78206-b470-4459-b237-cc6dc90f29a4)

## Git repo cloned into colab

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/3e5cae4b-fcb1-412a-891f-12f5746f6459)

## Training logs  84% Test accuracy achieved in 20 epochs

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/69f71926-551a-4d28-b53c-bbc549d3c6f9)

## Learning curves

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/1c27c514-c85c-4bb8-8e28-f87b9538598f)

## Misclassified Images

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/195a9915-ed57-4291-bce9-f568a86ad009)

## Grad-CAM output

 Channel size for layer 3 was 8X8 - suitable for Grad-CAM output. However, output is generated for all conv layers.

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/df646c63-c277-4a7a-873d-d213cf829b78)
