# Assignment 6 Part 1 and 2
# Part 1
## Manual Backpropagation for a Three layer (1 Input, 1 Hidden and 1 Output layer) Neural Network using different learning rates


### Neural Network Architecture

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/37d2ce50-3484-4807-9044-a11dd48d2768)

## Input Nodes:

i1

i2

## Target nodes :

t1

t2

## Hidden Layer Nodes:

h1 = w1*i1 + w2*i2

h2 = w3*i1 + w4*i2

### Activation values for Hidden Layer (Sigmoid):

a_h1 = σ(h1) = 1/(1 + exp(-h1))


a_h2 = σ(h2) = 1/(1 + exp(-h2))


## Output Layer Nodes:

o1 = w5*a_h1 + w6*a_h2

o2 = w7*a_h1 + w8*a_h2

### Activation values for Output Layer (Sigmoid):

a_o1 = σ(o1) = 1/(1 + exp(-o1))

a_o2 = σ(o2) = 1/(1 + exp(-o2))

## Error Calculations (Loss Function):

E1 = (1/2) * (t1 - a_o1)^2

E2 = (1/2) * (t2 - a_o2)^2

E_Total = E1 + E2

## Backpropagation to minimize Loss function E_Total

### Gradient of E_Total calculated with respect to all the weights (parameters) w1, w2, w3, w4, w5, w6, w7, w8 using Partial Derivatives

∂E_total/∂w5 = ∂(E1+E2)/∂w5

**As E2 is constant wrt w5 thus only E1 remains**

∂E_total/∂w5 = ∂E1/∂w5

**Applying Chain Rule**

∂E_total/∂w5 = ∂E1/∂w5 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5

∂E1/∂ao1 = ∂((1/2) * (t1 - a_o1)^2) / ∂a_o1 = (a_o1 - t1)

∂a_o1/∂o1 = ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)

∂o1/∂w5 = a_h1

Hence,

∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) * a_h1

Similar calculations done for weights w6, w7 & w8.

∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) * a_h2

∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) * a_h1

∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) * a_h2

### a_h1 is contributing to two output nodes in the next layer and hence E_Total. Therefore,

∂E_total/∂a_h1 = ∂E1/∂a_h1 + ∂E2/∂a_h1

∂E1/∂a_h1 can be represented as below with subsequent steps:

∂E1/∂a_h1 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1

∂E1/∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5

Similarly ∂E2/∂a_h1 can be written as:

∂E2/∂a_h1 = (a_o2 - t2) * a_o2 * (1 - a_o2) * w7

∂E_total/∂a_h1 is now:

∂E_total/∂a_h1 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7

Similar calculation is done for ∂E_total/∂a_h2:

∂E_total/∂a_h2 = (a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8

Gradient for w1 is represented as below:

∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1

∂E_total/∂a_h1 is already calculated above.

∂a_h1/∂h1 is sigmoid differential

∂h1/∂w1 is equal to i1

Overall equation for ∂E_total/∂w1 can be represented as:

∂E_total/∂w1 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * ( a_h1 * (1 - a_h1) ) * i1

Similar calculations can be performed on w2, w3 and w4. Following are their gradient respectively

∂E_total/∂w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w7) * ( a_h1 * (1 - a_h1) ) * i2

∂E_total/∂w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * ( a_h2 * (1 - a_h2) ) * i1

∂E_total/∂w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2) * w8) * ( a_h2 * (1 - a_h2) ) * i2

## Once we have gradient values for all, backpropogation starts. Weights are updated using these gradients using the specified Learning rate.

**for example, for the first parameter w1, updated w1 will be  ( w1 - LR * ∂E_total/∂w1 )

We analyse the Loss over **100 iterations**, for different learning rates and understand its impact on network convergence.


## Learning rate 0.1:

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/8c16f35e-cc0a-4ea9-a510-184ea5530cf5)


## Learning rate 0.2:

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/0ebb8efd-68e8-4b60-aabc-3e6ac5a1493f)


## Learning rate 0.5:

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/cf1909f8-3e44-47a2-bd36-77fa0edc0fad)


## Learning rate 0.8:

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/fa284bc0-fb5a-4883-8af4-ad97d6fc8484)


## Learning rate 1.0

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/2deedf88-e31d-48d1-b52d-9fb8b23824e8)


## Learning rate 2.0:

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/07073f5a-3ba2-4dfb-bb31-482916a06eae)


# Part 2

## MNIST CLassifier using Convolutional Neural Network with 99.4% Validation accuracy

### Constraints :

1.Numnber of parameters < 20K

2.Less than 20 epochs

### Network Summary

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/08f46c86-58d3-495a-bc1b-63823c7a3675)

**Highlights of this Squeeze and Expand Network Architecture:**

Network has total 9 layers.

Logic used for designing layers is CRB (Convolution-Relu-Batch Normalization).

Dropout of 0.05% is used after Batch Normalization layer.

Dropout & Batch Normalization not used after 1x1 convolution layer as there was a considerable gap observed between train and test accuracy.

1x1 convolution is used after two 3x3 convolutions followed by Max pooling.

Number of channels vary from 8 to 32 at different layers.

GAP is used near to last layer after convolution and a layer before fully connected layer.

Fully connected layer is the last layer of network.

Log Softmax used as last layer activation function with NLL Loss.

Trainable parameters for network are 16,562 (less than 20k).

## 99.4% test/validation accuracy from 15th epoch.

![image](https://github.com/RashiTech/ERA-V1/assets/90626052/7233815b-c570-48f0-9f52-4f034d40b6ba)


