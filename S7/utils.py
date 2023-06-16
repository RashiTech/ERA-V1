
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR	
from torchvision import datasets, transforms
from matplotlib import pyplot as plt








def plots(train_loader= None):
	import matplotlib.pyplot as plt

	batch_data, batch_label = next(iter(train_loader)) 

	fig = plt.figure()

	for i in range(12):
  		plt.subplot(3,4,i+1)
  		plt.tight_layout()
  		plt.imshow(batch_data[i].squeeze(0), cmap='gray')
  		plt.title(batch_label[i].item())
  		plt.xticks([])
  		plt.yticks([])
	return

def plot_graphs(train_losses, train_acc,test_losses,test_acc):
  

  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

