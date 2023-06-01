#Block 1
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR	
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

# Train data transformations
def transformation():
	train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])

# Test data transformations
	test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
	return train_transforms, test_transforms

#Block 4
def download_data(train_transforms=None,test_transforms=None):
   train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
   test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
   return train_data , test_data


#Block 5
def loaders(train_data, test_data):
	batch_size = 512

	kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

	test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
	train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
	return train_loader, test_loader



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

def optimize(model, device,train_loader,test_loader):
  import torch.optim as optim    
  optimizer = optim.SGD(model.parameters(), lr=.01, momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
  num_epochs = 20
  train_losses=[]
  train_acc=[]
  test_losses=[]
  test_acc=[]
  for epoch in range(1,num_epochs+1):
    print(f'Epoch {epoch}')
    train_losses, train_acc = train(model, device, train_loader, optimizer,train_losses, train_acc)
    test_losses, test_acc = test(model, device, test_loader,test_losses,test_acc)
    scheduler.step()
  return train_losses, train_acc, test_losses, test_acc

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()




def train(model, device, train_loader, optimizer,train_losses=None,train_acc=None):
  
  from tqdm import tqdm
  
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = F.nll_loss(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))
  return train_losses, train_acc

def test(model, device, test_loader,test_losses=None,test_acc=None):
    
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_losses,test_acc
     



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

