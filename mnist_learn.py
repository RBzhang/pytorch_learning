import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
#from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307),(0.3081))
])

train_dataset = datasets.MNIST(root='~/dataset/mnist',train=True,download=True,transform=transform)

train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)

test_dataset = datasets.MNIST(root='~/dataset/mnist',train=False,download=True,transform=transform)

test_loader = DataLoader(test_dataset,shuffle=True,batch_size=batch_size)