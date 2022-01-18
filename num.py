#from tkinter import W
from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])
    def __getitem__(self, index):
        return self.x_data[index] , self.y_data[index]
    def __len__(self):
        return self.len
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32,shuffle=True,num_workers=2)
model = Model()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
for epoch in range(100):
    for i , data in enumerate(train_loader , 0):
        input , labels = data

        y_pred = model(input)
        loss = criterion(y_pred , labels)
        print(epoch, i , loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
print('w=',model.linear1.weight.data)
print('b=',model.linear1.bias.data)

#x_test = torch.Tensor([[4.0]])
#y_test = model(x_test)
#print('y_pred = ', y_test.data)
train_set = torchvision.datasets.CIFAR10(root='~/dataset/CIFAR10',train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='~/dataset/CIFAR10',train=False, download=True)
