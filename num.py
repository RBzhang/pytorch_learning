#from tkinter import W
from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])
class LogisticModel(torch.nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred
model = LogisticModel()
criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
for epoch in range(1000):
    y_pred =  model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch , loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
train_set = torchvision.datasets.CIFAR10(root='~/dataset/CIFAR10',train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root='~/dataset/CIFAR10',train=False, download=True)
