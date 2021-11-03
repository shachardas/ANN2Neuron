
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models.binarized_modules import  BinarizeLinear,BinarizeConv2d
from models.binarized_modules import  Binarize,HingeLoss

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = BinarizeLinear(1, 1, bias=False)
        self.fc2 = BinarizeLinear(1, 1, bias=False)
        self.fc3 = BinarizeLinear(1, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = Net()
with torch.no_grad():
    model.fc1.weight[0, 0] = -1.0
    model.fc2.weight[0, 0] = 1.0
    model.fc3.weight[0, 0] = -1.0
    #print(model(torch.as_tensor([[1.0]])))


#print(model.state_dict())

#torch.save(model.state_dict(), "mnist_trained_dummy2.pt")

