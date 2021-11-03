
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
"""
def Binarize(tensor, include_zero=True):
    if include_zero:
        N_std = 
        up = (tensor > 0 + N_std * tensor.std()).float()
        down = (tensor < 0 - N_std * tensor.std()).float().mul(-1)
        return (up + down)
    else:
        return tensor.sign()

"""
"""
class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = BinarizeLinear(784, 50, bias=False)
        self.fc2 = BinarizeLinear(50, 10, bias=False)
        self.activity = []

    def forward(self, x):
        x = x.view(-1, 28*28)
        self.activity.append(x)
        x = self.fc1(x)
        self.activity.append(x)
        x = self.fc2(x)
        self.activity.append(x)
        return x

model = Net()
name = "mnist_trained_tiny_and_L1_zeroTrim"
model.load_state_dict(torch.load(name+".pt", map_location=torch.device('cpu')))
model.eval()

import pickle

sampleData = None
with open("sampleData", 'rb') as file:
    sampleData = pickle.load(file)

cleanSamples = {}
for data, label in sampleData[20:]:
    cleanSamples.update({label: (data >= 0).int()})
"""
layerACT = {}
for i in range(10):
    print(i, model(torch.as_tensor(cleanSamples[i]).float()))
    layerACT.update({i:model.activity})
import pickle
pickle.dump(layerACT,open(name+".act",'wb'))
"""
def checkClean():
    for i in range(10):
        if torch.sum(model(torch.as_tensor(cleanSamples[i]).float())>0) != 1:
            print(f"line {i} was a fluke")
            print(model(torch.as_tensor(cleanSamples[i]).float()))
            return False
    return True
#print("ACT", model.activity[1])
model(torch.as_tensor(cleanSamples[8]).float())
for outSig, weights in zip(model.activity[1][0], model.fc1.weight):
    sum=0
    for weight, inSig in zip(weights, model.activity[0][0]):
        sum += weight*inSig
    print(f"{sum}*{weight}={sum*weight} -> {outSig}")
#print(model.fc2.weight)