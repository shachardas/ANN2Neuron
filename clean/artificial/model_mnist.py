import torch
import torch.nn as nn
import numpy as np

class customLoss(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.cost = nn.CrossEntropyLoss()
        self.activation = nn.LogSoftmax()

    def forward(self, output, labels):
        tempOut = torch.clone(output)
        
        for i in range(labels.size(0)):
            tempOut[i,labels[i]] = -1 * tempOut[i,labels[i]]

        tempOut = torch.where(tempOut>-1,tempOut+2,torch.zeros_like(tempOut))

        return self.p    * torch.mean(tempOut) \
            + (1-self.p) * self.cost(self.activation(output),labels)


#criterion = nn.CrossEntropyLoss()
criterion = customLoss()

def Binarize(tensor, include_zero = False, minSig=3):
    if include_zero:
        P_std = 0.25
        up_lim = torch.max(0 + P_std*tensor.std(), torch.ones_like(tensor)*minSig)
        down_lim = torch.min(0 - P_std*tensor.std(), -1*torch.ones_like(tensor)*minSig)
        up_v = (tensor>up_lim).float()
        down_v = (tensor<down_lim).float().mul(-1)
        return (up_v + down_v)
    else:
        return tensor.sign()

"""def Binarize(tensor, include_zero = True):
        if include_zero:
            return ((tensor+0.5).sign()+(tensor-0.5).sign())/2
        else:
            return tensor.sign()"""
'''
class PositiveBinarizeLinear(nn.Linear):

        def __init__(self, *kargs, **kwargs):
            super(PositiveBinarizeLinear, self).__init__(*kargs, **kwargs)
    
        def forward(self, input):
            
            if input.size(1) != 784:
                input.data=Binarize(input.data)
                input.data = input.data.add(1).div(2)
            #zero = torch.zeros_like(input.data)
            #input.data = torch.where(input.data > 0, input.data, zero)
            input.data=Binarize(input.data)
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.data=Binarize(self.weight.org)
            out = nn.functional.linear(input, self.weight)
            if not self.bias is None:
                self.bias.org=self.bias.data.clone()
                out += self.bias.view(1, -1).expand_as(out)

            return out
'''



class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

# simplified batchnorm with no mean normalization and separate learnable parameters for positive and negatives
class SignSensitiveBatchNorm1d(nn.BatchNorm1d):

    def __init__(self, size):
        super(SignSensitiveBatchNorm1d, self).__init__(size)
        self.eps = 1e-5
        self.l1 = nn.Parameter(torch.ones(size))
        self.l2 = nn.Parameter(torch.ones(size))
        self.running_var = torch.ones(size)
        self.momentum = 0.1

    def forward(self, input):
        device = input.get_device()
        if input.size()[0] != 1:
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * torch.var(input, keepdim=True, dim=0)
            bottom = torch.sqrt(torch.var(input, keepdim=True, dim=0) + self.eps)
        else:
            bottom = torch.sqrt(self.running_var + self.eps)

        top = input * torch.sigmoid(10 * input) * self.l1 + input * torch.sigmoid(-10 * input) * self.l2
        
        out = top/bottom

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 1024*self.infl_ratio, bias=False)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = SignSensitiveBatchNorm1d(1024*self.infl_ratio)
        self.fc2 = BinarizeLinear(1024*self.infl_ratio, 1024*self.infl_ratio, bias=False)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = SignSensitiveBatchNorm1d(1024*self.infl_ratio)
        self.fc3 = BinarizeLinear(1024*self.infl_ratio, 1024*self.infl_ratio, bias=False)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = SignSensitiveBatchNorm1d(1024*self.infl_ratio)
        self.fc4 = BinarizeLinear(1024*self.infl_ratio, 10, bias=False)
        self.logsoftmax=nn.LogSoftmax()
        self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x) 
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return x#self.logsoftmax(x)
