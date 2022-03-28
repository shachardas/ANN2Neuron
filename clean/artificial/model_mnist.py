import torch
import torch.nn as nn
import numpy as np

class customLoss(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.cost = nn.CrossEntropyLoss()
        self.cost2 = nn.L1Loss()

    def forward(self, output, labels, scale_true=10, scale_wrong=-10):
        # scale_wrong = max(sum(output[0])/len(output[0]), 2) * scale
        # scale_true = scale_wrong * 10
        tempOut = torch.where(output>0,output*scale_wrong,torch.zeros_like(output))
        tempLabels = torch.ones_like(output) * scale_wrong
        for i in range(labels.size(0)):
            j = labels[i]
            #tempLabels[i,j] = scale_true 
            tempLabels[i,j] = torch.max(torch.abs(torch.mean(output[i])), torch.abs(output[i,j]))*scale_true
        #output_softmax = nn.LogSoftmax()(output)
        return self.p    * self.cost2(tempOut, tempLabels) \
            + (1-self.p) * self.cost(output,labels)


criterion = nn.CrossEntropyLoss()
#criterion = customLoss()

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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio=3
        self.fc1 = BinarizeLinear(784, 1024*self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(1024*self.infl_ratio)
        self.fc2 = BinarizeLinear(1024*self.infl_ratio, 1024*self.infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(1024*self.infl_ratio)
        self.fc3 = BinarizeLinear(1024*self.infl_ratio, 1024*self.infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(1024*self.infl_ratio)
        self.fc4 = nn.Linear(1024*self.infl_ratio, 10)
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
        return self.logsoftmax(x)
