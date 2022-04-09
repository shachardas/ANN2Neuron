import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from binarized_modules import  BinarizeLinear,BinarizeConv2d
from binarized_modules import  Binarize,HingeLoss
import random


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).float().to(device = 'cuda')

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_()
            sampled_noise = sampled_noise * scale
            x = x + sampled_noise
        return x

class randomChooser(nn.Module):
    def __init__(self, scale = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if self.training:
            try:
                chosenIndex = random.choice((torch.sum(x.detach(),0)>=0).nonzero())
            except:
                chosenIndex = 0
            x = torch.ones_like(x)*-1.0*self.scale
            x[torch.arange(x.size(0)), chosenIndex] = torch.tensor(1).to(device = 'cuda')*self.scale
            x.requires_grad = True

        return x


class customLoss(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.cost = nn.CrossEntropyLoss()
        self.cost2 = nn.L1Loss()
        #self.noise = torch.tensor(0).float().to(device = 'cuda')

    def forward(self, output, labels, scale_wrong=2, scale_true=2):
        # tempOut = torch.where(output>0,output*scale_wrong,torch.zeros_like(output))
        # tempLabels = torch.zeros_like(output)
        # for i in range(labels.size(0)):
        #     j = labels[i]
        #     tempLabels[i,j] = torch.max(torch.abs(torch.mean(output[i])), torch.abs(output[i,j]))*scale_true
        # return self.p*self.cost2(tempOut, tempLabels) + (1-self.p)*self.cost(output-torch.ones_like(output),labels)
        mean = torch.abs(torch.mean(output))*2
        tempLabels = torch.ones_like(output)*-20
        for i in range(labels.size(0)):
            j = labels[i]
            tempLabels[i,j] = 100
        # tempOut = torch.where(output>0,torch.max(output) - output,torch.min(output) + output)
        # return 0.2*self.cost(tempOut,labels) + self.cost(output,labels)
        return 0.2*self.cost2(output,tempLabels) + self.cost(output,labels)


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()
        self.weight.data=Binarize(self.weight.org)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.cv1 = BinarizeConv2d(1, 8, 5, bias=False, stride=1, padding=0)
        #self.htanh0 = nn.Hardtanh()
        #self.mp = nn.MaxPool2d(kernel_size=2,stride=2)
        #self.cv2 = BinarizeConv2d(8, 16, 5, bias=False, stride=1, padding=2)
        #self.htanh01 = nn.Hardtanh()
        #self.mp2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = BinarizeLinear(784, 400, bias=False)
        #self.bn1 = nn.BatchNorm1d(50)
        #self.htanh1 = nn.Hardtanh()
        self.fc2 = BinarizeLinear(400, 200, bias=False)
        #self.htanh2 = nn.Hardtanh()
        #self.bn2 = nn.BatchNorm1d(7*7*4)
        self.fc3 = BinarizeLinear(200, 10, bias=False)
        #self.htanh3 = nn.Hardtanh()
        #self.bn3 = nn.BatchNorm1d(7*7*2)
        #self.fc4 = BinarizeLinear(50, 10, bias=False)
        #self.htanh4 = nn.Hardtanh()
        #self.bn4 = nn.BatchNorm1d(7*7*2)
        #self.fc5 = BinarizeLinear(7*7*2, 7*7, bias=False)
        #self.htanh5 = nn.Hardtanh()
        #self.bn5 = nn.BatchNorm1d(7*7)
        #self.fc6 = BinarizeLinear(7*7, 25, bias=False)
        #self.htanh6 = nn.Hardtanh()
        #self.gn = GaussianNoise(sigma=1)
        #self.bn6 = nn.BatchNorm1d(25)
        #self.fc7 = BinarizeLinear(25, 10, bias=False)
        #self.htanh7 = nn.Hardtanh()
        #self.bn7 = nn.BatchNorm1d(10)7326
        #self.fc8 = BinarizeLinear(25, 10, bias=False)
        #self.logsoftmax=nn.LogSoftmax()
        #self.drop=nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28*28)
        #x = self.cv1(x)
        #x = self.htanh0(x)
        #x = self.mp(x)
        #x = self.cv2(x)
        #x = self.htanh01(x)
        #x = self.mp2(x)
        #x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        #x = self.bn1(x)
        #x = self.htanh1(x)
        x = self.fc2(x)
        #x = self.gn(x)
        #x = self.bn2(x)
        #x = self.htanh2(x)
        x = self.fc3(x)
        #x = self.drop(x)
        #x = self.bn3(x)
        #x = self.htanh3(x)
        #x = self.fc4(x)
        #x = self.bn4(x)
        #x = self.htanh4(x)
        #x = self.fc5(x)
        #x = self.bn5(x)
        #x = self.htanh5(x)
        #x = self.fc6(x)
        #x = self.bn6(x)
        #x = self.htanh6(x)
        #x = self.fc7(x)
        #x = self.bn7(x)
        #x = self.htanh7(x)
        #x = self.fc8(x)
        #x = self.logsoftmax(x)
        return x
