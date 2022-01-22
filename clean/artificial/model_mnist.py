import torch
import torch.nn as nn

class customLoss(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.cost = nn.CrossEntropyLoss()
        self.cost2 = nn.L1Loss()

    def forward(self, output, labels, scale_wrong=20, scale_true=100):
        tempLabels = torch.ones_like(output)*-1*scale_wrong
        for i in range(labels.size(0)):
            j = labels[i]
            tempLabels[i,j] = scale_true
        output_softmax = nn.LogSoftmax()(output)
        return self.p    * self.cost2(output, tempLabels) \
            + (1-self.p) * self.cost(output_softmax,labels)


#criterion = nn.CrossEntropyLoss()
criterion = customLoss()

def Binarize(tensor, include_zero = False, minSig=3):
    if include_zero:
        P_std = 0.25
        up_lim = torch.min(0 + P_std*tensor.std(), torch.ones_like(tensor)*minSig)
        down_lim = torch.max(0 - P_std*tensor.std(), -1*torch.ones_like(tensor)*minSig)
        up_v = (tensor>up_lim).float()
        down_v = (tensor<down_lim).float().mul(-1)
        return (up_v + down_v)
    else:
        return tensor.sign()

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
        self.fc1 = BinarizeLinear(784, 400, bias=False)
        self.bn1 = nn.BatchNorm1d(400)
        self.htanh1 = nn.Hardtanh()
        self.drop=nn.Dropout(0.5)
        self.fc3 = BinarizeLinear(400, 200, bias=False)
        self.bn3 = nn.BatchNorm1d(200)
        self.htanh3 = nn.Hardtanh()
        self.fc4 = BinarizeLinear(200, 10, bias=False)
        self.logsoftmax=nn.LogSoftmax()

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        #x = self.bn1(x)
        #print("b4", x)
        x = self.htanh1(x)
        #print("after", x)
        #exit()
        x = self.drop(x)
        x = self.fc3(x)
        #x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        #x = self.logsoftmax(x)
        return x

    def test(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

