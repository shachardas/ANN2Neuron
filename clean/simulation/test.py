print("prepre")
import modules.model2sim as sim
print("post post")
import torch
import torch.nn as nn
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class customLoss(nn.Module):
    def __init__(self, p=0.3):
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
        self.running_var = torch.ones((1,size))
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


class BinarizedRNN(nn.Module):
    # Elman Network

    def __init__(self, inputSize, hiddenSize, outputSize, n_iters, withBias = False):
        super(BinarizedRNN, self).__init__()
        self.hiddenSize = hiddenSize
        self.inputSize = inputSize
        self.outputSize = outputSize
        
        self.InputLinear = BinarizeLinear(inputSize, hiddenSize, bias = withBias)
        self.hiddenLinear = BinarizeLinear(hiddenSize, hiddenSize, bias = withBias)
        self.outputLinear = BinarizeLinear(hiddenSize, outputSize, bias = withBias)

        self.gates = nn.Parameter(torch.ones(n_iters))
        self.hiddenLinear.weight = nn.Parameter(torch.eye(hiddenSize))

        self.tanh = nn.Hardtanh()
        self.hiddenBatchNorm = SignSensitiveBatchNorm1d(hiddenSize)
        self.outputBatchNorm = SignSensitiveBatchNorm1d(outputSize)

        self.hiddenOutput = torch.zeros(hiddenSize).to(device)
        

    def forward(self, x):
        self.hiddenOutput = torch.zeros(1,self.hiddenSize).to(device)
        out = torch.zeros(x.size()[0],x.size()[1],self.outputSize).to(device)
        for i, x_i in enumerate(x):
            hidden = self.InputLinear(x_i) + self.gates[i]*self.hiddenLinear(self.hiddenOutput)
            hidden = self.tanh(hidden)
            hidden = self.hiddenBatchNorm(hidden)
            self.hiddenOutput = hidden

            out_i = self.outputLinear(hidden)
            out[i] = out_i
        return out

class Net(nn.Module):
    def __init__(self, inputSize = 2, outputSize = 5, hiddenSize = 50, n_iters = 20):
        super(Net, self).__init__()
        
        self.BRNN1 = BinarizedRNN(inputSize=inputSize,hiddenSize=hiddenSize,outputSize=outputSize,n_iters=n_iters)

    def forward(self,x):
        x = self.BRNN1(x)
        return x[-1]


if __name__ == "__main__":
    print("started")
    
    batchSize = 500 
    n_classes = 10
    data_bits = 2
    seq_len = 5
    hidden_size = 50
    learning_rate = 0.01

    model = Net(inputSize=data_bits, outputSize=n_classes, hiddenSize=hidden_size, n_iters=seq_len)
    print("1")
    model.load_state_dict(torch.load("/home/shachar/nestDocker/ANN2Neuron/clean/simulation/trained_models/rnn.pt",map_location=torch.device('cpu')))
    print("2")
    model.eval()
    print("model loaded")
    seqs = torch.load('/home/shachar/nestDocker/ANN2Neuron/clean/simulation/trained_models/rnn.squences')
    print("3")
    res, model_act = sim.testRNN(model, seqs)