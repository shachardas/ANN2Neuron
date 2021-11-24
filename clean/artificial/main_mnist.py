
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from binarized_modules import  BinarizeLinear,BinarizeConv2d
from binarized_modules import  Binarize,HingeLoss
import random
import os.path
import pickle

ROOT_DIR = "/home/shachar/nestDocker/ANN2Neuron/clean/"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64*8, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu-id', default=1,
                    help='gpu id used in cuda.set_device()')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

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


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        Parallized_model = torch.nn.DataParallel(model)
        output = Parallized_model((data>0).float())
        loss = criterion(output, target)

        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1
        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p,'org'):
                p.org.copy_(p.data.clamp_(-1,1))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            Parallized_model = torch.nn.DataParallel(model)
            output = Parallized_model((data>0).float())
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

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


if __name__ == "__main__":
    print("*** program started ***")
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # dataSet = torch.utils.data.Subset(datasets.MNIST(ROOT_DIR+'data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])), list(range(600))*100)
    dataSet = datasets.MNIST(ROOT_DIR+'data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    train_loader = torch.utils.data.DataLoader(
        dataSet,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataSet,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    #if args.cuda:
        #device = torch.device("cuda")
        #torch.cuda.set_device(args.gpu_id)
    model.to(device)#cuda()

    #criterion = HingeLoss()
    criterion = customLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # consistent sample of the dataset, to better evaluate how well the model behaves in sim
    SAMPLE_DATA = ROOT_DIR + "mnist.sample"
    sampleData = None
    if os.path.isfile(SAMPLE_DATA):
        sampleData = None
        with open(SAMPLE_DATA, 'rb') as file:
            sampleData = pickle.load(file)
    else:
        sampleData = {}
        i = 0
        while len(sampleData.keys()) < 10:
            label = dataSet[i][1]
            singleData = dataSet[i][0][0] 
            sampleData.update({label: (singleData.to(device) >= 0).int()})
            i = i+1
        with open(SAMPLE_DATA,'wb') as file:
            pickle.dump(sampleData,file)

    # for data, label in singleData[20:]:
    #     sampleData.update({label: (data.cuda()) >= 0).int()})

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.5

        torch.save(model.state_dict(), ROOT_DIR + "trained_models/test.pt")

        passed = True
        for i in range(10):
            if torch.count_nonzero(
                torch.greater(
                    model(torch.as_tensor(sampleData[i]).float())
                    ,0)
                    ) != 1:
                print(f"line {i} was a fluke")
                print(model(torch.as_tensor(sampleData[i]).float()))
                passed = False
                break
        if passed:
            print("bingo")
            for i in range(10):
                print(model(torch.as_tensor(sampleData[i]).float()))
            exit()
'''
    criterion = customLoss(10,10)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs*100 + 1):
        train(epoch)
        test()
        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        torch.save(model.state_dict(), "mnist_trained_with_test.pt")

'''