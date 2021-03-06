#origin https://github.com/itayhubara/BinaryNet.pytorch/blob/master/main_mnist.py

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
from model_mnist import Net, criterion
import config



# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64*8, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu-id', default=1,
                    help='gpu id used in cuda.set_device()')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--overfit', action='store_true', default=False,
                    help='(bool) use sampled dataset in order to overfit')
parser.add_argument('--halt-on-find', action='store_true', default=False,
                    help='stop training the model when the constant sample is currectly predicted')
parser.add_argument('--model-name', type=str, default="test", metavar='N',
                    help='how would you name your model (will be used to generate the notebook)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


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
    signed_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            Parallized_model = torch.nn.DataParallel(model)
            output = Parallized_model((data>0).float())
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max output
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            indicator_out = torch.where(output.data>0,torch.ones_like(output.data),torch.zeros_like(output.data))
            signed_pred = indicator_out.sum(1, keepdim=True)
            signed_correct += (pred.eq(target.data.view_as(pred)).logical_and(signed_pred.eq(torch.ones_like(signed_pred)))).cpu().sum() # make sure the sign is currect

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Signed Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset),
        signed_correct, len(test_loader.dataset),
        100. * signed_correct / len(test_loader.dataset)))


# consistent sample of the dataset, to better evaluate how well the model behaves in sim
def getConstantSample(dataSet):
    SAMPLE_DATA = config.SIMULATION_DIR + "mnist.sample"
    sampleData = None
    if os.path.isfile(SAMPLE_DATA):
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
    return sampleData

# evauate what the results should be in simulation
def evaluateForSim(model, dataSet):
    sampleData = getConstantSample(dataSet)

    passed = True
    for i in range(10):
        if torch.count_nonzero(
            torch.greater(
                model(torch.as_tensor(sampleData[i]).float().to(device))
                ,0)
                ) != 1:
            print(f"For label {i} there wasn't just one positive logit")
            print(model(torch.as_tensor(sampleData[i]).float().to(device)))
            
            passed = False
            break
    if passed:
        print("All the sample examples were currectly classified")
        
        if args.halt_on_find:
            for i in range(10):
                print(f"for label {i} the output was \n{model(torch.as_tensor(sampleData[i]).float().to(device))}")
            
            exit() # stop when the results are what we wanted

def generateTestingNotebook():
    notebookContent = ""
    with open(config.SIMULATION_DIR + "testing.template.ipynb",'r') as f:
        notebookContent = f.read()

    with open(config.ANN_DIR + "model_mnist.py", 'r') as f:
        notebookContent = notebookContent.replace("#model#",f.read().replace("\\","\\\\").replace("\"","\\\"").replace("\t","\\t").replace("\n","\\n\",\n\t\""))
    
    notebookContent = notebookContent.replace("#name#", args.model_name)

    with open(config.SIMULATION_DIR + args.model_name + ".sim.ipynb",'w') as f:
        f.write(notebookContent)
    

if __name__ == "__main__":
    print("*** program started ***")
    
    generateTestingNotebook()
    print("*** generated testing notebook ***")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    sample = torch.utils.data.Subset(datasets.MNIST(config.ROOT_DIR+'data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), list(range(600))*100)
    fullData = datasets.MNIST(config.ROOT_DIR+'data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    dataSet = sample if args.overfit else fullData
    train_loader = torch.utils.data.DataLoader(
        dataSet,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataSet,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    print("*** training started ***")
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()
        if epoch%40==0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*0.1

        torch.save(model.state_dict(), config.TRAINED_MODELS_DIR + args.model_name + ".pt")
        
        evaluateForSim(model, dataSet)


