import pickle
import torch
from .dishNetwork import DishNetwork
import numpy as np
import matplotlib.pyplot as plt
import os

# generate a biological network simulation that correlates to the trained ANN
def SimfromModel(trainedModel):
    weights = []
    layers = []
    state = trainedModel.state_dict()
    nextLayer = 0
    for key, value in state.items():
        #print("parsing layer:",key, value.size())
        if key.split(".")[1] == "weight" and key.split(".")[0].startswith("fc"):
            layers.append(value[0].size()[0]*2)
            nextLayer = value.size()[0]*2
            W1 = torch.cat((value, -1*value) , 0)
            W2 = torch.cat((-1*value, value) , 0)
            weights.append(torch.transpose(torch.cat((W1,W2),1),0,1))

    layers.append(nextLayer)

    # account for the positive layers
    for i, (key, value) in enumerate(state.items()):
        if key.split(".")[1] == "weight" and key.split(".")[0].startswith("fcp"):
            layers[i] = int(layers[i]/2)
            weights[i] = weights[i].narrow(0,0,int(weights[i].size()[0]/2))
            if i > 0:
                weights[i-1] = weights[i-1].narrow(1,0,int(weights[i-1].size()[1]/2))


    # trim the input and output layers
    layers[0] = int(layers[0]/2)
    layers[-1] = int(layers[-1]/2)
    weights[0] = weights[0].narrow(0,0,int(weights[0].size()[0]/2))
    weights[-1] = weights[-1].narrow(1,0,int(weights[-1].size()[1]/2))
    return DishNetwork(layers, weights)

# convert samples input to a correlating stimuli 
def samples2stim(sample):
    arr = []
    for row in sample:
        for cell in row:
            if cell == 1:
                arr.append(1)
            else:
                arr.append(0)
    return arr

# visualize the network input and output (as MNIST)
def testMNIST(model):
    events = {}
    model_act = {}
    cleanSamples = None
    with open("mnist.sample", 'rb') as file:
        cleanSamples = pickle.load(file)
    for label in cleanSamples:  

        pixels = cleanSamples[label]
        pixels = np.array(pixels, dtype='uint8')
        pixels = pixels.reshape((28, 28))
        plt.figure()
        plt.title('Label is {label}'.format(label=label))
        plt.imshow(pixels, cmap='gray')
        plt.show()
        
        print("ANN result:", model(torch.as_tensor(cleanSamples[label]).float()))

        net = SimfromModel(model)      
        plt.figure(figsize=(20,5))
        plt.title("simulating for "+str(label))
        res = net.stimulate(samples2stim(cleanSamples[label]), simLen = 100, verbosity=1)
        events.update({label:res})
        print("simulation result:")
        plt.show()
        #model_act.update({label:model.activity})
        model_act.update({label:None})
    return (events, model_act)

# investigate the activity of the simulated network activity on the sample
# model: the torch model
# res, model_act: correlate to (events, model_act) from testMNIST
# number: the label of the specific run from the samples 
# layer: the layer in the model we want to investigate as source of signals
# targetInNextLayer: the index of the node in the next layer that will be the 
#                    recieving end of the investigated signals
def testSim(model, res, model_act, number,layer,targetInNextLayer):
    modelLayer = getattr(model, "fc"+str(layer+1))
    
    spikes = res[number][layer]
    weights = torch.cat((modelLayer.weight[targetInNextLayer],-1*modelLayer.weight[targetInNextLayer]),0)

    model_act = model_act[number][layer][0]
    sum = 0
    parse = {}
    hist = []
    for spike, weight, i in zip(spikes, weights, list(range(int(len(weights)/2)))*2):
        signal = 1 if list(spike) != [] else 0
        sum += signal*weight
        if i in parse.keys():
            if list(spike) != []:
                hist.append(-1*spike[0])
            parse[i].update({"N-x":f"{signal}*{weight}::{signal*weight}"})
            parse[i].update({"N-a":signal})
            parse[i].update({"N-t":spike})
            parse[i].update({"signals are not negative together":parse[i]["P-a"]+parse[i]["N-a"]==1 or (parse[i]["P-a"] and parse[i]["N-a"]==0)})
            parse[i].update({"M-a":int(model_act[i])})
            parse[i].update({"singal match model":(parse[i]["P-a"] == int(model_act[i])) or (parse[i]["N-a"] == -1*int(model_act[i]))})
        else:
            if list(spike) != []:
                hist.append(spike[0])
            parse.update({i:{"P-s":f"{signal}*{weight}::{signal*weight}",
                             "P-a":signal,
                             "P-t":spike}})
    plt.figure()
    plt.hist(hist,bins=50)
    return (parse, int(sum))

if __name__ == '__main__':
    #tests

    import torch
    import torch.nn as nn
    def Binarize(tensor, include_zero = False):
        if include_zero:
            return ((tensor+0.5).sign()+(tensor-0.5).sign())/2
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

    class PositiveBinarizeLinear(nn.Linear):

        def __init__(self, *kargs, **kwargs):
            super(PositiveBinarizeLinear, self).__init__(*kargs, **kwargs)
    
        def forward(self, input):
            zero = torch.zeros_like(input.data)
            input.data = torch.where(input.data > 0, input.data, zero)
            input.data=Binarize(input.data)
            if not hasattr(self.weight,'org'):
                self.weight.org=self.weight.data.clone()
            self.weight.data=Binarize(self.weight.org)
            out = nn.functional.linear(input, self.weight)
            if not self.bias is None:
                self.bias.org=self.bias.data.clone()
                out += self.bias.view(1, -1).expand_as(out)

            return out

    class Predefined_policy(nn.Module):
        def __init__(self):
            super(Predefined_policy, self).__init__()
            self.fc = BinarizeLinear(4, 2, bias = False)
            self.fcp = PositiveBinarizeLinear(2, 1, bias = False)
            self.fc.weight = nn.Parameter(torch.tensor([[1.0,0.0,-1.0,0],[0.0,1.0,0,-1.0]]))
            self.fcp.weight = nn.Parameter(torch.tensor([[1.0,1.0]]))
            

            self.saved_log_probs = []
            self.rewards = []

        def parseInput(self, x):
            theta, w = x[0][2:4]
            res = torch.zeros([1,4])
            res[0][0] = float(theta > 0)
            res[0][1] = float(w > 0)
            res[0][2] = float(abs(theta) < 0.03)
            res[0][3] = float(abs(theta) >= 0.03)
            return res

        def forward(self, x):
            x = self.parseInput(x)
            x = self.fc(x)
            action_scores = self.fcp(x)
            return action_scores


    def parseInput(x):
        theta, w = x[2:4]
        res = torch.zeros([4])
        res[0] = float(theta > 0)
        res[1] = float(w > 0)
        res[2] = float(abs(theta) < 0.03)
        res[3] = float(abs(theta) >= 0.03)
        return res

    model = Predefined_policy()
    model.load_state_dict(torch.load("/home/shachar/nestDocker/ANN2Neuron/clean/simulation/trained_models/predefined-CartPole.pt",map_location=torch.device('cpu')))
    net = SimfromModel(model)
    res = net.stimulate([1,1,1,0], simLen = 30, verbosity=1)
    print(len(res[0][0]["times"]))

    import gym
    env = gym.make('CartPole-v1')
    state = env.reset()
    frames = []
    for t in range(1, 1000):  # Don't infinite loop while learning
        net = SimfromModel(model)
        print(state, parseInput(state))
        res = net.stimulate(parseInput(state), simLen = 30, verbosity=1)
        action = len(res[0][0]["times"])
        state, reward, done, _ = env.step(action)
        frames.append(env.render(mode="rgb_array"))
        if done:
            print("done!")
            break