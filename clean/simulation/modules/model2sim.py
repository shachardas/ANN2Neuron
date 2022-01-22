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

