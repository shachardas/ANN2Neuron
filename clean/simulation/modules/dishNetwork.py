from re import I
from modules.dish import Dish
import torch
import time


class DishNetwork():
    def __init__(self, layers, weights):
        self.dish = Dish()
        self.inputLayer = []
        self.layers = layers
        self.weights = weights
        self._mkANN(layers, weights)
    
    def _connectLayers(self, this, that, weights, delay = None):
        print(f"connecting {len(this)} neurons to {len(that)} neurons")
        #from tensor to lists
        t = time.process_time()
        weights = torch.tensor(weights)
        self.dish.connect(this, that, weights.t(), delay)
        elapsed_time = time.process_time() - t
        print(f"this took {elapsed_time} seconds")

        """
        for n1 in range(len(this)):
            if n1%100 == 50:
                print("connected",n1,"nodes in the layer")
            for n2 in range(len(that)):
                if(weights[n1][n2] != 0):
                    #print(n1,"->",n2,float(weights[n1][n2]))
                    self.dish.connect(this[n1], that[n2], float(weights[n1][n2]), delay)
        """
        
    def _connectRecurrentLayer(self, layer, weights):
        #print("adding recurrent layer of",layer)
        recLayer = self.dish.node(len(layer))
        self._connectLayers(layer, recLayer, [[1 if i == j else 0 for i in range(len(layer))] for j in range(len(layer))], 0.5)
        self._connectLayers(recLayer, layer, weights, 0.5)
    
    def _mkANN(self, layers, weights):
        assert(len(layers)==len(weights)+1)
        print("simulating ANN of layers",layers)
        print(f"creating {layers[0]} nodes")
        self.inputLayer = self.dish.node(size=layers[0])
        lastLayer = self.inputLayer
        
        for layer, weight, idx in zip(layers[1:], weights, range(len(weights))):
            print(f"creating {layer} nodes")
            nextLayer = self.dish.node(size=layer)
            # recurrent weights
            if type(weight) is tuple:
                self._connectRecurrentLayer(lastLayer, weight[1])
                weight = weight[0]
            self._connectLayers(lastLayer, nextLayer, weight)
            '''
            #to monitor all
            [self.dish.monitorNode(node) for node in lastLayer]   
            
            #to mute repetition
            [self.dish.mutableNode(node,freq=3,start=6+3*idx, magnitude=10000) for node in lastLayer]
            
            #noise supression
            if idx<len(weights)-1:
                k = int(layer/2)
                neg = nextLayer[k:]
                pos = nextLayer[:k]
                self._connectLayers(neg, pos, (-1*torch.eye(k)).tolist())
                self._connectLayers(pos, neg, (-1*torch.eye(k)).tolist())
            '''
            lastLayer = nextLayer
            
        self.dish.monitorNode(lastLayer) 
    
    # stimulate the network
    # accepts a function as stimulation f(i,t)=1 iff the node i spikes at time t, f(i,t)=0 otherwise
    # accepts a 2D list where list[i][t]=1 iff the node i spikes at time t, list[i][t]=0 otherwise
    def stimulate(self, stim = None, simLen = 1000, verbosity = 3, isSimple = True):
        self.dish.stimuli_list = []
        self.dish.simLength = simLen
        print("creating stimulation")
        """
        for i, neuron in enumerate(self.inputLayer):
            if i%20 == 0:
                pass #print("connected stimulation to",i,"nodes")
            if not stim:
                self.dish.stimulateNode(neuron)
            elif callable(stim):
                self.dish.stimulateNode(neuron, lambda t: stim(i,t))
            elif type(stim) is list:
                self.dish.stimulateNode(neuron, stim[i])
        """
        if not stim:
            for i, neuron in enumerate(self.inputLayer):
                self.dish.stimulateNode(neuron)
        elif isSimple:
            self.dish.SimpleStimulateNodes(self.inputLayer,stim)
        else:
            for i, neuron in enumerate(self.inputLayer):
                if callable(stim):
                    self.dish.stimulateNode(neuron, lambda t: stim(i,t))
                elif type(stim) is list:
                    self.dish.stimulateNode(neuron, stim[i])
        print("starting to record")
        events = self.dish.record(verbosity)
        layeredEvents = []
        if events != []:
            for layer in self.layers:
                layeredEvents.append(events[:layer])
                events = events[layer:]
        return layeredEvents
