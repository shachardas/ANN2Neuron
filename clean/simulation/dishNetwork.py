from dish import Dish
import torch


class DishNetwork():
    def __init__(self, layers, weights):
        self.dish = Dish()
        self.inputLayer = []
        self.layers = layers
        self.weights = weights
        self._mkANN(layers, weights)
    
    def _connectLayers(self, this, that, weights, delay = None):
        print("connecting layer", len(this), len(that))
        for n1 in range(len(this)):
            if n1%100 == 0:
                print("connected",n1,"nodes in the layer")
            for n2 in range(len(that)):
                if(weights[n1][n2] != 0):
                    #print(n1,"->",n2,float(weights[n1][n2]))
                    self.dish.connect(this[n1], that[n2], float(weights[n1][n2]), delay)
        
    def _connectRecurrentLayer(self, layer, weights):
        #print("adding recurrent layer of",layer)
        recLayer = [self.dish.node() for _ in range(len(layer))]
        self._connectLayers(layer, recLayer, [[1 if i == j else 0 for i in range(len(layer))] for j in range(len(layer))], 0.5)
        self._connectLayers(recLayer, layer, weights, 0.5)
    
    def _mkANN(self, layers, weights):
        assert(len(layers)==len(weights)+1)
        #print("simulating ANN of layers",layers)
        self.inputLayer = [self.dish.node() for i in range(layers[0])]
        lastLayer = self.inputLayer
        
        for layer, weight, idx in zip(layers[1:], weights, range(len(weights))):
            nextLayer = [self.dish.node() for i in range(layer)]
            # recurrent weights
            if type(weight) is tuple:
                self._connectRecurrentLayer(lastLayer, weight[1])
                weight = weight[0]
            self._connectLayers(lastLayer, nextLayer, weight)
            #to monitor all
            #[self.dish.monitorNode(node) for node in lastLayer]   
            
            #to mute repetition
            [self.dish.mutableNode(node,freq=3,start=6+3*idx, magnitude=10000) for node in lastLayer]
            
            # noise supression
            if idx<len(weights)-1:
                k = int(layer/2)
                neg = nextLayer[k:]
                pos = nextLayer[:k]
                self._connectLayers(neg, pos, (-1*torch.eye(k)).tolist())
                self._connectLayers(pos, neg, (-1*torch.eye(k)).tolist())
            lastLayer = nextLayer
        [self.dish.monitorNode(node) for node in lastLayer] 
    
    # stimulate the network
    # accepts a function as stimulation f(i,t)=1 iff the node i spikes at time t, f(i,t)=0 otherwise
    # accepts a 2D list where list[i][t]=1 iff the node i spikes at time t, list[i][t]=0 otherwise
    def stimulate(self, stim = None, simLen = 1000, verbosity = 3):
        self.dish.stimuli_list = []
        self.dish.simLength = simLen
        for i, neuron in enumerate(self.inputLayer):
            if i%20 == 0:
                pass #print("connected stimulation to",i,"nodes")
            if not stim:
                self.dish.stimulateNode(neuron)
            elif callable(stim):
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
