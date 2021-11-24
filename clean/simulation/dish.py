
import nest
import matplotlib.pyplot as plt

class Dish():
    def __init__(self, quick=False):
        self.simLength = 1000 #ms
        self.recorders_list = []
        self.massRecorders_list = []
        self.stimuli_list = []
        self.quick = quick
        nest.ResetKernel()

        # Simulation kernel
        nest.SetKernelStatus({
            "local_num_threads": 1,
            "resolution": 0.1
        })

    # run the simulation and display the results
    # verbosity defines which graphs are displayed: 1-none, 2-spikes & stimulation, 3-full multimeter recording
    def record(self, verbose = 4):
        # Run simulation
        nest.Simulate(self.simLength)

        # Define function getting activity from the recorder
        def getActivity(node):
            if node.get("model") == "spike_recorder":
                nodeIds = nest.GetConnections(None, node).sources()
            else:
                nodeIds = nest.GetConnections(node).targets()
            return {
                "events": node.get("events"),
                "nodeIds": list(nodeIds),
                "model": node.get("model")
            }

        # Collect activities
        recordersActivity = [getActivity(recorder) for recorder in self.recorders_list]
        results = []
        counter = 0
        for data in recordersActivity:
            events = None
            if data["model"] == "spike_recorder" and verbose == 1:
                freq = "of frequency "+ str(np.mean(np.diff(np.array(data["events"]["times"])))) if len(data["events"]["times"])>1 else ""
                print("for", counter, "found", len(data["events"]["times"]), "spikes", freq)
                plt.plot(data["events"]["times"], [counter for _ in data["events"]["times"]] , 'rx')
                events = data["events"]["senders"]
                counter = counter + 1
            if data["model"] == "spike_recorder" and verbose > 1:
                plt.figure(figsize=(20,5))
                plt.title(data["nodeIds"])
                plt.plot(data["events"]["times"], data["events"]["senders"], 'rx')
                events = data["events"]["senders"]
                if verbose > 2:
                    for i,stim in enumerate(self.stimuli_list):
                        plt.plot([j for j in range(len(stim)) if stim[int(j)]==1],[i for j in stim if j==1], 'd')
                
            elif data["model"] == "multimeter" and verbose > 3:
                plt.figure(figsize=(20,5))
                plt.title(data["nodeIds"])
                plt.plot(data["events"]["times"], data["events"]["V_m"])
                events = data["events"]["V_m"]
            
            results.append({
                "type": data["model"],
                "times": data["events"]["times"], 
                "events": events
            })
        
        # return debugging information
        events = []
        if not self.quick:
            for data in [getActivity(recorder) for recorder in self.massRecorders_list]:
                events.append(data["events"]["times"])
        return events
        
    
    # add a spike recorder and a multimeter monitors to the node
    def monitorNode(self, node):
        if not self.quick:
            multimeter = nest.Create("multimeter", params={"record_from": ["V_m"]})
            self.recorders_list.append(multimeter)
            self.connect(multimeter, node)

        spikeRecorder = nest.Create("spike_recorder")        
        self.recorders_list.append(spikeRecorder)
        self.connect(node, spikeRecorder,1)
        
        return node
        
    
    # make simple neuron node
    def node(self):
        node = nest.Create("iaf_psc_alpha", 1, params={"C_m": 250.0,"tau_m": 10.0})
        
        # spike recording of the whole dish
        if not self.quick:
            spikeRecorder = nest.Create("spike_recorder")
            self.connect(node, spikeRecorder,1)
            self.massRecorders_list.append(spikeRecorder)
            
        return node


    # generators
    def stimulateNode(self, node, spikes = None):
        stimulator = None
        if spikes is None:
            self.connect(
                nest.Create("dc_generator", 1, params={"amplitude": 400.0}), 
                node)
            self.stimuli_list.append([1 for _ in range(self.simLength)])
        else:
            if callable(spikes):
                spikes = [i for i in range(1,self.simLength+1) if spikes(i)>0.95]
            self.connect(
                nest.Create("spike_generator", 1, params={
                    "spike_times": [x*1.0 for x in spikes] }),
                node,
                1)
            self.stimuli_list.append([1 if j*1.0 in spikes else 0 for j in range(self.simLength)])
        return node
    
    
    # mutable node
    def mutableNode(self, node, freq=3, start=1, magnitude=1000):
        self.connect(
            nest.Create("spike_generator", 1, params={
                "spike_times": [x*1.0 for x in range(int(start+freq/2),self.simLength,freq)]}),
            node,
            -1*magnitude) # inhibitory muting connection
        
    
    # conections
    # weight of 1 means that the spike will propegate to the next node
    # delay is used to make the RNN nodes
    def connect(self, src, dst, weight = None, delay = None):
        # set connection weight - 1 will induce a spike by its own
        synSpec = {} if weight == None else {"weight": 1200 * weight}
        
        # set connection delay 
        synSpec.update({} if delay == None else {"delay": 23.0 * delay})
        
        nest.Connect(src, dst, syn_spec=synSpec)
        
        return {"src":src, "dst":dst}
        