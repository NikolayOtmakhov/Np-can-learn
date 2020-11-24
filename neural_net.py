from forward import Forward
from loss import CrossEntropy
from check_data import Setup
import numpy as np

class myNN():

    def __init__(self, input_layer, output_layer, neurons_per_hidden_layer=0):
        # Make sure the data/inputs provided will work
        self.input_layer, self.input_features \
                = Setup.input_layer_check(input_layer)
        self.neurons_per_hidden_layer , self.nb_hidden_layers \
                = Setup.hidden_layer_check(neurons_per_hidden_layer)
        self.output_layer, self.nb_unique_outputs \
                = Setup.output_layer_check(output_layer) 
        # Initialize
        self.weights = self.weights_init()
        self.biases = self.bias_init()
        # Placeholders
        self.layers = [None for _ in range(self.nb_hidden_layers)]
        self.prediction = None
        self.loss_result = None
        self.acc_result = None

    def test(self):
        import activation
        self.input000 = activation.Activation(self.prediction).SoftMax(direction="back")

    # Main Functions
    def forward(self,show=False):
        F = Forward(input_layer = self.input_layer,
                    layers = self.layers,
                    nb_hidden_layers=self.nb_hidden_layers, 
                    biases=self.biases, 
                    weights = self.weights,)
        self.layers, self.prediction = F.classify()

        # Test Print - for predictions layer
        if show!=False:
            try:
                print(self.prediction[:show])
                print(f">>> {show} out of " + str(self.prediction.shape[0]))                  
            except:
                print("Scilent Error: wrong show parameter in forward")

    def loss(self, show=False):
        self.loss_result = CrossEntropy().get_loss(
                            self.prediction,self.output_layer)
        self.acc_result = CrossEntropy().get_accuracy(
                            self.prediction,self.output_layer)
        
        # Test Print - for predictions layer
        if show!=False:
            print(f"LOSS: {self.loss_result}")
            print(f"ACC: {self.acc_result}")
        
    
    # Init Functions
    def weights_init(self):
        list_of_layers = []
        for i in range(self.nb_hidden_layers+1):
            # Count neurons on the left (input) side
            if i == 0:
                start_neurons = self.input_features
            else:
                start_neurons = self.neurons_per_hidden_layer[i-1]
            # Count neurons on the right (output) side
            if i == self.nb_hidden_layers:
                end_neurons = self.nb_unique_outputs
            else:
                end_neurons = self.neurons_per_hidden_layer[i]
            # Initialize with very small values
            weight_layer = 0.01*np.random.rand(start_neurons,end_neurons)
            list_of_layers.append(weight_layer)
        return list_of_layers 

    def bias_init(self):
        list_of_layers = []
        for i in range(self.nb_hidden_layers+1):
            # Count neurons in layer
            if i == self.nb_hidden_layers:
                nb_biases = self.nb_unique_outputs
            else:
                nb_biases = self.neurons_per_hidden_layer[i]
            # Initialize all layers with zeros
            list_of_layers.append(np.zeros((1, nb_biases)))
        return list_of_layers