from activation import Activation
import numpy as np

class Backpropogation():
    def __init__(self, input_layer, layers, nb_hidden_layers, biases, weights, prediction):
        self.input_layer = input_layer
        self.layers = layers
        self.nb_hidden_layers = nb_hidden_layers
        self.biases = biases
        self.weights = weights
        self.prediction = prediction
        #placeholders
        self.drv_current_layer = None
        self.drv_inputs_layer = None
        self.drv_weights = [None for _ in range(self.nb_hidden_layers+1)]
        self.drv_biases = [None for _ in range(self.nb_hidden_layers+1)]
        # deriv_chain needed

    def adjust(self, learning_rate=0.01):

        # backprop activation
        def _SoftMax_b(x):
            return = Activation(x).SoftMax(direction=back)
        def _ReLU_b(x)
            return = Activation(x).ReLU(direction=back)

        for step in range(self.nb_hidden_layers,-1,-1):

            if step == self.nb_hidden_layers:
                self.drv_end = _SoftMax_b(self.prediction)       
            elif step != 0:
                self.drv_end = _SoftMax_b(self.layers[step])

            self.drv_biases = np.sum(self.drv_current_layer, axis=0, keepdims=True) #needs testing

            if step != 0:
                self.drv_weights[step] = self.layers[step-1].T@self.drv_current_layer
            else:
                self.drv_weights[step] = self.input_layer.T@self.drv_current_layer
        
        return self.drv_weights, self.drv_biases

        
        
