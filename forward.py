from activation import Activation
import numpy as np

class Forward():
    def __init__(self, input_layer, layers, nb_hidden_layers, biases, weights):
        self.input_layer = input_layer
        self.layers = layers
        self.nb_hidden_layers = nb_hidden_layers
        self.biases = biases
        self.weights = weights

    def classify(self):
        for step in range(self.nb_hidden_layers+1):
            if step == 0:
                current_layer = self.input_layer
            else:
                current_layer = self.layers[step - 1]
            current_layer = np.dot(current_layer, self.weights[step]) + self.biases[step]
            # ReLU for hidden layers
            if step != self.nb_hidden_layers:
                current_layer = Activation(current_layer).ReLU()
                self.layers[step] = current_layer
            # SoftMax for final layer
            else:
                current_layer = Activation(current_layer).SoftMax()
        forward_result = current_layer
        return self.layers, forward_result
        
