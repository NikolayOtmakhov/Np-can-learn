import numpy as np

class Activation():
    def __init__(self, input):
        self.input = input

    def ReLU(self,, direction="front"):
        if direction.lower() == "front":
            return np.maximum(0, self.input)
        if direction.lower() == "back":
            stop_reference = self.input.copy()
            stop_reference[stop_reference <= 0] = 0
            return stop_reference

        
    def SoftMax(self, direction="front"):
        if direction.lower() == "front":
            # Removed highest value to prevent overflow
            exp = np.exp(self.input - np.max(self.input, axis=1,keepdims=True))
            return exp / np.sum(exp, axis=1,keepdims=True)
        if direction.lower() == "back":
            return self.input

