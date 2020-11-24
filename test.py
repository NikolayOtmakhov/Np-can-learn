from neural_net import myNN
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np

X, y = spiral_data(samples=100, classes=3)
# plt.scatter(X[:,0],X[:,1])
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')

t = myNN(X, y, neurons_per_hidden_layer=[4,4]) 
t.forward(show=3)
np.argmax(t.prediction,axis=1)
t.loss(show=True)

# plt.scatter(X[:, 0], X[:, 1], c=np.argmax(t.prediction, axis=1), cmap='brg')
