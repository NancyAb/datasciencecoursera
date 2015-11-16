import numpy as np
import matplotlib.pyplot as plt

class Neural_Network(object):
    def __init__(self):
        # Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3
        
        #Weights (Parameters)

        self.W1 = np.random.randn(self.inputLayerSize, \
                                 self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, \
                                 self.outputLayerSize)

    def forward(self, X):
        #Propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function
        return 1/(1+np.exp(-z))

if __file__ == '__main__':

    testInput = np.arange(-6, 6, 0.01)
    nn = Neural_Network()
    plt.plot(testInput, nn.sigmoid(testInput), linewidth = 2)
    plt.grid(1)
    plt.show()
