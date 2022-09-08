"""
 File used for neural network layer structure
"""

import numpy as np
from utils import Activation


class Layer:
    def __init__(self, n_inputs: int, n_neurons: int):
        """
        Constructor
        :param n_inputs: Number of input data in a single layer of a batch
        :param n_neurons: Number of neurons for the network layer instance
        """
        self.output = None
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.05
        self.biases = np.zeros((1, n_neurons))

    def advance(self, inputs, activation: Activation):
        """
        Processes the input and produces an output with the specified activation function
        :param inputs: Inputs needed to produce output data from the instance
        :param activation: Type of activation function, either Rectified Linear Unit or Softmax
        :return: The output from this instance, ready to be fed to the next layer or be treated as final output
        """
        output_raw = np.dot(inputs, self.weights) + self.biases
        if activation == Activation.ReLU:
            self.output = np.maximum(0, output_raw)
        elif activation == Activation.Softmax:
            exp_output = np.exp(output_raw - np.max(output_raw, axis=1, keepdims=True))
            self.output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
        else:
            raise Exception("Unknown activation function")  # Should not be reachable, but just in case
