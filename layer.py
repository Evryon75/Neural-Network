import numpy as np

from utils import Activations


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.05
        self.biases = np.zeros((1, n_neurons))

    def advance(self, inputs, activation):
        output_raw = np.dot(inputs, self.weights) + self.biases
        if activation == Activations.ReLU:
            self.output = np.maximum(0, output_raw)
        elif activation == Activations.Softmax:
            exp_output = np.exp(output_raw - np.max(output_raw, axis=1, keepdims=True))
            self.output = exp_output / np.sum(exp_output, axis=1, keepdims=True)
