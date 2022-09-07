import enum

import numpy as np

X = np.random.randn(4, 10)

np.random.seed(0)


class Activations(enum.Enum):
    ReLU = 1
    Softmax = 2


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


def calculate_loss(prediction, expected):
    """
    Readable version:
        length = len(prediction)
        clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        neg_log = -np.log(clipped[range(length), expected])
        return np.mean(neg_log)
    """
    return np.mean(-np.log(np.clip(prediction, 1e-7, 1 - 1e-7)[range(len(prediction)), expected]))


input_layer = Layer(10, 5)
layer_one = Layer(5, 5)
layer_two = Layer(5, 5)
output_layer = Layer(5, 2)

input_layer.advance(X, Activations.ReLU)
layer_one.advance(input_layer.output, Activations.ReLU)
layer_two.advance(layer_one.output, Activations.ReLU)
output_layer.advance(layer_two.output, Activations.Softmax)

print(output_layer.output)

loss = calculate_loss(output_layer.output, [1, 1, 0, 0])
print(loss)

# https://www.youtube.com/watch?v=dEXPMQXoiLc
