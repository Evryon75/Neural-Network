import numpy as np

X = [
    [1.32, 2.12, 1.5],
    [-2.1, -0.2, 4.1],
    [3.63, 2.12, -3.3]
]

np.random.seed(0)


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.output = None
        self.weights = 0.2 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def advance(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Layer(3, 5)
layer2 = Layer(5, 2)

layer1.advance(X)
layer2.advance(layer1.output)

print("L1: {\n", layer1.output, "\n{")
print("L2: {\n", layer2.output, "\n}")
