import numpy as np

from layer import Layer, Activations
from utils import calculate_loss

X = np.random.randn(4, 10)

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
