"""
 Main file
"""

import numpy as np
from layer import Layer
from optimization import calculate_loss
from utils import Activation

X = np.random.randn(4, 10)

input_layer = Layer(10, 5)
layer_one = Layer(5, 5)
layer_two = Layer(5, 5)
output_layer = Layer(5, 2)

input_layer.advance(X, Activation.ReLU)
layer_one.advance(input_layer.output, Activation.ReLU)
layer_two.advance(layer_one.output, Activation.ReLU)
output_layer.advance(layer_two.output, Activation.Softmax)

print(output_layer.output)

loss = calculate_loss(output_layer.output, [1, 1, 0, 0])
print(loss)
