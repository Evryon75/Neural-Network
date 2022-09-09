"""
 Main file
"""

import warnings
import numpy as np
import pandas as pd
from layer import Layer
from optimization import calculate_loss, back_propagation
from utils import Activation, parse_pass, batcher

warnings.simplefilter(action='ignore', category=FutureWarning)  # Annoying future warnings from pandas

# Reading from data set
data = pd.read_csv("data set/student-mat.csv", sep=";")[[
    "G1",  # Max: 19, Min: 3
    "G2",  # Max: 19, Min: 0
    "G3",  # Max: 20, Min: 0
    "studytime",  # Max: 4, Min: 1
    "failures",  # Max: 3, Min: 0
    "absences"  # Max: 75, Min: 0
]]
raw_y = parse_pass(np.array(data.get("G3")))
expected = np.array(batcher(raw_y[0:350], 10))
raw_inputs = np.array(data.drop("G3", 1))
X = np.array(batcher(raw_inputs.tolist()[0:350], 10))
test_X = raw_inputs[350:396]
test_Y = raw_y[350:396]

input_layer = Layer(5, 8)
hidden_layer = Layer(8, 8)
output_layer = Layer(8, 2)

for i, y in zip(X, expected):
    input_layer.advance(i, Activation.ReLU)
    hidden_layer.advance(input_layer.output, Activation.ReLU)
    output_layer.advance(hidden_layer.output, Activation.Softmax)
    back_propagation([input_layer, hidden_layer, output_layer], i, y)
