"""
 Main file
"""

import warnings
import numpy as np
import pandas as pd
from layer import Layer
from optimization import calculate_loss
from utils import Activation, parse_pass

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
predictions = parse_pass(np.array(data.get("G3")))
raw_inputs = np.array(data.drop("G3", 1))

# TODO: Split inputs into training data and testing data, make a utility function for that maybe

input_layer = Layer(5, 8)
hidden_layer = Layer(8, 8)
output_layer = Layer(8, 2)

input_layer.advance(raw_inputs, Activation.ReLU)
hidden_layer.advance(input_layer.output, Activation.ReLU)
output_layer.advance(hidden_layer.output, Activation.Softmax)

loss = calculate_loss(output_layer.output, predictions)
