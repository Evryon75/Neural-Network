"""
 Main file
"""
import time
import warnings

import numpy as np
import pandas as pd
from colorama import Fore, Style

from layer import Layer
from optimization import back_propagate
from utils import Activation, parse_pass, batcher

warnings.simplefilter(action='ignore', category=FutureWarning)  # Annoying future warnings from pandas
np.random.seed(0)
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
expected = np.array(batcher(raw_y[0:350], 5))
raw_inputs = np.array(data.drop("G3", 1))
X = np.array(batcher(raw_inputs.tolist()[0:350], 5))
test_X = raw_inputs[350:396]
test_Y = raw_y[350:396]

hidden_layer = Layer(5, 5)
output_layer = Layer(5, 5)


def advance_network(X_input):
    """
    Helper function to advance the network here
    :param X_input: Input data
    """
    hidden_layer.advance(np.array(X_input), Activation.ReLU)
    output_layer.advance(hidden_layer.output, Activation.Softmax)


batch_counter = 1
for i, y in zip(X, expected):
    advance_network(i)
    back_propagate(hidden_layer, output_layer, y, i)
    print(Fore.YELLOW + "Batch:", batch_counter)
    batch_counter += 1
    for j, k in zip(output_layer.output, y):
        print(Fore.BLUE + Style.BRIGHT + str(j[k] * 100)[0:5] + Style.RESET_ALL, "%")
    print()
    time.sleep(0.3)

print(Fore.YELLOW + "Training finished, testing data results:")
advance_network(test_X)
correct = 0
wrong = 0
for output, y in zip(output_layer.output, test_Y):
    if output[y] > 0.5:
        correct += 1
        print(Fore.GREEN + Style.BRIGHT + "This student", "passed" if y > 2 else "failed", "G3")
        print(Fore.LIGHTBLACK_EX + Style.NORMAL + "Predicted with an accuracy of", str(output[y] * 100)[0:5], "%")
        time.sleep(0.3)
    else:
        wrong += 1
        print(Fore.RED + Style.BRIGHT + "This student", "passed" if y > 2 else "failed", "G3")
        print(Fore.LIGHTBLACK_EX + Style.NORMAL + "Predicted with an accuracy of", str(output[y] * 100)[0:5], "%")
        time.sleep(1)

print(Style.RESET_ALL)
print("Correct answers:", str(correct) + " - ", str(round(correct / (correct + wrong) * 100, 2)) + "%")
print("Wrong answers:", str(wrong) + " - ", str(round(wrong / (correct + wrong) * 100, 2)) + "%")
