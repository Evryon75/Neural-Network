"""
 File used for loss calculation weight optimization
"""
import random

import numpy as np
from colorama import Style

from layer import Layer


def calculate_loss(prediction: np.ndarray, expected: list):
    """
    Calculates the loss from the network's prediction
    :param prediction: The final layer's output
    :param expected: The desired output
    :return: The loss from the final layer's output
    """
    """
    Readable version:
        length = len(prediction)  # The length of the predictions list
        clipped = np.clip(prediction, 1e-7, 1 - 1e-7)  # In case any value is 0
        neg_log = -np.log(clipped[range(length), expected])  # Calculating the loss on each element
        return np.mean(neg_log)  # Average of all losses
    I used a one-liner for memory efficiency and possibly faster execution
    """
    return np.mean(-np.log(np.clip(prediction, 1e-7, 1 - 1e-7)[range(len(prediction)), expected]))


def sigmoid_deriv(s):
    return s * (1 - s)


def back_propagate(hidden_layer, output_layer, expected, inputs):
    output_error = []
    for z, y in zip(output_layer.output, expected):
        temp = []
        for j in range(5):
            temp.append(0 - z[j])
        temp[y] = 1 - z[y]
        output_error.append(temp)
    output_delta = output_error * sigmoid_deriv(output_layer.output)
    hidden_delta = output_delta.dot(hidden_layer.weights) * sigmoid_deriv(hidden_layer.output)

    hidden_layer.weights += inputs.T.dot(hidden_delta) * 10
    output_layer.weights += output_layer.output.T.dot(output_delta) * 10
