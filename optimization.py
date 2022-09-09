"""
 File used for loss calculation weight optimization
"""

import numpy as np

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


def back_propagate(layers: list, inputs, expected):
    error_deltas = []

    for layer in reversed(layers):
        output = layer.output
        error_deltas.append(calculate_loss(output, expected) * sigmoid_deriv(output))
