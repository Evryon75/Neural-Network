"""
 File used for loss calculation weight optimization
"""

import numpy as np


def calculate_loss(prediction, expected):
    """
    Calculates the loss from the network's prediction
    :param prediction: The final layer's output
    :param expected: The desired output
    :return: The loss from the final layer's output
    """
    """
    Readable version:
        length = len(prediction)
        clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        neg_log = -np.log(clipped[range(length), expected])
        return np.mean(neg_log)
    I used a one-liner for memory efficiency and possibly faster execution
    """
    return np.mean(-np.log(np.clip(prediction, 1e-7, 1 - 1e-7)[range(len(prediction)), expected]))
