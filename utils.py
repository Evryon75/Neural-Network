import enum

import numpy as np


class Activations(enum.Enum):
    ReLU = 1
    Softmax = 2


def calculate_loss(prediction, expected):
    """
    Readable version:
        length = len(prediction)
        clipped = np.clip(prediction, 1e-7, 1 - 1e-7)
        neg_log = -np.log(clipped[range(length), expected])
        return np.mean(neg_log)
    """
    return np.mean(-np.log(np.clip(prediction, 1e-7, 1 - 1e-7)[range(len(prediction)), expected]))
