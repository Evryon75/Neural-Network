"""
 File used for project utilities
"""
import numpy as np

import enum


class Activation(enum.Enum):
    """
    Used for deciding which activation an output should go through
    It is an enum because that makes the code more rigid, as opposed to passing strings as parameters,
    and it is also easy to expand
    """
    ReLU = 1
    Softmax = 2


def parse_pass(grades: np.ndarray) -> list:
    """
    Creates a list with 1 or 0, it is 1 if the grade is above 10 and 0 if it is below 10
    :param grades: List with grades
    :return: List with 0 or 1 for failed or passed
    """
    temp = []
    for i in grades:
        temp.append(1 if i > 10 else 0)
    return temp


def batcher(raw_inputs: list, batch_size: int) -> list:
    """
    Creates batches from raw inputs
    :param raw_inputs: Inputs such as training data or expected predictions
    :param batch_size: How big one batch should be
    :return: The formatted batch list
    """
    outer = []
    counter = 0
    inner = []
    for i in raw_inputs:
        inner.append(i)
        counter += 1
        if counter == batch_size:
            counter = 0
            outer.append(inner)
            inner = []
    return outer
