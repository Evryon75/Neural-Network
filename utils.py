"""
 File used for project utilities
"""

import enum


class Activation(enum.Enum):
    """
    Used for deciding which activation an output should go through
    It is an enum because that makes the code more rigid, as opposed to passing strings as parameters,
    and it is also easy to expand
    """
    ReLU = 1
    Softmax = 2


def parse_pass(grades):
    temp = []
    for i in grades:
        temp.append(1 if i > 10 else 0)
    return temp
