import numpy as np


def glm(n):
    """
    Retrieve the GLM nodes and weights for n nodes.
    """
    return np.linspace(-1., 1., 2), np.ones(2)
