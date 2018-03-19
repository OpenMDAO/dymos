from __future__ import division

import numpy as np
from dymos.glm.ozone.methods.method import GLMMethod


class RungeKutta(GLMMethod):

    def __init__(self, A, B):
        A = np.atleast_2d(A)
        B = np.atleast_2d(B)

        U = np.ones((A.shape[0], 1))
        V = np.array([[1.]])

        abscissa = np.sum(A, 1)
        starting_method = None

        super(RungeKutta, self).__init__(A, B, U, V, abscissa, starting_method)
