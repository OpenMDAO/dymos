from __future__ import division
import numpy as np


class GLMMethod(object):
    """
    Base class for all GLM methods.
    """
    def __init__(self, A, B, U, V, abscissa, starting_method):

        s, s2 = A.shape

        if s != s2:
            raise ValueError('GLM Matrix A must be square. Received {}x{}'.format(s, s2))

        us, r = U.shape

        if us != s:
            raise ValueError('GLM Matrix U must have {} rows to match A. '
                             'Received U: {}x{}'.format(s, s, s, us, r))

        br, bs = B.shape

        if br != r or bs != s:
            raise ValueError('GLM Matrix B must have {} rows and {} columns to match A and U. '
                             'Received B: {}x{}'.format(s, r, br, bs))

        vr, vr2 = V.shape

        if vr != r or vr2 != r:
            raise ValueError('GLM Matrix V must have {} rows and {} columns to match A and U. '
                             'Received V: {}x{}'.format(s, r, vr, vr2))

        self.num_stages = s
        self.num_values = r
        self.abscissa = abscissa
        self.A = A
        self.B = B
        self.U = U
        self.V = V
        self.starting_method = starting_method

        lower = np.tril(A, -1)
        err = np.linalg.norm(lower - A)
        self.explicit = err < 1e-15
