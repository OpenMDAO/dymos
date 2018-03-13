from __future__ import division

import numpy as np

from dymos.glm.ozone.methods.runge_kutta.runge_kutta import RungeKutta


_gl_coeffs = {
    2: (
        np.array([.5]),
        np.array([1.])
    ),
    4: (
        np.array([[1 / 4, 1 / 4 - np.sqrt(3) / 6],
                  [1 / 4 + np.sqrt(3) / 6, 1 / 4]]),
        np.array([1 / 2, 1 / 2])
    ),
    6: (
        np.array([[5 / 36, 2 / 9 - np.sqrt(15) / 15, 5 / 36 - np.sqrt(15) / 30],
                  [5 / 36 + np.sqrt(15) / 24, 2 / 9, 5 / 36 - np.sqrt(15) / 24],
                  [5 / 36 + np.sqrt(15) / 30, 2 / 9 + np.sqrt(15) / 15, 5 / 36]]),
        np.array([5 / 18, 4 / 9, 5 / 18])
    )
}


class GaussLegendre(RungeKutta):

    def __init__(self, order=4):
        self.order = order

        if order not in _gl_coeffs:
            raise ValueError('GaussLegendre order must be one of the following: {}'.format(
                sorted(_gl_coeffs.keys())
            ))
        A, B = _gl_coeffs[order]
        super(GaussLegendre, self).__init__(A=A, B=B)
