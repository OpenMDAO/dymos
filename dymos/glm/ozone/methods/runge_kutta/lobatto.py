from __future__ import division

import numpy as np

from dymos.glm.ozone.methods.runge_kutta.runge_kutta import RungeKutta


_lobatto_coeffs = {
    2: (
        np.array([[0, 0],
                  [1 / 2, 1 / 2]]),
        np.array([1 / 2, 1 / 2])
    ),
    4: (
        np.array([[0, 0, 0],
                  [5 / 24, 1 / 3, -1 / 24],
                  [1 / 6, 2 / 3, 1 / 6]]),
        np.array([1 / 6, 2 / 3, 1 / 6])
    )
}

class LobattoIIIA(RungeKutta):

    def __init__(self, order=4):
        self.order = order

        if order not in _lobatto_coeffs:
            raise ValueError('LobattoIIIA order must be one of the following: {}'.format(
                sorted(_lobatto_coeffs.keys())
            ))
        A, B = _lobatto_coeffs[order]
        super(LobattoIIIA, self).__init__(A=A, B=B)
