import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from dymos.utils.hermite import hermite_matrices


class TestHermiteMatrices(unittest.TestCase):

    def test_quadratic(self):

        # Interpolate with values and rates provided at [-1, 1] in tau space
        tau_given = [-1.0, 1.0]
        tau_eval = np.linspace(-1, 1, 100)

        # In time space use the boundaries [-2, 2]
        dt_dtau = 4.0 / 2.0

        # Provide values for y = t**2 and its time-derivative
        y_given = [4.0, 4.0]
        ydot_given = [-4.0, 4.0]

        # Get the hermite matrices.
        Ai, Bi, Ad, Bd = hermite_matrices(tau_given, tau_eval)

        # Interpolate y and ydot at tau_eval points in tau space.
        y_i = np.dot(Ai, y_given) + dt_dtau * np.dot(Bi, ydot_given)
        ydot_i = (1.0 / dt_dtau) * np.dot(Ad, y_given) + np.dot(Bd, ydot_given)

        # Compute our function as a point of comparison.
        y_computed = (tau_eval * dt_dtau)**2
        ydot_computed = 2.0 * (tau_eval * dt_dtau)

        # Check results
        assert_almost_equal(y_i, y_computed)
        assert_almost_equal(ydot_i, ydot_computed)

    def test_cubic(self):

        # Interpolate with values and rates provided at [-1, 1] in tau space
        tau_given = [-1.0, 0.0, 1.0]
        tau_eval = np.linspace(-1, 1, 101)

        # In time space use the boundaries [-2, 2]
        dt_dtau = 4.0 / 2.0

        # Provide values for y = t**2 and its time-derivative
        y_given = [-8.0, 0.0, 8.0]
        ydot_given = [12.0, 0.0, 12.0]

        # Get the hermite matrices.
        Ai, Bi, Ad, Bd = hermite_matrices(tau_given, tau_eval)

        # Interpolate y and ydot at tau_eval points in tau space.
        y_i = np.dot(Ai, y_given) + dt_dtau * np.dot(Bi, ydot_given)
        ydot_i = (1.0 / dt_dtau) * np.dot(Ad, y_given) + np.dot(Bd, ydot_given)

        # Compute our function as a point of comparison.
        y_computed = (tau_eval * dt_dtau)**3
        ydot_computed = 3.0 * (tau_eval * dt_dtau)**2

        # Check results
        assert_almost_equal(y_i, y_computed)
        assert_almost_equal(ydot_i, ydot_computed)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
