import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from dymos.utils.cgl import cgl

# Known solutions
x_i = {2: [-1.0, 1.0],
       3: [-1.0, 0.0, 1.0],
       4: [-1.0, -0.5, 0.5, 1.0],
       5: [-1.0, -1/np.sqrt(2), 0.0, 1/np.sqrt(2), 1.0]}

w_i = {2: [np.pi/2, np.pi/2],
       3: [np.pi/4, np.pi/2, np.pi/4],
       4: [np.pi/6, np.pi/3, np.pi/3, np.pi/6],
       5: [np.pi/8, np.pi/4, np.pi/4, np.pi/4, np.pi/8]}


class TestCGL(unittest.TestCase):

    def test_nodes_and_weights_2(self):
        x_2, w_2 = cgl(2)
        assert_almost_equal(x_2, x_i[2], decimal=6)
        assert_almost_equal(w_2, w_i[2], decimal=6)

    def test_nodes_and_weights_3(self):
        x_3, w_3 = cgl(3)
        assert_almost_equal(x_3, x_i[3], decimal=6)
        assert_almost_equal(w_3, w_i[3], decimal=6)

    def test_nodes_and_weights_4(self):
        x_4, w_4 = cgl(4)
        assert_almost_equal(x_4, x_i[4], decimal=6)
        assert_almost_equal(w_4, w_i[4], decimal=6)

    def test_nodes_and_weights_5(self):
        x_5, w_5 = cgl(5)
        assert_almost_equal(x_5, x_i[5], decimal=6)
        assert_almost_equal(w_5, w_i[5], decimal=6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
