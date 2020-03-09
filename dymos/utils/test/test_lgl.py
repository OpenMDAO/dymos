import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from dymos.utils.lgl import lgl

# Known solutions
x_i = {2: [-1.0, 1.0],
       3: [-1.0, 0.0, 1.0],
       4: [-1.0, -np.sqrt(5)/5, np.sqrt(5)/5, 1.0],
       5: [-1.0, -np.sqrt(21)/7, 0.0, np.sqrt(21)/7, 1.0],
       6: [-1.0, -np.sqrt((7 + 2 * np.sqrt(7)) / 21), -np.sqrt((7 - 2 * np.sqrt(7)) / 21),
           np.sqrt((7 - 2 * np.sqrt(7)) / 21), np.sqrt((7 + 2 * np.sqrt(7)) / 21), 1.0]}

w_i = {2: [1.0, 1.0],
       3: [1/3, 4/3, 1/3],
       4: [1/6, 5/6, 5/6, 1/6],
       5: [1/10, 49/90, 32/45, 49/90, 1/10],
       6: [1/15, (14 - np.sqrt(7))/30, (14 + np.sqrt(7))/30,
           (14 + np.sqrt(7))/30, (14 - np.sqrt(7))/30, 1/15]}


class TestLGL(unittest.TestCase):

    def test_nodes_and_weights_2(self):
        x_2, w_2 = lgl(2)
        assert_almost_equal(x_2, x_i[2], decimal=6)
        assert_almost_equal(w_2, w_i[2], decimal=6)

    def test_nodes_and_weights_3(self):
        x_3, w_3 = lgl(3)
        assert_almost_equal(x_3, x_i[3], decimal=6)
        assert_almost_equal(w_3, w_i[3], decimal=6)

    def test_nodes_and_weights_4(self):
        x_4, w_4 = lgl(4)
        assert_almost_equal(x_4, x_i[4], decimal=6)
        assert_almost_equal(w_4, w_i[4], decimal=6)

    def test_nodes_and_weights_5(self):
        x_5, w_5 = lgl(5)
        assert_almost_equal(x_5, x_i[5], decimal=6)
        assert_almost_equal(w_5, w_i[5], decimal=6)

    def test_nodes_and_weights_6(self):
        x_6, w_6 = lgl(6)
        assert_almost_equal(x_6, x_i[6], decimal=6)
        assert_almost_equal(w_6, w_i[6], decimal=6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
