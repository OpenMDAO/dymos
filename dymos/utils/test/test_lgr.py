import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from dymos.utils.lgr import lgr

# Known solutions
x_i = {2: [-1.0, 1 / 3.0],
       3: [-1.0, -0.289898, 0.689898],
       4: [-1.0, -0.575319, 0.181066, 0.822824],
       5: [-1.0, -0.72048, -0.167181, 0.446314, 0.885792]}

w_i = {2: [0.5, 1.5],
       3: [2.0 / 9.0, (16 + np.sqrt(6)) / 18.0, (16 - np.sqrt(6)) / 18.0],
       4: [0.125, 0.657689, 0.776387, 0.440924],
       5: [0.08, 0.446208, 0.623653, 0.562712, 0.287427]}


class TestLGR(unittest.TestCase):

    def test_nodes_and_weights_2(self):
        x_2, w_2 = lgr(2)
        assert_almost_equal(x_2, x_i[2], decimal=6)
        assert_almost_equal(w_2, w_i[2], decimal=6)

        x_2, w_2 = lgr(2, include_endpoint=True)
        assert_almost_equal(x_2, x_i[2] + [1], decimal=6)
        assert_almost_equal(w_2, w_i[2] + [0], decimal=6)

    def test_nodes_and_weights_3(self):
        x_3, w_3 = lgr(3)
        assert_almost_equal(x_3, x_i[3], decimal=6)
        assert_almost_equal(w_3, w_i[3], decimal=6)

        x_3, w_3 = lgr(3, include_endpoint=True)
        assert_almost_equal(x_3, x_i[3] + [1], decimal=6)
        assert_almost_equal(w_3, w_i[3] + [0], decimal=6)

    def test_nodes_and_weights_4(self):
        x_4, w_4 = lgr(4)
        assert_almost_equal(x_4, x_i[4], decimal=6)
        assert_almost_equal(w_4, w_i[4], decimal=6)

        x_4, w_4 = lgr(4, include_endpoint=True)
        assert_almost_equal(x_4, x_i[4] + [1], decimal=6)
        assert_almost_equal(w_4, w_i[4] + [0], decimal=6)

    def test_nodes_and_weights_5(self):
        x_5, w_5 = lgr(5)
        assert_almost_equal(x_5, x_i[5], decimal=6)
        assert_almost_equal(w_5, w_i[5], decimal=6)

        x_5, w_5 = lgr(5, include_endpoint=True)
        assert_almost_equal(x_5, x_i[5] + [1], decimal=6)
        assert_almost_equal(w_5, w_i[5] + [0], decimal=6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
