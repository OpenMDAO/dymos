import unittest

from numpy.testing import assert_almost_equal

from dymos.utils.lg import lg

# Known solutions
x_i = {2: [-0.57735, 0.57735],
       3: [-0.774597, 0.0, 0.774597],
       4: [-0.861136, -0.339981, 0.339981, 0.861136],
       5: [-0.90618, -0.538469, 0.0, 0.538469, 0.90618]}

w_i = {2: [1.0, 1.0],
       3: [0.555556, 0.888889, 0.555556],
       4: [0.347855, 0.652145, 0.652145, 0.347855],
       5: [0.236927, 0.478629, 0.568889, 0.478629, 0.236927]}


class TestLG(unittest.TestCase):

    def test_nodes_and_weights_2(self):
        x_2, w_2 = lg(2)
        assert_almost_equal(x_2, x_i[2], decimal=6)
        assert_almost_equal(w_2, w_i[2], decimal=6)

    def test_nodes_and_weights_3(self):
        x_3, w_3 = lg(3)
        assert_almost_equal(x_3, x_i[3], decimal=6)
        assert_almost_equal(w_3, w_i[3], decimal=6)

    def test_nodes_and_weights_4(self):
        x_4, w_4 = lg(4)
        assert_almost_equal(x_4, x_i[4], decimal=6)
        assert_almost_equal(w_4, w_i[4], decimal=6)

    def test_nodes_and_weights_5(self):
        x_5, w_5 = lg(5)
        assert_almost_equal(x_5, x_i[5], decimal=6)
        assert_almost_equal(w_5, w_i[5], decimal=6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
