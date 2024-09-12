import unittest

from numpy.testing import assert_almost_equal

from dymos.utils.birkhoff import birkhoff_matrix
from dymos.utils.cgl import cgl, clenshaw_curtis
from dymos.utils.lgl import lgl


class TestBirkhoffMatrix(unittest.TestCase):

    def test_birkhoff_matrix_lgl(self):
        x, w = lgl(9)
        B = birkhoff_matrix(x, w, grid_type='lgl')
        assert_almost_equal(B[-1, :], w)

    def test_birkhoff_matrix_cgl(self):
        x, w = cgl(9)
        B = birkhoff_matrix(x, w, grid_type='cgl')
        _, wB = clenshaw_curtis(9)
        assert_almost_equal(B[-1, :], wB)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
