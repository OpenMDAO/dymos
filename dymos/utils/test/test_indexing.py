import unittest

import numpy as np
from numpy.testing import assert_array_equal

from dymos.utils.indexing import get_src_indices_by_row


class TestIndexing(unittest.TestCase):

    def test_get_src_indices_by_row_vector_target(self):
        row_idxs = (0, 1, 2)
        shape = (5,)
        idxs = get_src_indices_by_row(row_idxs, shape, flat=True)
        expected = np.reshape(np.arange(15, dtype=int), (3, 5))
        assert_array_equal(idxs, expected)

    def test_get_src_indices_by_row_matrix_target(self):
        row_idxs = (0, 2, 4)
        shape = (5, 2)
        idxs = get_src_indices_by_row(row_idxs, shape, flat=True)

        expected = np.zeros((3, 5, 2), dtype=int)
        expected[0, ...] = np.reshape(np.arange(10, dtype=int), newshape=shape)
        expected[1, ...] = expected[0, ...] + 20
        expected[2, ...] = expected[0, ...] + 40

        assert_array_equal(idxs, expected)

    def test_get_src_indices_by_row_raises_if_not_flat(self):

        with self.assertRaises(NotImplementedError) as e:
            get_src_indices_by_row((0, 1, 2), (3, 3), flat=False)

        self.assertEqual(str(e.exception),
                         'Currently get_src_indices_by_row only returns flat source indices.')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
