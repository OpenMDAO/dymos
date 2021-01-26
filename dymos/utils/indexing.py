import numpy as np


def get_src_indices_by_row(row_idxs, shape, flat=True):
    """
    Provide the src_indices when connecting a vectorized variable from an output to an input.

    Indices are selected by choosing the first indices to be passed, corresponding to node
    index in Dymos.

    Parameters
    ----------
    row_idxs : array_like
        The rows/node indices to be connected from the source to the target.
    shape : tuple
        The shape of the variable at each node (ignores the first dimension).
    flat : bool
        If True, return the source indices in flat source indices form.

    Returns
    -------
    array_like
        If flat, a numpy array of shape `(row_idxs,) + shape` where each element is the index
        of the source of that element in the source array, in C-order.
    """
    if not flat:
        raise NotImplementedError('Currently get_src_indices_by_row only returns '
                                  'flat source indices.')

    num_src_rows = np.max(row_idxs) + 1
    src_shape = (num_src_rows,) + shape
    other_idxs = [np.arange(n, dtype=int) for n in shape]
    ixgrid = np.ix_(row_idxs, *other_idxs)
    a = np.reshape(np.arange(np.prod(src_shape), dtype=int), newshape=src_shape)
    src_idxs = a[ixgrid]
    return src_idxs
