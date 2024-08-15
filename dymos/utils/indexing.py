from collections.abc import Iterable
import numpy as np

from openmdao.utils.indexer import indexer


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
    a = np.reshape(np.arange(np.prod(src_shape), dtype=int), src_shape)
    return a[ixgrid]


def get_constraint_flat_idxs(con):
    """
    Return the flat indices for a constraint at single point in time.

    Indices are always returned as non-negative.

    Parameters
    ----------
    con : dict
        The ConstraintOptionsDictionary for the constraint in question.

    Returns
    -------
    np.array
        The flat indices of a constraint at a single point in time.
    """
    if con['indices'] is None:
        flat_idxs = np.arange(np.prod(con['shape'], dtype=int), dtype=int)
    else:
        # Use shaped_array to force all indices to be non-negative.
        flat_idxs = indexer(con['indices'], src_shape=con['shape'],
                            flat_src=con['flat_indices']).shaped_array(flat=True)

    return flat_idxs


def get_desvar_indices(size, num_nodes, fix_initial, fix_final):
    """
    Compute design var index array.

    Parameters
    ----------
    size : int
        Size of one node of the design var array.
    num_nodes : int
        Number of nodes in the design var array.
    fix_initial : bool
        If True, the initial value of the array is fixed and not a design variable.
    fix_final : bool
        If True, the final value of the array is fixed and not a design variable.

    Returns
    -------
    ndarray
        The design var index array.
    """
    if fix_initial or fix_final:
        start = end = mask = None

        if fix_initial:
            if isinstance(fix_initial, Iterable):
                idxs_to_fix = np.where(np.asarray(fix_initial))[0]
                mask = np.ones(size * num_nodes, dtype=bool)
                mask[idxs_to_fix] = False
            else:
                start = size

        if fix_final:
            if isinstance(fix_final, Iterable):
                idxs_to_fix = np.where(np.asarray(fix_final))[0] - size
                if mask is None:
                    mask = np.ones(size * num_nodes, dtype=bool)
                mask[idxs_to_fix] = False
            else:
                end = size * (num_nodes - 1)

        if mask is None:
            start = 0 if start is None else start
            end = size * num_nodes if end is None else end
            return np.arange(start, end)
        else:
            if start is not None:
                mask[:start] = False
            if end is not None:
                mask[end:] = False
            return np.arange(size * num_nodes)[mask]

    return np.arange(size * num_nodes)
