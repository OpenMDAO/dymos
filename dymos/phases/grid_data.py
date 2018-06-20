from __future__ import print_function, division, absolute_import

import numpy as np

from six import iteritems

from scipy.linalg import block_diag

from dymos.utils.lg import lg
from dymos.utils.lgl import lgl
from dymos.utils.lgr import lgr
from dymos.utils.hermite import hermite_matrices
from dymos.utils.lagrange import lagrange_matrices


def gauss_lobatto_subsets(n):
    """
    Returns the subset dictionary corresponding to the Gauss-Lobatto transcription.

    Parameters
    ----------
    num_nodes : int
        The total number of nodes in the Gauss-Lobatto segment.  Must be
        an odd number.

    Returns
    -------
    subsets : A dictionary with the following keys:
        'disc' gives the indices of the state discretization nodes (deprecated)
        'state_disc' gives the indices of the state discretization nodes
        'control_disc' gives the indices of the control discretization nodes
        'segment_ends' gives the indices of the nodes at the start (even) and end (odd) of a segment
        'col' gives the indices of the collocation nodes
        'all' gives all node indices
    """
    if n % 2 == 0:
        raise ValueError('A Gauss-Lobatto scheme must use an odd number of points')

    subsets = {
        'disc': np.arange(0, n, 2, dtype=int),
        'state_disc': np.arange(0, n, 2, dtype=int),
        'control_disc': np.arange(n, dtype=int),
        'segment_ends': np.array([0, n-1], dtype=int),
        'col': np.arange(1, n, 2, dtype=int),
        'all': np.arange(n, dtype=int),
    }

    return subsets


def radau_pseudospectral_subsets(n):
    """
    Returns the subset dictionary corresponding to the Radau Pseudospectral
    transcription.

    Parameters
    ----------
    num_nodes : int
        The total number of nodes in the Radau Pseudospectral segment (including right endpoint).

    Returns
    -------
    subsets : A dictionary with the following keys:
        'disc' gives the indices of the state discretization nodes (deprecated)
        'state_disc' gives the indices of the state discretization nodes
        'control_disc' gives the indices of the control discretization nodes
        'segment_ends' gives the indices of the nodes at the start (even) and end (odd) of a segment
        'col' gives the indices of the collocation nodes
        'all' gives all node indices
    """
    node_indices = {
        'disc': np.arange(n),
        'state_disc': np.arange(n, dtype=int),
        'control_disc': np.arange(n, dtype=int),
        'segment_ends': np.array([0, n - 1], dtype=int),
        'col': np.arange(n - 1, dtype=int),
        'all': np.arange(n, dtype=int),
    }
    return node_indices


class GridData(object):
    """
    Properties associated with the GridData of a phase.

    GridData contains properties associated
    with the "grid" or "mesh" of a phase - the number of segments, the
    polynomial order of each segment, and the relative lengths of the segments.
    In turn, these three defining properties determine various other properties,
    such as indexing arrays used to extract the discretization or collocation
    nodes from a list of all nodes within the phase.

    """

    def __init__(self, num_segments, transcription, transcription_order=None,
                 segment_ends=None, compressed=False):
        """
        Initialize and compute all attributes.

        Parameters
        ----------
        num_segments : int
            The number of segments in the phase.
        transcription : str
            Case-insensitive transcription scheme (e.g., ('gauss-lobatto', 'radau-ps')).
        transcription_order : int or int ndarray[:]
            The order of the state transcription in each segment, as a scalar or a vector.
        segment_ends : Iterable[num_segments + 1] or None
            The segments nodes on some arbitrary interval.
            This will be normalized to the interval [-1, 1].
        compressed : bool
            If the transcription is compressed, then states and controls at shared
            nodes of adjacent segments are only specified once, and then broadcast
            to the appropriate indices.

        Attributes
        ----------
        num_segments : int
            The number of segments in the phase
        segment_ends : ndarray or None
            The segment boundaries in non-dimensional phase time (phase tau space).
            If given as a Iterable, it must be of length (num_segments+1) and it
            must be monotonically increasing.
            If None, then the segments are equally spaced.
        num_nodes : int
            The total number of nodes in the phase
        node_stau : ndarray
            The locations of each node in non-dimensional segment time (segment tau space).
        node_ptau : ndarray
            The locations of each node in non-dimensional phase time (phase tau space).
        node_dptau_dstau : ndarray
            The ratio of phase tau to segment tau at each node.
        segment_indices : int ndarray[:,2]
            Array where each row contains the start and end indices into the nodes.
        subset_node_indices : dict of int ndarray[:]
            Dict keyed by subset name where each entry are the indices of the nodes
            belonging to that given subset.
        subset_segment_indices: dict of int ndarray[num_seg,:]
            Dict keyed by subset name where each entry are the indices of the nodes
            belonging to the given subset, indexed into subset_node_indices!
        subset_num_nodes: dict of int
            A dict keyed by subset name that provides the total number of
            nodes in the phase which belong to the given subset.
        subset_num_nodes_per_segment: dict of list
            A dict keyed by subset name that provides a list of ints giving the number of
            nodes which belong to the given subset in each segment.
        compressed: bool
            True if the transcription is compressed (connecting nodes of adjacent segments
            are not duplicated in the inputs).
        input_maps: dict of int ndarray[:]
            Dict keyed by the map name that provides a mapping for src_indices to
            and from "compressed" form.

        """
        if segment_ends is None:
            segment_ends = np.linspace(-1, 1, num_segments + 1)
        else:
            if len(segment_ends) != num_segments + 1:
                raise ValueError('segment_ends must be of length (num_segments + 1)')
            # Assert monotonic increasing
            if not np.all(np.diff(segment_ends) > 0):
                raise ValueError('segment_ends must be monotonically increasing')
            segment_ends = np.atleast_1d(segment_ends)

        v0 = segment_ends[0]
        v1 = segment_ends[-1]
        segment_ends = -1. + 2 * (segment_ends - v0) / (v1 - v0)

        # List of all GridData attributes

        self.num_segments = num_segments

        self.segment_ends = segment_ends

        self.num_nodes = 0

        self.node_stau = None

        self.node_ptau = None

        self.node_dptau_dstau = None

        self.segment_indices = None

        self.subset_node_indices = {}

        self.subset_segment_indices = {}

        self.subset_num_nodes = {}

        self.subset_num_nodes_per_segment = {}

        self.num_dynamic_control_input_nodes = 0

        self.num_state_input_nodes = 0

        self.compressed = compressed

        self.input_maps = {'state_input_to_disc': np.empty(0, dtype=int),
                           'dynamic_control_input_to_disc': np.empty(0, dtype=int)}

        # Define get_subsets and node points based on the transcription scheme
        if transcription.lower() == 'gauss-lobatto':
            get_subsets = gauss_lobatto_subsets
            get_points = lgl
        elif transcription.lower() == 'radau-ps':
            get_subsets = radau_pseudospectral_subsets

            def get_points(n):
                return lgr(n, include_endpoint=True)
        else:
            raise ValueError('Unknown transcription: {0}'.format(transcription))

        # Make sure transcription_order is a vector
        if np.isscalar(transcription_order):
            transcription_order = np.ones(num_segments, int) * transcription_order
        self.transcription_order = transcription_order

        # Determine the list of subset_names
        subset_names = get_subsets(1).keys()

        # Initialize num_nodes and subset_num_nodes
        self.num_nodes = 0
        for name in subset_names:
            self.subset_num_nodes[name] = 0
            self.subset_num_nodes_per_segment[name] = []

        # Initialize segment_indices and subset_segment_indices
        self.segment_indices = np.empty((self.num_segments, 2), int)
        for name in subset_names:
            self.subset_segment_indices[name] = np.empty((self.num_segments, 2), int)

        # Compute the number of nodes in the phase (total and by subset)
        for iseg in range(num_segments):
            segment_nodes, _ = get_points(transcription_order[iseg])
            segment_subsets = get_subsets(len(segment_nodes))

            self.num_nodes += len(segment_nodes)
            for name, val in iteritems(segment_subsets):
                self.subset_num_nodes[name] += len(val)
                self.subset_num_nodes_per_segment[name].append(len(val))

                # Build the state decompression map
                if name == 'state_disc':
                    idxs = np.arange(len(val))
                    if iseg > 0:
                        idxs += self.input_maps['state_input_to_disc'][-1]
                        if not compressed:
                            idxs += 1
                    self.input_maps['state_input_to_disc'] = \
                        np.concatenate((self.input_maps['state_input_to_disc'], idxs))

                # Build the control decompression map
                elif name == 'control_disc':
                    idxs = np.arange(len(val))
                    if iseg > 0:
                        idxs += self.input_maps['dynamic_control_input_to_disc'][-1]
                        if not compressed:
                            idxs += 1
                    self.input_maps['dynamic_control_input_to_disc'] = \
                        np.concatenate((self.input_maps['dynamic_control_input_to_disc'], idxs))

        self.num_state_input_nodes = len(set(self.input_maps['state_input_to_disc']))
        self.num_dynamic_control_input_nodes = \
            len(set(self.input_maps['dynamic_control_input_to_disc']))

        # Now that we know the sizes, allocate arrays
        self.node_stau = np.empty(self.num_nodes)
        self.node_ptau = np.empty(self.num_nodes)
        self.node_dptau_dstau = np.empty(self.num_nodes)
        for name in subset_names:
            self.subset_node_indices[name] = np.empty(self.subset_num_nodes[name], int)

        # Populate the arrays
        ind0 = 0
        ind1 = 0
        subset_ind0 = {name: 0 for name in subset_names}
        subset_ind1 = {name: 0 for name in subset_names}
        for iseg in range(num_segments):
            segment_nodes, _ = get_points(transcription_order[iseg])
            segment_subsets = get_subsets(len(segment_nodes))

            ind1 += len(segment_nodes)
            for name in subset_names:
                subset_ind1[name] += len(segment_subsets[name])

            self.segment_indices[iseg, 0] = ind0
            self.segment_indices[iseg, 1] = ind1
            for name in subset_names:
                self.subset_segment_indices[name][iseg, 0] = subset_ind0[name]
                self.subset_segment_indices[name][iseg, 1] = subset_ind1[name]

            v0 = segment_ends[iseg]
            v1 = segment_ends[iseg + 1]
            self.node_stau[ind0:ind1] = segment_nodes
            self.node_ptau[ind0:ind1] = v0 + 0.5 * (segment_nodes + 1) * (v1 - v0)
            self.node_dptau_dstau[ind0:ind1] = 0.5 * (v1 - v0)

            for name in subset_names:
                self.subset_node_indices[name][subset_ind0[name]:subset_ind1[name]] = \
                    segment_subsets[name] + ind0

            ind0 += len(segment_nodes)
            for name in subset_names:
                subset_ind0[name] += len(segment_subsets[name])

    def phase_lagrange_matrices(self, given_set_name, eval_set_name):
        """
        Compute the matrices mapping values at some nodes to values and derivatives at new nodes.

        The values are mapped using the equation:
        .. math:: x_{eval} = \left[ L \right] x_{given}

        And the derivatives are mapped with the equation:
        .. math:: \dot{x}_{eval} = \left[ D \right] x_{given} \frac{d \tau}{dt}

        Parameters
        ----------
        given_set_name : str
            Name of the set of nodes with which to perform the interpolation.
        eval_set_name : str
            Name of the set of nodes at which to evaluate the values and derivatives.

        Returns
        -------
        ndarray[num_eval_set, num_given_set]
            Matrix that yields the values at the new nodes.
        ndarray[num_eval_set, num_given_set]
            Matrix that yields the time derivatives at the new nodes.
        """
        L_blocks = []
        D_blocks = []

        for iseg in range(self.num_segments):
            i1, i2 = self.subset_segment_indices[given_set_name][iseg, :]
            indices = self.subset_node_indices[given_set_name][i1:i2]
            nodes_given = self.node_stau[indices]

            i1, i2 = self.subset_segment_indices[eval_set_name][iseg, :]
            indices = self.subset_node_indices[eval_set_name][i1:i2]
            nodes_eval = self.node_stau[indices]

            L_block, D_block = lagrange_matrices(nodes_given, nodes_eval)

            L_blocks.append(L_block)
            D_blocks.append(D_block)

        L = block_diag(*L_blocks)
        D = block_diag(*D_blocks)

        return L, D

    def phase_hermite_matrices(self, given_set_name, eval_set_name):
        """
        Compute the matrices mapping values at some nodes to values and derivatives at new nodes.

        The equation for Hermite interpolation of the values is:
        .. math:: x_{eval} = \left[ A_i \right] x_{given}
                             + \frac{dt}{d\tau} \left[ B_i \right] \dot{x}_{given}

        Hermite interpolation of the derivatives is performed as:
        .. math:: \dot{x}_{eval} = \frac{d\tau}{dt} \left[ A_d \right] x_{given}
                                   + \left[ B_d \right] \dot{x}_{given}

        Parameters
        ----------
        given_set_name : str
            Name of the set of nodes with which to perform the interpolation.
        eval_set_name : str
            Name of the set of nodes at which to evaluate the values and derivatives.

        Returns
        -------
        ndarray[num_eval_set, num_given_set]
            Matrix that maps values at given nodes to values at eval nodes.
            This is A_i in the equations above.
        ndarray[num_eval_set, num_given_set]
            Matrix that maps derivatives at given nodes to values at eval nodes.
            This is B_i in the equations above.
        ndarray[num_eval_set, num_given_set]
            Matrix that maps values at given nodes to derivatives at eval nodes.
            This is A_d in the equations above.
        ndarray[num_eval_set, num_given_set]
            Matrix that maps derivatives at given nodes to derivatives at eval nodes.
            This is A_d in the equations above.
        """
        Ai_list = []
        Bi_list = []
        Ad_list = []
        Bd_list = []

        for iseg in range(self.num_segments):
            i1, i2 = self.subset_segment_indices[given_set_name][iseg, :]
            indices = self.subset_node_indices[given_set_name][i1:i2]
            nodes_given = self.node_stau[indices]

            i1, i2 = self.subset_segment_indices[eval_set_name][iseg, :]
            indices = self.subset_node_indices[eval_set_name][i1:i2]
            nodes_eval = self.node_stau[indices]

            Ai_seg, Bi_seg, Ad_seg, Bd_seg = hermite_matrices(nodes_given, nodes_eval)

            Ai_list.append(Ai_seg)
            Bi_list.append(Bi_seg)
            Ad_list.append(Ad_seg)
            Bd_list.append(Bd_seg)

        Ai = block_diag(*Ai_list)
        Bi = block_diag(*Bi_list)
        Ad = block_diag(*Ad_list)
        Bd = block_diag(*Bd_list)

        return Ai, Bi, Ad, Bd
