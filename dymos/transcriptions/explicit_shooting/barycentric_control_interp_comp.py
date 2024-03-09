import functools
# from itertools import combinations, permutations
import numpy as np
from numpy.typing import ArrayLike
import openmdao.api as om

try:
    from numba import njit, prange
except ImportError:
    # If numba is not available, just write a dummy njit wrapper.
    # Code will still run at a significant performance hit.
    def njit(*args, **kwargs):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Call the original function
                return func(*args, **kwargs)
            return wrapper
        return decorator


from ...utils.lgl import lgl
from ...utils.misc import get_rate_units


@njit()
def _compute_dl_dg(tau: float,
                   taus: ArrayLike,
                   mask: ArrayLike,
                   l: ArrayLike,
                   dl_dg: ArrayLike,
                   d2l_dg2: ArrayLike,
                   d3l_dg3: ArrayLike | None = None,):
    """Compute the Lagrange polynomials and the first three derivatives wrt g at the nodes.

    This function achieves good performance with numba.njit (and uses a do-nothing njit decorator
    if numba is not available).

    Parameters
    ----------
    tau : float
        The value of the independent variable at which the interpolation is being requested.
    taus : ArrayLike
        An n-vector giving location of the polynomial nodes in the independent variable dimension.
    mask : ArrayLike
        A boolean array used to mask off portions of g for faster products.
    l : ArrayLike
        An n-vector giving the value of the Lagrange polynomial at each node.
    dl_dg : ArrayLike
        An n x n - vector giving the derivative of l wrt g.
    d2l_dg2 : ArrayLike
        An n x n x n - vector giving the second derivative of l wrt g.
    d3l_dg3 : ArrayLike
        An n x n x n - vector giving the second derivative of l wrt g.
    """
    n = len(taus)
    g = tau - taus

    mask[...] = True

    for i in range(n):
        mask[i] = False
        # l[i] = np.prod(np.delete(g, [i]))
        l[i] = np.prod(g[mask])
        # dl_dg is symmetric, so fill two elements at once
        if dl_dg is not None:
            for j in range(i+1, n):
                mask[j] = False
                dl_dg[i, j] = dl_dg[j, i] = np.prod(g[mask])
                if d2l_dg2 is not None:
                    for k in range(n):
                        if k != i and k != j:
                            mask[k] = False
                            val = np.prod(g[mask])
                            d2l_dg2[i, j, k] = d2l_dg2[j, i, k] = val
                            if d3l_dg3 is not None:
                                # We only need this if derivs of second deriv wrt tau are needed.
                                for ii in range(k + 1, n):
                                    if ii not in {i, j, k}:
                                        mask[ii] = False
                                        val = np.prod(g[mask])
                                        d3l_dg3[i, j, k, ii] = d3l_dg3[j, i, k, ii] = \
                                            d3l_dg3[i, j, ii, k] = d3l_dg3[j, i, ii, k] = val
                                        mask[ii] = True
                            mask[k] = True
                mask[j] = True
        mask[i] = True


class BarycentricControlInterpComp(om.ExplicitComponent):
    """A component which interpolates control values in 1D using Vandermonde interpolation.

    Takes training values for control variables at given _input_ nodes,
    broadcasts them to _discretization_ nodes, and then interpolates the discretization values
    to provide a control variable at a given segment tau or phase tau.

    For dynamic controls, the current segment is given as a discrete input and the interpolation is
    a smooth polynomial along the given segment.

    OpenMDAO assumes sizes of variables at setup time, and we don't want to need to change the
    size of the control input nodes when we evaluate different segments. Instead, this component
    will take in the control values of all segments and internally use the appropriate one.

    Parameters
    ----------
    grid_data : GridData
        A GridData instance that details information on how the control input and discretization
        nodes are layed out.
    control_options : dict of {str: ControlOptionsDictionary}
        A mapping that maps the name of each control to a ControlOptionsDictionary of its options.
    polynomial_control_options : dict of {str: PolynomialControlOptionsDictionary}
        A mapping that maps the name of each polynomial control to an OptionsDictionary of its options.
    time_units : str
        The time units pertaining to the control rates.
    standalone_mode : bool
        If True, this component runs its configuration steps during setup. This is useful for
        unittests in which the component does not exist in a larger group.
    **kwargs
        Keyword arguments passed to ExplicitComponent.
    """
    def __init__(self, grid_data, control_options=None, polynomial_control_options=None,
                 time_units=None, standalone_mode=False, compute_derivs=True, **kwargs):
        self._grid_data = grid_data
        self._control_options = {} if control_options is None else control_options
        self._polynomial_control_options = {} if polynomial_control_options is None else polynomial_control_options
        self._time_units = time_units
        self._standalone_mode = standalone_mode
        self._compute_derivs = compute_derivs
        self._under_complex_step_prev = False
        self._taus_seg = {}

        self._inputs_hash_cache = None

        # Cache formatted strings: { control_name : (input_name, output_name) }
        self._control_io_names = {}

        # The Lagrange interpolation matrix L_id maps control values as given at the input nodes
        # to values at the discretization nodes.
        num_disc_nodes = grid_data.subset_num_nodes['control_disc']
        num_input_nodes = grid_data.subset_num_nodes['control_input']
        self._L_id = {}
        self._L_id['controls'] = np.zeros((num_disc_nodes, num_input_nodes), dtype=float)
        self._L_id['controls'][np.arange(num_disc_nodes, dtype=int),
                               self._grid_data.input_maps['dynamic_control_input_to_disc']] = 1.0

        super().__init__(**kwargs)

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('segment_index', default=None, types=int, desc='index of the current segment')
        self.options.declare('vec_size', types=int, default=1,
                             desc='number of points at which the control will be evaluated. This is not'
                                  'necessarily the same as the number of nodes in the GridData.')

    def set_segment_index(self, idx, alloc_complex=False):
        """Set the index of the segment being interpolated.

        Parameters
        ----------
        idx : int
            The index of the segment being interpolated.
        alloc_complex : bool
            If True, allocate storage for complex step.
        """
        self.options['segment_index'] = idx
        disc_node_idxs = self._disc_node_idxs_by_segment[idx]

        i1, i2 = self._grid_data.subset_segment_indices['control_disc'][idx, :]
        indices = self._grid_data.subset_node_indices['control_disc'][i1:i2]
        taus_seg = self._grid_data.node_stau[indices]

        n = len(taus_seg)
        ar_n = np.arange(n, dtype=int)
        ptaus = {pc_name: lgl(pc_options['order'] + 1)[0]
                 for pc_name, pc_options in self._polynomial_control_options.items()}
        self._compute_barycentric_weights(taus_seg, ptaus)
        dtype = complex if alloc_complex else float

        # The arrays pertaining to the collocated controls are stored with the 'controls' key
        self._l = {'controls': np.ones(n, dtype=dtype)}
        self._dl_dg = {'controls': np.zeros((n, n), dtype=dtype)}
        self._d2l_dg2 = {'controls': np.zeros((n, n, n), dtype=dtype)}
        self._dl_dtau = {'controls': np.ones((n, 1), dtype=dtype)}
        self._d2l_dtau2 = {'controls': np.ones((n, 1), dtype=dtype)}

        if self._compute_derivs:
            self._d3l_dg3 = {'controls': np.zeros((n, n, n, n), dtype=dtype)}
            self._d3l_dtau3 = {'controls': np.ones((n, 1), dtype=dtype)}
        else:
            self._d3l_dg3 = {'controls': None}
            self._d3l_dtau3 = {'controls': None}

        self._taus_seg['controls'] = taus_seg

        # Arrays pertaining to polynomial controls are stored with their name as a key
        for pc_name, pc_options in self._polynomial_control_options.items():
            n = pc_options['order'] + 1
            ar_n = np.arange(n, dtype=int)
            self._taus_seg[pc_name] = lgl(n)[0]
            self._l[pc_name] = np.ones(n, dtype=dtype)
            self._dl_dg[pc_name] = np.zeros((n, n), dtype=dtype)
            self._d2l_dg2[pc_name] = np.zeros((n, n, n), dtype=dtype)
            self._dl_dtau[pc_name] = np.ones((n, 1), dtype=dtype)
            self._d2l_dtau2[pc_name] = np.ones((n, 1), dtype=dtype)

            if self._compute_derivs:
                self._d3l_dg3[pc_name] = np.zeros((n, n, n, n), dtype=dtype)
                self._d3l_dtau3[pc_name] = np.ones((n, 1), dtype=dtype)
            else:
                self._d3l_dg3[pc_name] = None
                self._d3l_dtau3[pc_name] = None

    def _configure_controls(self):
        vec_size = self.options['vec_size']
        gd = self._grid_data

        self._disc_node_idxs_by_segment = []
        self._input_node_idxs_by_segment = []

        first_disc_node_in_seg = first_input_node_in_seg = 0

        for seg_idx in range(gd.num_segments):
            # Number of control discretization nodes per segment
            ncdnps = gd.subset_num_nodes_per_segment['control_disc'][seg_idx]
            ncinps = gd.subset_num_nodes_per_segment['control_input'][seg_idx]

            ar_control_disc_nodes = np.arange(ncdnps, dtype=int)
            disc_idxs_in_seg = first_disc_node_in_seg + np.arange(ncdnps, dtype=int)
            input_idxs_in_seg = first_input_node_in_seg + np.arange(ncinps, dtype=int)

            first_disc_node_in_seg += ncdnps
            first_input_node_in_seg += ncinps

            # The indices of the discretization node u vector pertaining to the given segment
            self._disc_node_idxs_by_segment.append(disc_idxs_in_seg)

            # The indices of the input u vector pertaining to the given segment
            self._input_node_idxs_by_segment.append(input_idxs_in_seg)

        if not self._control_options:
            return

        num_uhat_nodes = gd.subset_num_nodes['control_input']
        ar = np.arange(vec_size, dtype=int)
        for control_name, options in self._control_options.items():
            shape = options['shape']
            units = options['units']
            input_name = f'controls:{control_name}'
            output_name = f'control_values:{control_name}'
            rate_name = f'control_rates:{control_name}_rate'
            rate2_name = f'control_rates:{control_name}_rate2'
            rate_units = get_rate_units(units, self._time_units)
            rate2_units = get_rate_units(units, self._time_units, deriv=2)
            uhat_shape = (num_uhat_nodes,) + shape
            output_shape = (vec_size,) + shape
            self.add_input(input_name, shape=uhat_shape, units=units)
            self.add_output(output_name, shape=output_shape, units=units)
            self.add_output(rate_name, shape=output_shape, units=rate_units)
            self.add_output(rate2_name, shape=output_shape, units=rate2_units)
            self._control_io_names[control_name] = (input_name, output_name, rate_name, rate2_name)
            self.declare_partials(of=output_name, wrt=input_name)
            self.declare_partials(of=output_name, wrt='stau', rows=ar, cols=ar)
            self.declare_partials(of=rate_name, wrt=input_name)
            self.declare_partials(of=rate_name, wrt='stau', rows=ar, cols=ar)
            self.declare_partials(of=rate_name, wrt='dstau_dt')
            self.declare_partials(of=rate2_name, wrt=input_name)
            self.declare_partials(of=rate2_name, wrt='stau', rows=ar, cols=ar)
            self.declare_partials(of=rate2_name, wrt='dstau_dt')

    def _configure_polynomial_controls(self):
        vec_size = self.options['vec_size']
        ar = np.arange(vec_size, dtype=int)

        for pc_name, options in self._polynomial_control_options.items():
            order = options['order']

            shape = options['shape']
            units = options['units']
            input_name = f'polynomial_controls:{pc_name}'
            output_name = f'polynomial_control_values:{pc_name}'
            rate_name = f'polynomial_control_rates:{pc_name}_rate'
            rate2_name = f'polynomial_control_rates:{pc_name}_rate2'
            rate_units = get_rate_units(units, self._time_units)
            rate2_units = get_rate_units(units, self._time_units, deriv=2)
            input_shape = (order + 1,) + shape
            output_shape = (vec_size,) + shape
            self.add_input(input_name, shape=input_shape, units=units)
            self.add_output(output_name, shape=output_shape, units=units)
            self.add_output(rate_name, shape=output_shape, units=rate_units)
            self.add_output(rate2_name, shape=output_shape, units=rate2_units)
            self._control_io_names[pc_name] = (input_name, output_name, rate_name, rate2_name)
            self.declare_partials(of=output_name, wrt=input_name)
            self.declare_partials(of=output_name, wrt='ptau', rows=ar, cols=ar)
            self.declare_partials(of=rate_name, wrt=input_name)
            self.declare_partials(of=rate_name, wrt='ptau', rows=ar, cols=ar)
            self.declare_partials(of=rate_name, wrt='t_duration')
            self.declare_partials(of=rate2_name, wrt=input_name)
            self.declare_partials(of=rate2_name, wrt='ptau', rows=ar, cols=ar)
            self.declare_partials(of=rate2_name, wrt='t_duration')

    def setup(self):
        """
        Perform the I/O creation if operating in _standalone_mode.
        """
        if self._standalone_mode:
            self.configure_io()

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units for the controls.
        """
        vec_size = self.options['vec_size']

        # self.add_discrete_input('segment_index', val=0, desc='index of the segment')
        self.add_input('stau', shape=(vec_size,), units=None)
        self.add_input('dstau_dt', val=1.0, units=f'1/{self._time_units}')
        self.add_input('t_duration', val=1.0, units=self._time_units)
        self.add_input('ptau', shape=(vec_size,), units=None)

        self._configure_controls()
        self._configure_polynomial_controls()

    def _compute_barycentric_weights(self, taus: ArrayLike, ptaus: dict):
        """Computes the barycentric weights given a set of nodes.

        Parameters
        ----------
        taus : ArrayLike
            The nodes of the collocated control polynomial.
        ptaus : dict[ArrayLike]
            The nodes of each polynomial control polynomial.
        """
        n = len(taus)
        self._w_b = {'controls': np.ones((n, 1))}

        for j in range(n):
            self._w_b['controls'][j, 0] = 1. / np.prod(taus[j] - np.delete(taus, j))

        for pc_name, _ptaus in ptaus.items():
            n = len(_ptaus)
            self._w_b[pc_name] = np.ones((n, 1))
            for j in range(n):
                self._w_b[pc_name][j, 0] = 1. / np.prod(_ptaus[j] - np.delete(_ptaus, j))

    def _compute_dl_dg(self, tau, taus, l, dl_dg, d2l_dg2, d3l_dg3=None,
                       dl_dg_prod_idx_map=None, d2l_dg2_prod_idx_map=None,
                       d3l_dg3_prod_idx_map=None):
        """
        Compute the Lagrange polynomial coefficients and the first 3 derivatives wrt g at the nodes.

        When, be sure to enable the _combo_idxs and _ijk indexing arrays in set_segment index.

        Parameters
        ----------
        tau : _type_
            _description_
        taus : _type_
            _description_
        name : _type_
            _description_
        """
        n = len(taus)
        ar_n = np.arange(n, dtype=int)
        g = tau - taus

        # l = self._l[name]
        # dl_dg = self._dl_dg[name]
        # d2l_dg2 = self._d2l_dg2[name]
        # d3l_dg3 = self._d3l_dg3[name]

        # Compute l by populating dl_dg as temporary storage,
        # and then using a reducing multiply.
        dl_dg[...] = np.tile(g, (n, 1))
        np.fill_diagonal(dl_dg, 1.0)
        l[...] = np.prod(dl_dg, axis=1)

        # Now populate dl_dg.  The diagonals are one and the lower and upper triangular parts are equal.
        prod_g_idxs, put_idxs = dl_dg_prod_idx_map
        dl_dg_vals = np.prod(g.take(prod_g_idxs), axis=1)
        dl_dg[...] = 0.0
        dl_dg_i, dl_dg_j = put_idxs
        dl_dg[dl_dg_i, dl_dg_j] = dl_dg_vals

        # Assign the lower half to be the same as the upper half
        dl_dg[...] = dl_dg + dl_dg.T - np.diag(np.diag(dl_dg))

        # For the higher derivative matrices use a mapping of
        # coordinates to the product of g values at those indices.
        if d2l_dg2_prod_idx_map is not None:
            for prod_g_idxs, put_idxs in d2l_dg2_prod_idx_map.items():
                d2l_dg2[tuple(zip(*put_idxs))] = np.prod(g.take(prod_g_idxs))

        if d3l_dg3 is not None:
            # We only need d3l_dg3 for the derivatives of rate2.
            for prod_g_idxs, put_idxs in d3l_dg3_prod_idx_map.items():
                d3l_dg3[tuple(zip(*put_idxs))] = np.prod(g.take(prod_g_idxs))

    def _compute_controls(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute interpolated control values and rates for the collocated controls.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete_inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete_outputs.
        """
        gd = self._grid_data
        seg_idx = self.options['segment_index']
        stau = inputs['stau']
        dstau_dt = inputs['dstau_dt']

        disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
        input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
        # V_stau = np.vander(stau, N=n, increasing=True)
        # dV_stau, dV2_stau, _ = self._dvander(V_stau)
        taus_seg = self._taus_seg['controls']

        # Retrieve the storage vectors that pertain to the collocated controls
        l = self._l['controls']
        dl_dg = self._dl_dg['controls']
        d2l_dg2 = self._d2l_dg2['controls']

        dl_dstau = self._dl_dtau['controls']
        d2l_dstau2 = self._d2l_dtau2['controls']
        w_b = self._w_b['controls']

        mask = np.ones_like(taus_seg, dtype=bool)

        # dl_dg_prod_idx_map = self._dl_dg_prod_idx_map['controls']
        # d2l_dg2_prod_idx_map = self._d2l_dg2_prod_idx_map['controls']

        # self._compute_dl_dg(stau, taus_seg, l, dl_dg, d2l_dg2, d3l_dg3=None,
        #                     dl_dg_prod_idx_map=dl_dg_prod_idx_map,
        #                     d2l_dg2_prod_idx_map=d2l_dg2_prod_idx_map,
        #                     d3l_dg3_prod_idx_map=None)
        _compute_dl_dg(stau, taus_seg, mask, l, dl_dg, d2l_dg2, d3l_dg3=None)


        # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
        dl_dstau[...] = np.sum(dl_dg, axis=-1, keepdims=True)

        # d2l_dg @ dg_dtau + dl_dg @ d2g_dtau2 but d2g_dtau2 is zero.
        d2l_dstau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)

        L_id = self._L_id['controls']
        L_seg = L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                     input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

        for control_name in self._control_options:
            input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

            # Translate the input nodes to the discretization nodes.
            u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])

            # Perform a row_wise multiplication of w_b and u_hat
            wbuhat = np.einsum("ij,i...->i...", w_b, u_hat)

            outputs[output_name] = np.einsum('i...,i...->...', l, wbuhat)
            outputs[rate_name] = wbuhat.T @ dl_dstau * dstau_dt
            outputs[rate2_name] = wbuhat.T @ d2l_dstau2 * dstau_dt ** 2

            print(self.options['segment_index'], outputs[output_name])

    def _compute_polynomial_controls(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute interpolated control values and rates for the polynomial controls.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete_inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete_outputs.
        """
        gd = self._grid_data
        seg_idx = self.options['segment_index']
        ptau = inputs['ptau']
        dptau_dt = 2. / inputs['t_duration']

        disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
        input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
        # V_stau = np.vander(stau, N=n, increasing=True)
        # dV_stau, dV2_stau, _ = self._dvander(V_stau)

        for pc_name in self._polynomial_control_options:
            # Retrieve the storage vectors that pertain to the polynomial control
            taus_seg = self._taus_seg[pc_name]
            l = self._l[pc_name]
            dl_dg = self._dl_dg[pc_name]
            d2l_dg2 = self._d2l_dg2[pc_name]

            dl_dptau = self._dl_dtau[pc_name]
            d2l_dptau2 = self._d2l_dtau2[pc_name]
            w_b = self._w_b[pc_name]

            mask = np.ones_like(taus_seg, dtype=bool)

            # dl_dg_prod_idx_map = self._dl_dg_prod_idx_map[pc_name]
            # d2l_dg2_prod_idx_map = self._d2l_dg2_prod_idx_map[pc_name]

            # self._compute_dl_dg(ptau, taus_seg, l, dl_dg, d2l_dg2, d3l_dg3=None,
            #                     dl_dg_prod_idx_map=dl_dg_prod_idx_map,
            #                     d2l_dg2_prod_idx_map=d2l_dg2_prod_idx_map,
            #                     d3l_dg3_prod_idx_map=None)
            _compute_dl_dg(ptau, taus_seg, mask, l, dl_dg, d2l_dg2)

            # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
            dl_dptau[...] = np.sum(dl_dg, axis=-1, keepdims=True)

            # d2l_dg @ dg_dtau + dl_dg @ d2g_dtau2 but d2g_dtau2 is zero.
            d2l_dptau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)

            input_name, output_name, rate_name, rate2_name = self._control_io_names[pc_name]

            # Translate the input nodes to the discretization nodes.
            u_hat = inputs[input_name]

            # Perform a row_wise multiplication of w_b and u_hat
            wbuhat = np.einsum("ij,i...->i...", w_b, u_hat)

            # outputs[output_name] = np.einsum('i...,i...->...', l, wbuhat)
            outputs[output_name] = wbuhat.T @ l
            outputs[rate_name] = wbuhat.T @ dl_dptau * dptau_dt
            outputs[rate2_name] = wbuhat.T @ d2l_dptau2 * dptau_dt ** 2

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute interpolated control values and rates.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete_inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete_outputs.
        """
        if self.under_complex_step != self._under_complex_step_prev:
            self.set_segment_index(self.options['segment_index'], alloc_complex=self.under_complex_step)
            self._under_complex_step_prev = self.under_complex_step

        inputs_hash = inputs.get_hash()
        if inputs_hash != self._inputs_hash_cache:
            # Do the compute if our inputs have changed
            if self._control_options:
                self._compute_controls(inputs, outputs, discrete_inputs, discrete_outputs)
            if self._polynomial_control_options:
                self._compute_polynomial_controls(inputs, outputs, discrete_inputs, discrete_outputs)
            self._inputs_hash_cache = inputs_hash

    def _compute_partials_controls(self, inputs, partials, discrete_inputs=None):
        """
        Compute partials of interpolated control values and rates for the collocated controls.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete_inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete_outputs.
        """
        if self._control_options:
            gd = self._grid_data
            seg_idx = self.options['segment_index']
            n = gd.transcription_order[seg_idx]
            stau = inputs['stau']
            dstau_dt = inputs['dstau_dt']

            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
            # V_stau = np.vander(stau, N=n, increasing=True)
            # dV_stau, dV2_stau, _ = self._dvander(V_stau)
            taus_seg = gd.node_stau[disc_node_idxs]

            # Retrieve the storage vectors that pertain to the collocated controls
            l = self._l['controls']
            dl_dg = self._dl_dg['controls']
            d2l_dg2 = self._d2l_dg2['controls']
            d3l_dg3 = self._d3l_dg3['controls']
            dl_dstau = self._dl_dtau['controls']
            d2l_dstau2 = self._d2l_dtau2['controls']
            d3l_dstau3 = self._d3l_dtau3['controls']
            w_b = self._w_b['controls']

            mask = np.ones_like(taus_seg, dtype=bool)

            # Update self._l, self._dl_dg, self._d2l_dg2, and self._d3l_dg3 in place.
            _compute_dl_dg(stau, taus_seg, mask, l, dl_dg, d2l_dg2, d3l_dg3)

            # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
            dl_dstau[...] = np.sum(dl_dg, axis=-1, keepdims=True)
            d2l_dstau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)

            if self._compute_derivs:
                d3l_dstau3[...] = np.sum(np.sum(np.sum(d3l_dg3, axis=-1), axis=-1), axis=-1, keepdims=True)

            L_id = self._L_id['controls']
            L_seg = L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                         input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            for control_name in self._control_options:
                input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

                # Translate the input nodes to the discretization nodes.
                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])

                # Perform a row_wise multiplication of w_b and u_hat
                wbuhat = np.einsum("ij,i...->i...", w_b, u_hat)

                # outputs[output_name] = np.einsum('i...,i...->...', l, wbuhat)
                partials[output_name, 'stau'] = dl_dstau.T @ wbuhat

                partials[rate_name, 'stau'] = d2l_dstau2.T @ wbuhat * dstau_dt
                partials[rate_name, 'dstau_dt'] = partials[output_name, 'stau']

                partials[rate2_name, 'stau'] = d3l_dstau3.T @ wbuhat * dstau_dt ** 2
                partials[rate2_name, 'dstau_dt'] = 2 * wbuhat.T @ d2l_dstau2 * dstau_dt

                # Assign only thos jacobian columns due to the current segment, since
                # other segments cannot impact interpolation in this one.
                partials[output_name, input_name] = 0.0
                partials[output_name, input_name][..., input_node_idxs] = \
                    ((l * w_b.T) @ L_seg)

                partials[rate_name, input_name] = 0.0
                partials[rate_name, input_name][..., input_node_idxs] = \
                    (dl_dstau * w_b * dstau_dt).T @ L_seg

                partials[rate2_name, input_name] = 0.0
                partials[rate2_name, input_name][..., input_node_idxs] = \
                    (d2l_dstau2 * w_b * dstau_dt ** 2).T @ L_seg

    def _compute_partials_polynomial_controls(self, inputs, partials, discrete_inputs=None):
        """
        Compute partials of interpolated control values and rates for the collocated controls.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        discrete_inputs : `Vector`
            `Vector` containing discrete_inputs.
        discrete_outputs : `Vector`
            `Vector` containing discrete_outputs.
        """
        ptau = inputs['ptau']
        t_duration = inputs['t_duration']
        dptau_dt = 2.0 / t_duration

        for pc_name, pc_options in self._polynomial_control_options.items():
            gd = self._grid_data
            n = pc_options['order'] + 1

            taus_seg = self._taus_seg[pc_name]

            # Retrieve the storage vectors that pertain to the collocated controls
            l = self._l[pc_name]
            dl_dg = self._dl_dg[pc_name]
            d2l_dg2 = self._d2l_dg2[pc_name]
            d3l_dg3 = self._d3l_dg3[pc_name]
            dl_dstau = self._dl_dtau[pc_name]
            d2l_dstau2 = self._d2l_dtau2[pc_name]
            d3l_dstau3 = self._d3l_dtau3[pc_name]
            w_b = self._w_b[pc_name]

            mask = np.ones_like(taus_seg, dtype=bool)

            # dl_dg_prod_idx_map = self._dl_dg_prod_idx_map[pc_name]
            # d2l_dg2_prod_idx_map = self._d2l_dg2_prod_idx_map[pc_name]
            # d3l_dg3_prod_idx_map = self._d3l_dg3_prod_idx_map[pc_name]

            # self._compute_dl_dg(ptau, taus_seg, l, dl_dg, d2l_dg2, d3l_dg3,
            #                     dl_dg_prod_idx_map=dl_dg_prod_idx_map,
            #                     d2l_dg2_prod_idx_map=d2l_dg2_prod_idx_map,
            #                     d3l_dg3_prod_idx_map=d3l_dg3_prod_idx_map)
            _compute_dl_dg(ptau, taus_seg, mask, l, dl_dg, d2l_dg2, d3l_dg3)

            # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
            dl_dstau[...] = np.sum(dl_dg, axis=-1, keepdims=True)
            d2l_dstau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)
            d3l_dstau3[...] = np.sum(np.sum(np.sum(d3l_dg3, axis=-1), axis=-1), axis=-1, keepdims=True)

            input_name, output_name, rate_name, rate2_name = self._control_io_names[pc_name]

            # Translate the input nodes to the discretization nodes.
            u_hat = inputs[input_name]

            # Perform a row_wise multiplication of w_b and u_hat
            wbuhat = np.einsum("ij,i...->i...", w_b, u_hat)

            # outputs[output_name] = np.einsum('i...,i...->...', l, wbuhat)

            d_dptau_dt_d_t_duration = -2.0 / t_duration ** 2

            partials[output_name, 'ptau'] = dl_dstau.T @ wbuhat

            partials[rate_name, 'ptau'] = d2l_dstau2.T @ wbuhat * dptau_dt
            partials[rate_name, 't_duration'] = partials[output_name, 'ptau'] * d_dptau_dt_d_t_duration

            partials[rate2_name, 'ptau'] = d3l_dstau3.T @ wbuhat * dptau_dt ** 2
            partials[rate2_name, 't_duration'] = 2 * wbuhat.T @ d2l_dstau2 * dptau_dt * d_dptau_dt_d_t_duration

            # Assign only thos jacobian columns due to the current segment, since
            # other segments cannot impact interpolation in this one.
            # partials[output_name, input_name] = 0.0
            partials[output_name, input_name][...] = ((l * w_b.T).T).T

            # partials[rate_name, input_name] = 0.0
            partials[rate_name, input_name][...] = (dl_dstau * w_b * dptau_dt).T

            # partials[rate2_name, input_name] = 0.0
            partials[rate2_name, input_name][...] = (d2l_dstau2 * w_b * dptau_dt ** 2).T

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        """
        Compute derivatives of interpolated control values and rates wrt the inputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        discrete_inputs : Vector
            Unscaled, discrete input variables keyed by variable name.
        """
        if self._control_options:
            self._compute_partials_controls(inputs, partials, discrete_inputs)

        if self._polynomial_control_options:
            self._compute_partials_polynomial_controls(inputs, partials, discrete_inputs)

            # for control_name, options in self._control_options.items():
            #     input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

            #     u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs].real)
            #     a = self._V_hat_inv[seg_order] @ u_hat

            #     da_duhat = self._V_hat_inv[seg_order] @ L_seg
            #     dV_a = dV_stau @ a
            #     dV2_a = dV2_stau @ a
            #     dV3_a = dV3_stau @ a

            #     partials[output_name, input_name][...] = 0.0
            #     partials[output_name, input_name][..., u_idxs] = V_stau @ da_duhat
            #     partials[output_name, 'stau'] = dV_a.ravel()

            #     pudot_pa = dstau_dt * dV_stau
            #     pa_puhat = self._V_hat_inv[seg_order]
            #     partials[rate_name, input_name][...] = 0.0
            #     partials[rate_name, input_name][..., u_idxs] = pudot_pa @ pa_puhat
            #     partials[rate_name, 'dstau_dt'][...] = dV_a
            #     partials[rate_name, 'stau'][...] = dV2_a.ravel()

            #     pu2dot_pa = dstau_dt**2 * dV2_stau
            #     partials[rate2_name, input_name][...] = 0.0
            #     partials[rate2_name, input_name][..., u_idxs] = pu2dot_pa @ pa_puhat
            #     partials[rate2_name, 'dstau_dt'][...] = 2 * dstau_dt * dV2_a
            #     partials[rate2_name, 'stau'][...] = dV3_a.ravel()

        # for pc_name, options in self._polynomial_control_options.items():
        #     input_name, output_name, rate_name, rate2_name = self._control_io_names[pc_name]
        #     order = options['order']

        #     V_ptau = np.vander(ptau, N=order+1, increasing=True)
        #     dV_ptau, dV2_ptau, dV3_ptau = self._dvander(V_ptau)

        #     u_hat = inputs[input_name].real
        #     a = self._V_hat_inv[order] @ u_hat

        #     dV_a = dV_ptau @ a
        #     dV2_a = dV2_ptau @ a
        #     dV3_a = dV3_ptau @ a

        #     da_duhat = self._V_hat_inv[order]

        #     partials[output_name, input_name][...] = V_ptau @ da_duhat
        #     partials[output_name, 'ptau'][...] = dV_a.ravel()

        #     pudot_pa = dptau_dt * dV_ptau
        #     pa_puhat = self._V_hat_inv[order]
        #     partials[rate_name, input_name][...] = pudot_pa @ pa_puhat
        #     partials[rate_name, 't_duration'][...] = ddptau_dt_dtduration * dV_a
        #     partials[rate_name, 'ptau'][...] = dptau_dt * dV2_a.ravel()

        #     pu2dot_pa = dptau_dt ** 2 * dV2_ptau
        #     partials[rate2_name, input_name][...] = pu2dot_pa @ pa_puhat
        #     partials[rate2_name, 't_duration'][...] = 2 * dptau_dt * ddptau_dt_dtduration * dV2_a
        #     partials[rate2_name, 'ptau'][...] = dptau_dt**2 * dV3_a.ravel()
