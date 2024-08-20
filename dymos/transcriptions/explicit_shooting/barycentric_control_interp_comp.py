import functools
from typing import Union
import numpy as np
from numpy.typing import ArrayLike
import openmdao.api as om

try:
    from numba import njit
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
                   l: ArrayLike,  # noqa: E741, allow ambiguous variable name 'l'
                   dl_dg: ArrayLike,
                   d2l_dg2: ArrayLike,
                   d3l_dg3: Union[ArrayLike, None] = None,):
    """Compute the Lagrange polynomials and the first three derivatives wrt g at the nodes.

    This function achieves good performance with numba.njit (and uses a do-nothing njit decorator
    if numba is not available).

    Parameters
    ----------
    tau : float
        The value of the independent variable at which the interpolation is being requested.
    taus : ArrayLike
        An n-vector giving location of the polynomial nodes in the independent variable dimension.
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

    for i in range(n):
        i_save = g[i]
        g[i] = 1.0
        l[i] = np.prod(g)
        # dl_dg is symmetric, so fill two elements at once
        if dl_dg is not None:
            for j in range(i+1, n):
                j_save = g[j]
                g[j] = 1.0
                val = np.prod(g)
                dl_dg[i, j] = dl_dg[j, i] = val
                if d2l_dg2 is not None:
                    for k in range(n):
                        if k != i and k != j:
                            k_save = g[k]
                            g[k] = 1.0
                            d2l_dg2[i, j, k] = d2l_dg2[j, i, k] = np.prod(g)
                            if d3l_dg3 is not None:
                                # We only need this if derivs of second deriv wrt tau are needed.
                                for ii in range(k + 1, n):
                                    if ii not in {i, j, k}:
                                        ii_save = g[ii]
                                        g[ii] = 1
                                        d3l_dg3[i, j, k, ii] = d3l_dg3[j, i, k, ii] = \
                                            d3l_dg3[i, j, ii, k] = d3l_dg3[j, i, ii, k] = np.prod(g)
                                        g[ii] = ii_save
                            g[k] = k_save
                g[j] = j_save
        g[i] = i_save


class BarycentricControlInterpComp(om.ExplicitComponent):
    """
    A component which interpolates control values in 1D using Vandermonde interpolation.

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
    time_units : str
        The time units pertaining to the control rates.
    standalone_mode : bool
        If True, this component runs its configuration steps during setup. This is useful for
        unittests in which the component does not exist in a larger group.
    compute_derivs : bool
        Set to True if derivatives of rate2 need to be computed, otherwise False.
    **kwargs : dict, optional
        Keyword arguments passed to ExplicitComponent.
    """
    def __init__(self, grid_data, control_options=None,
                 time_units=None, standalone_mode=False, compute_derivs=True, **kwargs):
        self._grid_data = grid_data
        self._control_options = {} if control_options is None else control_options
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

    def set_segment_index(self, idx, alloc_complex=False):
        """
        Set the index of the segment being interpolated.

        Parameters
        ----------
        idx : int
            The index of the segment being interpolated.
        alloc_complex : bool
            If True, allocate storage for complex step.
        """
        self.options['segment_index'] = idx

        i1, i2 = self._grid_data.subset_segment_indices['control_disc'][idx, :]
        indices = self._grid_data.subset_node_indices['control_disc'][i1:i2]
        taus_seg = self._grid_data.node_stau[indices]

        n = len(taus_seg)
        ptaus = {}
        for control_name, options in self._control_options.items():
            if options['control_type'] == 'polynomial':
                order = options['order']
                ptaus[control_name] = lgl(order + 1)[0]
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
        for pc_name in ptaus:
            pc_options = self._control_options[pc_name]
            n = pc_options['order'] + 1
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
        gd = self._grid_data

        self._disc_node_idxs_by_segment = []
        self._input_node_idxs_by_segment = []

        first_disc_node_in_seg = 0

        for seg_idx in range(gd.num_segments):
            # Number of control discretization nodes per segment
            ncdnps = gd.subset_num_nodes_per_segment['control_disc'][seg_idx]
            ar_control_disc_nodes = np.arange(ncdnps, dtype=int)
            disc_idxs_in_seg = first_disc_node_in_seg + ar_control_disc_nodes
            first_disc_node_in_seg += ncdnps

            # The indices of the discretization node u vector pertaining to the given segment
            self._disc_node_idxs_by_segment.append(disc_idxs_in_seg)

            # The indices of the input u vector pertaining to the given segment
            self._input_node_idxs_by_segment.append(gd.input_maps['dynamic_control_input_to_disc'][disc_idxs_in_seg])

        if not self._control_options:
            return

        num_uhat_nodes = gd.subset_num_nodes['control_input']
        for control_name, options in self._control_options.items():
            if options['control_type'] == 'full':
                shape = options['shape']
                units = options['units']
                input_name = f'controls:{control_name}'
                output_name = f'control_values:{control_name}'
                rate_name = f'control_rates:{control_name}_rate'
                rate2_name = f'control_rates:{control_name}_rate2'
                rate_units = get_rate_units(units, self._time_units)
                rate2_units = get_rate_units(units, self._time_units, deriv=2)
                uhat_shape = (num_uhat_nodes,) + shape
                output_shape = (1,) + shape
                self.add_input(input_name, shape=uhat_shape, units=units)
                self.add_output(output_name, shape=output_shape, units=units)
                self.add_output(rate_name, shape=output_shape, units=rate_units)
                self.add_output(rate2_name, shape=output_shape, units=rate2_units)
                self._control_io_names[control_name] = (input_name, output_name, rate_name, rate2_name)
                self.declare_partials(of=output_name, wrt=input_name)
                self.declare_partials(of=output_name, wrt='stau')
                self.declare_partials(of=rate_name, wrt=input_name)
                self.declare_partials(of=rate_name, wrt='stau')
                self.declare_partials(of=rate_name, wrt='dstau_dt')

                if self._compute_derivs:
                    self.declare_partials(of=rate2_name, wrt=input_name)
                    self.declare_partials(of=rate2_name, wrt='stau')
                    self.declare_partials(of=rate2_name, wrt='dstau_dt')

            elif options['control_type'] == 'polynomial':
                order = options['order']

                shape = options['shape']
                units = options['units']
                input_name = f'controls:{control_name}'
                output_name = f'control_values:{control_name}'
                rate_name = f'control_rates:{control_name}_rate'
                rate2_name = f'control_rates:{control_name}_rate2'
                rate_units = get_rate_units(units, self._time_units)
                rate2_units = get_rate_units(units, self._time_units, deriv=2)
                input_shape = (order + 1,) + shape
                output_shape = (1,) + shape
                self.add_input(input_name, shape=input_shape, units=units)
                self.add_output(output_name, shape=output_shape, units=units)
                self.add_output(rate_name, shape=output_shape, units=rate_units)
                self.add_output(rate2_name, shape=output_shape, units=rate2_units)
                self._control_io_names[control_name] = (input_name, output_name, rate_name, rate2_name)
                self.declare_partials(of=output_name, wrt=input_name)
                self.declare_partials(of=output_name, wrt='ptau')
                self.declare_partials(of=rate_name, wrt=input_name)
                self.declare_partials(of=rate_name, wrt='ptau')
                self.declare_partials(of=rate_name, wrt='t_duration')

                if self._compute_derivs:
                    self.declare_partials(of=rate2_name, wrt=input_name)
                    self.declare_partials(of=rate2_name, wrt='ptau')
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

        # self.add_discrete_input('segment_index', val=0, desc='index of the segment')
        self.add_input('stau', shape=(1,), units=None)
        self.add_input('dstau_dt', val=1.0, units=f'1/{self._time_units}')
        self.add_input('t_duration', val=1.0, units=self._time_units)
        self.add_input('ptau', shape=(1,), units=None)

        self._configure_controls()

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
        seg_idx = self.options['segment_index']
        stau = inputs['stau']
        dstau_dt = inputs['dstau_dt']

        disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
        input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
        # V_stau = np.vander(stau, N=n, increasing=True)
        # dV_stau, dV2_stau, _ = self._dvander(V_stau)
        taus_seg = self._taus_seg['controls']

        # Retrieve the storage vectors that pertain to the collocated controls
        l = self._l['controls']  # noqa: E741, allow ambiguous variable name 'l'
        dl_dg = self._dl_dg['controls']
        d2l_dg2 = self._d2l_dg2['controls']

        dl_dstau = self._dl_dtau['controls']
        d2l_dstau2 = self._d2l_dtau2['controls']
        w_b = self._w_b['controls']

        # _compute_dl_dg(stau, taus_seg, l, dl_dg, d2l_dg2, d3l_dg3=None)
        _compute_dl_dg(tau=stau, taus=taus_seg, l=l, dl_dg=dl_dg, d2l_dg2=d2l_dg2, d3l_dg3=None)

        # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
        dl_dstau[...] = np.sum(dl_dg, axis=-1, keepdims=True)

        # d2l_dg @ dg_dtau + dl_dg @ d2g_dtau2 but d2g_dtau2 is zero.
        d2l_dstau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)

        L_id = self._L_id['controls']
        L_seg = L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                     input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

        seg_idx = self.options['segment_index']
        ptau = inputs['ptau']
        dptau_dt = 2. / inputs['t_duration']

        disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
        input_node_idxs = self._input_node_idxs_by_segment[seg_idx]

        for control_name, options in self._control_options.items():
            if options['control_type'] == 'full':
                input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

                # Translate the input nodes to the discretization nodes.
                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])

                # Perform a row_wise multiplication of w_b and u_hat
                wbuhat = np.einsum("ij,i...->i...", w_b, u_hat)

                outputs[output_name] = np.einsum('i...,i...->...', l, wbuhat)
                outputs[rate_name] = wbuhat.T @ dl_dstau * dstau_dt
                outputs[rate2_name] = wbuhat.T @ d2l_dstau2 * dstau_dt ** 2

            else:
                # Retrieve the storage vectors that pertain to the polynomial control
                taus_seg = self._taus_seg[control_name]
                l = self._l[control_name]  # noqa: E741, allow ambiguous variable name 'l'
                dl_dg = self._dl_dg[control_name]
                d2l_dg2 = self._d2l_dg2[control_name]

                dl_dptau = self._dl_dtau[control_name]
                d2l_dptau2 = self._d2l_dtau2[control_name]
                w_b = self._w_b[control_name]

                _compute_dl_dg(ptau, taus_seg, l, dl_dg, d2l_dg2)

                # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
                dl_dptau[...] = np.sum(dl_dg, axis=-1, keepdims=True)

                # d2l_dg @ dg_dtau + dl_dg @ d2g_dtau2 but d2g_dtau2 is zero.
                d2l_dptau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)

                input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

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
            stau = inputs['stau']
            dstau_dt = inputs['dstau_dt']

            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
            # V_stau = np.vander(stau, N=n, increasing=True)
            # dV_stau, dV2_stau, _ = self._dvander(V_stau)
            taus_seg = gd.node_stau[disc_node_idxs]

            # Retrieve the storage vectors that pertain to the collocated controls
            l = self._l['controls']  # noqa: E741, allow ambiguous variable name 'l'
            dl_dg = self._dl_dg['controls']
            d2l_dg2 = self._d2l_dg2['controls']
            d3l_dg3 = self._d3l_dg3['controls']
            dl_dstau = self._dl_dtau['controls']
            d2l_dstau2 = self._d2l_dtau2['controls']
            d3l_dstau3 = self._d3l_dtau3['controls']
            w_b = self._w_b['controls']

            # Update self._l, self._dl_dg, self._d2l_dg2, and self._d3l_dg3 in place.
            _compute_dl_dg(stau, taus_seg, l, dl_dg, d2l_dg2, d3l_dg3)

            # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
            dl_dstau[...] = np.sum(dl_dg, axis=-1, keepdims=True)
            d2l_dstau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)

            if self._compute_derivs:
                d3l_dstau3[...] = np.sum(np.sum(np.sum(d3l_dg3, axis=-1), axis=-1), axis=-1, keepdims=True)

            L_id = self._L_id['controls']
            L_seg = L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                         input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            ptau = inputs['ptau']
            t_duration = inputs['t_duration']
            dptau_dt = 2.0 / t_duration

            for control_name, options in enumerate(self._control_options):
                if options['control_type'] == 'full':
                    input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

                    # Translate the input nodes to the discretization nodes.
                    u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])

                    # Perform a row_wise multiplication of w_b and u_hat
                    wbuhat = np.einsum("ij,i...->i...", w_b, u_hat)

                    # outputs[output_name] = np.einsum('i...,i...->...', l, wbuhat)
                    partials[output_name, 'stau'] = dl_dstau.T @ wbuhat

                    partials[rate_name, 'stau'] = d2l_dstau2.T @ wbuhat * dstau_dt
                    partials[rate_name, 'dstau_dt'] = partials[output_name, 'stau']

                    if self._compute_derivs:
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

                    if self._compute_derivs:
                        partials[rate2_name, input_name] = 0.0
                        partials[rate2_name, input_name][..., input_node_idxs] = \
                            (d2l_dstau2 * w_b * dstau_dt ** 2).T @ L_seg

                else:
                    gd = self._grid_data

                    taus_seg = self._taus_seg[control_name]

                    # Retrieve the storage vectors that pertain to the collocated controls
                    l = self._l[control_name]  # noqa: E741, allow ambiguous variable name 'l'
                    dl_dg = self._dl_dg[control_name]
                    d2l_dg2 = self._d2l_dg2[control_name]
                    d3l_dg3 = self._d3l_dg3[control_name]
                    dl_dstau = self._dl_dtau[control_name]
                    d2l_dstau2 = self._d2l_dtau2[control_name]
                    d3l_dstau3 = self._d3l_dtau3[control_name]
                    w_b = self._w_b[control_name]

                    _compute_dl_dg(ptau, taus_seg, l, dl_dg, d2l_dg2, d3l_dg3)

                    # Equivalent of multiplying dl_dg @ dg_dtau, where dg_dtau is a column vector of n ones.
                    dl_dstau[...] = np.sum(dl_dg, axis=-1, keepdims=True)
                    d2l_dstau2[...] = np.sum(np.sum(d2l_dg2, axis=-1), axis=-1, keepdims=True)

                    input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

                    # Translate the input nodes to the discretization nodes.
                    u_hat = inputs[input_name]

                    # Perform a row_wise multiplication of w_b and u_hat
                    wbuhat = np.einsum("ij,i...->i...", w_b, u_hat)

                    # outputs[output_name] = np.einsum('i...,i...->...', l, wbuhat)

                    d_dptau_dt_d_t_duration = -2.0 / t_duration ** 2

                    partials[output_name, 'ptau'] = dl_dstau.T @ wbuhat

                    partials[rate_name, 'ptau'] = d2l_dstau2.T @ wbuhat * dptau_dt
                    partials[rate_name, 't_duration'] = partials[output_name, 'ptau'] * d_dptau_dt_d_t_duration

                    # Assign only thos jacobian columns due to the current segment, since
                    # other segments cannot impact interpolation in this one.
                    # partials[output_name, input_name] = 0.0
                    partials[output_name, input_name][...] = ((l * w_b.T).T).T

                    # partials[rate_name, input_name] = 0.0
                    partials[rate_name, input_name][...] = (dl_dstau * w_b * dptau_dt).T

                    # partials[rate2_name, input_name] = 0.0
                if self._compute_derivs:
                    d3l_dstau3[...] = np.sum(np.sum(np.sum(d3l_dg3, axis=-1), axis=-1), axis=-1, keepdims=True)
                    partials[rate2_name, 'ptau'] = d3l_dstau3.T @ wbuhat * dptau_dt ** 2
                    partials[rate2_name, 't_duration'] = 2 * wbuhat.T @ d2l_dstau2 * dptau_dt * d_dptau_dt_d_t_duration
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
