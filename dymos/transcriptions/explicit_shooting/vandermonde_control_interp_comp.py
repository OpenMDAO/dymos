import numpy as np
import openmdao.api as om

from ...transcriptions.grid_data import GridData
from ...utils.lgl import lgl


class VandermondeControlInterpComp(om.ExplicitComponent):
    """
    A component which interpolates control values in 1D using Vandermonde interpolation.

    Takes training values for control variables at given _input_ nodes,
    broadcaasts them to _discretization_ nodes, and then interpolates the discretization values
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
                 time_units=None, standalone_mode=False, **kwargs):
        self._grid_data = grid_data
        self._control_options = {} if control_options is None else control_options
        self._polynomial_control_options = {} if polynomial_control_options is None else polynomial_control_options
        self._time_units = time_units
        self._standalone_mode = standalone_mode

        # Storage for the Vandermonde matrix and its inverse for each control
        self._V_hat = {}
        self._V_hat_inv = {}

        # Storage for factors used in the derivatives of Vandermonde matrices.
        self._fac = {}

        # Cache formatted strings: { control_name : (input_name, output_name) }
        self._control_io_names = {}

        # The Lagrange interpolation matrix L_id maps control values as given at the input nodes
        # to values at the discretization nodes.
        num_disc_nodes = grid_data.subset_num_nodes['control_disc']
        num_input_nodes = grid_data.subset_num_nodes['control_input']
        self._L_id = np.zeros((num_disc_nodes, num_input_nodes), dtype=float)
        self._L_id[np.arange(num_disc_nodes, dtype=int),
                   self._grid_data.input_maps['dynamic_control_input_to_disc']] = 1.0

        super().__init__(**kwargs)

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('segment_index', types=int, desc='index of the current segment')
        self.options.declare('vec_size', types=int, default=1,
                             desc='number of points at which the control will be evaluated. This is not'
                                  'necessarily the same as the number of nodes in the GridData.')

    def _configure_controls(self):
        vec_size = self.options['vec_size']
        gd = self._grid_data

        self._V_hat = {}
        self._V_hat_inv = {}
        self._disc_node_idxs_by_segment = []
        self._input_node_idxs_by_segment = []

        if not self._control_options:
            return

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

            # Indices of the control disc nodes belonging to the current segment
            control_disc_seg_idxs = gd.subset_segment_indices['control_disc'][seg_idx]

            # Segment tau values for the control disc nodes in the phase
            control_disc_stau = gd.node_stau[gd.subset_node_indices['control_disc']]

            # Segment tau values for the control disc nodes in the given segment
            control_disc_seg_stau = control_disc_stau[control_disc_seg_idxs[0]:
                                                      control_disc_seg_idxs[1]]

            seg_control_order = gd.transcription_order[seg_idx] - 1
            if seg_control_order not in self._V_hat:
                self._V_hat[seg_control_order] = np.vander(control_disc_seg_stau, increasing=True)
                self._V_hat_inv[seg_control_order] = np.linalg.inv(self._V_hat[seg_control_order])
            if seg_control_order + 1 not in self._fac:
                self._fac[seg_control_order + 1] = np.arange(seg_control_order + 1, dtype=int)

        num_uhat_nodes = gd.subset_num_nodes['control_input']
        ar = np.arange(vec_size, dtype=int)
        for control_name, options in self._control_options.items():
            shape = options['shape']
            units = options['units']
            input_name = f'controls:{control_name}'
            output_name = f'control_values:{control_name}'
            rate_name = f'control_rates:{control_name}_rate'
            rate2_name = f'control_rates:{control_name}_rate2'
            uhat_shape = (num_uhat_nodes,) + shape
            output_shape = (vec_size,) + shape
            self.add_input(input_name, shape=uhat_shape, units=units)
            self.add_output(output_name, shape=output_shape, units=units)
            self.add_output(rate_name, shape=output_shape, units=units)
            self.add_output(rate2_name, shape=output_shape, units=units)
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
            input_shape = (order + 1,) + shape
            output_shape = (vec_size,) + shape
            self.add_input(input_name, shape=input_shape, units=units)
            self.add_output(output_name, shape=output_shape, units=units)
            self.add_output(rate_name, shape=output_shape, units=units)
            self.add_output(rate2_name, shape=output_shape, units=units)
            self._control_io_names[pc_name] = (input_name, output_name, rate_name, rate2_name)
            self.declare_partials(of=output_name, wrt=input_name)
            self.declare_partials(of=output_name, wrt='ptau', rows=ar, cols=ar)
            self.declare_partials(of=rate_name, wrt=input_name)
            self.declare_partials(of=rate_name, wrt='ptau', rows=ar, cols=ar)
            self.declare_partials(of=rate_name, wrt='t_duration')
            self.declare_partials(of=rate2_name, wrt=input_name)
            self.declare_partials(of=rate2_name, wrt='ptau', rows=ar, cols=ar)
            self.declare_partials(of=rate2_name, wrt='t_duration')

            if order not in self._V_hat:
                pc_disc_seg_ptau, _ = lgl(order + 1)
                self._V_hat[order] = np.vander(pc_disc_seg_ptau, increasing=True)
                self._V_hat_inv[order] = np.linalg.inv(self._V_hat[order])
            if order + 1 not in self._fac:
                self._fac[order + 1] = np.arange(order + 1, dtype=int)

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

        self._V_hat = {}
        self._V_hat_inv = {}

        # self.add_discrete_input('segment_index', val=0, desc='index of the segment')
        self.add_input('stau', shape=(vec_size,), units=None)
        self.add_input('dstau_dt', val=1.0, units=f'1/{self._time_units}')
        self.add_input('t_duration', val=1.0, units=self._time_units)
        self.add_input('ptau', shape=(vec_size,), units=None)

        self._configure_controls()
        self._configure_polynomial_controls()

    def _dvander(self, v):
        """
        Return the derivative of a Vandermonde matrix wrt the independent variable _in increasing order_.

        Parameters
        ----------
        v : np.array
            The Vandermonde matrix for which the derivatives are requested.

        Returns
        -------
        dV : np.array
            The derivative of v with respect to the independent variable.
        dv2 : np.array
            The second derivative of v wrt the independent variable.
        dv3 : np.array
            The third derivative of v wrt the independent variable.
        """
        p, n = v.shape
        dv = np.zeros_like(v)
        dv2 = dv.copy()
        dv3 = dv.copy()
        dv[:, 1:] = v[:, :-1]
        dv2[:, 2:] = v[:, :-2]
        dv3[:, 3:] = v[:, :-3]

        fac = self._fac[n]
        fac2 = fac[:-1]
        fac3 = fac[:-2]

        dv[:, :] = dv * fac[np.newaxis, :]
        dv2[:, 1:] = dv2[:, 1:] * fac2[np.newaxis, :] * fac[np.newaxis, 1:]
        dv3[:, 2:] = dv3[:, 2:] * fac3[np.newaxis, :] * fac2[np.newaxis, 1:] * fac[np.newaxis, 2:]
        return dv, dv2, dv3

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
        seg_idx = self.options['segment_index']
        n = self._grid_data.transcription_order[seg_idx]
        stau = inputs['stau']
        dstau_dt = inputs['dstau_dt']
        ptau = inputs['ptau']
        dptau_dt = 2 / inputs['t_duration']

        if self._control_options:
            seg_order = n - 1
            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
            V_stau = np.vander(stau, N=n, increasing=True)
            dV_stau, dV2_stau, _ = self._dvander(V_stau)

            L_seg = self._L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                               input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            for control_name, options in self._control_options.items():
                input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]
                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])
                a = np.atleast_2d(self._V_hat_inv[seg_order] @ u_hat)
                outputs[output_name] = V_stau @ a
                outputs[rate_name] = dstau_dt * (dV_stau @ a)
                outputs[rate2_name] = dstau_dt**2 * (dV2_stau @ a)

        for pc_name, options in self._polynomial_control_options.items():
            input_name, output_name, rate_name, rate2_name = self._control_io_names[pc_name]
            order = options['order']
            V_ptau = np.vander(ptau, N=order+1, increasing=True)
            dV_ptau, dV2_ptau, _ = self._dvander(V_ptau)
            a = np.atleast_2d(self._V_hat_inv[order] @ inputs[input_name])
            outputs[output_name] = V_ptau @ a
            outputs[rate_name] = dptau_dt * (dV_ptau @ a)
            outputs[rate2_name] = dptau_dt**2 * (dV2_ptau @ a)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        """
        Compute derivatives interpolated control values and rates wrt the inputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        discrete_inputs : Vector
            Unscaled, discrete input variables keyed by variable name.
        """
        seg_idx = self.options['segment_index']
        n = self._grid_data.transcription_order[seg_idx]
        stau = inputs['stau'].real
        dstau_dt = inputs['dstau_dt'].real
        ptau = inputs['ptau'].real
        t_duration = inputs['t_duration'].real
        dptau_dt = 2.0 / t_duration
        ddptau_dt_dtduration = -2.0 / t_duration**2

        if self._control_options:
            u_idxs = self._input_node_idxs_by_segment[seg_idx]
            seg_order = self._grid_data.transcription_order[seg_idx] - 1

            V_stau = np.vander(stau, N=n, increasing=True)
            dV_stau, dV2_stau, dV3_stau = self._dvander(V_stau)

            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]

            L_seg = self._L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                               input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            for control_name, options in self._control_options.items():
                input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs].real)
                a = self._V_hat_inv[seg_order] @ u_hat

                da_duhat = self._V_hat_inv[seg_order] @ L_seg
                dV_a = dV_stau @ a
                dV2_a = dV2_stau @ a
                dV3_a = dV3_stau @ a

                partials[output_name, input_name][...] = 0.0
                partials[output_name, input_name][..., u_idxs] = V_stau @ da_duhat
                partials[output_name, 'stau'] = dV_a.ravel()

                pudot_pa = dstau_dt * dV_stau
                pa_puhat = self._V_hat_inv[seg_order]
                partials[rate_name, input_name][...] = 0.0
                partials[rate_name, input_name][..., u_idxs] = pudot_pa @ pa_puhat
                partials[rate_name, 'dstau_dt'][...] = dV_a
                partials[rate_name, 'stau'][...] = dV2_a.ravel()

                pu2dot_pa = dstau_dt**2 * dV2_stau
                partials[rate2_name, input_name][...] = 0.0
                partials[rate2_name, input_name][..., u_idxs] = pu2dot_pa @ pa_puhat
                partials[rate2_name, 'dstau_dt'][...] = 2 * dstau_dt * dV2_a
                partials[rate2_name, 'stau'][...] = dV3_a.ravel()

        for pc_name, options in self._polynomial_control_options.items():
            input_name, output_name, rate_name, rate2_name = self._control_io_names[pc_name]
            order = options['order']

            V_ptau = np.vander(ptau, N=order+1, increasing=True)
            dV_ptau, dV2_ptau, dV3_ptau = self._dvander(V_ptau)

            u_hat = inputs[input_name].real
            a = self._V_hat_inv[order] @ u_hat

            dV_a = dV_ptau @ a
            dV2_a = dV2_ptau @ a
            dV3_a = dV3_ptau @ a

            da_duhat = self._V_hat_inv[order]

            partials[output_name, input_name][...] = V_ptau @ da_duhat
            partials[output_name, 'ptau'][...] = dV_a.ravel()

            pudot_pa = dptau_dt * dV_ptau
            pa_puhat = self._V_hat_inv[order]
            partials[rate_name, input_name][...] = pudot_pa @ pa_puhat
            partials[rate_name, 't_duration'][...] = ddptau_dt_dtduration * dV_a
            partials[rate_name, 'ptau'][...] = dptau_dt * dV2_a.ravel()

            pu2dot_pa = dptau_dt ** 2 * dV2_ptau
            partials[rate2_name, input_name][...] = pu2dot_pa @ pa_puhat
            partials[rate2_name, 't_duration'][...] = 2 * dptau_dt * ddptau_dt_dtduration * dV2_a
            partials[rate2_name, 'ptau'][...] = dptau_dt**2 * dV3_a.ravel()
