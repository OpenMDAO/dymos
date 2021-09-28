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

        # Storage for the Vandermonde matrix and its inverse for each segment
        self._V_pc = None
        self._V_pc_inv = None

        # Storage for the Vandermonde matrix and its inverse for each polynomial control
        self._V_pc = {}
        self._V_pc_inv = {}

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

    def _configure_controls(self):
        gd = self._grid_data

        self._V_pc = {}
        self._V_pc_inv = {}
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
            if seg_control_order not in self._V_pc:
                self._V_pc[seg_control_order] = np.vander(control_disc_seg_stau, increasing=True)
                self._V_pc_inv[seg_control_order] = np.linalg.inv(self._V_pc[seg_control_order])
                self._k[seg_control_order] = np.atleast_2d(np.arange(seg_control_order + 1, dtype=int)).T
                self._k1[seg_control_order] = self._k[seg_control_order] + 1
                self._k2[seg_control_order] = self._k1[seg_control_order] * (self._k[seg_control_order] + 2)
                self._k3[seg_control_order] = self._k2[seg_control_order] * (self._k[seg_control_order] + 3)

        num_uhat_nodes = gd.subset_num_nodes['control_input']
        for control_name, options in self._control_options.items():
            shape = options['shape']
            units = options['units']
            input_name = f'controls:{control_name}'
            output_name = f'control_values:{control_name}'
            rate_name = f'control_rates:{control_name}_rate'
            rate2_name = f'control_rates:{control_name}_rate2'
            uhat_shape = (num_uhat_nodes,) + shape
            self.add_input(input_name, shape=uhat_shape, units=units)
            self.add_output(output_name, shape=shape, units=units)
            self.add_output(rate_name, shape=shape, units=units)
            self.add_output(rate2_name, shape=shape, units=units)
            self._control_io_names[control_name] = (input_name, output_name, rate_name, rate2_name)
            self.declare_partials(of=output_name, wrt=input_name, val=1.0)
            self.declare_partials(of=output_name, wrt='stau', val=1.0)
            self.declare_partials(of=rate_name, wrt=input_name, val=1.0)
            self.declare_partials(of=rate_name, wrt='stau', val=1.0)
            self.declare_partials(of=rate_name, wrt='dstau_dt', val=1.0)
            self.declare_partials(of=rate2_name, wrt=input_name, val=1.0)
            self.declare_partials(of=rate2_name, wrt='stau', val=1.0)
            self.declare_partials(of=rate2_name, wrt='dstau_dt', val=1.0)

    def _configure_polynomial_controls(self):
        for pc_name, options in self._polynomial_control_options.items():
            order = options['order']
            shape = options['shape']
            units = options['units']
            input_name = f'polynomial_controls:{pc_name}'
            output_name = f'polynomial_control_values:{pc_name}'
            rate_name = f'polynomial_control_rates:{pc_name}_rate'
            rate2_name = f'polynomial_control_rates:{pc_name}_rate2'
            input_shape = (order + 1,) + shape
            self.add_input(input_name, shape=input_shape, units=units)
            self.add_output(output_name, shape=shape, units=units)
            self.add_output(rate_name, shape=shape, units=units)
            self.add_output(rate2_name, shape=shape, units=units)
            self._control_io_names[pc_name] = (input_name, output_name, rate_name, rate2_name)
            self.declare_partials(of=output_name, wrt=input_name, val=1.0)
            self.declare_partials(of=output_name, wrt='ptau', val=1.0)
            self.declare_partials(of=rate_name, wrt=input_name, val=1.0)
            self.declare_partials(of=rate_name, wrt='ptau', val=1.0)
            self.declare_partials(of=rate_name, wrt='t_duration', val=1.0)
            self.declare_partials(of=rate2_name, wrt=input_name, val=1.0)
            self.declare_partials(of=rate2_name, wrt='ptau', val=1.0)
            self.declare_partials(of=rate2_name, wrt='t_duration', val=1.0)

            if order not in self._V_pc:
                pc_disc_seg_ptau, _ = lgl(order + 1)
                self._V_pc[order] = np.vander(pc_disc_seg_ptau, increasing=True)
                self._V_pc_inv[order] = np.linalg.inv(self._V_pc[order])
                self._k[order] = np.atleast_2d(np.arange(order + 1, dtype=int)).T
                self._k1[order] = self._k[order] + 1
                self._k2[order] = self._k1[order] * (self._k[order] + 2)
                self._k3[order] = self._k2[order] * (self._k[order] + 3)

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
        self._V_pc = []
        self._V_pc_inv = []

        self._V_pc = {}
        self._V_pc_inv = {}

        self._k = {}
        self._k1 = {}
        self._k2 = {}
        self._k3 = {}

        # self.add_discrete_input('segment_index', val=0, desc='index of the segment')
        self.add_input('stau', val=0.0, units=None)
        self.add_input('dstau_dt', val=1.0, units=f'1/{self._time_units}')
        self.add_input('t_duration', val=1.0, units=self._time_units)
        self.add_input('ptau', val=0.0, units=None)

        self._configure_controls()
        self._configure_polynomial_controls()

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
        stau = inputs['stau']
        dstau_dt = inputs['dstau_dt']
        ptau = inputs['ptau']
        dptau_dt = 2 / inputs['t_duration']

        if self._control_options:
            seg_order = self._grid_data.transcription_order[seg_idx] - 1
            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
            k = self._k[seg_order]
            k1 = self._k1[seg_order]
            k2 = self._k2[seg_order]
            stau_vec = np.power(stau, k)

            L_seg = self._L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                               input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            for control_name, options in self._control_options.items():
                input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]
                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])
                a = np.atleast_2d(self._V_pc_inv[seg_order] @ u_hat)
                outputs[output_name] = a.T @ stau_vec
                outputs[rate_name] = dstau_dt * (a.T[..., 1:] @ (k1[:-1] * stau_vec[:-1]))
                outputs[rate2_name] = dstau_dt**2 * (a.T[..., 2:] @ (k2[:-2] * stau_vec[:-2]))

        for pc_name, options in self._polynomial_control_options.items():
            input_name, output_name, rate_name, rate2_name = self._control_io_names[pc_name]
            order = options['order']
            k = self._k[order]
            k1 = self._k1[order]
            k2 = self._k2[order]
            ptau_vec = np.power(ptau, k)
            a = np.atleast_2d(self._V_pc_inv[order] @ inputs[input_name])
            outputs[output_name] = a.T @ ptau_vec
            outputs[rate_name] = dptau_dt * (a.T[..., 1:] @ (k1[:-1] * ptau_vec[:-1]))
            outputs[rate2_name] = dptau_dt**2 * (a.T[..., 2:] @ (k2[:-2] * ptau_vec[:-2]))

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
        stau = inputs['stau'].real
        dstau_dt = inputs['dstau_dt'].real
        ptau = inputs['ptau'].real
        t_duration = inputs['t_duration'].real
        dptau_dt = 2.0 / t_duration

        if self._control_options:
            u_idxs = self._input_node_idxs_by_segment[seg_idx]
            seg_order = self._grid_data.transcription_order[seg_idx] - 1

            k = self._k[seg_order]
            k1 = self._k1[seg_order]
            k2 = self._k2[seg_order]
            k3 = self._k3[seg_order]
            stau_vec = np.power(stau, k)

            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]

            L_seg = self._L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                               input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            for control_name, options in self._control_options.items():
                input_name, output_name, rate_name, rate2_name = self._control_io_names[control_name]

                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs].real)
                a = self._V_pc_inv[seg_order] @ u_hat

                da_duhat = self._V_pc_inv[seg_order] @ L_seg

                partials[output_name, input_name][...] = 0.0
                partials[output_name, input_name][..., u_idxs] = stau_vec.T @ da_duhat
                partials[output_name, 'stau'] = a.T[..., 1:] @ (k1[:-1] * stau_vec[:-1])

                pudot_pa = (k1[:-1] * stau_vec[:-1])
                partials[rate_name, input_name][...] = 0.0
                partials[rate_name, input_name][..., u_idxs] = dstau_dt * (pudot_pa.T @ self._V_pc_inv[seg_order][1:, :])
                partials[rate_name, 'dstau_dt'][...] = partials[output_name, 'stau']
                partials[rate_name, 'stau'][...] = dstau_dt * (a.T[..., 2:] @ (k2[:-2] * stau_vec[:-2]))

                pudotdot_pa = (k2[:-2] * stau_vec[:-2])
                partials[rate2_name, input_name][...] = 0.0
                partials[rate2_name, input_name][..., u_idxs] = dstau_dt**2 * (pudotdot_pa.T @ self._V_pc_inv[seg_order][2:, :])
                partials[rate2_name, 'stau'][...] = dstau_dt**2 * (a.T[..., 3:] @ (k3[:-3] * stau_vec[:-3]))
                partials[rate2_name, 'dstau_dt'][...] = 2 * dstau_dt * (a.T[..., 2:] @ (k2[:-2] * stau_vec[:-2]))

        for pc_name, options in self._polynomial_control_options.items():
            input_name, output_name, rate_name, rate2_name = self._control_io_names[pc_name]
            order = options['order']

            k = self._k[order]
            k1 = self._k1[order]
            k2 = self._k2[order]
            k3 = self._k3[order]
            ptau_vec = np.power(ptau, k)

            u_hat = inputs[input_name].real
            a = self._V_pc_inv[order] @ u_hat

            da_duhat = self._V_pc_inv[order]

            partials[output_name, input_name][...] = ptau_vec.T.real @ da_duhat.real
            partials[output_name, 'ptau'][...] = a.T[..., 1:] @ (k1[:-1] * ptau_vec[:-1])

            pudot_pa = (k1[:-1] * ptau_vec[:-1])
            partials[rate_name, input_name][...] = dptau_dt * (pudot_pa.T @ self._V_pc_inv[order][1:, :])
            partials[rate_name, 'ptau'][...] = dptau_dt * a.T[..., 2:] @ (k2[:-2] * ptau_vec[:-2])
            partials[rate_name, 't_duration'][...] = -2 * (a.T[..., 1:] @ (k1[:-1] * ptau_vec[:-1])) / t_duration**2

            pudotdot_pa = (k2[:-2] * ptau_vec[:-2])
            partials[rate2_name, input_name][...] = dptau_dt**2 * (pudotdot_pa.T @ self._V_pc_inv[order][2:, :])
            partials[rate2_name, 'ptau'] = dptau_dt**2 * a.T[..., 3:] @ (k3[:-3] * ptau_vec[:-3])
            partials[rate2_name, 't_duration'] = -8 * (a.T[..., 2:] @ (k2[:-2] * ptau_vec[:-2])) / t_duration**3
