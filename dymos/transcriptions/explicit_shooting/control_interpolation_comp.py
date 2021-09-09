import numpy as np
import openmdao.api as om

from ...phase.options import ControlOptionsDictionary, PolynomialControlOptionsDictionary
from ...transcriptions.grid_data import GridData
from ...utils.lgl import lgl


class ControlInterpolationComp(om.ExplicitComponent):
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
        self._V_u = None
        self._V_u_inv = None

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
        pass

    def _configure_controls(self):
        gd = self._grid_data

        self._V_u = {}
        self._V_u_inv = {}
        self._disc_node_idxs_by_segment = []
        self._input_node_idxs_by_segment = []
        self._u_exponents = {}

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
            if seg_control_order not in self._V_u:
                self._V_u[seg_control_order] = np.vander(control_disc_seg_stau)
                self._V_u_inv[seg_control_order] = np.linalg.inv(self._V_u[seg_control_order])
                self._u_exponents[seg_control_order] = np.arange(seg_control_order + 1, dtype=int)[::-1]

        num_uhat_nodes = gd.subset_num_nodes['control_input']
        for control_name, options in self._control_options.items():
            shape = options['shape']
            units = options['units']
            input_name = f'controls:{control_name}'
            output_name = f'control_values:{control_name}'
            uhat_shape = (num_uhat_nodes,) + shape
            self.add_input(input_name, shape=uhat_shape, units=units)
            self.add_output(output_name, shape=shape, units=units)
            self._control_io_names[control_name] = (input_name, output_name)
            self.declare_partials(of=output_name, wrt=input_name, val=1.0)
            self.declare_partials(of=output_name, wrt='stau', val=1.0)

    def _configure_polynomial_controls(self):
        for pc_name, options in self._polynomial_control_options.items():
            order = options['order']
            shape = options['shape']
            units = options['units']
            input_name = f'polynomial_controls:{pc_name}'
            output_name = f'polynomial_control_values:{pc_name}'
            input_shape = (order + 1,) + shape
            self.add_input(input_name, shape=input_shape, units=units)
            self.add_output(output_name, shape=shape, units=units)
            self._control_io_names[pc_name] = (input_name, output_name)
            self.declare_partials(of=output_name, wrt=input_name, val=1.0)
            self.declare_partials(of=output_name, wrt='ptau', val=1.0)

            if order not in self._V_pc:
                pc_disc_seg_ptau, _ = lgl(order + 1)
                self._V_pc[order] = np.vander(pc_disc_seg_ptau)
                self._V_pc_inv[order] = np.linalg.inv(self._V_pc[order])
                self._u_exponents[order] = np.arange(order + 1, dtype=int)[::-1]

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
        self._V_u = []
        self._V_u_inv = []

        self._V_pc = {}
        self._V_pc_inv = {}

        self.add_discrete_input('segment_index', val=0, desc='index of the segment')
        self.add_input('stau', val=0.0, units=None)
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
        seg_idx = int(discrete_inputs['segment_index'])
        stau = inputs['stau']
        ptau = inputs['ptau']

        if self._control_options:
            seg_order = self._grid_data.transcription_order[seg_idx] - 1
            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]
            exponents = self._u_exponents[seg_order]
            stau_array = np.power(stau, exponents)

            L_seg = self._L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                               input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            for control_name, options in self._control_options.items():
                input_name, output_name = self._control_io_names[control_name]
                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])
                a = self._V_u_inv[seg_order] @ u_hat
                outputs[output_name] = stau_array @ a

        for pc_name, options in self._polynomial_control_options.items():
            input_name, output_name = self._control_io_names[pc_name]
            order = options['order']
            exponents = self._u_exponents[order]
            ptau_array = np.power(ptau, exponents)
            a = self._V_pc_inv[order] @ inputs[input_name]
            outputs[output_name] = ptau_array @ a

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
        seg_idx = int(discrete_inputs['segment_index'])
        stau = inputs['stau']
        ptau = inputs['ptau']

        if self._control_options:
            u_idxs = self._input_node_idxs_by_segment[seg_idx]
            seg_order = self._grid_data.transcription_order[seg_idx] - 1
            exponents = self._u_exponents[seg_order]
            dir_exponents = (exponents-1)[:len(exponents)-1]
            stau_array = np.power(stau, exponents)
            dstau_array_dstau = np.atleast_2d(np.power(stau, dir_exponents))
            pu_pa = stau_array
            disc_node_idxs = self._disc_node_idxs_by_segment[seg_idx]
            input_node_idxs = self._input_node_idxs_by_segment[seg_idx]

            L_seg = self._L_id[disc_node_idxs[0]:disc_node_idxs[0] + len(disc_node_idxs),
                               input_node_idxs[0]:input_node_idxs[0] + len(input_node_idxs)]

            for control_name, options in self._control_options.items():
                input_name, output_name = self._control_io_names[control_name]

                da_duhat = self._V_u_inv[seg_order] @ L_seg

                partials[output_name, input_name][...] = 0.0
                partials[output_name, input_name][..., u_idxs] = pu_pa.real @ da_duhat.real

                u_hat = np.dot(L_seg, inputs[input_name][input_node_idxs])
                a = self._V_u_inv[seg_order] @ u_hat

                partials[output_name, 'stau'] = exponents[:-1] * dstau_array_dstau @ a[:len(exponents)-1]

        for pc_name, options in self._polynomial_control_options.items():
            input_name, output_name = self._control_io_names[pc_name]
            order = options['order']
            exponents = self._u_exponents[order]
            dir_exponents = (exponents-1)[:len(exponents)-1]

            ptau_array = np.power(ptau, exponents)
            pu_pa = ptau_array
            da_duhat = self._V_pc_inv[order]

            partials[output_name, input_name][...] = 0.0
            partials[output_name, input_name][...] = pu_pa @ da_duhat

            u_hat = inputs[input_name]
            a = self._V_pc_inv[order] @ u_hat

            dptau_array_dptau = np.atleast_2d(np.power(ptau, dir_exponents))

            partials[output_name, 'ptau'] = exponents[:-1] * dptau_array_dptau @ a[:len(exponents)-1]
