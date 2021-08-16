import numpy as np
import openmdao.api as om

from ...utils.lgl import lgl


class ControlInterpolationComp(om.ExplicitComponent):
    """
    A segmented component which divides the interpolated region into some number of segments
    num_seg.

    OpenMDAO assumes sizes of variables at setup time, and we don't want to need to change the
    size of the control input nodes when we evaluate different segments. Instead, this component
    will take in the control values of all segments and internally use the appropriate one.

    """
    def __init__(self, grid_data, control_options=None, polynomial_control_options=None,
                 time_units=None, **kwargs):
        self._grid_data = grid_data
        self._control_options = {} if control_options is None else control_options
        self._polynomial_control_options = {} if polynomial_control_options is None else polynomial_control_options
        self._time_units = time_units

        # Storage for the vandermonde matrix and its inverse for each segment
        self._V_u = None
        self._V_u_inv = None

        # Storage for the vandermonde matrix and its inverse for each polynomial control
        self._V_pc = {}
        self._V_pc_inv = {}

        # Cache formatted strings: { control_name : (input_name, output_name) }
        self._control_io_names = {}

        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare('segment_index', types=int, default=0)

    def _setup_controls(self):
        gd = self._grid_data

        self._V_u = {}
        self._V_u_inv = {}
        self._u_node_idxs_by_segment = []
        self._u_exponents = {}

        if not self._control_options:
            return

        for seg_idx in range(gd.num_segments):
            i1, i2 = gd.subset_segment_indices['control_disc'][seg_idx, :]
            control_disc_idxs_in_seg = gd.subset_node_indices['control_disc'][i1:i2]
            self._u_node_idxs_by_segment.append(control_disc_idxs_in_seg)
            tau_seg = gd.node_stau[control_disc_idxs_in_seg]
            seg_order = gd.transcription_order[seg_idx] - 1
            if seg_order not in self._V_u:
                self._V_u[seg_order] = np.vander(tau_seg)
                self._V_u_inv[seg_order] = np.linalg.inv(self._V_u[seg_order])
                self._u_exponents[seg_order] = np.arange(seg_order + 1, dtype=int)[::-1]

        num_uhat_nodes = gd.subset_num_nodes['control_disc']
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

    def _setup_polynomial_controls(self):
        for pc_name, options in self._polynomial_control_options.items():
            order = options['order']
            shape = options['shape']
            units = options['units']
            input_name = f'polynomial_controls:{pc_name}'
            output_name = f'polynomial_control_values:{pc_name}'
            self.add_input(input_name, shape=shape, units=units)
            self.add_output(output_name, shape=shape, units=units)
            self._control_io_names[pc_name] = (input_name, output_name)
            self.declare_partials(of=output_name, wrt=input_name, val=1.0)
            self.declare_partials(of=output_name, wrt='ptau', val=1.0)

            if order not in self._V_pc:
                tau_phase, _ = lgl(order + 1)
                self._V_pc[order] = np.vander(tau_phase)
                self._V_pc_inv[order] = np.linalg.inv(self._V_pc[order])
                self._u_exponents[order] = np.arange(order + 1, dtype=int)[::-1]

    def setup(self):
        self._V_u = []
        self._V_u_inv = []

        self._V_pc = {}
        self._V_pc_inv = {}

        self.add_discrete_input('segment_index', val=0, desc='index of the segment')
        self.add_input('stau', val=0.0, units=None)
        self.add_input('ptau', val=0.0, units=None)

        self._setup_controls()
        self._setup_polynomial_controls()

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        seg_idx = int(discrete_inputs['segment_index'])
        stau = inputs['stau']
        ptau = inputs['ptau']

        seg_order = self._grid_data.transcription_order[seg_idx] - 1
        exponents = self._u_exponents[seg_order]
        stau_array = np.power(stau, exponents)

        for control_name, options in self._control_options.items():
            input_name, output_name = self._control_io_names[control_name]
            u_hat = inputs[input_name][self._u_node_idxs_by_segment[seg_idx]]
            a = self._V_u_inv[seg_order] @ u_hat
            outputs[output_name] = stau_array @ a
            # 4 Different wats to achieve matrix multiplication
            # outputs[output_name] = np.sum(a * stau_array[..., np.newaxis])
            # outputs[output_name] = np.einsum('ij,i', a, stau_array)
            # outputs[output_name] = np.matmul(stau_array, a)

        for pc_name, options in self._polynomial_control_options.items():
            input_name, output_name = self._control_io_names[pc_name]
            order = options['order']
            ptau_array = np.power(ptau, range(order))
            a = self._V_pc_inv[order] @ inputs[input_name]
            outputs[output_name] = ptau_array @ a

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        seg_idx = int(discrete_inputs['segment_index'])
        u_idxs = self._u_node_idxs_by_segment[seg_idx]
        stau = inputs['stau']
        ptau = inputs['ptau']

        seg_order = self._grid_data.transcription_order[seg_idx] - 1
        exponents = self._u_exponents[seg_order]
        dir_exponents = (exponents-1)[:len(exponents)-1]
        stau_array = np.power(stau, exponents)
        dstau_array_dstau = np.atleast_2d(np.power(stau, dir_exponents))

        pu_pa = stau_array

        for control_name, options in self._control_options.items():
            input_name, output_name = self._control_io_names[control_name]

            da_duhat = self._V_u_inv[seg_order]

            partials[output_name, input_name] = 0.0
            partials[output_name, input_name][..., u_idxs] = pu_pa @ da_duhat

            u_hat = inputs[input_name][self._u_node_idxs_by_segment[seg_idx]]
            a = self._V_u_inv[seg_order] @ u_hat

            partials[output_name, 'stau'] = exponents[:-1] * dstau_array_dstau @ a[:len(exponents)-1]

        for pc_name, options in self._polynomial_control_options.items():
            order = options['order']
            ptau_array = np.power(ptau, range(order))
            a = self._V_pc_inv[order] @ inputs[f'polynomial_controls:{pc_name}']
            outputs[f'polynomial_control_values:{pc_name}'] = ptau_array @ a

