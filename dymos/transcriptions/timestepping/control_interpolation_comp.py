import numpy as np
import openmdao.api as om

from ...utils.lgl import lgl


class ControlInterpolationComp(om.ExplicitComponent):
    """
    A segmented component which divides the interpolated region into some number of segments
    num_seg.
    In each of these segments (i) the interpolant is provided for some number of nodes (n_segi)
    """
    def __init__(self, grid_data, control_options=None, polynomial_control_options=None,
                 time_units=None, **kwargs):
        self._grid_data = grid_data
        self._control_options = {} if control_options is None else control_options
        self._polynomial_control_options = {} if polynomial_control_options is None else polynomial_control_options
        self._time_units = time_units

        self._max_seg_order = max([grid_data.transcription_order[seg_idx] - 1
                                   for seg_idx in range(grid_data.num_segments)])

        # Storage for the vandermonde matrix and its inverse for each segment
        self._v = {}
        self._v_inv = {}

        # Storage for the vandermonde matrix and its inverse for each polynomial control
        self._vpc = {}
        self._vpc_inv = {}

        # Cache formatted strings: { control_name : (input_name, output_name) }
        self._control_io_names = {}

        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare('segment_index', types=int, default=0)

    def setup(self):
        gd = self._grid_data

        if self._control_options:
            for seg_idx in range(gd.num_segments):
                i1, i2 = gd.subset_segment_indices['control_input'][seg_idx, :]
                control_disc_idxs_in_seg = gd.subset_node_indices['control_input'][i1:i2]
                tau_seg = gd.node_stau[control_disc_idxs_in_seg]
                order = gd.transcription_order[seg_idx] - 1
                if order not in self._v:
                    self._v[order] = np.vander(tau_seg)
                    self._v_inv[order] = np.linalg.inv(self._v[order])

        for pc, options in self._polynomial_control_options.items():
            order = options['order']
            if order not in self._vpc:
                tau_phase, _ = lgl(order + 1)
                self._vpc[order] = np.vander(tau_phase)
                self._vpc_inv[order] = np.linalg.inv(self._vpc[order])

        self.add_discrete_input('segment_index', val=0, desc='index of the segment')
        self.add_input('stau', val=0.0, units=None)
        self.add_input('ptau', val=0.0, units=None)

        for control_name, options in self._control_options.items():
            shape = options['shape']
            units = options['units']
            uhat_shape = (self._max_seg_order + 1,) + shape
            input_name = f'controls:{control_name}'
            output_name = f'control_values:{control_name}'
            self.add_input(input_name, shape=uhat_shape, units=units)
            self.add_output(output_name, shape=shape, units=units)
            self._control_io_names[control_name] = (input_name, output_name)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        seg_idx = discrete_inputs['segment_index']
        seg_order = self._grid_data.transcription_order[seg_idx] - 1

        stau = inputs['stau']
        stau_array = np.power(stau, range(self._max_seg_order + 1)[::-1])


        for control_name, options in self._control_options.items():
            input_name, output_name = self._control_io_names[control_name]
            u_hat = inputs[input_name]
            a = self._v_inv[seg_order] @ u_hat
            # 4 Different wats to achieve matrix multiplication
            # outputs[output_name] = np.sum(a * stau_array[..., np.newaxis])
            # outputs[output_name] = np.einsum('ij,i', a, stau_array)
            # outputs[output_name] = np.matmul(stau_array, a)
            outputs[output_name] = stau_array @ a
            print(control_name, u_hat, outputs[output_name])

        for pc_name, options in self._polynomial_control_options.items():
            order = options['order']
            ptau_array = np.power(ptau, range(order))
            a = self._v_inv[order] @ inputs[f'polynomial_controls:{pc_name}']
            outputs[f'polynomial_control_values:{pc_name}'] = ptau_array @ a

    def compute_partials(self, inputs, partials, discrete_inputs=None):

        seg_idx = discrete_inputs['segment_index']
        tau = inputs['tau']

        self._tau_array[...] = np.power(tau, range(self._max_order+1))[::-1]

        if seg_idx >= 0:
            order = self._grid_data.transcription_order[seg_idx]

        for control_name, options in self._control_options.items():
            a = self._v_inv[order] @ inputs[f'controls:{control_name}']

            # Power series of tau in decreasing order
            tau_array = self._tau_array[:order + 1]

            da_duhat = self._v_inv[order]
            pu_pa = self._tau_array[::-1][:order + 1].reshape((order+1, 1))

            # du/d_uhat
            partials[f'control_values:{control_name}',
                     f'controls:{control_name}'] = pu_pa @ da_duhat

            # du/d_tau
            partials[f'control_values:{control_name}', 'tau'] = np.dot(a, self._tau_array[:order+1])

        for pc_name, options in self._polynomial_control_options.items():
            order = options['order']
            a = self._v_inv[order] @ inputs[f'polynomial_controls:{pc_name}']
            partials[f'polynomial_control_values:{pc_name}'] = np.dot(a, self._tau_array[:order+1])





    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pass


