import numpy as np
import openmdao.api as om
from ...grid_data import GridData
from ....options import options as dymos_options


class ControlEndpointDefectComp(om.ExplicitComponent):
    r""" Compute/enforce the control endpoint defect when using the Radau Pseudospectral method.

    For each dynamic control, take the control values at all nodes.  Use a Radau interpolation
    matrix to compute the terminal value of the control, and compute the difference between this
    interpolated terminal value and the given terminal value.  Apply a constraint such that this
    value is zero.

    .. math:: {u}_{f} = \left[ L_{LGR} \right] vec{u}_{col}

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):

        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'control_options', types=dict,
            desc='Dictionary of control names/options for the phase')

    def setup(self):
        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['all']

        control_options = self.options['control_options']

        # We only need the last row of the Lagrange interpolation matrix since we
        # desire the interpolated control value at the endpoint of the phase.
        L, _ = gd.phase_lagrange_matrices('col', 'all')
        self._num_disc_end_segment = gd.subset_num_nodes_per_segment['col'][-1]
        self._L = np.atleast_2d(L[-1, -self._num_disc_end_segment:])

        self._input_str = {}
        self._output_str = {}

        for control_name, options in control_options.items():
            if not (options['dynamic'] and options['opt']):
                continue

            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            self._input_str[control_name] = 'controls:{0}'.format(control_name)
            self._output_str[control_name] = 'control_endpoint_defects:{0}'.format(control_name)

            self.add_input(name=self._input_str[control_name],
                           shape=(num_nodes,) + shape,
                           desc='Values of control {0} at all nodes'.format(control_name),
                           units=units)

            self.add_output(
                name=self._output_str[control_name],
                shape=(1,) + shape,
                units=units,
                desc='Interpolated value of {0} at endpoint'.format(control_name))

            self.add_constraint(name=self._output_str[control_name],
                                equals=0.0,
                                scaler=1.0,
                                linear=False)

            row_idxs = np.repeat(np.arange(size, dtype=int),
                                 gd.subset_num_nodes_per_segment['all'][-1])

            initial_col = sum([gd.subset_num_nodes_per_segment['all'][i] * size
                               for i in range(gd.num_segments - 1)])
            final_col = sum([gd.subset_num_nodes_per_segment['all'][i] * size
                             for i in range(gd.num_segments)])

            col_idxs = []
            for i in range(size):
                col_idxs.extend(np.arange(initial_col + i, final_col, size, dtype=int))

            dout_din = np.tile(np.concatenate((np.ravel(-self._L), [1.0])), size)

            self.declare_partials(of=self._output_str[control_name],
                                  wrt=self._input_str[control_name],
                                  rows=row_idxs, cols=col_idxs, val=dout_din)

    def compute(self, inputs, outputs):
        gd = self.options['grid_data']

        for name in self._input_str:
            u_col = inputs[self._input_str[name]][gd.subset_node_indices['col']]
            u_col_end_segment = np.atleast_2d(u_col[-self._num_disc_end_segment:])
            np.set_printoptions(linewidth=1024)
            outputs[self._output_str[name]] = inputs[self._input_str[name]][-1, :] - \
                np.tensordot(self._L, u_col_end_segment, axes=(1, 0))
