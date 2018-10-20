from __future__ import print_function, division, absolute_import

from six import iteritems, string_types

import numpy as np

from openmdao.api import ExplicitComponent
from dymos.phases.grid_data import GridData
from dymos.utils.rk_methods import rk_methods
from dymos.utils.lagrange import lagrange_matrices
from dymos.utils.misc import get_rate_units


class StageControlComp(ExplicitComponent):
    """
    Computes the values of the states to pass to the ODE for a given stage
    """

    def initialize(self):
        super(StageControlComp, self).initialize()

        self.options.declare('index', types=int, desc='The index of this segment in the phase.')
        self.options.declare('method', types=str, default='rk4')
        self.options.declare('num_steps', types=(int,))
        self.options.declare('control_options', types=dict, desc='Dictionary of options for '
                                                                 'the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

        # Save the names of the dynamic controls/parameters
        self._dynamic_names = []
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

        # Save the names of the dynamic controls/parameters
        self._dynamic_names = []
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

        # Data structures for storing partial data
        self.val_jacs = {}
        self.rate_jacs = {}
        self.rate2_jacs = {}
        self.val_jac_rows = {}
        self.val_jac_cols = {}
        self.rate_jac_rows = {}
        self.rate_jac_cols = {}
        self.rate2_jac_rows = {}
        self.rate2_jac_cols = {}
        self.sizes = {}

    def _setup_controls(self):
        control_options = self.options['control_options']
        idx = self.options['index']
        gd = self.options['grid_data']
        num_control_disc_nodes = gd.subset_num_nodes_per_segment['control_disc'][idx]
        time_units = self.options['time_units']
        num_steps = self.options['num_steps']
        method = self.options['method']
        num_stages = rk_methods[method]['num_stages']
        num_nodes = num_steps * num_stages

        for name, options in iteritems(control_options):
            self._input_names[name] = 'disc_controls:{0}'.format(name)
            self._output_val_names[name] = 'stage_control_values:{0}'.format(name)
            self._output_rate_names[name] = 'stage_control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'stage_control_rates:{0}_rate2'.format(name)
            shape = options['shape']
            input_shape = (num_control_disc_nodes,) + shape
            output_shape = (num_steps, num_stages) + shape

            units = options['units']
            rate_units = get_rate_units(units, time_units)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self._dynamic_names.append(name)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)

            self.add_output(self._output_val_names[name], shape=output_shape, units=units)

            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)

            self.add_output(self._output_rate2_names[name], shape=output_shape,
                            units=rate2_units)

            size = np.prod(shape)
            self.val_jacs[name] = np.zeros((num_nodes, size, num_control_disc_nodes, size))
            self.rate_jacs[name] = np.zeros((num_nodes, size, num_control_disc_nodes, size))
            self.rate2_jacs[name] = np.zeros((num_nodes, size, num_control_disc_nodes, size))
            for i in range(size):
                self.val_jacs[name][:, i, :, i] = self.L
                self.rate_jacs[name][:, i, :, i] = self.D
                self.rate2_jacs[name][:, i, :, i] = self.D2
            self.val_jacs[name] = self.val_jacs[name].reshape((num_nodes * size,
                                                              num_control_disc_nodes * size),
                                                              order='C')
            self.rate_jacs[name] = self.rate_jacs[name].reshape((num_nodes * size,
                                                                num_control_disc_nodes * size),
                                                                order='C')
            self.rate2_jacs[name] = self.rate2_jacs[name].reshape((num_nodes * size,
                                                                  num_control_disc_nodes * size),
                                                                  order='C')
            self.val_jac_rows[name], self.val_jac_cols[name] = \
                np.where(self.val_jacs[name] != 0)
            self.rate_jac_rows[name], self.rate_jac_cols[name] = \
                np.where(self.rate_jacs[name] != 0)
            self.rate2_jac_rows[name], self.rate2_jac_cols[name] = \
                np.where(self.rate2_jacs[name] != 0)

            self.sizes[name] = size

            rs, cs = self.val_jac_rows[name], self.val_jac_cols[name]
            self.declare_partials(of=self._output_val_names[name],
                                  wrt=self._input_names[name],
                                  rows=rs, cols=cs, val=self.val_jacs[name][rs, cs])

            rs = np.concatenate([np.arange(0, num_nodes * size, size, dtype=int) + i
                                 for i in range(size)])

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=np.zeros_like(rs))

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.rate_jac_rows[name], cols=self.rate_jac_cols[name])

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=np.zeros_like(rs))

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.rate2_jac_rows[name], cols=self.rate2_jac_cols[name])

    def setup(self):
        idx = self.options['index']
        method = self.options['method']
        num_steps = self.options['num_steps']
        gd = self.options['grid_data']
        time_units = self.options['time_units']
        c = rk_methods[method]['c']

        stau_step = np.linspace(-1, 1, num_steps + 1)

        step_stau_span = 2.0 / num_steps

        # For each step, linear transform c (on [0, 1]) onto stau of the step boundaries
        # TODO: Change this to accommodate variable step sizes within the segment
        stau_stage = np.asarray([stau_step[i] + c * step_stau_span for i in range(num_steps)])
        stau_stage_flat = stau_stage.ravel()

        i1, i2 = gd.segment_indices[idx, :]
        stau_disc = gd.node_stau[i1:i2]

        self.L, self.D = lagrange_matrices(stau_disc, stau_stage_flat)
        _, D_dd = lagrange_matrices(stau_disc, stau_disc)
        self.D2 = np.dot(self.D, D_dd)

        self.add_input(name='dt_dstau', val=1.0, units=time_units,
                       desc='Ratio of segment time duration to segment tau duration (2).')

        self._setup_controls()

    def compute(self, inputs, outputs):
        control_options = self.options['control_options']

        for name, options in iteritems(control_options):

            u = inputs[self._input_names[name]]

            a = np.tensordot(self.D, u, axes=(1, 0)).T
            b = np.tensordot(self.D2, u, axes=(1, 0)).T

            # divide each "row" by dt_dstau or dt_dstau**2
            outputs[self._output_val_names[name]].flat[:] = \
                np.tensordot(self.L, u, axes=(1, 0)).ravel()

            outputs[self._output_rate_names[name]].flat[:] = \
                (a / inputs['dt_dstau']).T.ravel()

            outputs[self._output_rate2_names[name]].flat[:] = \
                (b / inputs['dt_dstau'] ** 2).T.ravel()

    def compute_partials(self, inputs, partials):
        gd = self.options['grid_data']
        control_options = self.options['control_options']
        idx = self.options['index']
        num_control_disc_nodes = gd.subset_num_nodes_per_segment['control_disc'][idx]

        for name, options in iteritems(control_options):
            control_name = self._input_names[name]

            size = self.sizes[name]

            rate_name = self._output_rate_names[name]
            rate2_name = self._output_rate2_names[name]

            # Unroll matrix-shaped controls into an array at each node
            u_d = np.reshape(inputs[control_name], (num_control_disc_nodes, size))

            dt_dstau = inputs['dt_dstau']
            dt_dstau_tile = np.tile(dt_dstau, size)

            partials[rate_name, 'dt_dstau'] = \
                (-np.dot(self.D, u_d).ravel(order='F') / dt_dstau_tile ** 2)

            partials[rate2_name, 'dt_dstau'] = \
                -2.0 * (np.dot(self.D2, u_d).ravel(order='F') / dt_dstau_tile ** 3)

            dt_dstau_x_size = np.repeat(dt_dstau, size)[:, np.newaxis]

            r_nz, c_nz = self.rate_jac_rows[name], self.rate_jac_cols[name]
            partials[rate_name, control_name] = \
                (self.rate_jacs[name] / dt_dstau_x_size)[r_nz, c_nz]

            r_nz, c_nz = self.rate2_jac_rows[name], self.rate2_jac_cols[name]
            partials[rate2_name, control_name] = \
                (self.rate2_jacs[name] / dt_dstau_x_size ** 2)[r_nz, c_nz]
