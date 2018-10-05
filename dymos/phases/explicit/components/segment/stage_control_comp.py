from __future__ import print_function, division, absolute_import

from six import iteritems

import numpy as np

from dymos.phases.components.control_interp_comp import ControlInterpComp
from dymos.utils.rk_methods import rk_methods
from dymos.utils.lagrange import lagrange_matrices
from dymos.utils.misc import get_rate_units


class StageControlComp(ControlInterpComp):
    """
    Computes the values of the states to pass to the ODE for a given stage
    """

    def initialize(self):
        super(StageControlComp, self).initialize()

        self.options.declare('index', types=int, desc='The index of this segment in the phase.')
        self.options.declare('method', types=str, default='rk4')
        self.options.declare('num_steps', types=(int,))

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
            self._output_val_names[name] = 'control_values:{0}'.format(name)
            self._output_rate_names[name] = 'control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'control_rates:{0}_rate2'.format(name)
            shape = options['shape']
            input_shape = (num_control_disc_nodes,) + shape
            output_shape = (num_nodes,) + shape

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

            cs = np.tile(np.arange(num_nodes, dtype=int), reps=size)
            rs = np.concatenate([np.arange(0, num_nodes * size, size, dtype=int) + i
                                 for i in range(size)])

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.rate_jac_rows[name], cols=self.rate_jac_cols[name])

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

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
