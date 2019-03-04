from __future__ import division, print_function, absolute_import

import numpy as np
from six import iteritems, string_types

from openmdao.api import Group, ExplicitComponent, IndepVarComp

from ..grid_data import GridData
from ...utils.lgl import lgl
from ...utils.lagrange import lagrange_matrices
from ...utils.misc import get_rate_units


class LGLInterpolatedControlComp(ExplicitComponent):
    """
    Component which interpolates controls as a single polynomial across the entire phase.
    """

    def initialize(self):
        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')

        self._matrices = {}

    def setup(self):

        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['all']
        eval_nodes = gd.node_ptau

        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}
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

        self.add_input('t_duration', val=1.0, units=self.options['time_units'],
                       desc='duration of the phase to which this interpolated control group '
                            'belongs')

        for name, options in iteritems(self.options['control_options']):
            disc_nodes = lgl(options['num_points'])
            num_control_input_nodes = len(disc_nodes)
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            rate_units = get_rate_units(units, self.options['time_units'], deriv=1)
            rate2_units = get_rate_units(units, self.options['time_units'], deriv=1)

            input_shape = (num_control_input_nodes,) + shape
            output_shape = (num_nodes,) + shape

            L_de, D_de = lagrange_matrices(disc_nodes, eval_nodes)
            _, D_dd = lagrange_matrices(disc_nodes, disc_nodes)
            D2_de = np.dot(D_de, D_dd)

            self._matrices[name] = L_de, D_de, D2_de

            self._input_names[name] = 'controls:{0}'.format(name)
            self._output_val_names[name] = 'control_values:{0}'.format(name)
            self._output_rate_names[name] = 'control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'control_rates:{0}_rate2'.format(name)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)
            self.add_output(self._output_val_names[name], shape=output_shape, units=units)
            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)
            self.add_output(self._output_rate2_names[name], shape=output_shape, units=rate2_units)

            self.val_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))
            self.rate_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))
            self.rate2_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))

            for i in range(size):
                self.val_jacs[name][:, i, :, i] = L_de
                self.rate_jacs[name][:, i, :, i] = D_de
                self.rate2_jacs[name][:, i, :, i] = D2_de

            self.val_jacs[name] = self.val_jacs[name].reshape((num_nodes * size,
                                                              num_control_input_nodes * size),
                                                              order='C')
            self.rate_jacs[name] = self.rate_jacs[name].reshape((num_nodes * size,
                                                                num_control_input_nodes * size),
                                                                order='C')
            self.rate2_jacs[name] = self.rate2_jacs[name].reshape((num_nodes * size,
                                                                  num_control_input_nodes * size),
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

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        dt_dptau = 0.5 * inputs['t_duration']

        for name, options in self.options['interp_controls']:
            L_de, D_de, D2_de = self._matrices[name]

            u = inputs[self._input_names[name]]

            a = np.tensordot(D_de, u, axes=(1, 0)).T
            b = np.tensordot(D2_de, u, axes=(1, 0)).T

            # divide each "row" by dt_dptau or dt_dptau**2
            outputs[self._output_val_names[name]] = np.tensordot(self.L, u, axes=(1, 0))
            outputs[self._output_rate_names[name]] = (a / dt_dptau).T
            outputs[self._output_rate2_names[name]] = (b / dt_dptau ** 2).T

    def compute_partials(self, inputs, partials):
        control_options = self.options['control_options']
        num_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']

        for name, options in iteritems(control_options):
            control_name = self._input_names[name]

            size = self.sizes[name]
            rate_name = self._output_rate_names[name]
            rate2_name = self._output_rate2_names[name]

            # Unroll matrix-shaped controls into an array at each node
            u_d = np.reshape(inputs[control_name], (num_input_nodes, size))

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


class InterpolatedControlGroup(Group):

    def initialize(self):
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

        self._interp_controls = {}

    def setup(self):

        ivc = IndepVarComp()

        self.add_subsystem('control_inputs', subsys=ivc, promotes_outputs=['*'])

        self.add_subsystem('control_comp',
                           subsys=LGLInterpolatedControlComp(time_units=self.options['time_units'],
                                                             grid_data=self.options['grid_data'],
                                                             control_options=self.options['control_options']),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

        for name, options in iteritems(self._interp_controls):
            print(name)
