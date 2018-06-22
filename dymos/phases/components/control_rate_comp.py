from __future__ import print_function, division

import numpy as np
from openmdao.api import ExplicitComponent
from six import string_types, iteritems

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units


class ControlRateComp(ExplicitComponent):
    """
    Compute the approximated control rate given the values of a control at a set of nodes.

    .. math:: \dot{u}_c = \frac{d\tau_s}{dt} \left[ D \right] u_d

    where :math:`u_d` are the values of the control at the control discretization nodes,
    :math:`\dot{u}_c` are the time-derivatives of the control at the control discretization nodes,
    :math:`D` is the lagrange differentiation matrix, and :math:`\frac{d\tau_s}{dt}` is the ratio
    of segment duration in segment tau space [-1 1] to segment duration in time.

    Similarly, the second time-derivative of the control is

    .. math:: \ddot{u}_c = \left( \frac{d\tau_s}{dt} \right)^2 \left[ D_2 \right] u_d

    """

    def initialize(self):
        self.options.declare(
            'control_options', types=dict,
            desc='Dictionary of options for the dynamic controls')
        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        # Save the names of the dynamic controls/parameters
        self._dynamic_names = []
        self._input_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

    def _setup_controls(self):
        control_options = self.options['control_options']
        num_nodes = self.num_nodes
        time_units = self.options['time_units']

        for name, options in iteritems(control_options):
            self._input_names[name] = 'controls:{0}'.format(name)
            self._output_rate_names[name] = 'control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'control_rates:{0}_rate2'.format(name)
            shape = options['shape']
            units = options['units']
            io_shape = (num_nodes,) + shape

            rate_units = get_rate_units(units, time_units)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self._dynamic_names.append(name)

            self.add_input(self._input_names[name], val=np.ones(io_shape), units=units)

            self.add_output(self._output_rate_names[name], shape=io_shape, units=rate_units)

            self.add_output(self._output_rate2_names[name], shape=io_shape,
                            units=rate2_units)

            size = np.prod(shape)
            self.jacs[name] = np.zeros((num_nodes, size, num_nodes, size))
            self.jacs2[name] = np.zeros((num_nodes, size, num_nodes, size))
            for i in range(size):
                self.jacs[name][:, i, :, i] = self.D
                self.jacs2[name][:, i, :, i] = self.D2
            self.jacs[name] = self.jacs[name].reshape((num_nodes * size, num_nodes * size),
                                                      order='C')
            self.jacs2[name] = self.jacs2[name].reshape((num_nodes * size, num_nodes * size),
                                                        order='C')

            self.jac_rows[name], self.jac_cols[name] = np.where(self.jacs[name] != 0)
            self.jac2_rows[name], self.jac2_cols[name] = np.where(self.jacs2[name] != 0)

            self.sizes[name] = size

            cs = np.tile(np.arange(num_nodes, dtype=int), reps=size)
            rs = np.concatenate([np.arange(0, num_nodes * size, size, dtype=int) + i
                                 for i in range(size)])

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.jac_rows[name], cols=self.jac_cols[name])

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.jac2_rows[name], cols=self.jac2_cols[name])

    def setup(self):
        num_nodes = self.options['grid_data'].num_nodes
        time_units = self.options['time_units']

        self.add_input('dt_dstau', shape=num_nodes, units=time_units)

        self.jacs = {}
        self.jacs2 = {}
        self.jac_rows = {}
        self.jac_cols = {}
        self.jac2_rows = {}
        self.jac2_cols = {}
        self.sizes = {}
        self.num_nodes = num_nodes

        _, self.D = self.options['grid_data'].phase_lagrange_matrices('control_disc', 'all')
        self.D2 = np.dot(self.D, self.D)

        self._setup_controls()

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        control_options = self.options['control_options']

        for name, options in iteritems(control_options):

            u = inputs[self._input_names[name]]

            a = np.tensordot(self.D, u, axes=(1, 0)).T
            b = np.tensordot(self.D2, u, axes=(1, 0)).T

            # divide each "row" by dt_dstau or dt_dstau**2
            outputs[self._output_rate_names[name]] = (a / inputs['dt_dstau']).T
            outputs[self._output_rate2_names[name]] = (b / inputs['dt_dstau'] ** 2).T

    def compute_partials(self, inputs, partials):
        control_options = self.options['control_options']

        for name, options in iteritems(control_options):
            control_name = self._input_names[name]

            if options['dynamic']:
                size = self.sizes[name]
                nn = self.num_nodes

                rate_name = self._output_rate_names[name]
                rate2_name = self._output_rate2_names[name]

                # Unroll matrix-shaped controls into an array at each node
                u_d = np.reshape(inputs[control_name], (nn, size))

                dt_dstau = inputs['dt_dstau']
                dt_dstau_tile = np.tile(dt_dstau, size)

                partials[rate_name, 'dt_dstau'] = \
                    (-np.dot(self.D, u_d).ravel(order='F') / dt_dstau_tile ** 2)

                partials[rate2_name, 'dt_dstau'] = \
                    -2.0 * (np.dot(self.D2, u_d).ravel(order='F') / dt_dstau_tile ** 3)

                dt_dstau_x_size = np.repeat(dt_dstau, size)[:, np.newaxis]

                r_nz, c_nz = self.jac_rows[name], self.jac_cols[name]
                partials[rate_name, control_name] = \
                    (self.jacs[name] / dt_dstau_x_size)[r_nz, c_nz]

                r_nz, c_nz = self.jac2_rows[name], self.jac2_cols[name]
                partials[rate2_name, control_name] = \
                    (self.jacs2[name] / dt_dstau_x_size ** 2)[r_nz, c_nz]
