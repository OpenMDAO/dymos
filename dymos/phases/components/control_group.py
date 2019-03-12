from __future__ import print_function, division

from six import string_types, iteritems

import numpy as np

from openmdao.api import ExplicitComponent, Group, IndepVarComp

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units, CoerceDesvar
from dymos.utils.constants import INF_BOUND


class ControlInterpComp(ExplicitComponent):
    """
    Compute the approximated control values and rates given the values of a control at all nodes,
    given values at the control discretization nodes.

    Notes
    -----
    .. math::

        u = \\left[ L \\right] u_d

        \\dot{u} = \\frac{d\\tau_s}{dt} \\left[ D \\right] u_d

        \\ddot{u} = \\left( \\frac{d\\tau_s}{dt} \\right)^2 \\left[ D_2 \\right] u_d

    where
    :math:`u_d` are the values of the control at the control discretization nodes,
    :math:`u` are the values of the control at all nodes,
    :math:`\\dot{u}` are the time-derivatives of the control at all nodes,
    :math:`\\ddot{u}` are the second time-derivatives of the control at all nodes,
    :math:`L` is the Lagrange interpolation matrix,
    :math:`D` is the Lagrange differentiation matrix,
    and :math:`\\frac{d\\tau_s}{dt}` is the ratio of segment duration in segment tau space
    [-1 1] to segment duration in time.
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
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

    def _setup_controls(self):
        control_options = self.options['control_options']
        num_nodes = self.num_nodes
        num_control_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']
        time_units = self.options['time_units']

        for name, options in iteritems(control_options):
            self._input_names[name] = 'controls:{0}'.format(name)
            self._output_val_names[name] = 'control_values:{0}'.format(name)
            self._output_rate_names[name] = 'control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'control_rates:{0}_rate2'.format(name)
            shape = options['shape']
            input_shape = (num_control_input_nodes,) + shape
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
            self.val_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))
            self.rate_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))
            self.rate2_jacs[name] = np.zeros((num_nodes, size, num_control_input_nodes, size))

            for i in range(size):
                self.val_jacs[name][:, i, :, i] = self.L
                self.rate_jacs[name][:, i, :, i] = self.D
                self.rate2_jacs[name][:, i, :, i] = self.D2
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

    def setup(self):
        num_nodes = self.options['grid_data'].num_nodes
        time_units = self.options['time_units']
        gd = self.options['grid_data']

        self.add_input('dt_dstau', shape=num_nodes, units=time_units)

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
        self.num_nodes = num_nodes

        num_disc_nodes = gd.subset_num_nodes['control_disc']
        num_input_nodes = gd.subset_num_nodes['control_input']

        # Find the indexing matrix that, multiplied by the values at the input nodes,
        # gives the values at the discretization nodes
        L_id = np.zeros((num_disc_nodes, num_input_nodes), dtype=float)
        L_id[np.arange(num_disc_nodes, dtype=int),
             gd.input_maps['dynamic_control_input_to_disc']] = 1.0

        # Matrices L_da and D_da interpolate values and rates (respectively) at all nodes from
        # values specified at control discretization nodes.
        L_da, D_da = gd.phase_lagrange_matrices('control_disc', 'all')
        self.L = np.dot(L_da, L_id)
        self.D = np.dot(D_da, L_id)

        # Matrix D_dd interpolates rates at discretization nodes from values given at control
        # discretization nodes.
        _, D_dd = gd.phase_lagrange_matrices('control_disc', 'control_disc')

        # Matrix D2 provides second derivatives at all nodes given values at input nodes.
        self.D2 = np.dot(D_da, np.dot(D_dd, L_id))

        self._setup_controls()

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        control_options = self.options['control_options']

        for name, options in iteritems(control_options):

            u = inputs[self._input_names[name]]

            a = np.tensordot(self.D, u, axes=(1, 0)).T
            b = np.tensordot(self.D2, u, axes=(1, 0)).T

            # divide each "row" by dt_dstau or dt_dstau**2
            outputs[self._output_val_names[name]] = np.tensordot(self.L, u, axes=(1, 0))
            outputs[self._output_rate_names[name]] = (a / inputs['dt_dstau']).T
            outputs[self._output_rate2_names[name]] = (b / inputs['dt_dstau'] ** 2).T

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


class ControlGroup(Group):

    def initialize(self):
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=string_types,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

    def setup(self):

        ivc = IndepVarComp()

        # opts = self.options
        gd = self.options['grid_data']
        control_options = self.options['control_options']
        time_units = self.options['time_units']

        if len(control_options) < 1:
            return

        opt_controls = [name for (name, opts) in iteritems(control_options) if opts['opt']]

        if len(opt_controls) > 0:
            ivc = self.add_subsystem('indep_controls', subsys=IndepVarComp(),
                                     promotes_outputs=['*'])

        self.add_subsystem(
            'control_interp_comp',
            subsys=ControlInterpComp(time_units=time_units, grid_data=gd,
                                     control_options=control_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

        for name, options in iteritems(control_options):
            if options['opt']:
                num_input_nodes = gd.subset_num_nodes['control_input']

                desvar_indices = list(range(gd.subset_num_nodes['control_input']))
                if options['fix_initial']:
                    desvar_indices.pop(0)
                if options['fix_final']:
                    desvar_indices.pop()

                if len(desvar_indices) > 0:
                    coerce_desvar = CoerceDesvar(gd.subset_num_nodes['control_disc'],
                                                 desvar_indices, options)

                    lb = -INF_BOUND if coerce_desvar('lower') is None else coerce_desvar('lower')
                    ub = INF_BOUND if coerce_desvar('upper') is None else coerce_desvar('upper')

                    self.add_design_var(name='controls:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar('scaler'),
                                        adder=coerce_desvar('adder'),
                                        ref0=coerce_desvar('ref0'),
                                        ref=coerce_desvar('ref'),
                                        indices=desvar_indices)

                ivc.add_output(name='controls:{0}'.format(name),
                               val=options['val'],
                               shape=(num_input_nodes, np.prod(options['shape'])),
                               units=options['units'])
