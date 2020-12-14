from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp

import openmdao.api as om

from ..grid_data import GridData
from ...utils.misc import get_rate_units, CoerceDesvar
from ...utils.constants import INF_BOUND
from ...options import options as dymos_options


class ControlInterpComp(om.ExplicitComponent):
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        self.options.declare(
            'control_options', types=dict,
            desc='Dictionary of options for the dynamic controls')
        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        # Save the names of the dynamic controls/parameters
        # self._dynamic_names = []
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

    def _configure_controls(self):
        control_options = self.options['control_options']
        num_nodes = self.options['grid_data'].num_nodes
        num_control_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']
        time_units = self.options['time_units']

        for name, options in control_options.items():
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

            # self._dynamic_names.append(name)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)

            self.add_output(self._output_val_names[name], shape=output_shape, units=units)

            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)

            self.add_output(self._output_rate2_names[name], shape=output_shape,
                            units=rate2_units)

            size = np.prod(shape)
            self.sizes[name] = size

            # The partial of interpolated value wrt the control input values is linear
            # and can be computed as the kronecker product of the interpolation matrix (L)
            # and eye(size).
            J_val = sp.kron(self.L, sp.eye(size), format='csr')
            rs, cs, data = sp.find(J_val)
            self.declare_partials(of=self._output_val_names[name],
                                  wrt=self._input_names[name],
                                  rows=rs, cols=cs, val=data)

            # The partials of the output rate and second derivative wrt dt_dstau
            rs = np.arange(num_nodes * size, dtype=int)
            cs = np.repeat(np.arange(num_nodes, dtype=int), size)

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            # The partials of the rates and second derivatives are nonlinear but the sparsity
            # pattern is obtained from the kronecker product of the 1st and 2nd differentiation
            # matrices (D and D2) and eye(size).
            self.rate_jacs[name] = sp.kron(sp.csr_matrix(self.D), sp.eye(size), format='csr')
            rs, cs = self.rate_jacs[name].nonzero()

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=rs, cols=cs)

            self.rate2_jacs[name] = sp.kron(sp.csr_matrix(self.D2), sp.eye(size), format='csr')
            rs, cs = self.rate2_jacs[name].nonzero()

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt=self._input_names[name],
                                  rows=rs, cols=cs)

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        num_nodes = self.options['grid_data'].num_nodes
        time_units = self.options['time_units']
        gd = self.options['grid_data']

        self.add_input('dt_dstau', shape=num_nodes, units=time_units)

        self.rate_jacs = {}
        self.rate2_jacs = {}
        self.sizes = {}

        num_disc_nodes = gd.subset_num_nodes['control_disc']
        num_input_nodes = gd.subset_num_nodes['control_input']

        # Find the indexing matrix that, multiplied by the values at the input nodes,
        # gives the values at the discretization nodes
        L_id = np.zeros((num_disc_nodes, num_input_nodes), dtype=float)
        L_id[np.arange(num_disc_nodes, dtype=int),
             gd.input_maps['dynamic_control_input_to_disc']] = 1.0
        L_id = sp.csr_matrix(L_id)

        # Matrices L_da and D_da interpolate values and rates (respectively) at all nodes from
        # values specified at control discretization nodes.
        L_da, D_da = gd.phase_lagrange_matrices('control_disc', 'all', sparse=True)
        self.L = L_da.dot(L_id)
        self.D = D_da.dot(L_id)

        # Matrix D_dd interpolates rates at discretization nodes from values given at control
        # discretization nodes.
        _, D_dd = gd.phase_lagrange_matrices('control_disc', 'control_disc', sparse=True)

        # Matrix D2 provides second derivatives at all nodes given values at input nodes.
        self.D2 = D_da.dot(D_dd.dot(L_id))

        self._configure_controls()

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        control_options = self.options['control_options']
        num_nodes = self.options['grid_data'].num_nodes
        num_control_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']

        for name, options in control_options.items():
            size = np.prod(options['shape'])

            u_flat = np.reshape(inputs[self._input_names[name]],
                                newshape=(num_control_input_nodes, size))

            a = self.D.dot(u_flat)
            b = self.D2.dot(u_flat)

            val = np.reshape(self.L.dot(u_flat), (num_nodes,) + options['shape'])

            rate = a / inputs['dt_dstau'][:, np.newaxis]
            rate = np.reshape(rate, (num_nodes,) + options['shape'])

            rate2 = b / inputs['dt_dstau'][:, np.newaxis] ** 2
            rate2 = np.reshape(rate2, (num_nodes,) + options['shape'])

            outputs[self._output_val_names[name]] = val
            outputs[self._output_rate_names[name]] = rate
            outputs[self._output_rate2_names[name]] = rate2

    def compute_partials(self, inputs, partials):
        control_options = self.options['control_options']
        num_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']

        dstau_dt = np.reciprocal(inputs['dt_dstau'])
        dstau_dt2 = (dstau_dt ** 2)[:, np.newaxis]
        dstau_dt3 = (dstau_dt ** 3)[:, np.newaxis]

        for name, options in control_options.items():
            control_name = self._input_names[name]

            size = self.sizes[name]
            rate_name = self._output_rate_names[name]
            rate2_name = self._output_rate2_names[name]

            # Unroll shaped controls into an array at each node
            u_flat = np.reshape(inputs[control_name], (num_input_nodes, size))

            partials[rate_name, 'dt_dstau'] = (-self.D.dot(u_flat) * dstau_dt2).ravel()
            partials[rate2_name, 'dt_dstau'] = (-2.0 * self.D2.dot(u_flat) * dstau_dt3).ravel()

            dstau_dt_x_size = np.repeat(dstau_dt, size)[:, np.newaxis]
            dstau_dt2_x_size = np.repeat(dstau_dt2, size)[:, np.newaxis]

            partials[rate_name, control_name] = self.rate_jacs[name].multiply(dstau_dt_x_size).data

            partials[rate2_name, control_name] = self.rate2_jacs[name].multiply(dstau_dt2_x_size).data


class ControlGroup(om.Group):

    def initialize(self):
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

    def setup(self):
        gd = self.options['grid_data']
        control_options = self.options['control_options']
        time_units = self.options['time_units']

        if len(control_options) < 1:
            return

        opt_controls = [name for (name, opts) in control_options.items() if opts['opt']]

        if len(opt_controls) > 0:
            self.add_subsystem('indep_controls', subsys=om.IndepVarComp(), promotes_outputs=['*'])

        self.add_subsystem(
            'control_interp_comp',
            subsys=ControlInterpComp(time_units=time_units, grid_data=gd,
                                     control_options=control_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        control_options = self.options['control_options']
        gd = self.options['grid_data']
        self.control_interp_comp.configure_io()

        for name, options in control_options.items():
            size = np.prod(options['shape'])
            if options['opt']:
                num_input_nodes = gd.subset_num_nodes['control_input']
                desvar_indices = list(range(size * num_input_nodes))

                if options['fix_initial']:
                    if isinstance(options['fix_initial'], Iterable):
                        idxs_to_fix = np.where(np.asarray(options['fix_initial']))[0]
                        for idx_to_fix in reversed(sorted(idxs_to_fix)):
                            del desvar_indices[idx_to_fix]
                    else:
                        del desvar_indices[:size]

                if options['fix_final']:
                    if isinstance(options['fix_final'], Iterable):
                        idxs_to_fix = np.where(np.asarray(options['fix_final']))[0]
                        for idx_to_fix in reversed(sorted(idxs_to_fix)):
                            del desvar_indices[-size + idx_to_fix]
                    else:
                        del desvar_indices[-size:]

                if len(desvar_indices) > 0:
                    coerce_desvar_option = CoerceDesvar(num_input_nodes, desvar_indices,
                                                        options)

                    lb = np.zeros_like(desvar_indices, dtype=float)
                    lb[:] = -INF_BOUND if coerce_desvar_option('lower') is None else \
                        coerce_desvar_option('lower')

                    ub = np.zeros_like(desvar_indices, dtype=float)
                    ub[:] = INF_BOUND if coerce_desvar_option('upper') is None else \
                        coerce_desvar_option('upper')

                    self.add_design_var(name='controls:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar_option('scaler'),
                                        adder=coerce_desvar_option('adder'),
                                        ref0=coerce_desvar_option('ref0'),
                                        ref=coerce_desvar_option('ref'),
                                        indices=desvar_indices)

                self.indep_controls.add_output(name='controls:{0}'.format(name),
                                               val=options['val'],
                                               shape=(num_input_nodes, np.prod(options['shape'])),
                                               units=options['units'])
