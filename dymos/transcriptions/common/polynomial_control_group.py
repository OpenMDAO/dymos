import numpy as np

import openmdao.api as om

from ..grid_data import GridData
from ...utils.lgl import lgl
from ...utils.lagrange import lagrange_matrices
from ...utils.misc import get_rate_units, reshape_val
from ...utils.constants import INF_BOUND

from ...options import options as dymos_options


class LGLPolynomialControlComp(om.ExplicitComponent):
    """
    Component which interpolates controls as a single polynomial across the entire phase.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')
        self.options.declare('polynomial_control_options', types=dict,
                             desc='Dictionary of options for the polynomial controls')

        self._matrices = {}

        self._no_check_partials = not dymos_options['include_check_partials']

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units for the states.
        """
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

        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['all']
        eval_nodes = gd.node_ptau

        for name, options in self.options['polynomial_control_options'].items():
            disc_nodes, _ = lgl(options['order'] + 1)
            num_control_input_nodes = len(disc_nodes)
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            rate_units = get_rate_units(units, self.options['time_units'], deriv=1)
            rate2_units = get_rate_units(units, self.options['time_units'], deriv=2)

            input_shape = (num_control_input_nodes,) + shape
            output_shape = (num_nodes,) + shape

            L_de, D_de = lagrange_matrices(disc_nodes, eval_nodes)
            _, D_dd = lagrange_matrices(disc_nodes, disc_nodes)
            D2_de = np.dot(D_de, D_dd)

            self._matrices[name] = L_de, D_de, D2_de

            self._input_names[name] = 'polynomial_controls:{0}'.format(name)
            self._output_val_names[name] = 'polynomial_control_values:{0}'.format(name)
            self._output_rate_names[name] = 'polynomial_control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'polynomial_control_rates:{0}_rate2'.format(name)

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

            rs = np.concatenate([np.arange(0, num_nodes * size, size, dtype=int) + i
                                 for i in range(size)])

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt='t_duration', rows=rs, cols=np.zeros_like(rs))

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.rate_jac_rows[name], cols=self.rate_jac_cols[name])

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt='t_duration', rows=rs, cols=np.zeros_like(rs))

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt=self._input_names[name],
                                  rows=self.rate2_jac_rows[name], cols=self.rate2_jac_cols[name])

    def compute(self, inputs, outputs):
        """
        Interpolate control outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        dt_dptau = 0.5 * inputs['t_duration']

        for name, options in self.options['polynomial_control_options'].items():
            L_de, D_de, D2_de = self._matrices[name]

            u = inputs[self._input_names[name]]

            a = np.tensordot(D_de, u, axes=(1, 0)).T
            b = np.tensordot(D2_de, u, axes=(1, 0)).T

            # divide each "row" by dt_dptau or dt_dptau**2
            outputs[self._output_val_names[name]] = np.tensordot(L_de, u, axes=(1, 0))
            outputs[self._output_rate_names[name]] = (a / dt_dptau).T
            outputs[self._output_rate2_names[name]] = (b / dt_dptau ** 2).T

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        nn = self.options['grid_data'].num_nodes

        for name, options in self.options['polynomial_control_options'].items():
            control_name = self._input_names[name]
            num_input_nodes = options['order'] + 1
            L_de, D_de, D2_de = self._matrices[name]

            size = self.sizes[name]
            rate_name = self._output_rate_names[name]
            rate2_name = self._output_rate2_names[name]

            # Unroll matrix-shaped controls into an array at each node
            u_d = np.reshape(inputs[control_name], (num_input_nodes, size))

            t_duration = inputs['t_duration']
            t_duration_tile = np.tile(t_duration, size * nn)

            partials[rate_name, 't_duration'] = \
                0.5 * (-np.dot(D_de, u_d).ravel(order='F') / (0.5 * t_duration_tile) ** 2)

            partials[rate2_name, 't_duration'] = \
                -1.0 * (np.dot(D2_de, u_d).ravel(order='F') / (0.5 * t_duration_tile) ** 3)

            t_duration_x_size = np.repeat(t_duration, size * nn)[:, np.newaxis]

            r_nz, c_nz = self.rate_jac_rows[name], self.rate_jac_cols[name]
            partials[rate_name, control_name] = \
                (self.rate_jacs[name] / (0.5 * t_duration_x_size))[r_nz, c_nz]

            r_nz, c_nz = self.rate2_jac_rows[name], self.rate2_jac_cols[name]
            partials[rate2_name, control_name] = \
                (self.rate2_jacs[name] / (0.5 * t_duration_x_size) ** 2)[r_nz, c_nz]


class PolynomialControlGroup(om.Group):
    """
    Group that contains and manages the LGLPolynomialControlComp.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare group options.
        """
        self.options.declare('polynomial_control_options', types=dict,
                             desc='Dictionary of options for the polynomial controls')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

    def setup(self):
        """
        Define the structure of the polynomial control group.
        """
        opts = self.options

        # Pull out the interpolated controls
        num_opt = 0
        for name, options in opts['polynomial_control_options'].items():
            if options['order'] < 1:
                raise ValueError('Interpolation order must be >= 1 (linear)')
            if options['opt']:
                num_opt += 1

        if num_opt > 0:
            self.add_subsystem('indep_polynomial_controls', subsys=om.IndepVarComp(),
                               promotes_outputs=['*'])

        self.add_subsystem(
            'interp_comp',
            subsys=LGLPolynomialControlComp(time_units=opts['time_units'],
                                            grid_data=opts['grid_data'],
                                            polynomial_control_options=opts['polynomial_control_'
                                                                            'options']),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units for the states.
        """
        self.interp_comp.configure_io()

        # For any interpolated control with `opt=True`, add an indep var comp output and
        # setup the design variable for optimization.
        for name, options in self.options['polynomial_control_options'].items():
            num_input_nodes = options['order'] + 1
            shape = options['shape']
            if options['opt']:
                default_val = reshape_val(options['val'], shape, num_input_nodes)

                self.indep_polynomial_controls.add_output(f'polynomial_controls:{name}',
                                                          shape=(num_input_nodes,) + shape,
                                                          val=default_val,
                                                          units=options['units'])

                desvar_indices = list(range(num_input_nodes))
                if options['fix_initial']:
                    desvar_indices.pop(0)
                if options['fix_final']:
                    desvar_indices.pop()

                lb = -INF_BOUND if options['lower'] is None else options['lower']
                ub = INF_BOUND if options['upper'] is None else options['upper']

                self.add_design_var('polynomial_controls:{0}'.format(name),
                                    lower=lb,
                                    upper=ub,
                                    ref=options['ref'],
                                    ref0=options['ref0'],
                                    adder=options['adder'],
                                    scaler=options['scaler'],
                                    indices=desvar_indices)
