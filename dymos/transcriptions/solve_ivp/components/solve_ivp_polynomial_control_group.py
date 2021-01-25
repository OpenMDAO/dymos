import numpy as np
import openmdao.api as om

from ...grid_data import GridData
from ....utils.lgl import lgl
from ....utils.lagrange import lagrange_matrices
from ....utils.misc import get_rate_units


class SolveIVPLGLPolynomialControlComp(om.ExplicitComponent):
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
        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

        self._matrices = {}

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        output_nodes_per_seg = self.options['output_nodes_per_seg']
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        num_nodes = gd.subset_num_nodes['all']
        all_nodes_ptau = gd.node_ptau

        if output_nodes_per_seg is None:
            output_nodes_ptau = all_nodes_ptau
        else:
            output_nodes_ptau = np.empty(0, dtype=float)
            for iseg in range(num_seg):
                i1, i2 = gd.subset_segment_indices['all'][iseg, :]
                ptau1 = all_nodes_ptau[i1]
                ptau2 = all_nodes_ptau[i2-1]
                output_nodes_ptau = np.concatenate((output_nodes_ptau,
                                                    np.linspace(ptau1, ptau2, output_nodes_per_seg)))

        num_output_nodes = len(output_nodes_ptau)

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

        for name, options in self.options['polynomial_control_options'].items():
            disc_nodes, _ = lgl(options['order'] + 1)
            num_control_input_nodes = len(disc_nodes)
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            rate_units = get_rate_units(units, self.options['time_units'], deriv=1)
            rate2_units = get_rate_units(units, self.options['time_units'], deriv=2)

            input_shape = (num_control_input_nodes,) + shape
            output_shape = (num_output_nodes,) + shape

            L_do, D_do = lagrange_matrices(disc_nodes, output_nodes_ptau)
            _, D_dd = lagrange_matrices(disc_nodes, disc_nodes)
            D2_do = np.dot(D_do, D_dd)

            self._matrices[name] = L_do, D_do, D2_do

            self._input_names[name] = 'polynomial_controls:{0}'.format(name)
            self._output_val_names[name] = 'polynomial_control_values:{0}'.format(name)
            self._output_rate_names[name] = 'polynomial_control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'polynomial_control_rates:{0}_rate2'.format(name)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)
            self.add_output(self._output_val_names[name], shape=output_shape, units=units)
            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)
            self.add_output(self._output_rate2_names[name], shape=output_shape, units=rate2_units)

            self.val_jacs[name] = np.zeros((num_output_nodes, size, num_control_input_nodes, size))
            self.rate_jacs[name] = np.zeros((num_output_nodes, size, num_control_input_nodes, size))
            self.rate2_jacs[name] = np.zeros((num_output_nodes, size, num_control_input_nodes, size))

            for i in range(size):
                self.val_jacs[name][:, i, :, i] = L_do
                self.rate_jacs[name][:, i, :, i] = D_do
                self.rate2_jacs[name][:, i, :, i] = D2_do

            self.val_jacs[name] = self.val_jacs[name].reshape((num_output_nodes * size,
                                                              num_control_input_nodes * size),
                                                              order='C')
            self.rate_jacs[name] = self.rate_jacs[name].reshape((num_output_nodes * size,
                                                                num_control_input_nodes * size),
                                                                order='C')
            self.rate2_jacs[name] = self.rate2_jacs[name].reshape((num_output_nodes * size,
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
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        dt_dptau = 0.5 * inputs['t_duration']

        for name, options in self.options['polynomial_control_options'].items():
            L_do, D_do, D2_do = self._matrices[name]

            u = inputs[self._input_names[name]]

            a = np.tensordot(D_do, u, axes=(1, 0)).T
            b = np.tensordot(D2_do, u, axes=(1, 0)).T

            # divide each "row" of the rates by dt_dptau or dt_dptau**2
            outputs[self._output_val_names[name]] = np.tensordot(L_do, u, axes=(1, 0))
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
            L_do, D_do, D2_do = self._matrices[name]

            size = self.sizes[name]
            rate_name = self._output_rate_names[name]
            rate2_name = self._output_rate2_names[name]

            # Unroll matrix-shaped controls into an array at each node
            u_d = np.reshape(inputs[control_name], (num_input_nodes, size))

            t_duration = inputs['t_duration']
            t_duration_tile = np.tile(t_duration, size * nn)

            partials[rate_name, 't_duration'] = \
                0.5 * (-np.dot(D_do, u_d).ravel(order='F') / (0.5 * t_duration_tile) ** 2)

            partials[rate2_name, 't_duration'] = \
                -1.0 * (np.dot(D2_do, u_d).ravel(order='F') / (0.5 * t_duration_tile) ** 3)

            t_duration_x_size = np.repeat(t_duration, size * nn)[:, np.newaxis]

            r_nz, c_nz = self.rate_jac_rows[name], self.rate_jac_cols[name]
            partials[rate_name, control_name] = \
                (self.rate_jacs[name] / (0.5 * t_duration_x_size))[r_nz, c_nz]

            r_nz, c_nz = self.rate2_jac_rows[name], self.rate2_jac_cols[name]
            partials[rate2_name, control_name] = \
                (self.rate2_jacs[name] / (0.5 * t_duration_x_size) ** 2)[r_nz, c_nz]


class SolveIVPPolynomialControlGroup(om.Group):
    """
    Group containing the SolveIVPLGLPolynomialControlComp.

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
        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

    def setup(self):
        """
        Build the group hierarchy.
        """
        ivc = om.IndepVarComp()

        opts = self.options
        pcos = self.options['polynomial_control_options']
        output_nodes_per_seg = self.options['output_nodes_per_seg']

        # Pull out the interpolated controls
        num_opt = 0
        for name, options in opts['polynomial_control_options'].items():
            if options['order'] < 1:
                raise ValueError('Interpolation order must be >= 1 (linear)')
            if options['opt']:
                num_opt += 1

        if num_opt > 0:
            ivc = self.add_subsystem('control_inputs', subsys=ivc, promotes_outputs=['*'])

        self.add_subsystem(
            'control_comp',
            subsys=SolveIVPLGLPolynomialControlComp(time_units=opts['time_units'],
                                                    grid_data=opts['grid_data'],
                                                    polynomial_control_options=pcos,
                                                    output_nodes_per_seg=output_nodes_per_seg),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine variable shape and units.
        """
        ivc = self.control_inputs
        self.control_comp.configure_io()

        # For any interpolated control with `opt=True`, add an indep var comp output and
        # setup the design variable for optimization.
        for name, options in self.options['polynomial_control_options'].items():
            num_input_nodes = options['order'] + 1
            shape = options['shape']
            if options['opt']:
                ivc.add_output('polynomial_controls:{0}'.format(name),
                               val=np.ones((num_input_nodes,) + shape),
                               units=options['units'])
