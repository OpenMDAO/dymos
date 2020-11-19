import numpy as np
from scipy.linalg import block_diag

import openmdao.api as om

from ...grid_data import GridData
from dymos.utils.misc import get_rate_units
from ....utils.lagrange import lagrange_matrices


class SolveIVPControlInterpComp(om.ExplicitComponent):
    """
    Compute the approximated control values and rates given the values of a control at output nodes
    and the approximated values at output nodes, given values at the control input nodes.

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
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')
        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

        # Save the names of the dynamic controls/parameters
        self._dynamic_names = []
        self._input_names = {}
        self._output_val_names = {}
        self._output_val_all_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

    def _configure_controls(self):
        control_options = self.options['control_options']
        num_nodes_all = self.num_nodes_all
        num_nodes_output = self.num_nodes_output
        num_control_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']
        time_units = self.options['time_units']

        for name, options in control_options.items():
            self._input_names[name] = 'controls:{0}'.format(name)
            self._output_val_all_names[name] = 'control_values_all:{0}'.format(name)
            self._output_val_names[name] = 'control_values:{0}'.format(name)
            self._output_rate_names[name] = 'control_rates:{0}_rate'.format(name)
            self._output_rate2_names[name] = 'control_rates:{0}_rate2'.format(name)
            shape = options['shape']
            input_shape = (num_control_input_nodes,) + shape
            all_shape = (num_nodes_all,) + shape
            output_shape = (num_nodes_output,) + shape

            units = options['units']
            rate_units = get_rate_units(units, time_units)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self._dynamic_names.append(name)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)

            self.add_output(self._output_val_all_names[name], shape=all_shape, units=units)

            self.add_output(self._output_val_names[name], shape=output_shape, units=units)

            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)

            self.add_output(self._output_rate2_names[name], shape=output_shape,
                            units=rate2_units)

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        output_nodes_per_seg = self.options['output_nodes_per_seg']
        time_units = self.options['time_units']
        gd = self.options['grid_data']
        num_seg = gd.num_segments
        num_nodes_all = gd.subset_num_nodes['all']

        if output_nodes_per_seg is None:
            num_nodes_output = num_nodes_all
        else:
            num_nodes_output = num_seg * output_nodes_per_seg

        self.add_input('dt_dstau', shape=num_nodes_output, units=time_units)

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
        self.num_nodes_all = num_nodes_all
        self.num_nodes_output = num_nodes_output

        num_disc_nodes = gd.subset_num_nodes['control_disc']
        num_input_nodes = gd.subset_num_nodes['control_input']

        # Find the indexing matrix that, multiplied by the values at the input nodes,
        # gives the values at the discretization nodes
        L_id = np.zeros((num_disc_nodes, num_input_nodes), dtype=float)
        L_id[np.arange(num_disc_nodes, dtype=int),
             gd.input_maps['dynamic_control_input_to_disc']] = 1.0

        # Matrices L_do and D_do interpolate values and rates (respectively) at output nodes from
        # values specified at control discretization nodes.
        L_da, _ = gd.phase_lagrange_matrices('control_disc', 'all')

        L_do_blocks = []
        D_do_blocks = []

        for iseg in range(num_seg):
            i1, i2 = gd.subset_segment_indices['control_disc'][iseg, :]
            indices = gd.subset_node_indices['control_disc'][i1:i2]
            nodes_given = gd.node_stau[indices]

            if output_nodes_per_seg is None:
                i1, i2 = gd.subset_segment_indices['all'][iseg, :]
                indices = gd.subset_node_indices['all'][i1:i2]
                nodes_eval = gd.node_stau[indices]
            else:
                nodes_eval = np.linspace(-1, 1, output_nodes_per_seg)

            L_block, D_block = lagrange_matrices(nodes_given, nodes_eval)

            L_do_blocks.append(L_block)
            D_do_blocks.append(D_block)

        L_do = block_diag(*L_do_blocks)
        D_do = block_diag(*D_do_blocks)

        self.L = np.dot(L_do, L_id)
        self.L_all = np.dot(L_da, L_id)
        self.D = np.dot(D_do, L_id)

        # Matrix D_dd interpolates rates at discretization nodes from values given at control
        # discretization nodes.
        _, D_dd = gd.phase_lagrange_matrices('control_disc', 'control_disc')

        # Matrix D2 provides second derivatives at output nodes given values at input nodes.
        self.D2 = np.dot(D_do, np.dot(D_dd, L_id))

        self._configure_controls()

        self.set_check_partial_options('*', method='cs')

    def compute(self, inputs, outputs):
        control_options = self.options['control_options']

        for name, options in control_options.items():

            u = inputs[self._input_names[name]]

            a = np.tensordot(self.D, u, axes=(1, 0)).T
            b = np.tensordot(self.D2, u, axes=(1, 0)).T

            # divide each "row" by dt_dstau or dt_dstau**2
            outputs[self._output_val_names[name]] = np.tensordot(self.L, u, axes=(1, 0))
            outputs[self._output_val_all_names[name]] = np.tensordot(self.L_all, u, axes=(1, 0))
            outputs[self._output_rate_names[name]] = (a / inputs['dt_dstau']).T
            outputs[self._output_rate2_names[name]] = (b / inputs['dt_dstau'] ** 2).T


class SolveIVPControlGroup(om.Group):

    def initialize(self):
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')
        self.options.declare('output_nodes_per_seg', default=None, types=(int,), allow_none=True,
                             desc='If None, results are provided at the all nodes within each'
                                  'segment.  If an int (n) then results are provided at n '
                                  'equally distributed points in time within each segment.')

    def setup(self):
        gd = self.options['grid_data']
        control_options = self.options['control_options']
        time_units = self.options['time_units']

        if len(control_options) < 1:
            return

        opt_controls = [name for (name, opts) in control_options.items() if opts['opt']]

        if len(opt_controls) > 0:
            self.add_subsystem('indep_controls', subsys=om.IndepVarComp(),
                               promotes_outputs=['*'])

        self.add_subsystem(
            'control_interp_comp',
            subsys=SolveIVPControlInterpComp(time_units=time_units, grid_data=gd,
                                             control_options=control_options,
                                             output_nodes_per_seg=self.options['output_nodes_per_seg']),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

    def configure_io(self):
        """
        I/O creation is delayed until configure so that we can determine the shape and units for
        the states.
        """
        gd = self.options['grid_data']
        control_options = self.options['control_options']

        self.control_interp_comp.configure_io()

        for name, options in control_options.items():
            if options['opt']:
                num_input_nodes = gd.subset_num_nodes['control_input']

                self.indep_controls.add_output(name='controls:{0}'.format(name),
                                               val=options['val'],
                                               shape=(num_input_nodes, np.prod(options['shape'])),
                                               units=options['units'])
