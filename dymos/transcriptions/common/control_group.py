import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

import openmdao.api as om

from ..grid_data import GridData
from ...utils.misc import get_rate_units, CoerceDesvar, reshape_val
from ...utils.lagrange import lagrange_matrices
from ...utils.indexing import get_desvar_indices
from ...utils.constants import INF_BOUND
from ..._options import options as dymos_options


class ControlInterpComp(om.ExplicitComponent):
    """
    Class definition for the ControlInterpComp.

    Compute the approximated control values and rates given the values of a control at all nodes,
    given values at the control discretization nodes.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

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
        """
        Declare component options.
        """
        self.options.declare(
            'control_options', types=dict,
            desc='Dictionary of options for the dynamic controls')
        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info for the control inputs.')
        self.options.declare('output_grid_data', types=GridData, allow_none=True, default=None,
                             desc='GridData object for the output grid. If None, use the same grid_data as the inputs.')

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}

    def setup(self):
        """
        Perform setup procedure for the Control interpolation component.
        """
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd

        if not gd.is_aligned_with(ogd):
            raise RuntimeError(f'{self.pathname}: The input grid and the output grid must have the same number of '
                               f'segments and segment spacing, but the input grid segment ends are '
                               f'\n{gd.segment_ends}\n and the output grid segment ends are \n'
                               f'{ogd.segment_ends}.')

    def _configure_controls(self):
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        control_options = self.options['control_options']
        num_output_nodes = ogd.num_nodes
        num_control_input_nodes = gd.subset_num_nodes['control_input']
        time_units = self.options['time_units']

        for name, options in control_options.items():
            self._input_names[name] = f'controls:{name}'
            self._output_val_names[name] = f'control_values:{name}'
            self._output_rate_names[name] = f'control_rates:{name}_rate'
            self._output_rate2_names[name] = f'control_rates:{name}_rate2'
            shape = options['shape']
            input_shape = (num_control_input_nodes,) + shape
            output_shape = (num_output_nodes,) + shape

            units = options['units']
            rate_units = get_rate_units(units, time_units)
            rate2_units = get_rate_units(units, time_units, deriv=2)

            self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)

            self.add_output(self._output_val_names[name], shape=output_shape, units=units)

            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)

            self.add_output(self._output_rate2_names[name], shape=output_shape,
                            units=rate2_units)

            size = np.prod(shape)
            self.sizes[name] = size
            sp_eye = sp.eye(size, format='csr')

            # The partial of interpolated value wrt the control input values is linear
            # and can be computed as the kronecker product of the interpolation matrix (L)
            # and eye(size).
            J_val = sp.kron(self.L, sp_eye, format='csr')
            rs, cs, data = sp.find(J_val)
            self.declare_partials(of=self._output_val_names[name],
                                  wrt=self._input_names[name],
                                  rows=rs, cols=cs, val=data)

            # The partials of the output rate and second derivative wrt dt_dstau
            rs = np.arange(num_output_nodes * size, dtype=int)
            cs = np.repeat(np.arange(num_output_nodes, dtype=int), size)

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt='dt_dstau',
                                  rows=rs, cols=cs)

            # The partials of the rates and second derivatives are nonlinear but the sparsity
            # pattern is obtained from the kronecker product of the 1st and 2nd differentiation
            # matrices (D and D2) and eye(size).
            self.rate_jacs[name] = sp.kron(self.D, sp_eye, format='csr')
            rs, cs = self.rate_jacs[name].nonzero()

            self.declare_partials(of=self._output_rate_names[name],
                                  wrt=self._input_names[name],
                                  rows=rs, cols=cs)

            self.rate2_jacs[name] = sp.kron(self.D2, sp_eye, format='csr')
            rs, cs = self.rate2_jacs[name].nonzero()

            self.declare_partials(of=self._output_rate2_names[name],
                                  wrt=self._input_names[name],
                                  rows=rs, cols=cs)

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units for the states.
        """
        time_units = self.options['time_units']
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        output_num_nodes = ogd.subset_num_nodes['all']
        num_seg = gd.num_segments

        self.add_input('dt_dstau', shape=output_num_nodes, units=time_units)

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
        # If the output grid is different than the input grid, then we have to build these these matrices ourselves.
        if ogd is gd:
            L_da, D_da = gd.phase_lagrange_matrices('control_disc', 'all', sparse=True)
        else:
            L_blocks = []
            D_blocks = []

            for iseg in range(num_seg):
                i1, i2 = gd.subset_segment_indices['control_disc'][iseg, :]
                indices = gd.subset_node_indices['control_disc'][i1:i2]
                nodes_given = gd.node_stau[indices]

                i1, i2 = ogd.subset_segment_indices['all'][iseg, :]
                indices = ogd.subset_node_indices['all'][i1:i2]
                nodes_eval = ogd.node_stau[indices]

                L_block, D_block = lagrange_matrices(nodes_given, nodes_eval)

                L_blocks.append(L_block)
                D_blocks.append(D_block)

            L_da = sp.csr_matrix(block_diag(*L_blocks))
            D_da = sp.csr_matrix(block_diag(*D_blocks))

        self.L = L_da.dot(L_id)
        self.D = D_da.dot(L_id)

        # Matrix D_dd interpolates rates at discretization nodes from values given at control
        # discretization nodes.
        _, D_dd = gd.phase_lagrange_matrices('control_disc', 'control_disc', sparse=True)

        # Matrix D2 provides second derivatives at all nodes given values at input nodes.
        self.D2 = D_da.dot(D_dd.dot(L_id))

        self._configure_controls()

    def compute(self, inputs, outputs):
        """
        Compute interpolated control values and rates.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        control_options = self.options['control_options']
        num_control_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']
        num_output_nodes = ogd.num_nodes

        for name, options in control_options.items():
            size = np.prod(options['shape'])

            u_flat = np.reshape(inputs[self._input_names[name]],
                                newshape=(num_control_input_nodes, size))

            a = self.D.dot(u_flat)
            b = self.D2.dot(u_flat)

            val = np.reshape(self.L.dot(u_flat), (num_output_nodes,) + options['shape'])

            rate = a / inputs['dt_dstau'][:, np.newaxis]
            rate = np.reshape(rate, (num_output_nodes,) + options['shape'])

            rate2 = b / inputs['dt_dstau'][:, np.newaxis] ** 2
            rate2 = np.reshape(rate2, (num_output_nodes,) + options['shape'])

            outputs[self._output_val_names[name]] = val
            outputs[self._output_rate_names[name]] = rate
            outputs[self._output_rate2_names[name]] = rate2

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
        control_options = self.options['control_options']
        num_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']

        dstau_dt = np.reciprocal(inputs['dt_dstau'])
        dstau_dt2 = (dstau_dt ** 2)[:, np.newaxis]
        dstau_dt3 = (dstau_dt ** 3)[:, np.newaxis]

        for name in control_options:
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
    """
    Class definition for the ControlGroup.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def initialize(self):
        """
        Declare group options.
        """
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls.')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time.')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info for the control inputs.')
        self.options.declare('output_grid_data', types=GridData, allow_none=True, default=None,
                             desc='GridData object for the output grid. If None, use the same grid_data as the inputs.')

    def setup(self):
        """
        Define the structure of the control group.
        """
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or self.options['grid_data']
        control_options = self.options['control_options']
        time_units = self.options['time_units']

        if len(control_options) < 1:
            return

        self.add_subsystem(
            'control_interp_comp',
            subsys=ControlInterpComp(time_units=time_units, grid_data=gd, output_grid_data=ogd,
                                     control_options=control_options),
            promotes_inputs=['*'],
            promotes_outputs=['*'])

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units for the states.
        """
        control_options = self.options['control_options']
        gd = self.options['grid_data']
        num_input_nodes = gd.subset_num_nodes['control_input']

        self.control_interp_comp.configure_io()

        for name, options in control_options.items():
            dvname = f'controls:{name}'
            shape = options['shape']
            size = np.prod(shape)
            if options['opt']:
                desvar_indices = get_desvar_indices(size, num_input_nodes,
                                                    options['fix_initial'], options['fix_final'])

                if len(desvar_indices) > 0:
                    coerce_desvar_option = CoerceDesvar(num_input_nodes, desvar_indices,
                                                        options=options)

                    lb = np.zeros_like(desvar_indices, dtype=float)
                    lb[:] = -INF_BOUND if coerce_desvar_option('lower') is None else \
                        coerce_desvar_option('lower')

                    ub = np.zeros_like(desvar_indices, dtype=float)
                    ub[:] = INF_BOUND if coerce_desvar_option('upper') is None else \
                        coerce_desvar_option('upper')

                    self.add_design_var(name=dvname,
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar_option('scaler'),
                                        adder=coerce_desvar_option('adder'),
                                        ref0=coerce_desvar_option('ref0'),
                                        ref=coerce_desvar_option('ref'),
                                        indices=desvar_indices,
                                        flat_indices=True)

            default_val = reshape_val(options['val'], shape, num_input_nodes)

            self.set_input_defaults(name=dvname, val=default_val, units=options['units'])
