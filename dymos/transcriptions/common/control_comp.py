import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

import openmdao.api as om

from ..grid_data import GridData
from ...utils.misc import get_rate_units
from ...utils.constants import INF_BOUND
from ...utils.lgl import lgl
from ...utils.lagrange import lagrange_matrices
from ..._options import options as dymos_options


class ControlComp(om.ExplicitComponent):
    r"""
    Class definition for the ControlComp.

    Compute the approximated control values and rates given the values of a control at all nodes,
    given values at the control discretization nodes.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    Notes
    -----
    .. math::

        u = \left[ L \right] u_d

        \dot{u} = \frac{d\tau_s}{dt} \left[ D \right] u_d

        \ddot{u} = \left( \frac{d\tau_s}{dt} \right)^2 \left[ D_2 \right] u_d

    where
    :math:`u_d` are the values of the control at the control discretization nodes,
    :math:`u` are the values of the control at all nodes,
    :math:`\dot{u}` are the time-derivatives of the control at all nodes,
    :math:`\ddot{u}` are the second time-derivatives of the control at all nodes,
    :math:`L` is the Lagrange interpolation matrix,
    :math:`D` is the Lagrange differentiation matrix,
    and :math:`\frac{d\tau_s}{dt}` is the ratio of segment duration in segment tau space
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
        self.options.declare('compressed', types=bool, default=False,
                             desc='Specifies whether the controls are giving in a compressed '
                             'transcription (values are specified once at segment boundaries), '
                             'or not (each end of the boundary can have a unique value). Controls '
                             'in the Radau transcription are never compressed.')
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info for the control inputs.')
        self.options.declare('output_grid_data', types=GridData, allow_none=True, default=None,
                             desc='GridData object for the output grid. If None, use the same grid_data as the inputs.')

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}
        self._output_cnty_defect_names = {}
        self._output_rate_cnty_defect_names = {}
        self._output_rate2_cnty_defect_names = {}
        self._matrices = {}

    def setup(self):
        """
        Perform setup procedure for the Control interpolation component.
        """
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        if self.options['output_grid_data'] is None:
            self.options['output_grid_data'] = ogd

        if not gd.is_aligned_with(ogd):
            raise RuntimeError(f'{self.pathname}: The input grid and the output grid must have the same number of '
                               f'segments and segment spacing, but the input grid segment ends are '
                               f'\n{gd.segment_ends}\n and the output grid segment ends are \n'
                               f'{ogd.segment_ends}.')

    def _configure_controls(self):
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        num_seg = gd.num_segments
        segment_end_idxs = gd.subset_node_indices['segment_ends']
        eval_nodes = ogd.node_ptau
        control_options = self.options['control_options']
        num_output_nodes = ogd.num_nodes
        time_units = self.options['time_units']

        for name, options in control_options.items():
            if 'control_type' not in options:
                options['control_type'] = 'full'
            if options['control_type'] == 'polynomial':
                disc_nodes, _ = lgl(options['order'] + 1)
                num_control_input_nodes = len(disc_nodes)
                shape = options['shape']
                size = np.prod(shape)
                units = options['units']
                rate_units = get_rate_units(units, self.options['time_units'], deriv=1)
                rate2_units = get_rate_units(units, self.options['time_units'], deriv=2)

                input_shape = (num_control_input_nodes,) + shape
                output_shape = (num_output_nodes,) + shape

                L_de, D_de = lagrange_matrices(disc_nodes, eval_nodes)
                _, D_dd = lagrange_matrices(disc_nodes, disc_nodes)
                D2_de = np.dot(D_de, D_dd)

                self._matrices[name] = L_de, D_de, D2_de

                self._input_names[name] = f'controls:{name}'
                self._output_val_names[name] = f'control_values:{name}'
                self._output_rate_names[name] = f'control_rates:{name}_rate'
                self._output_rate2_names[name] = f'control_rates:{name}_rate2'

                self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)
                self.add_output(self._output_val_names[name], shape=output_shape, units=units)
                self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)
                self.add_output(self._output_rate2_names[name], shape=output_shape, units=rate2_units)

                self.val_jacs[name] = np.zeros((num_output_nodes, size, num_control_input_nodes, size))
                self.rate_jacs[name] = np.zeros((num_output_nodes, size, num_control_input_nodes, size))
                self.rate2_jacs[name] = np.zeros((num_output_nodes, size, num_control_input_nodes, size))

                for i in range(size):
                    self.val_jacs[name][:, i, :, i] = L_de
                    self.rate_jacs[name][:, i, :, i] = D_de
                    self.rate2_jacs[name][:, i, :, i] = D2_de

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

                rs = np.concatenate([np.arange(0, num_output_nodes * size, size, dtype=int) + i
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

            else:
                self._input_names[name] = f'controls:{name}'
                self._output_val_names[name] = f'control_values:{name}'
                self._output_rate_names[name] = f'control_rates:{name}_rate'
                self._output_rate2_names[name] = f'control_rates:{name}_rate2'
                self._output_cnty_defect_names[name] = f'cnty_defect_controls:{name}'
                self._output_rate_cnty_defect_names[name] = f'cnty_defect_control_rates:{name}_rate'
                self._output_rate2_cnty_defect_names[name] = f'cnty_defect_control_rates:{name}_rate2'

                shape = options['shape']
                num_control_input_nodes = gd.subset_num_nodes['control_input']
                ncinps = gd.subset_num_nodes_per_segment['control_input']
                input_shape = (num_control_input_nodes,) + shape
                output_shape = (num_output_nodes,) + shape
                cnty_output_shape = (num_seg - 1,) + shape

                units = options['units']
                rate_units = get_rate_units(units, time_units)
                rate2_units = get_rate_units(units, time_units, deriv=2)

                self.add_input(self._input_names[name], val=np.ones(input_shape), units=units)

                self.add_output(self._output_val_names[name], shape=output_shape, units=units)

                self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)

                self.add_output(self._output_rate2_names[name], shape=output_shape,
                                units=rate2_units)

                if gd.num_segments > 1 and not self.options['compressed']:
                    if options['continuity']:
                        self.add_output(self._output_cnty_defect_names[name],
                                        shape=cnty_output_shape, units=units)
                        self.add_constraint(name=self._output_cnty_defect_names[name],
                                            scaler=options['continuity_scaler'],
                                            ref=options['continuity_ref'],
                                            equals=0.0, linear=True)

                    if options['rate_continuity']:
                        self.add_output(self._output_rate_cnty_defect_names[name],
                                        shape=cnty_output_shape, units=rate_units)
                        self.add_constraint(name=self._output_rate_cnty_defect_names[name],
                                            scaler=options['rate_continuity_scaler'],
                                            ref=options['rate_continuity_ref'],
                                            equals=0.0, linear=True)

                    if options['rate2_continuity']:
                        self.add_output(self._output_rate2_cnty_defect_names[name],
                                        shape=cnty_output_shape, units=rate2_units)
                        self.add_constraint(name=self._output_rate2_cnty_defect_names[name],
                                            scaler=options['rate2_continuity_scaler'],
                                            ref=options['rate2_continuity_ref'],
                                            equals=0.0, linear=True)

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

                # The jacobian for continuity defects is similar to the jacobian of the inteprpolated values
                if options['continuity'] and gd.num_segments > 1 and not self.options['compressed']:
                    J_cnty_def = -J_val[segment_end_idxs, ...][1:-1:2, ...]
                    J_cnty_def = J_cnty_def.tolil(copy=True)
                    one_col_idxs = np.cumsum(np.asarray(ncinps[:-1]))
                    J_cnty_def[np.arange(num_seg -1, dtype=int), one_col_idxs] = 1.0

                    rs, cs, data = sp.find(J_cnty_def)

                    self.declare_partials(of=self._output_cnty_defect_names[name],
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

                # Similar with the continuity rate defects.
                if options['rate_continuity'] and gd.num_segments > 1 and not self.options['compressed']:
                    J_rate_cnty_def = sp.kron(self.D, sp_eye, format='lil')[segment_end_idxs, ...]
                    J_rate_cnty_def = J_rate_cnty_def[1:-1]
                    J_rate_cnty_def[::2] *= -1
                    J_rate_cnty_def /= gd.node_dptau_dstau[segment_end_idxs, np.newaxis][1:-1]
                    # This gives twice as many rows as we want. Each odd-index row needs to
                    # have its elements moved to the row before it.
                    rs, cs, data = sp.find(J_rate_cnty_def)
                    rs = rs // 2

                    self.declare_partials(of=self._output_rate_cnty_defect_names[name],
                                        wrt=self._input_names[name],
                                        rows=rs, cols=cs, val=data)

                self.rate2_jacs[name] = sp.kron(self.D2, sp_eye, format='csr')

                rs, cs = self.rate2_jacs[name].nonzero()

                self.declare_partials(of=self._output_rate2_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

                # Similar with the continuity rate defects.
                if options['rate2_continuity'] and gd.num_segments > 1 and not self.options['compressed']:
                    J_rate2_cnty_def = sp.kron(self.D2, sp_eye, format='lil')[segment_end_idxs, ...]
                    J_rate2_cnty_def = J_rate2_cnty_def[1:-1]
                    J_rate2_cnty_def[::2] *= -1
                    J_rate2_cnty_def /= gd.node_dptau_dstau[segment_end_idxs, np.newaxis][1:-1]
                    # This gives twice as many rows as we want. Each odd-index row needs to
                    # have its elements moved to the row before it.
                    rs, cs, data = sp.find(J_rate2_cnty_def)
                    rs = rs // 2

                    self.declare_partials(of=self._output_rate2_cnty_defect_names[name],
                                          wrt=self._input_names[name],
                                          rows=rs, cols=cs, val=data)

            if options['opt']:
                desvar_indices = np.arange(num_control_input_nodes, dtype=int)
                if options['fix_initial']:
                    desvar_indices = desvar_indices[1:]
                if options['fix_final']:
                    desvar_indices = desvar_indices[:-1]
                if not(options['fix_initial'] or options['fix_final']):
                    desvar_indices = None

                lb = -INF_BOUND if options['lower'] is None else options['lower']
                ub = INF_BOUND if options['upper'] is None else options['upper']

                self.add_design_var(f'controls:{name}',
                                    lower=lb,
                                    upper=ub,
                                    ref=options['ref'],
                                    ref0=options['ref0'],
                                    adder=options['adder'],
                                    scaler=options['scaler'],
                                    indices=desvar_indices,
                                    flat_indices=True)


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
        seg_end_idxs = gd.subset_node_indices['segment_ends']
        dt_dptau = 0.5 * inputs['t_duration']
        dptau_dstau = gd.node_dptau_dstau

        for name, options in control_options.items():
            if control_options[name]['control_type'] == 'polynomial':
                L_de, D_de, D2_de = self._matrices[name]

                u = inputs[self._input_names[name]]

                a = np.tensordot(D_de, u, axes=(1, 0)).T
                b = np.tensordot(D2_de, u, axes=(1, 0)).T

                # divide each "row" by dt_dptau or dt_dptau**2
                outputs[self._output_val_names[name]] = np.tensordot(L_de, u, axes=(1, 0))
                outputs[self._output_rate_names[name]] = (a / dt_dptau).T
                outputs[self._output_rate2_names[name]] = (b / dt_dptau ** 2).T
            else:
                size = np.prod(options['shape'])

                u_flat = np.reshape(inputs[self._input_names[name]],
                                    (num_control_input_nodes, size))

                a = self.D.dot(u_flat)  # du/dstau
                b = self.D2.dot(u_flat) # d2u/dstau2

                val = np.reshape(self.L.dot(u_flat), (num_output_nodes,) + options['shape'])

                rate = a / inputs['dt_dstau'][:, np.newaxis]
                rate = np.reshape(rate, (num_output_nodes,) + options['shape'])

                rate2 = b / inputs['dt_dstau'][:, np.newaxis] ** 2
                rate2 = np.reshape(rate2, (num_output_nodes,) + options['shape'])

                outputs[self._output_val_names[name]] = val
                outputs[self._output_rate_names[name]] = rate
                outputs[self._output_rate2_names[name]] = rate2

                if options['continuity'] and gd.num_segments > 1 and not self.options['compressed']:
                    # output_name = self._output_val_names[name]
                    cnty_output_name = self._output_cnty_defect_names[name]
                    segment_end_values = val[seg_end_idxs, ...]
                    end_vals = segment_end_values[1:-1:2, ...]
                    start_vals = segment_end_values[2:-1:2, ...]
                    outputs[cnty_output_name] = start_vals - end_vals

                if options['rate_continuity'] and gd.num_segments > 1 and not self.options['compressed']:
                    # output_name = self._output_rate_names[name]
                    rate_in_ptau = a / dptau_dstau[:, np.newaxis]
                    rate_in_ptau = np.reshape(rate_in_ptau, (num_output_nodes,) + options['shape'])

                    cnty_output_name = self._output_rate_cnty_defect_names[name]
                    segment_end_values = rate_in_ptau[seg_end_idxs, ...]
                    end_vals = segment_end_values[1:-1:2, ...]
                    start_vals = segment_end_values[2:-1:2, ...]
                    outputs[cnty_output_name] = (start_vals - end_vals)

                if options['rate2_continuity'] and gd.num_segments > 1 and not self.options['compressed']:
                    # output_name  = self._output_rate2_names[name]
                    cnty_output_name = self._output_rate2_cnty_defect_names[name]
                    segment_end_values = rate2[seg_end_idxs, ...]
                    end_vals = segment_end_values[1:-1:2, ...]
                    start_vals = segment_end_values[2:-1:2, ...]
                    outputs[cnty_output_name] = (start_vals - end_vals) #* dt_dptau ** 2

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

        dstau_dt = np.reciprocal(inputs['dt_dstau'])
        dstau_dt2 = (dstau_dt ** 2)[:, np.newaxis]
        dstau_dt3 = (dstau_dt ** 3)[:, np.newaxis]

        t_duration = inputs['t_duration']

        for name in control_options:
            if control_options[name]['control_type'] == 'polynomial':
                options = control_options[name]
                control_name = self._input_names[name]
                num_input_nodes = options['order'] + 1
                L_de, D_de, D2_de = self._matrices[name]
                nn = self.options['output_grid_data'].num_nodes

                size = self.sizes[name]
                rate_name = self._output_rate_names[name]
                rate2_name = self._output_rate2_names[name]

                # Unroll matrix-shaped controls into an array at each node
                u_d = np.reshape(inputs[control_name], (num_input_nodes, size))

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
            else:
                control_name = self._input_names[name]

                size = self.sizes[name]
                rate_name = self._output_rate_names[name]
                rate2_name = self._output_rate2_names[name]
                num_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']

                # Unroll shaped controls into an array at each node
                u_flat = np.reshape(inputs[control_name], (num_input_nodes, size))

                partials[rate_name, 'dt_dstau'] = (-self.D.dot(u_flat) * dstau_dt2).ravel()
                partials[rate2_name, 'dt_dstau'] = (-2.0 * self.D2.dot(u_flat) * dstau_dt3).ravel()

                dstau_dt_x_size = np.repeat(dstau_dt, size)[:, np.newaxis]
                dstau_dt2_x_size = np.repeat(dstau_dt2, size)[:, np.newaxis]

                partials[rate_name, control_name] = self.rate_jacs[name].multiply(dstau_dt_x_size).data

                partials[rate2_name, control_name] = self.rate2_jacs[name].multiply(dstau_dt2_x_size).data