from functools import lru_cache
import numpy as np
import scipy.sparse as sp
from scipy.linalg import block_diag

import openmdao.api as om
from openmdao.utils.general_utils import determine_adder_scaler

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units, CoerceDesvar, reshape_val
from dymos.utils.lgl import lgl
from dymos.utils.lagrange import lagrange_matrices
from dymos.utils.indexing import get_desvar_indices
from dymos.utils.constants import INF_BOUND
from dymos._options import options as dymos_options


class ControlInterpComp(om.ExplicitComponent):
    """
    Class definition for the ControlInterpComp.

    Compute the approximated control values and rates given the values of a control at all nodes,
    given values at the control discretization nodes.

    This component handles both polynomial controls and "full" controls.

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
        self.options.declare('control_options', types=dict,
                             desc='Dictionary of options for the dynamic controls')
        self.options.declare('time_units', default=None, allow_none=True, types=str,
                             desc='Units of time')
        self.options.declare('grid_data', types=GridData,
                             desc='Container object for grid info for the control inputs.')
        self.options.declare('output_grid_data', types=GridData, allow_none=True, default=None,
                             desc='GridData object for the output grid. If None, use the same grid_data as the inputs.')
        self.options.declare('compute_continuity', types=bool, default=False,
                             desc='Switch to enable calculation of segment continuity if necessary.')
        self.options.declare('enforce_continuity', types=bool, default=False,
                             desc='Switch to enable constraint of segment continuity.')

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_val_names = {}
        self._output_rate_names = {}
        self._output_rate2_names = {}
        self._output_boundary_val_names = {}
        self._output_boundary_rate_names = {}
        self._output_boundary_rate2_names = {}
        self._output_val_cnty_defect_names = {}
        self._output_rate_cnty_defect_names = {}
        self._output_rate2_cnty_defect_names = {}
        self._matrices = {}  # Interpolation, differentiation, selection matrices
        self._dcnty_dnode_vals_kron_eye = {}  # Used in partials

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

    @lru_cache
    def _is_val_cnty(self, control_name):
        """
        Return a flag indicating if control value continuity should be computed.

        Parameters
        ----------
        control_name : str
            The name of the control.

        Returns
        -------
        bool
            True if value continuity should be computed.
        """
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        options = self.options['control_options'][control_name]
        return (options['control_type'] == 'full' and
                (gd.grid_type == 'lgr' or not gd.compressed) and
                ogd.num_segments > 1 and
                options['continuity'] and
                self.options['compute_continuity'])

    @lru_cache
    def _is_rate_cnty(self, control_name):
        """
        Return a flag indicating if control rate continuity should be computed.

        Parameters
        ----------
        control_name : str
            The name of the control.

        Returns
        -------
        bool
            True if rate continuity should be computed.
        """
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        options = self.options['control_options'][control_name]
        return (options['control_type'] == 'full' and
                ogd.num_segments > 1 and
                options['rate_continuity'] and
                self.options['compute_continuity'])

    @lru_cache
    def _is_rate2_cnty(self, control_name):
        """
        Return a flag indicating if control rate2 continuity should be computed.

        Parameters
        ----------
        control_name : str
            The name of the control.

        Returns
        -------
        bool
            True if rate2 continuity should be computed.
        """
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        options = self.options['control_options'][control_name]
        return (options['control_type'] == 'full' and
                ogd.num_segments > 1 and
                options['rate2_continuity'] and
                self.options['compute_continuity'])

    def _configure_controls(self):
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        eval_nodes = ogd.node_ptau
        control_options = self.options['control_options']
        num_input_nodes = gd.subset_num_nodes['control_input']
        num_output_nodes = ogd.num_nodes
        num_output_segs = ogd.num_segments

        for name, options in control_options.items():
            if 'control_type' not in options:
                options['control_type'] = 'full'

            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            output_shape = (num_output_nodes,) + shape

            rate_units = get_rate_units(units, self.options['time_units'], deriv=1)
            rate2_units = get_rate_units(units, self.options['time_units'], deriv=2)

            self._input_names[name] = f'controls:{name}'

            self._output_val_names[name] = f'control_values:{name}'
            self._output_rate_names[name] = f'control_rates:{name}_rate'
            self._output_rate2_names[name] = f'control_rates:{name}_rate2'

            self._output_boundary_val_names[name] = f'control_boundary_values:{name}'
            self._output_boundary_rate_names[name] = f'control_boundary_rates:{name}_rate'
            self._output_boundary_rate2_names[name] = f'control_boundary_rates:{name}_rate2'

            self._output_val_cnty_defect_names[name] = f'control_continuity_defects:{name}'
            self._output_rate_cnty_defect_names[name] = f'control_rate_continuity_defects:{name}'
            self._output_rate2_cnty_defect_names[name] = f'control_rate2_continuity_defects:{name}'

            self.add_output(self._output_val_names[name], shape=output_shape, units=units)
            self.add_output(self._output_rate_names[name], shape=output_shape, units=rate_units)
            self.add_output(self._output_rate2_names[name], shape=output_shape, units=rate2_units)

            self.add_output(self._output_boundary_val_names[name], shape=(2,) + shape, units=units)
            self.add_output(self._output_boundary_rate_names[name], shape=(2,) + shape, units=rate_units)
            self.add_output(self._output_boundary_rate2_names[name], shape=(2,) + shape, units=rate2_units)

            if self._is_val_cnty(name):
                self.add_output(self._output_val_cnty_defect_names[name],
                                shape=(num_output_segs - 1,) + shape,
                                units=units)

            if self._is_rate_cnty(name):
                self.add_output(self._output_rate_cnty_defect_names[name],
                                shape=(num_output_segs - 1,) + shape,
                                units=rate_units)

            if self._is_rate2_cnty(name):
                self.add_output(self._output_rate2_cnty_defect_names[name],
                                shape=(num_output_segs - 1,) + shape,
                                units=rate2_units)

            if options['control_type'] == 'polynomial':
                num_input_nodes = options['order'] + 1
                disc_nodes, _ = lgl(num_input_nodes)
                num_control_input_nodes = len(disc_nodes)
                default_val = reshape_val(options['val'], shape, num_input_nodes)

                L_de, D_de = lagrange_matrices(disc_nodes, eval_nodes)
                _, D_dd = lagrange_matrices(disc_nodes, disc_nodes)
                D2_de = np.dot(D_de, D_dd)

                if options['order'] not in self._matrices:
                    self._matrices[options['order']] = L_de, D_de, D2_de

                self.add_input(self._input_names[name], val=default_val, units=units)

                rs, cs, vals = sp.find(sp.kron(L_de, sp.eye(size), format='csr'))
                self.declare_partials(of=self._output_val_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs, val=vals)

                J_val = sp.kron(L_de[[0, -1], ...], sp.eye(size), format='csr')
                rs, cs, data = sp.find(J_val)
                self.declare_partials(of=self._output_boundary_val_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs, val=data)

                rs = np.concatenate([np.arange(0, num_output_nodes * size, size, dtype=int) + i
                                    for i in range(size)])

                self.declare_partials(of=self._output_rate_names[name],
                                      wrt='t_duration', rows=rs, cols=np.zeros_like(rs))

                self.declare_partials(of=self._output_rate2_names[name],
                                      wrt='t_duration', rows=rs, cols=np.zeros_like(rs))

                rs = np.concatenate([np.arange(0, 2 * size, size, dtype=int) + i
                                    for i in range(size)])

                self.declare_partials(of=self._output_boundary_rate_names[name],
                                      wrt='t_duration', rows=rs, cols=np.zeros_like(rs))

                self.declare_partials(of=self._output_boundary_rate2_names[name],
                                      wrt='t_duration', rows=rs, cols=np.zeros_like(rs))

                rs, cs, vals = sp.find(sp.kron(D_de, sp.eye(size), format='csr'))
                self.declare_partials(of=self._output_rate_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

                rs, cs, vals = sp.find(sp.kron(D_de[[0, -1], ...], sp.eye(size), format='csr'))
                self.declare_partials(of=self._output_boundary_rate_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

                rs, cs, vals = sp.find(sp.kron(D2_de, sp.eye(size), format='csr'))
                self.declare_partials(of=self._output_rate2_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

                rs, cs, vals = sp.find(sp.kron(D2_de[[0, -1], ...], sp.eye(size), format='csr'))
                self.declare_partials(of=self._output_boundary_rate2_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

            else:
                L_de, D_de, D2_de, S = self._matrices['full']
                num_control_input_nodes = gd.subset_num_nodes['control_input']
                default_val = reshape_val(options['val'], shape, num_control_input_nodes)

                self.add_input(self._input_names[name], val=default_val, units=units)

                sp_eye = sp.eye(size, format='csr')

                # The partial of interpolated value wrt the control input values is linear
                # and can be computed as the kronecker product of the interpolation matrix (L)
                # and eye(size).
                d_ua_d_uin = sp.kron(L_de, sp_eye, format='csr')
                rs, cs, data = sp.find(d_ua_d_uin)
                self.declare_partials(of=self._output_val_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs, val=data)

                J_val = sp.kron(L_de[[0, -1], ...], sp_eye, format='csr')
                rs, cs, data = sp.find(J_val)
                self.declare_partials(of=self._output_boundary_val_names[name],
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

                self.declare_partials(of=self._output_boundary_rate_names[name],
                                      wrt='dt_dstau',
                                      rows=np.arange(2 * size, dtype=int),
                                      cols=np.concatenate((cs[:size], cs[-size:])))

                self.declare_partials(of=self._output_boundary_rate2_names[name],
                                      wrt='dt_dstau',
                                      rows=np.arange(2 * size, dtype=int),
                                      cols=np.concatenate((cs[:size], cs[-size:])))

                # The val continuity defects are dependent upon the control input values.
                # The calculation of the continuity defects can be expressed as
                # val_defects = ([S]@[U_a]) @ 1_n
                # where S is the defect selection matrix and 1_n is a column vector of ones at each
                # segment start/end involved in the calculation.
                # This is used for derivatives wrt dt_dstau
                self._d_cnty_d_node_vals = S.dot(sp.eye(num_output_nodes))

                # This is used for derivatives wrt to sized variables
                dcmat = self._dcnty_dnode_vals_kron_eye[size] = sp.kron(self._d_cnty_d_node_vals, sp_eye, format='csr')

                if self._is_val_cnty(name):
                    d_val_cnty_d_uin = dcmat.dot(d_ua_d_uin)
                    rs, cs, data = sp.find(d_val_cnty_d_uin)

                    self.declare_partials(of=self._output_val_cnty_defect_names[name],
                                          wrt=self._input_names[name],
                                          rows=rs, cols=cs,
                                          val=data)

                if self._is_rate_cnty(name):
                    rs, cs, data = sp.find(sp.kron(self._d_cnty_d_node_vals, np.ones((size, 1))))
                    self.declare_partials(of=self._output_rate_cnty_defect_names[name],
                                          wrt='dt_dstau',
                                          rows=rs, cols=cs)

                    d_urate_d_uin = sp.kron(D_de, sp_eye, format='csr')
                    d_rate_cnty_d_uin = dcmat.dot(d_urate_d_uin)
                    rs, cs, data = sp.find(d_rate_cnty_d_uin)

                    self.declare_partials(of=self._output_rate_cnty_defect_names[name],
                                          wrt=self._input_names[name],
                                          rows=rs, cols=cs,
                                          val=1.0)

                if self._is_rate2_cnty(name):
                    rs, cs, data = sp.find(sp.kron(self._d_cnty_d_node_vals, np.ones((size, 1))))
                    self.declare_partials(of=self._output_rate2_cnty_defect_names[name],
                                          wrt='dt_dstau',
                                          rows=rs, cols=cs)

                    d_urate2_d_uin = sp.kron(D2_de, sp_eye, format='csr')
                    d_rate2_cnty_d_uin = dcmat.dot(d_urate2_d_uin)
                    rs, cs, data = sp.find(d_rate2_cnty_d_uin)

                    self.declare_partials(of=self._output_rate2_cnty_defect_names[name],
                                          wrt=self._input_names[name],
                                          rows=rs, cols=cs,
                                          val=1.0)

                # The partials of the rates and second derivatives are nonlinear but the sparsity
                # pattern is obtained from the kronecker product of the 1st and 2nd differentiation
                # matrices (D and D2) and eye(size).
                # self.rate_jacs[name] = sp.kron(self.D, sp_eye, format='csr')
                rs, cs = sp.kron(D_de, sp_eye, format='csr').nonzero()
                self.declare_partials(of=self._output_rate_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

                rs, cs = sp.kron(D_de[[0, -1], ...], sp_eye, format='csr').nonzero()
                self.declare_partials(of=self._output_boundary_rate_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

                rs, cs = sp.kron(D2_de, sp_eye, format='csr').nonzero()
                self.declare_partials(of=self._output_rate2_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

                rs, cs = sp.kron(D2_de[[0, -1], ...], sp_eye, format='csr').nonzero()
                self.declare_partials(of=self._output_boundary_rate2_names[name],
                                      wrt=self._input_names[name],
                                      rows=rs, cols=cs)

    def _configure_desvars(self):
        control_options = self.options['control_options']
        gd = self.options['grid_data']

        for name, options in control_options.items():
            if options['control_type'] == 'polynomial':
                num_input_nodes = options['order'] + 1
                shape = options['shape']
                if options['opt']:

                    desvar_indices = np.arange(num_input_nodes, dtype=int)
                    if options['fix_initial']:
                        desvar_indices = desvar_indices[1:]
                    if options['fix_final']:
                        desvar_indices = desvar_indices[:-1]

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

            else:
                num_input_nodes = gd.subset_num_nodes['control_input']

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

    def _configure_constraints(self):
        if self.options['enforce_continuity']:
            control_options = self.options['control_options']
            for name, options in control_options.items():
                if self._is_val_cnty(name):
                    _, scaler = determine_adder_scaler(None if options['continuity_ref'] is None else 0.0,
                                                       options['continuity_ref'],
                                                       None if options['continuity_scaler'] is None else 0.0,
                                                       options['continuity_scaler'])
                    val_cnty_name = self._output_val_cnty_defect_names[name]
                    self.add_constraint(val_cnty_name, equals=0.0, scaler=scaler)

                if self._is_rate_cnty(name):
                    _, scaler = determine_adder_scaler(None if options['rate_continuity_ref'] is None else 0.0,
                                                       options['rate_continuity_ref'],
                                                       None if options['rate_continuity_scaler'] is None else 0.0,
                                                       options['rate_continuity_scaler'])
                    rate_cnty_name = self._output_rate_cnty_defect_names[name]
                    self.add_constraint(rate_cnty_name, equals=0.0, scaler=scaler)

                if self._is_rate2_cnty(name):
                    _, scaler = determine_adder_scaler(None if options['rate2_continuity_ref'] is None else 0.0,
                                                       options['rate2_continuity_ref'],
                                                       None if options['rate2_continuity_scaler'] is None else 0.0,
                                                       options['rate2_continuity_scaler'])
                    rate2_cnty_name = self._output_rate2_cnty_defect_names[name]
                    self.add_constraint(rate2_cnty_name, equals=0.0, scaler=options['rate2_continuity_scaler'])

    def configure_io(self):
        """
        I/O creation is delayed until configure so we can determine shape and units for the controls.
        """
        time_units = self.options['time_units']
        gd = self.options['grid_data']
        ogd = self.options['output_grid_data'] or gd
        output_num_nodes = ogd.subset_num_nodes['all']
        num_seg = gd.num_segments
        output_num_seg = ogd.num_segments

        self.add_input('dt_dstau', shape=output_num_nodes, units=time_units)

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

        L = L_da.dot(L_id)
        D = D_da.dot(L_id)

        # Matrix D_dd interpolates rates at discretization nodes from values given at control
        # discretization nodes.
        _, D_dd = gd.phase_lagrange_matrices('control_disc', 'control_disc', sparse=True)

        # Matrix D2 provides second derivatives at all nodes given values at input nodes.
        D2 = D_da.dot(D_dd.dot(L_id))

        # Matrix S provides a selection matrix_ that selects the segment start and end rows for
        # the continuity defects.
        seg_start_idxs = ogd.subset_node_indices['segment_ends'][2:-1:2]
        seg_end_idxs = ogd.subset_node_indices['segment_ends'][1:-1:2]
        selected_rows = sorted(seg_start_idxs.tolist() + seg_end_idxs.tolist())  # the rows being selected
        signs = [-1, 1] * (ogd.num_segments - 1)
        S = sp.coo_matrix((signs, (np.repeat(np.arange(output_num_seg - 1, dtype=int), 2), selected_rows)),
                          shape=(output_num_seg - 1, output_num_nodes))

        self._matrices['full'] = L, D, D2, S

        self._configure_controls()
        self._configure_desvars()
        self._configure_constraints()

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
        num_output_nodes = ogd.num_nodes

        for name, options in self.options['control_options'].items():
            if options['control_type'] == 'polynomial':
                L_de, D_de, D2_de = self._matrices[options['order']]
                num_control_input_nodes = options['order'] + 1
                dt_dtau = 0.5 * inputs['t_duration']
            else:
                L_de, D_de, D2_de, S = self._matrices['full']
                num_control_input_nodes = gd.subset_num_nodes['control_input']
                dt_dtau = inputs['dt_dstau'][:, np.newaxis]

            shape = options['shape']
            size = np.prod(shape)
            u_flat = np.reshape(inputs[self._input_names[name]],
                                (num_control_input_nodes, size))

            val_flat = L_de.dot(u_flat)
            val = np.reshape(val_flat, (num_output_nodes,) + shape)

            rate_flat = D_de.dot(u_flat) / dt_dtau
            rate = np.reshape(rate_flat, (num_output_nodes,) + shape)

            rate2_flat = D2_de.dot(u_flat) / dt_dtau ** 2
            rate2 = np.reshape(rate2_flat, (num_output_nodes,) + shape)

            outputs[self._output_val_names[name]] = val
            outputs[self._output_rate_names[name]] = rate
            outputs[self._output_rate2_names[name]] = rate2

            outputs[self._output_boundary_val_names[name]] = val[[0, -1], ...]
            outputs[self._output_boundary_rate_names[name]] = rate[[0, -1], ...]
            outputs[self._output_boundary_rate2_names[name]] = rate2[[0, -1], ...]

            if self._is_val_cnty(name):
                outputs[self._output_val_cnty_defect_names[name]] = S.dot(val_flat)
            if self._is_rate_cnty(name):
                outputs[self._output_rate_cnty_defect_names[name]] = S.dot(rate_flat)
            if self._is_rate2_cnty(name):
                outputs[self._output_rate2_cnty_defect_names[name]] = S.dot(rate2_flat)

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

        for name, options in control_options.items():
            control_type = options['control_type']
            size = np.prod(options['shape'])
            control_name = self._input_names[name]
            rate_name = self._output_rate_names[name]
            rate2_name = self._output_rate2_names[name]
            boundary_rate_name = self._output_boundary_rate_names[name]
            boundary_rate2_name = self._output_boundary_rate2_names[name]
            rate_cnty_name = self._output_rate_cnty_defect_names[name]
            rate2_cnty_name = self._output_rate2_cnty_defect_names[name]

            sp_eye = sp.eye(size)

            if options['control_type'] == 'polynomial':
                _, D_de, D2_de = self._matrices[options['order']]
                num_control_input_nodes = options['order'] + 1
                dt_dtau = 0.5 * inputs['t_duration']
            else:
                _, D_de, D2_de, S = self._matrices['full']
                num_control_input_nodes = self.options['grid_data'].subset_num_nodes['control_input']
                dt_dtau = inputs['dt_dstau']

            u_flat = np.reshape(inputs[control_name], (num_control_input_nodes, size))

            dtau_dt = np.reciprocal(dt_dtau)
            dtau_dt2 = (dtau_dt ** 2)
            dtau_dt3 = (dtau_dt ** 3)

            d_udot_ddt_dtau = -D_de.dot(u_flat) * dtau_dt2[:, np.newaxis]
            d_udotdot_ddt_dtau = -2.0 * (D2_de.dot(u_flat) * dtau_dt3[:, np.newaxis])

            drate_duin = sp.kron(D_de, sp_eye, format='csr').multiply(
                np.repeat(dtau_dt.ravel(), size)[:, np.newaxis])
            partials[rate_name, control_name] = drate_duin.data

            drate2_duin = sp.kron(D2_de, sp_eye, format='csr').multiply(
                np.repeat(dtau_dt2.ravel(), size)[:, np.newaxis])
            partials[rate2_name, control_name] = drate2_duin.data

            if control_type == 'polynomial':
                partials[rate_name, 't_duration'] = 0.5 * d_udot_ddt_dtau.ravel()
                partials[rate2_name, 't_duration'] = 0.5 * d_udotdot_ddt_dtau.ravel()

                partials[boundary_rate_name, 't_duration'][:size] = partials[rate_name, 't_duration'][:size]
                partials[boundary_rate_name, 't_duration'][-size:] = partials[rate_name, 't_duration'][-size:]

                partials[boundary_rate2_name, 't_duration'][:size] = partials[rate2_name, 't_duration'][:size]
                partials[boundary_rate2_name, 't_duration'][-size:] = partials[rate2_name, 't_duration'][-size:]
            else:
                partials[rate_name, 'dt_dstau'] = d_udot_ddt_dtau.ravel()
                partials[rate2_name, 'dt_dstau'] = d_udotdot_ddt_dtau.ravel()

                partials[boundary_rate_name, 'dt_dstau'][:size] = partials[rate_name, 'dt_dstau'][:size]
                partials[boundary_rate_name, 'dt_dstau'][-size:] = partials[rate_name, 'dt_dstau'][-size:]

                partials[boundary_rate2_name, 'dt_dstau'][:size] = partials[rate2_name, 'dt_dstau'][:size]
                partials[boundary_rate2_name, 'dt_dstau'][-size:] = partials[rate2_name, 'dt_dstau'][-size:]

                if self._is_rate_cnty(name):
                    Srs, Scs = S.nonzero()
                    num_output_seg_minus_one = S.shape[0]

                    rs = np.reshape(Scs, (-1, 2))
                    rs = np.repeat(rs, size, axis=0)
                    rs = rs.ravel()

                    cs = np.tile(np.repeat(np.arange(size, dtype=int), 2), num_output_seg_minus_one)

                    partials[rate_cnty_name, 'dt_dstau'] = d_udot_ddt_dtau[rs, cs]
                    partials[rate_cnty_name, 'dt_dstau'][::2] *= -1

                    dcmat = self._dcnty_dnode_vals_kron_eye[size]
                    result = dcmat.dot(drate_duin)
                    result.sort_indices()
                    partials[rate_cnty_name, control_name] = result.data

                if self._is_rate2_cnty(name):
                    Srs, Scs = S.nonzero()
                    num_output_seg_minus_one = S.shape[0]

                    rs = np.reshape(Scs, (-1, 2))
                    rs = np.repeat(rs, size, axis=0)
                    rs = rs.ravel()

                    cs = np.tile(np.repeat(np.arange(size, dtype=int), 2), num_output_seg_minus_one)

                    partials[rate2_cnty_name, 'dt_dstau'] = d_udotdot_ddt_dtau[rs, cs]
                    partials[rate2_cnty_name, 'dt_dstau'][::2] *= -1

                    dcmat = self._dcnty_dnode_vals_kron_eye[size]
                    result = dcmat.dot(drate2_duin)
                    result.sort_indices()
                    partials[rate2_cnty_name, control_name] = result.data

            par_size = partials[boundary_rate_name, control_name].size // 2
            partials[boundary_rate_name, control_name][:par_size] = partials[rate_name, control_name][:par_size]
            partials[boundary_rate_name, control_name][-par_size:] = partials[rate_name, control_name][-par_size:]

            par_size = partials[boundary_rate2_name, control_name].size // 2
            partials[boundary_rate2_name, control_name][:par_size] = partials[rate2_name, control_name][:par_size]
            partials[boundary_rate2_name, control_name][-par_size:] = partials[rate2_name, control_name][-par_size:]
