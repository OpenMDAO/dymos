from __future__ import division, print_function, absolute_import

from collections import Iterable

from six import iteritems

import numpy as np

from openmdao.api import IndepVarComp

from ..optimizer_based.components import CollocationDefectComp, CollocationBalanceComp, StateInterpComp
from ..components import EndpointConditionsComp
from ..phase_base import PhaseBase
from ...utils.constants import INF_BOUND
from ...utils.misc import CoerceDesvar, get_rate_units
from ...utils.lagrange import lagrange_matrices
from ...utils.indexing import get_src_indices_by_row


class OptimizerBasedPhaseBase(PhaseBase):
    """
    OptimizerBasedPhaseBase serves as the base class for GaussLobattoPhase and RadauPSPhase.

    Attributes
    ----------
    self.time_options : dict of TimeOptionsDictionary
        A dictionary of options for time (integration variable) in the phase.

    self.state_options : dict of StateOptionsDictionary
        A dictionary of options for the RHS states in the Phase.

    self.control_options : dict of ControlOptionsDictionary
        A dictionary of options for the controls in the Phase.

    self._ode_controls : dict of ControlOptionsDictionary
        A dictionary of the default options for controllable inputs of the Phase RHS

    """
    def setup(self):
        super(OptimizerBasedPhaseBase, self).setup()

        transcription = self.options['transcription']

        num_opt_controls = len([name for (name, options) in iteritems(self.control_options)
                                if options['opt']])

        num_controls = len(self.control_options)

        indep_controls = ['indep_controls'] if num_opt_controls > 0 else []
        design_params = ['design_params'] if self.design_parameter_options else []
        input_params = ['input_params'] if self.input_parameter_options else []
        traj_params = ['traj_params'] if self.traj_parameter_options else []
        control_interp_comp = ['control_interp_comp'] if num_controls > 0 else []

        order = self._time_extents + indep_controls + \
            input_params + design_params + traj_params + \
            ['indep_states', 'time'] + control_interp_comp + \
            ['indep_jumps', 'initial_conditions', 'final_conditions']

        if transcription == 'gauss-lobatto':
            order = order + ['rhs_disc', 'state_interp', 'rhs_col', 'collocation_constraint']
        elif transcription == 'radau-ps':
            order = order + ['state_interp', 'rhs_all', 'collocation_constraint']
        else:
            raise ValueError('Invalid transcription: {0}'.format(transcription))

        if self.grid_data.num_segments > 1:
            order.append('continuity_comp')
        if self._initial_boundary_constraints:
            order.append('initial_boundary_constraints')
        if self._final_boundary_constraints:
            order.append('final_boundary_constraints')
        if getattr(self, 'path_constraints', None) is not None:
            order.append('path_constraints')

        order.append('timeseries')

        self.set_order(order)

    def _setup_rhs(self):
        grid_data = self.grid_data
        time_units = self.time_options['units']
        map_input_indices_to_disc = self.grid_data.input_maps['state_input_to_disc']
        num_input_nodes = self.grid_data.subset_num_nodes['state_input']

        self.add_subsystem('state_interp',
                           subsys=StateInterpComp(grid_data=grid_data,
                                                  state_options=self.state_options,
                                                  time_units=time_units,
                                                  transcription=self.options['transcription']))

        self.connect(
            'time.dt_dstau', 'state_interp.dt_dstau',
            src_indices=grid_data.subset_node_indices['col'])

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])

            src_idxs_mat = np.reshape(np.arange(size * num_input_nodes, dtype=int),
                                      (num_input_nodes, size), order='C')

            src_idxs = src_idxs_mat[map_input_indices_to_disc, :]

            self.connect('states:{0}'.format(name),
                         'state_interp.state_disc:{0}'.format(name),
                         src_indices=src_idxs, flat_src_indices=True)

    def _setup_states(self):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']

        indep = IndepVarComp()
        for name, options in iteritems(self.state_options):
            indep.add_output(name='states:{0}'.format(name),
                             shape=(num_state_input_nodes, np.prod(options['shape'])),
                             units=options['units'])
        self.add_subsystem('indep_states', indep, promotes_outputs=['*'])

        for name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            if options['opt']:
                desvar_indices = list(range(size * num_state_input_nodes))

                if options['fix_initial']:
                    if options['initial_bounds'] is not None:
                        raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                         'initial_bounds for state {0}'.format(name))
                    if isinstance(options['fix_initial'], Iterable):
                        idxs_to_fix = np.where(np.asarray(options['fix_initial']))[0]
                        for idx_to_fix in reversed(sorted(idxs_to_fix)):
                            del desvar_indices[idx_to_fix]
                    else:
                        del desvar_indices[:size]
                if options['fix_final']:
                    if options['final_bounds'] is not None:
                        raise ValueError('Cannot specify \'fix_final=True\' and specify '
                                         'final_bounds for state {0}'.format(name))
                    if isinstance(options['fix_final'], Iterable):
                        idxs_to_fix = np.where(np.asarray(options['fix_final']))[0]
                        for idx_to_fix in reversed(sorted(idxs_to_fix)):
                            del desvar_indices[-size + idx_to_fix]
                    else:
                        del desvar_indices[-size:]

                if len(desvar_indices) > 0:
                    coerce_desvar_option = CoerceDesvar(num_state_input_nodes, desvar_indices,
                                                        options)

                    lb = np.zeros_like(desvar_indices, dtype=float)
                    lb[:] = -INF_BOUND if coerce_desvar_option('lower') is None else \
                        coerce_desvar_option('lower')

                    ub = np.zeros_like(desvar_indices, dtype=float)
                    ub[:] = INF_BOUND if coerce_desvar_option('upper') is None else \
                        coerce_desvar_option('upper')

                    if options['initial_bounds'] is not None:
                        lb[0] = options['initial_bounds'][0]
                        ub[0] = options['initial_bounds'][-1]

                    if options['final_bounds'] is not None:
                        lb[-1] = options['final_bounds'][0]
                        ub[-1] = options['final_bounds'][-1]

                    self.add_design_var(name='states:{0}'.format(name),
                                        lower=lb,
                                        upper=ub,
                                        scaler=coerce_desvar_option('scaler'),
                                        adder=coerce_desvar_option('adder'),
                                        ref0=coerce_desvar_option('ref0'),
                                        ref=coerce_desvar_option('ref'),
                                        indices=desvar_indices)

    def _setup_defects(self):
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        time_units = self.time_options['units']

        self.add_subsystem('collocation_constraint',
                           CollocationDefectComp(grid_data=grid_data,
                                                 state_options=self.state_options,
                                                 time_units=time_units))

        self.connect('time.dt_dstau', ('collocation_constraint.dt_dstau'),
                     src_indices=grid_data.subset_node_indices['col'])

        # Add the continuity constraint component if necessary
        if num_seg > 1:
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            state_disc_idxs = grid_data.subset_node_indices['state_disc']

            if not self.options['compressed']:
                state_input_subidxs = np.where(np.in1d(state_disc_idxs, segment_end_idxs))[0]

                for name, options in iteritems(self.state_options):
                    shape = options['shape']
                    flattened_src_idxs = get_src_indices_by_row(state_input_subidxs, shape=shape,
                                                                flat=True)
                    self.connect('states:{0}'.format(name),
                                 'continuity_comp.states:{}'.format(name),
                                 src_indices=flattened_src_idxs, flat_src_indices=True)

            for name, options in iteritems(self.control_options):
                control_src_name = 'control_interp_comp.control_values:{0}'.format(name)

                # The sub-indices of control_disc indices that are segment ends
                segment_end_idxs = grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(segment_end_idxs, options['shape'], flat=True)

                self.connect(control_src_name,
                             'continuity_comp.controls:{0}'.format(name),
                             src_indices=src_idxs, flat_src_indices=True)

                self.connect('control_rates:{0}_rate'.format(name),
                             'continuity_comp.control_rates:{}_rate'.format(name),
                             src_indices=src_idxs, flat_src_indices=True)

                self.connect('control_rates:{0}_rate2'.format(name),
                             'continuity_comp.control_rates:{}_rate2'.format(name),
                             src_indices=src_idxs, flat_src_indices=True)

    def _setup_endpoint_conditions(self):

        jump_comp = self.add_subsystem('indep_jumps', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        jump_comp.add_output('initial_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the start of the phase')

        jump_comp.add_output('final_jump:time', val=0.0, units=self.time_options['units'],
                             desc='discontinuity in time at the end of the phase')

        ic_comp = EndpointConditionsComp(loc='initial',
                                         time_options=self.time_options,
                                         state_options=self.state_options,
                                         control_options=self.control_options)

        self.add_subsystem(name='initial_conditions', subsys=ic_comp, promotes_outputs=['*'])

        fc_comp = EndpointConditionsComp(loc='final',
                                         time_options=self.time_options,
                                         state_options=self.state_options,
                                         control_options=self.control_options)

        self.add_subsystem(name='final_conditions', subsys=fc_comp, promotes_outputs=['*'])

        self.connect('time', 'initial_conditions.initial_value:time')
        self.connect('time', 'final_conditions.final_value:time')

        self.connect('initial_jump:time',
                     'initial_conditions.initial_jump:time')

        self.connect('final_jump:time',
                     'final_conditions.final_jump:time')

        for state_name, options in iteritems(self.state_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            jump_comp.add_output('initial_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(state_name))

            jump_comp.add_output('final_jump:{0}'.format(state_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(state_name))

            self.connect('states:{0}'.format(state_name),
                         'initial_conditions.initial_value:{0}'.format(state_name))
            self.connect('states:{0}'.format(state_name),
                         'final_conditions.final_value:{0}'.format(state_name))

            self.connect('initial_jump:{0}'.format(state_name),
                         'initial_conditions.initial_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(state_name),
                         'final_conditions.final_jump:{0}'.format(state_name),
                         src_indices=ar, flat_src_indices=True)

        for control_name, options in iteritems(self.control_options):
            size = np.prod(options['shape'])
            ar = np.arange(size)

            jump_comp.add_output('initial_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'start of the phase'.format(control_name))

            jump_comp.add_output('final_jump:{0}'.format(control_name),
                                 val=np.zeros(options['shape']),
                                 units=options['units'],
                                 desc='discontinuity in {0} at the '
                                      'end of the phase'.format(control_name))

            self.connect('control_interp_comp.control_values:{0}'.format(control_name),
                         'initial_conditions.initial_value:{0}'.format(control_name))

            self.connect('control_interp_comp.control_values:{0}'.format(control_name),
                         'final_conditions.final_value:{0}'.format(control_name))

            self.connect('initial_jump:{0}'.format(control_name),
                         'initial_conditions.initial_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

            self.connect('final_jump:{0}'.format(control_name),
                         'final_conditions.final_jump:{0}'.format(control_name),
                         src_indices=ar, flat_src_indices=True)

    def _get_boundary_constraint_src(self, var, loc):
        # Determine the path to the variable which we will be constraining
        time_units = self.time_options['units']
        var_type = self._classify_var(var)

        if var_type == 'time':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 'time'
        elif var_type == 'time_phase':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 'time_phase'
        elif var_type == 'state':
            state_shape = self.state_options[var]['shape']
            state_units = self.state_options[var]['units']
            shape = state_shape
            units = state_units
            linear = True
            constraint_path = 'states:{0}'.format(var)
        elif var_type in 'indep_control':
            control_shape = self.control_options[var]['shape']
            control_units = self.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'control_interp_comp.control_values:{0}'.format(var)
        elif var_type == 'input_control':
            control_shape = self.control_options[var]['shape']
            control_units = self.control_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'control_interp_comp.control_values:{0}'.format(var)
        elif var_type == 'design_parameter':
            control_shape = self.design_parameter_options[var]['shape']
            control_units = self.design_parameter_options[var]['units']
            shape = control_shape
            units = control_units
            linear = True
            constraint_path = 'design_parameters:{0}'.format(var)
        elif var_type == 'input_parameter':
            control_shape = self.input_parameter_options[var]['shape']
            control_units = self.input_parameter_options[var]['units']
            shape = control_shape
            units = control_units
            linear = False
            constraint_path = 'input_parameters:{0}_out'.format(var)
        elif var_type == 'control_rate':
            control_var = var[:-5]
            control_shape = self.control_options[control_var]['shape']
            control_units = self.control_options[control_var]['units']
            control_rate_units = get_rate_units(control_units, time_units, deriv=1)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        elif var_type == 'control_rate2':
            control_var = var[:-6]
            control_shape = self.control_options[control_var]['shape']
            control_units = self.control_options[control_var]['units']
            control_rate_units = get_rate_units(control_units, time_units, deriv=2)
            shape = control_shape
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        else:
            # Failed to find variable, assume it is in the RHS
            if self.options['transcription'] == 'gauss-lobatto':
                constraint_path = 'rhs_disc.{0}'.format(var)
            elif self.options['transcription'] == 'radau-ps':
                constraint_path = 'rhs_all.{0}'.format(var)
            else:
                raise ValueError('Invalid transcription')
            # TODO: Account for non-scalar variables here.
            shape = (1,)
            units = None
            linear = False

        return constraint_path, shape, units, linear

    def interpolate_values(self, var, times):
        """
        Interpolate a variable to the requested times within the phase, using the appropriate
        polynomial basis for each segment.

        Parameters
        ----------
        var : str
            The variable whose solution is to be interpolated.
        times : sequence
            The increasing-ordered times at which an interpolation of the given
            solution is requested. If the times fall beyond the extents of the phase,
            smooth extrapolation will be performed.

        Returns
        -------
        vals : numpy.ndarray
            Interpolated value of the specified variable at the given times.

        """
        gd = self.grid_data

        vals_all = self.get_values(var, nodes='all')
        # The values of the variable at all nodes in the phase

        times_all = self.get_values('time', nodes='all')
        # The values of time at all nodes in the phase

        seg_bins = np.unique(times_all[gd.subset_node_indices['segment_ends']])
        # An array of segment end times which provide bins for the given segments

        time_segs = np.clip(np.digitize(times, seg_bins) - 1, 0, gd.num_segments-1)
        # The segment into which each time in the given times array falls

        res = np.atleast_2d(np.zeros_like(times)).T

        for iseg in range(gd.num_segments):
            i1, i2 = gd.subset_segment_indices['all'][iseg, :]
            seg_idxs = gd.subset_node_indices['all'][i1:i2]
            nodes_given = gd.node_stau[seg_idxs]

            times_idxs_in_seg = np.where(time_segs == iseg)[0]

            times_eval = times[times_idxs_in_seg]

            times_seg = times_all[seg_idxs]
            t0_seg = times_seg[0]
            tf_seg = times_seg[-1]

            # put times_eval on [-1, 1]
            m = 2.0 / (tf_seg - t0_seg)
            b = 1.0 - (m * tf_seg)
            nodes_eval = m * times_eval + b

            L, _ = lagrange_matrices(nodes_given, nodes_eval)

            seg_given_vals = vals_all[seg_idxs]
            seg_interp_vals = np.dot(L, seg_given_vals)

            res[times_idxs_in_seg, ...] = seg_interp_vals

        return res

    def interpolate_rates(self, var, times):
        """
        Interpolate the rate of a variable to the requested times within the phase, using
        the appropriate polynomial basis for each segment.

        Parameters
        ----------
        var : str
            The variable whose solution is to be interpolated.
        times : sequence
            The increasing-ordered times at which an interpolation of the given
            solution is requested. If the times fall beyond the extents of the phase,
            smooth extrapolation will be performed.

        Returns
        -------
        vals : numpy.ndarray
            Interpolated rate of the specified variable at the given times.

        """
        gd = self.grid_data

        vals_all = self.get_values(var, nodes='all')
        # The values of the variable at all nodes in the phase

        times_all = self.get_values('time', nodes='all')
        # The values of time at all nodes in the phase

        seg_bins = np.unique(times_all[gd.subset_node_indices['segment_ends']])
        # An array of segment end times which provide bins for the given segments

        time_segs = np.clip(np.digitize(times, seg_bins) - 1, 0, gd.num_segments-1)
        # The segment into which each time in the given times array falls

        res = np.atleast_2d(np.zeros_like(times)).T

        for iseg in range(gd.num_segments):
            i1, i2 = gd.subset_segment_indices['all'][iseg, :]
            seg_idxs = gd.subset_node_indices['all'][i1:i2]
            nodes_given = gd.node_stau[seg_idxs]

            times_idxs_in_seg = np.where(time_segs == iseg)[0]

            times_eval = times[times_idxs_in_seg]

            times_seg = times_all[seg_idxs]
            t0_seg = times_seg[0]
            tf_seg = times_seg[-1]
            dt_dstau = 0.5 * (tf_seg - t0_seg)

            # put times_eval on [-1, 1]
            m = 2.0 / (tf_seg - t0_seg)
            b = 1.0 - (m * tf_seg)
            nodes_eval = m * times_eval + b

            _, D = lagrange_matrices(nodes_given, nodes_eval)

            seg_given_vals = vals_all[seg_idxs]
            seg_interp_rates = np.dot(D, seg_given_vals) / dt_dstau

            res[times_idxs_in_seg, ...] = seg_interp_rates

        return res
