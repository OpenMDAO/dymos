from collections.abc import Iterable

import numpy as np

import openmdao.api as om
from openmdao.utils.general_utils import warn_deprecation
from ..transcription_base import TranscriptionBase
from ..common import TimeComp, PseudospectralTimeseriesOutputComp
from .components import StateIndependentsComp, StateInterpComp, CollocationComp
from ...utils.misc import CoerceDesvar, get_rate_units, get_source_metadata
from ...utils.constants import INF_BOUND
from ...utils.indexing import get_src_indices_by_row


class PseudospectralBase(TranscriptionBase):
    """
    Base class for the pseudospectral transcriptions.
    """
    def initialize(self):
        self.options.declare(name='solve_segments', default=False,
                             values=(True, False, 'forward', 'backward'),
                             desc='Applies \'solve_segments\' behavior to _all_ states in the Phase. '
                                  'If True (deprecated) or \'forward\', collocation defects within each '
                                  'segment are solved with a Newton solver by fixing the initial value in the '
                                  'phase (if using compressed transcription) or segment (if not using '
                                  'compressed transcription). This provides a forward shooting (or multiple shooting) '
                                  'method.  If \'backward\', the final value in the phase or segment is fixed '
                                  'and a solver finds the other ones to mimic reverse propagation. Set '
                                  'to False (the default) to explicitly disable the use of a solver to '
                                  'converge the state time history.')

    def setup_time(self, phase):
        time_units = phase.time_options['units']
        grid_data = self.grid_data

        super(PseudospectralBase, self).setup_time(phase)

        time_comp = TimeComp(num_nodes=grid_data.num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau, units=time_units)

        phase.add_subsystem('time', time_comp, promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_time(self, phase):
        super(PseudospectralBase, self).configure_time(phase)
        phase.time.configure_io()

    def setup_states(self, phase):
        """
        Add an IndepVarComp for the states and setup the states as design variables.
        """
        grid_data = self.grid_data

        self.any_solved_segs = False
        self.any_connected_opt_segs = False
        for name, options in phase.state_options.items():
            # Transcription solve_segments overrides state solve_segments if its not set
            if options['solve_segments'] is None:
                options['solve_segments'] = self.options['solve_segments']

            if options['solve_segments']:
                self.any_solved_segs = True
            elif options['connected_initial']:
                self.any_connected_opt_segs = True

        if self.any_solved_segs or self.any_connected_opt_segs:
            indep = StateIndependentsComp(grid_data=grid_data,
                                          state_options=phase.state_options)
        else:
            indep = om.IndepVarComp()

        num_connected = len([s for (s, opts) in phase.state_options.items() if opts['connected_initial']])
        prom_inputs = ['initial_states:*'] if num_connected > 0 else None
        phase.add_subsystem('indep_states', indep, promotes_inputs=prom_inputs,
                            promotes_outputs=['*'])

    def configure_states(self, phase):
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']
        indep = phase.indep_states

        # state_idx_map holds the node indices provided by the solver (solver) and those
        # that are independent variables (indep)
        self.state_idx_map = {}

        # add all the des-vars (either from the IndepVarComp or from the indep-var-like
        # outputs of the collocation comp)
        for name, options in phase.state_options.items():
            self._configure_state_introspection(name, options, phase)
            self._configure_solve_segments(name, options, phase)

            size = np.prod(options['shape'])
            # In certain cases, we put an output on the IVC.
            if isinstance(indep, om.IndepVarComp):
                indep.add_output(name='states:{0}'.format(name),
                                 shape=(num_state_input_nodes, size),
                                 units=options['units'])

            if options['opt']:
                # Add the states as design variables.
                #
                # In the case of optimizer-driven collocation, this includes the values at all
                # nodes (excluding initial and/or final if fix_initial and/or fix_final is specified).
                #
                # In the case of solve_segments == 'forward', the design var nodes are the first
                # node in the  phase (when compressed) or the first node in each each segment
                # (when not compressed).
                #
                # When fix_initial is True, the first node in the phase is then removed from the
                # design variable node indices. (So when compressed, there is no design variable).
                #
                # In the case of solve_segments == 'backward', the design var nodes are the last
                # node in the phase (when compressed) or the last node in each segment (when not
                # compressed).
                #
                # When fix_final is True, the last node in the phase is then removed from the design
                # variable node indices.
                #
                desvar_node_idxs = np.asarray(self.state_idx_map[name]['indep'])

                # This matrix will contain 1's for every index that is to be a design variable,
                # otherwise the value will be zero
                state_input_shape = (num_state_input_nodes,) + options['shape']
                idx_mask = np.zeros(state_input_shape, dtype=int)
                idx_mask[desvar_node_idxs, ...] = 1

                if options['fix_initial']:
                    if options['initial_bounds'] is not None:
                        raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                         f'initial_bounds for state {name} in phase {phase.name}')
                    if options['connected_initial']:
                        raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                         f'\'connected_initial=True\' for state {name} '
                                         f'in phase {phase.name}')
                    idx_mask[0, ...] = np.asarray(np.logical_not(options['fix_initial']), dtype=int)
                elif options['connected_initial']:
                    if options['initial_bounds'] is not None:
                        raise ValueError('Cannot specify \'connected_initial=True\' and specify '
                                         f'initial_bounds for state {name} in phase {phase.name}')
                    idx_mask[0, ...] = np.asarray(np.logical_not(options['connected_initial']), dtype=int)

                if options['fix_final']:
                    if options['final_bounds'] is not None:
                        raise ValueError('Cannot specify \'fix_final=True\' and specify '
                                         f'final_bounds for state {name}')
                    idx_mask[-1, ...] = np.asarray(np.logical_not(options['fix_final']), dtype=int)

                # Now convert the masked array into actual flat indices
                desvar_indices = np.arange(idx_mask.size, dtype=int).reshape(state_input_shape)[idx_mask.nonzero()]

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

                    phase.add_design_var(name=f'states:{name}',
                                         lower=lb,
                                         upper=ub,
                                         scaler=coerce_desvar_option('scaler'),
                                         adder=coerce_desvar_option('adder'),
                                         ref0=coerce_desvar_option('ref0'),
                                         ref=coerce_desvar_option('ref'),
                                         indices=desvar_indices)

        if isinstance(indep, StateIndependentsComp):
            indep.configure_io(self.state_idx_map)

        if self.any_solved_segs or self.any_connected_opt_segs:
            for name, options in phase.state_options.items():
                if options['solve_segments']:
                    phase.connect('collocation_constraint.defects:{0}'.format(name),
                                  'indep_states.defects:{0}'.format(name))

    def setup_ode(self, phase):
        grid_data = self.grid_data
        transcription = grid_data.transcription
        time_units = phase.time_options['units']

        phase.add_subsystem('state_interp',
                            subsys=StateInterpComp(grid_data=grid_data,
                                                   state_options=phase.state_options,
                                                   time_units=time_units,
                                                   transcription=transcription))

    def configure_ode(self, phase):
        grid_data = self.grid_data
        map_input_indices_to_disc = grid_data.input_maps['state_input_to_disc']

        phase.state_interp.configure_io()

        phase.connect('dt_dstau', 'state_interp.dt_dstau',
                      src_indices=grid_data.subset_node_indices['col'])

        for name, options in phase.state_options.items():
            size = np.prod(options['shape'])

            phase.connect('states:{0}'.format(name),
                          'state_interp.state_disc:{0}'.format(name),
                          src_indices=om.slicer[map_input_indices_to_disc, ...])

    def setup_defects(self, phase):
        """
        Setup the Collocation and Continuity components as necessary.
        """
        grid_data = self.grid_data

        time_units = phase.time_options['units']

        phase.add_subsystem('collocation_constraint',
                            CollocationComp(grid_data=grid_data,
                                            state_options=phase.state_options,
                                            time_units=time_units))

    def _configure_solve_segments(self, state_name, options, phase):
        """
        Provides error checking for solve_segments and establishes necessary data structures.

        Parameters
        ----------
        state_name : str
            The name of the state being configured.
        options : StateOptionsDictionary
            The StateOptionsDictionary for the state being configured.
        phase : Phase
            The Dymos Phase associated with this transcription instance.
        """
        self.state_idx_map[state_name] = {'solver': None, 'indep': None}

        state_input_idxs = self.grid_data.subset_node_indices['state_input']
        num_state_input_nodes = self.grid_data.subset_num_nodes['state_input']
        compressed = self.options['compressed']

        # Transcription solve_segments overrides state solve_segments if its not set
        if options['solve_segments'] is None:
            options['solve_segments'] = self.options['solve_segments']

        # Flag deprecated solve_segments options:
        if options['solve_segments'] is True:
            ss = options['solve_segments']
            warn_deprecation(f'State {state_name} in phase {phase.name} has option '
                             f'\'solve_segments=True\'. Setting \'solve_segments=True\' now gives '
                             f'forward propagation. In Dymos 1.0 and later, only options '
                             f'\'forward\' and \'backward\' will be valid.')

        # Sanity-checks for solve segments
        # If solve_segments is used at all, we cannot fix the state at both ends of the phase.
        if options['solve_segments']:
            if options['fix_initial'] and options['fix_final']:
                raise ValueError(f'Can not use solve_segments for state ({state_name}) '
                                 f'in phase ({phase.name}) with both "fix_initial" and '
                                 '"fix_final" set to True.')
            if options['connected_initial'] and options['fix_final']:
                raise ValueError(f'Can not use solve_segments for state ({state_name}) '
                                 f'in phase ({phase.name}) with both "connected_initial" '
                                 f'and "fix_final" set to True .')

            # If solve_segments is 'forward', 'fix_final' may not be True
            if options['solve_segments'] in {'forward', True}:
                if options['fix_final']:
                    raise ValueError(f'Cannot use solve_segments in phase ({phase.name}) for state '
                                     f'({state_name}) with forward propagation when fix_final=True.'
                                     f' Either set fix_final=False or set solve_segments=\'reverse\'')

            # If solve_segments is 'backward', neither 'fix_initial' nor 'connected_initial' may be True.
            if options['solve_segments'] == 'backward':
                if options['fix_initial']:
                    raise ValueError(f'Cannot use solve_segments in phase ({phase.name}) with '
                                     f'backward propagation when fix_initial=True. Either set '
                                     f'fix_final=False or set solve_segments=\'reverse\'')
                elif options['connected_initial']:
                    raise ValueError(f'Cannot use solve_segments in phase ({phase.name}) with '
                                     f'backward propagation when connected_initial=True. Either set '
                                     f'connected_initial=False or set solve_segments=\'forward\'')

            # Forward propagation
            if options['solve_segments'] in {True, 'forward'}:
                if compressed:
                    self.state_idx_map[state_name]['solver'] = np.arange(1, num_state_input_nodes, dtype=int)
                    self.state_idx_map[state_name]['indep'] = np.zeros((1,), dtype=int)
                else:
                    left_idxs = self.grid_data.subset_node_indices['segment_ends'][0::2]
                    self.state_idx_map[state_name]['solver'] = [i for i in range(num_state_input_nodes)
                                                                if state_input_idxs[i] not in left_idxs]
                    self.state_idx_map[state_name]['indep'] = [i for i in range(num_state_input_nodes)
                                                               if state_input_idxs[i] in left_idxs]

            # Backward propagation
            elif options['solve_segments'] in {'backward'}:
                if compressed:
                    # The optimizer controls the last state input node, all others are solver-controlled
                    self.state_idx_map[state_name]['indep'] = np.array([num_state_input_nodes - 1], dtype=int)
                    self.state_idx_map[state_name]['solver'] = np.arange(num_state_input_nodes - 1, dtype=int)
                else:
                    # The optimizer controls the last state input node in each segment, all others are solver_controlled
                    right_idxs = self.grid_data.subset_node_indices['segment_ends'][1::2]
                    self.state_idx_map[state_name]['indep'] = [i for i in range(num_state_input_nodes)
                                                               if state_input_idxs[i] in right_idxs]
                    self.state_idx_map[state_name]['solver'] = [i for i in range(num_state_input_nodes)
                                                                if state_input_idxs[i] not in right_idxs]
        else:
            # No solver used to solve these nodes.  All state input nodes are the indep nodes.
            self.state_idx_map[state_name]['solver'] = []
            self.state_idx_map[state_name]['indep'] = np.arange(len(state_input_idxs), dtype=int)

    def configure_defects(self, phase):
        grid_data = self.grid_data
        num_seg = grid_data.num_segments

        phase.connect('dt_dstau', ('collocation_constraint.dt_dstau'),
                      src_indices=grid_data.subset_node_indices['col'])

        phase.collocation_constraint.configure_io()

        # Add the continuity constraint component if necessary
        if num_seg > 1:
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            state_disc_idxs = grid_data.subset_node_indices['state_disc']

            if not self.options['compressed']:
                state_input_subidxs = np.where(np.in1d(state_disc_idxs, segment_end_idxs))[0]

                for name, options in phase.state_options.items():
                    shape = options['shape']
                    flattened_src_idxs = get_src_indices_by_row(state_input_subidxs, shape=shape,
                                                                flat=True)
                    phase.connect('states:{0}'.format(name),
                                  'continuity_comp.states:{}'.format(name),
                                  src_indices=flattened_src_idxs, flat_src_indices=True)

            for name, options in phase.control_options.items():
                control_src_name = 'control_values:{0}'.format(name)

                # The sub-indices of control_disc indices that are segment ends
                segment_end_idxs = grid_data.subset_node_indices['segment_ends']
                src_idxs = get_src_indices_by_row(segment_end_idxs, options['shape'], flat=True)

                phase.connect(control_src_name,
                              'continuity_comp.controls:{0}'.format(name),
                              src_indices=src_idxs, flat_src_indices=True)

                phase.connect('control_rates:{0}_rate'.format(name),
                              'continuity_comp.control_rates:{}_rate'.format(name),
                              src_indices=src_idxs, flat_src_indices=True)

                phase.connect('control_rates:{0}_rate2'.format(name),
                              'continuity_comp.control_rates:{}_rate2'.format(name),
                              src_indices=src_idxs, flat_src_indices=True)

    def setup_solvers(self, phase):
        pass

    def configure_solvers(self, phase):
        if self.any_solved_segs or self.any_connected_opt_segs:
            newton = phase.nonlinear_solver = om.NewtonSolver()
            newton.options['solve_subsystems'] = True
            newton.options['maxiter'] = 100
            newton.options['iprint'] = -1
            newton.linesearch = om.BoundsEnforceLS()
            phase.linear_solver = om.DirectSolver()

    def setup_timeseries_outputs(self, phase):
        gd = self.grid_data

        for name, options in phase._timeseries.items():
            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = PseudospectralTimeseriesOutputComp(input_grid_data=gd,
                                                                 output_grid_data=ogd,
                                                                 output_subset=options['subset'])
            phase.add_subsystem(name, subsys=timeseries_comp)

    def _get_boundary_constraint_src(self, var, loc, phase):
        # Determine the path to the variable which we will be constraining
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

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
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            solve_segments = phase.state_options[var]['solve_segments']
            connected_initial = phase.state_options[var]['connected_initial']
            if not solve_segments and not connected_initial:
                linear = True
            elif solve_segments in {True, 'forward'} and not connected_initial and loc == 'initial':
                linear = True
            elif solve_segments == 'backward' and loc == 'final':
                linear = True
            else:
                linear = False
            constraint_path = f'states:{var}'
        elif var_type in 'indep_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = True
            constraint_path = 'control_values:{0}'.format(var)
        elif var_type == 'input_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            constraint_path = 'control_values:{0}'.format(var)
        elif var_type in 'indep_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = True
            constraint_path = 'polynomial_control_values:{0}'.format(var)
        elif var_type == 'input_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = False
            constraint_path = 'polynomial_control_values:{0}'.format(var)
        elif var_type == 'parameter':
            shape = phase.parameter_options[var]['shape']
            units = phase.parameter_options[var]['units']
            linear = True
            constraint_path = 'parameters:{0}'.format(var)
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5] if var_type == 'control_rate' else var[:-6]
            shape = phase.control_options[control_var]['shape']
            control_units = phase.control_options[control_var]['units']
            d = 2 if var_type == 'control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            constraint_path = 'control_rates:{0}'.format(var)
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5]
            shape = phase.polynomial_control_options[control_var]['shape']
            control_units = phase.polynomial_control_options[control_var]['units']
            d = 2 if var_type == 'polynomial_control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            constraint_path = f'polynomial_control_rates:{var}'
        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
            constraint_path = f'{self._rhs_source}.{var}'
            ode = phase._get_subsystem(self._rhs_source)
            shape, units = get_source_metadata(ode, var, user_units=None, user_shape=None)
            linear = False

        return constraint_path, shape, units, linear
