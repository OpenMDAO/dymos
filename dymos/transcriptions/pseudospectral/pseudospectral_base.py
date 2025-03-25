import numpy as np

import openmdao.api as om
from ..transcription_base import TranscriptionBase
from ..common import TimeComp
from .components import StateIndependentsComp, StateInterpComp, CollocationComp
from ..common.timeseries_output_comp import TimeseriesOutputComp
from ...utils.misc import CoerceDesvar, get_rate_units, reshape_val
from ...utils.introspection import get_promoted_vars, get_source_metadata
from ...utils.constants import INF_BOUND
from ...utils.indexing import get_src_indices_by_row


class PseudospectralBase(TranscriptionBase):
    """
    Base class for the pseudospectral transcriptions.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.any_solved_segs = False
        self.any_connected_opt_segs = False

    def initialize(self):
        """
        Declare transcription options.
        """
        self.options.declare(name='solve_segments', default=False,
                             values=(False, 'forward', 'backward'),
                             desc='Applies \'solve_segments\' behavior to _all_ states in the Phase. '
                                  'If \'forward\', collocation defects within each '
                                  'segment are solved with a Newton solver by fixing the initial value in the '
                                  'phase (if using compressed transcription) or segment (if not using '
                                  'compressed transcription). This provides a forward shooting (or multiple shooting) '
                                  'method.  If \'backward\', the final value in the phase or segment is fixed '
                                  'and a solver finds the other ones to mimic reverse propagation. Set '
                                  'to False (the default) to explicitly disable the use of a solver to '
                                  'converge the state time history.')

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        super(PseudospectralBase, self).setup_time(phase)

        time_comp = TimeComp(num_nodes=grid_data.num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau,
                             units=phase.time_options['units'],
                             initial_val=phase.time_options['initial_val'],
                             duration_val=phase.time_options['duration_val'])

        phase.add_subsystem('time', time_comp, promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super(PseudospectralBase, self).configure_time(phase)
        phase.time.configure_io()

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        for options in phase.state_options.values():
            # Transcription solve_segments overrides state solve_segments if its not set
            if options['solve_segments'] is None:
                options['solve_segments'] = self.options['solve_segments']

            if options['solve_segments']:
                self.any_solved_segs = True
            elif options['input_initial']:
                self.any_connected_opt_segs = True

        if self.any_solved_segs or self.any_connected_opt_segs:
            indep = StateIndependentsComp(grid_data=grid_data,
                                          state_options=phase.state_options)
        else:
            indep = om.IndepVarComp()

        num_connected = len([s for (s, opts) in phase.state_options.items() if opts['input_initial']])
        prom_inputs = ['initial_states:*'] if num_connected > 0 else None
        phase.add_subsystem('indep_states', indep, promotes_inputs=prom_inputs,
                            promotes_outputs=['*'])

    def configure_controls(self, phase):
        """
        Configure control I/O for the phase.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_controls(phase)

        if phase.control_options:
            phase.control_comp.configure_io()
            phase.promotes('control_comp',
                           any=['*controls:*', '*control_values:*', '*control_rates:*'])

            phase.connect('dt_dstau', 'control_comp.dt_dstau')

    def configure_states(self, phase):
        """
        Configure state connections post-introspection.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_states(phase)
        grid_data = self.grid_data
        num_state_input_nodes = grid_data.subset_num_nodes['state_input']
        indep = phase.indep_states

        # state_idx_map holds the node indices provided by the solver (solver) and those
        # that are independent variables (indep)
        self.state_idx_map = {}

        # add all the des-vars (either from the IndepVarComp or from the indep-var-like
        # outputs of the collocation comp)
        for name, options in phase.state_options.items():
            self._configure_solve_segments(name, options, phase)
            shape = options['shape']
            # In certain cases, we put an output on the IVC.
            if isinstance(indep, om.IndepVarComp):
                default_val = reshape_val(options['val'], shape, num_state_input_nodes)
                indep.add_output(name=f'states:{name}',
                                 shape=(num_state_input_nodes,) + shape,
                                 val=default_val,
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
                    if options['input_initial']:
                        raise ValueError('Cannot specify \'fix_initial=True\' and specify '
                                         f'\'connected_initial=True\' for state {name} '
                                         f'in phase {phase.name}')
                    idx_mask[0, ...] = np.asarray(np.logical_not(options['fix_initial']), dtype=int)
                elif options['input_initial']:
                    if options['initial_bounds'] is not None:
                        raise ValueError('Cannot specify \'connected_initial=True\' and specify '
                                         f'initial_bounds for state {name} in phase {phase.name}')
                    idx_mask[0, ...] = np.asarray(np.logical_not(options['input_initial']), dtype=int)

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
                                         indices=desvar_indices,
                                         flat_indices=True)

        if isinstance(indep, StateIndependentsComp):
            indep.configure_io(self.state_idx_map)

        if self.any_solved_segs or self.any_connected_opt_segs:
            for name, options in phase.state_options.items():
                if options['solve_segments']:
                    phase.connect(f'collocation_constraint.defects:{name}',
                                  f'indep_states.defects:{name}')

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data
        phase.add_subsystem('state_interp',
                            subsys=StateInterpComp(grid_data=grid_data,
                                                   state_options=phase.state_options,
                                                   time_units=phase.time_options['units'],
                                                   transcription=grid_data.transcription))

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data
        map_input_indices_to_disc = grid_data.input_maps['state_input_to_disc']

        phase.state_interp.configure_io()

        phase.connect('dt_dstau', 'state_interp.dt_dstau',
                      src_indices=grid_data.subset_node_indices['col'], flat_src_indices=True)

        for name in phase.state_options:
            phase.connect(f'states:{name}',
                          f'state_interp.state_disc:{name}',
                          src_indices=om.slicer[map_input_indices_to_disc, ...])

    def setup_defects(self, phase):
        """
        Setup the Collocation and Continuity components as necessary.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase.add_subsystem('collocation_constraint',
                            CollocationComp(grid_data=self.grid_data,
                                            state_options=phase.state_options,
                                            time_units=phase.time_options['units']))

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

        # Sanity-checks for solve segments
        # If solve_segments is used at all, we cannot fix the state at both ends of the phase.
        if options['solve_segments']:
            if options['fix_initial'] and options['fix_final']:
                raise ValueError(f'Can not use solve_segments for state ({state_name}) '
                                 f'in phase ({phase.name}) with both "fix_initial" and '
                                 '"fix_final" set to True.')
            if options['input_initial'] and options['fix_final']:
                raise ValueError(f'Can not use solve_segments for state ({state_name}) '
                                 f'in phase ({phase.name}) with both "input_initial" '
                                 f'and "fix_final" set to True .')

            # If solve_segments is 'forward', 'fix_final' may not be True
            if options['solve_segments'] in {'forward', True}:
                if options['fix_final']:
                    raise ValueError(f'Cannot use solve_segments in phase ({phase.name}) for state '
                                     f'({state_name}) with forward propagation when fix_final=True.'
                                     f' Either set fix_final=False or set solve_segments=\'backward\'')

            # Backward propagation
            if options['solve_segments'] == 'backward':
                # Neither 'fix_initial' nor 'input_initial' may be True.
                if options['fix_initial']:
                    raise ValueError(f'Cannot use solve_segments in phase ({phase.name}) with '
                                     f'backward propagation when fix_initial=True. Either set '
                                     f'fix_final=False or set solve_segments=\'backward\'')
                elif options['input_initial']:
                    raise ValueError(f'Cannot use solve_segments in phase ({phase.name}) with '
                                     f'backward propagation when input_initial=True. Either set '
                                     f'input_initial=False or set solve_segments=\'forward\'')

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
            # Forward propagation
            elif options['solve_segments'] in {'forward'}:
                if compressed:
                    self.state_idx_map[state_name]['solver'] = np.arange(1, num_state_input_nodes, dtype=int)
                    self.state_idx_map[state_name]['indep'] = np.zeros((1,), dtype=int)
                else:
                    left_idxs = self.grid_data.subset_node_indices['segment_ends'][0::2]
                    self.state_idx_map[state_name]['solver'] = [i for i in range(num_state_input_nodes)
                                                                if state_input_idxs[i] not in left_idxs]
                    self.state_idx_map[state_name]['indep'] = [i for i in range(num_state_input_nodes)
                                                               if state_input_idxs[i] in left_idxs]
        else:
            # No solver used to solve these nodes.  All state input nodes are the indep nodes.
            self.state_idx_map[state_name]['solver'] = []
            self.state_idx_map[state_name]['indep'] = np.arange(len(state_input_idxs), dtype=int)

    def configure_defects(self, phase):
        """
        Configure the continuity_comp and connect the collocation constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        phase.connect('dt_dstau', ('collocation_constraint.dt_dstau'),
                      src_indices=grid_data.subset_node_indices['col'], flat_src_indices=True)

        phase.collocation_constraint.configure_io()

        any_state_cnty, any_control_cnty, any_control_rate_cnty = self._requires_continuity_constraints(phase)

        if not any((any_state_cnty, any_control_cnty, any_control_rate_cnty)):
            return

        # Add the continuity constraint component if necessary
        segment_end_idxs = grid_data.subset_node_indices['segment_ends']
        state_disc_idxs = grid_data.subset_node_indices['state_disc']

        if any_state_cnty:
            state_input_subidxs = np.where(np.isin(state_disc_idxs, segment_end_idxs))[0]

            for name, options in phase.state_options.items():
                shape = options['shape']
                flattened_src_idxs = get_src_indices_by_row(state_input_subidxs, shape=shape,
                                                            flat=True)
                phase.connect(f'states:{name}',
                              f'continuity_comp.states:{name}',
                              src_indices=(flattened_src_idxs,), flat_src_indices=True)

        if any_control_rate_cnty:
            phase.connect('t_duration_val', 'continuity_comp.t_duration')

        for name, options in phase.control_options.items():
            # The sub-indices of control_disc indices that are segment ends
            segment_end_idxs = grid_data.subset_node_indices['segment_ends']
            src_idxs = get_src_indices_by_row(segment_end_idxs, options['shape'], flat=True)

            # enclose indices in tuple to ensure shaping of indices works
            src_idxs = (src_idxs,)

            if options['continuity']:
                phase.connect(f'control_values:{name}',
                              f'continuity_comp.controls:{name}',
                              src_indices=src_idxs, flat_src_indices=True)

            if options['rate_continuity']:
                phase.connect(f'control_rates:{name}_rate',
                              f'continuity_comp.control_rates:{name}_rate',
                              src_indices=src_idxs, flat_src_indices=True)

            if options['rate2_continuity']:
                phase.connect(f'control_rates:{name}_rate2',
                              f'continuity_comp.control_rates:{name}_rate2',
                              src_indices=src_idxs, flat_src_indices=True)

    def setup_solvers(self, phase):
        """
        Setup the solvers.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_solvers(self, phase, requires_solvers=None):
        """
        Configure the solvers.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        requires_solvers : dict[str: bool]
            A dictionary mapping a string descriptor of a reason why a solver is required,
            and whether a solver is required.
        """
        req_solvers = {'solved segments': self.any_solved_segs,
                       'input initial': self.any_connected_opt_segs}
        if requires_solvers is not None:
            req_solvers.update(requires_solvers)
        super().configure_solvers(phase, requires_solvers=req_solvers)

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        gd = self.grid_data

        for name, options in phase._timeseries.items():
            if options['transcription'] is None:
                ogd = None
            else:
                ogd = options['transcription'].grid_data

            timeseries_comp = TimeseriesOutputComp(input_grid_data=gd,
                                                   output_grid_data=ogd,
                                                   output_subset=options['subset'],
                                                   time_units=phase.time_options['units'])
            phase.add_subsystem(name, subsys=timeseries_comp)

            phase.connect('dt_dstau', f'{name}.dt_dstau', flat_src_indices=True)

    def _get_response_src(self, var, loc, phase, ode_outputs=None):
        """
        Return the path to the variable that will be used as a response..

        Parameters
        ----------
        var : str
            Name of the variable to be used as the response.
        loc : str
            The location of the response in the phase ['initial', 'final'].
        phase : dymos.Phase
            Phase object containing in which the objective resides.
        ode_outputs : dict or None
            A dictionary of ODE outputs as returned by get_promoted_vars.

        Returns
        -------
        obj_path : str
            Path to the source.
        shape : tuple
            Source shape.
        units : str
            Source units.
        linear : bool
            True if the objective quantity is linear.
        """
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

        if ode_outputs is None:
            ode_outputs = get_promoted_vars(phase._get_subsystem(self._rhs_source), 'output')

        if var_type == 't':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 't'
        elif var_type == 't_phase':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 't_phase'
        elif var_type == 'state':
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            solve_segments = phase.state_options[var]['solve_segments']
            connected_initial = phase.state_options[var]['input_initial']
            if not solve_segments and not connected_initial:
                linear = True
            elif solve_segments in {'forward'} and not connected_initial and loc == 'initial':
                linear = True
            elif solve_segments == 'backward' and loc == 'final':
                linear = True
            else:
                linear = False
            constraint_path = f'states:{var}'
        elif var_type == 'control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            constraint_path = f'control_values:{var}'
        elif var_type == 'parameter':
            shape = phase.parameter_options[var]['shape']
            units = phase.parameter_options[var]['units']
            linear = True
            constraint_path = f'parameter_vals:{var}'
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5] if var_type == 'control_rate' else var[:-6]
            shape = phase.control_options[control_var]['shape']
            control_units = phase.control_options[control_var]['units']
            d = 2 if var_type == 'control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            constraint_path = f'control_rates:{var}'
        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
            constraint_path = f'{self._rhs_source}.{var}'
            meta = get_source_metadata(ode_outputs, var, user_units=None, user_shape=None)
            shape = meta['shape']
            units = meta['units']
            linear = False

        return constraint_path, shape, units, linear

    def _get_num_timeseries_nodes(self):
        """
        Returns the number of nodes in the default timeseries for this transcription.

        Returns
        -------
        int
            The number of nodes in the default timeseries for this transcription.
        """
        return self.grid_data.num_nodes

    def _phase_set_state_val(self, phase, name, vals, time_vals, interpolation_kind):
        """
        Method to interpolate the provided input and return the variables that need to be set
        along with their appropriate value.

        Parameters
        ----------
        phase : dymos.Phase
            The phase to which this transcription applies.
        name : str
            The name of the phase variable to be set.
        vals : ndarray or Sequence or float
            Array of control/state/parameter values.
        time_vals : ndarray or Sequence or None
            Array of integration variable values.
        interpolation_kind : str
            Specifies the kind of interpolation, as per the scipy.interpolate package.
            One of ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.

        Returns
        -------
        input_data : dict
            Dict containing the values that need to be set in the phase

        """

        input_data = {}
        if np.isscalar(vals):
            interp_vals = vals
        else:
            interp_vals = phase.interp(name, ys=vals, xs=time_vals,
                                       nodes='state_input',
                                       kind=interpolation_kind)
        input_data[f'states:{name}'] = interp_vals

        return input_data
