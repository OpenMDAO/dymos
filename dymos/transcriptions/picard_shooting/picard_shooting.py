import numpy as np

import openmdao.api as om

from ..transcription_base import TranscriptionBase
from ..common import TimeComp, TimeseriesOutputComp, ControlInterpComp, GaussLobattoContinuityComp
from .multiple_shooting_iter_group import MultipleShootingIterGroup

from ..grid_data import GaussLobattoGrid, ChebyshevGaussLobattoGrid
from dymos.utils.introspection import get_promoted_vars, get_source_metadata, get_rate_units
from dymos.utils.indexing import get_constraint_flat_idxs, get_src_indices_by_row
from dymos.utils.misc import _format_phase_constraint_alias


class PicardShooting(TranscriptionBase):
    """
        Picard Shooting transcription.

        Parameters
        ----------
        **kwargs : dict
            Dictionary of optional arguments.

        References
        ----------
        TBD
        """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rhs_source = 'ode_iter_group.segment_prop_group.ode_all'
        self._has_initial_final_states = True

    def initialize(self):
        """
        Declare transcription options.
        """
        self.options.declare('num_segments', types=int, default=1,
                             desc='The number of segments in the grid.')

        self.options.declare('nodes_per_seg', types=int, default=10,
                             desc='The number of segments in the grid.')

        self.options.declare('grid_type', values=('cgl', 'lgl'), default='cgl',
                             desc='Specifies which type of grid is to be used. '
                                  'Options are Chebyshev-Gauss-Lobatto ("cgl") '
                                  'and Legendre-Gauss-Lobatto ("lgl")')

        self.options.declare(name='solve_segments', default='forward',
                             values=('forward', 'backward'),
                             desc='The default solve direction for states in this phase.'
                             'This value may be overridden by setting the solve_segments option of states')

        self.options.declare('ode_nonlinear_solver', default=om.NonlinearBlockGS(maxiter=100, use_aitken=True, iprint=0),
                             desc='Nonlinear solver used to resolve Picard iteration on each segment.',
                             recordable=False)

        self.options.declare('ode_linear_solver', default=om.DirectSolver(),
                             desc='Linear solver used to linearize the Picard iteration subsystem on each segment.',
                             recordable=False)

        self.options.declare('ms_nonlinear_solver', default=om.NonlinearBlockGS(maxiter=100, use_aitken=True, iprint=0),
                             desc='Nonlinear solver used to resolve Picard iteration differences between segments.',
                             recordable=False)

        self.options.declare('ms_linear_solver', default=om.DirectSolver(),
                             desc='Linear solver used to linearize the Picard iteration subsystem between segments.',
                             recordable=False)

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        if self.options['grid_type'] == 'cgl':
            grid_cls = ChebyshevGaussLobattoGrid
        elif self.options['grid_type'] == 'lgl':
            grid_cls = GaussLobattoGrid
        self.grid_data = grid_cls(num_segments=self.options['num_segments'],
                                  nodes_per_seg=self.options['nodes_per_seg'])

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data

        super().setup_time(phase)

        time_comp = TimeComp(num_nodes=grid_data.num_nodes, node_ptau=grid_data.node_ptau,
                             node_dptau_dstau=grid_data.node_dptau_dstau,
                             units=phase.time_options['units'],
                             initial_val=phase.time_options['initial_val'],
                             duration_val=phase.time_options['duration_val'])

        phase.add_subsystem('time', time_comp,
                            promotes_inputs=[('t_initial', 't_initial_val'), ('t_duration', 't_duration_val')],
                            promotes_outputs=['*'])

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        This method assumes that target introspection has already been performed by the phase and thus
        options['targets'], options['time_phase_targets'], options['t_initial_targets'],
        and options['t_duration_targets'] are all correctly populated.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_time(phase)
        phase.time.configure_io()
        options = phase.time_options
        ode = phase._get_subsystem(self._rhs_source)
        ode_inputs = get_promoted_vars(ode, iotypes='input')

        # The tuples here are (name, user_specified_targets, dynamic)
        for name, targets in [('t', options['targets']),
                              ('t_phase', options['time_phase_targets']),
                              ('dt_dstau', options['dt_dstau_targets'])]:
            if targets:
                src_idxs = self.grid_data.subset_node_indices['all']
                phase.connect(name, [f'ode_all.{t}' for t in targets], src_indices=src_idxs,
                              flat_src_indices=True)

        for name, targets in [('t_initial', options['t_initial_targets']),
                              ('t_duration', options['t_duration_targets']),
                              ('t_final', options['t_final_targets'])]:
            for t in targets:
                shape = ode_inputs[t]['shape']

                if shape == (1,):
                    src_idxs = None
                    flat_src_idxs = None
                    src_shape = None
                else:
                    src_idxs = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                    flat_src_idxs = True
                    src_shape = (1,)

                phase.promotes('ode_all', inputs=[(t, name)], src_indices=src_idxs,
                               flat_src_indices=flat_src_idxs, src_shape=src_shape)
            if targets:
                phase.set_input_defaults(name=name,
                                         val=np.ones((1,)),
                                         units=options['units'])

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        In the Birkhoff transcription, everything typically done in this
        method is instead done by the BirkhoffIterGroup.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        self.any_solved_segs = False
        self.any_connected_opt_segs = False
        for options in phase.state_options.values():
            # Transcription solve_segments overrides state solve_segments if its not set
            if options['solve_segments'] in (None, False):
                options['solve_segments'] = self.options['solve_segments']

            if options['solve_segments']:
                self.any_solved_segs = True
            elif options['input_initial']:
                self.any_connected_opt_segs = True

    def setup_controls(self, phase):
        """
        Setup the control group.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._check_control_options()

        if phase.control_options:
            control_comp = ControlInterpComp(control_options=phase.control_options,
                                             time_units=phase.time_options['units'],
                                             grid_data=self.grid_data)

            phase.add_subsystem('control_comp',
                                subsys=control_comp)

            phase.connect('t_duration_val', 'control_comp.t_duration')

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

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

        for name, options in phase.control_options.items():
            if options['targets']:
                phase.connect(f'control_values:{name}', [f'ode_all.{t}' for t in options['targets']])

            if options['rate_targets']:
                phase.connect(f'control_rates:{name}_rate',
                              [f'ode_all.{t}' for t in options['rate_targets']])

            if options['rate2_targets']:
                phase.connect(f'control_rates:{name}_rate2',
                              [f'ode_all.{t}' for t in options['rate2_targets']])

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """

        ODEClass = phase.options['ode_class']
        grid_data = self.grid_data
        ode_nonlinear_solver = self.options['ode_nonlinear_solver']
        ode_linear_solver = self.options['ode_linear_solver']
        ms_nonlinear_solver = self.options['ms_nonlinear_solver']
        ms_linear_solver = self.options['ms_linear_solver']

        ode_init_kwargs = phase.options['ode_init_kwargs']
        calc_exprs = phase._calc_exprs
        parameter_options = phase.parameter_options

        phase.add_subsystem('ode_iter_group',
                            subsys=MultipleShootingIterGroup(grid_data=grid_data,
                                                             state_options=phase.state_options,
                                                             time_units=phase.time_options['units'],
                                                             ode_class=ODEClass,
                                                             ode_init_kwargs=ode_init_kwargs,
                                                             ode_nonlinear_solver=ode_nonlinear_solver,
                                                             ode_linear_solver=ode_linear_solver,
                                                             ms_nonlinear_solver=ms_nonlinear_solver,
                                                             ms_linear_solver=ms_linear_solver,
                                                             calc_exprs=calc_exprs,
                                                             parameter_options=parameter_options),
                            promotes_inputs=['*'], promotes_outputs=['*'])

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._get_subsystem('ode_iter_group').configure_io(phase)
        phase.connect('dt_dstau', 'picard_update_comp.dt_dstau')

        num_nodes = self.grid_data.subset_num_nodes['all']
        for name, options in phase.state_options.items():
            rate_source_type = phase.classify_var(options['rate_source'])
            rate_src_path = self._get_rate_source_path(name, phase)
            if rate_src_path.startswith('parameter_vals:'):
                src_idxs = om.slicer[np.zeros(num_nodes, dtype=int), ...]
            else:
                src_idxs = None

            if rate_source_type not in ('state', 'ode'):
                phase.connect(rate_src_path, f'picard_update_comp.f_computed:{name}', src_indices=src_idxs)

    def setup_defects(self, phase):
        """
        Create the continuity_comp to house the defects.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if any(self._requires_continuity_constraints(phase)):
            phase.add_subsystem('continuity_comp',
                                GaussLobattoContinuityComp(grid_data=self.grid_data,
                                                           state_options={},
                                                           control_options=phase.control_options,
                                                           time_units=phase.time_options['units']))

    def configure_defects(self, phase):
        """
        Connect the collocation constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        grid_data = self.grid_data
        any_state_cnty, any_control_cnty, any_control_rate_cnty = self._requires_continuity_constraints(phase)

        if any((any_state_cnty, any_control_cnty, any_control_rate_cnty)):
            phase._get_subsystem('continuity_comp').configure_io()

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
        pass

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for timeseries_name, timeseries_options in phase._timeseries.items():
            timeseries_comp = phase._get_subsystem(timeseries_name)
            ts_inputs_to_promote = []
            for input_name, src, src_idxs in timeseries_comp._configure_io(timeseries_options):
                # If the src was added, promote it if it was a state,
                # or connect it otherwise.
                if src.startswith('states:'):
                    state_name = src.split(':')[-1]
                    ts_inputs_to_promote.append((input_name, f'states:{state_name}'))
                else:
                    phase.connect(src_name=src,
                                  tgt_name=f'{timeseries_name}.{input_name}',
                                  src_indices=src_idxs)
            phase.promotes(timeseries_name, inputs=ts_inputs_to_promote)

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

    def _get_constraint_kwargs(self, constraint_type, options, phase):
        """
        Given the constraint options provide the keyword arguments for the OpenMDAO add_constraint method.

        Parameters
        ----------
        constraint_type : str
            One of 'initial', 'final', or 'path'.
        options : dict
            The constraint options.
        phase : Phase
            The dymos phase to which the constraint applies.

        Returns
        -------
        con_output : str
            The phase-relative path being constrained.
        constraint_kwargs : dict
            Keyword arguments for the OpenMDAO add_constraint method.
        """
        num_nodes = self._get_num_timeseries_nodes()

        constraint_kwargs = {key: options for key, options in options.items()}

        # Determine the path to the variable which we will be constraining
        var = options['name']
        var_type = phase.classify_var(var)

        # These are the flat indices at a single point in time used
        # in either initial, final, or path constraints.
        idxs_in_initial = phase._indices_in_constraints(var, 'initial')
        idxs_in_final = phase._indices_in_constraints(var, 'final')
        idxs_in_path = phase._indices_in_constraints(var, 'path')

        size = np.prod(options['shape'], dtype=int)

        flat_idxs = get_constraint_flat_idxs(options)

        # Now we need to convert the indices given by the user at any given point
        # to flat indices to be given to OpenMDAO as flat indices spanning the phase.
        if var_type == 'parameter':
            if any([idxs_in_initial.intersection(idxs_in_final),
                    idxs_in_initial.intersection(idxs_in_path),
                    idxs_in_final.intersection(idxs_in_path)]):
                raise RuntimeError(f'In phase {phase.pathname}, parameter `{var}` is subject to multiple boundary '
                                   f'or path constraints.\nParameters are single values that do not change in '
                                   f'time, and may only be used in a single boundary or path constraint.')
            constraint_kwargs['indices'] = flat_idxs
        elif var_type == 'state':
            if constraint_type in ('initial', 'final'):
                constraint_kwargs['indices'] = flat_idxs
            else:
                path_idxs = []
                for i in range(num_nodes):
                    path_idxs.extend(size * i + flat_idxs)

                constraint_kwargs['indices'] = path_idxs
        else:
            if constraint_type == 'initial':
                constraint_kwargs['indices'] = flat_idxs
            elif constraint_type == 'final':
                constraint_kwargs['indices'] = (num_nodes - 1) * size + flat_idxs
            else:
                # Path
                path_idxs = []
                for i in range(num_nodes):
                    path_idxs.extend(size * i + flat_idxs)

                constraint_kwargs['indices'] = path_idxs

        con_path = constraint_kwargs.pop('constraint_path')
        con_name = constraint_kwargs.pop('constraint_name')

        constraint_kwargs['alias'] = _format_phase_constraint_alias(phase, con_name,
                                                                    constraint_type,
                                                                    options['indices'])
        constraint_kwargs.pop('name')
        constraint_kwargs.pop('shape')
        constraint_kwargs['flat_indices'] = True

        return con_path, constraint_kwargs

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
            linear = False
            if loc == 'initial':
                constraint_path = f'initial_states:{var}'
            elif loc == 'final':
                constraint_path = f'final_states:{var}'
            else:
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
            constraint_path = f'timeseries.{var}'
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

    def _get_rate_source_path(self, state_name, phase):
        """
        Return the rate source location and indices for a given state name.

        Parameters
        ----------
        state_name : str
            Name of the state.
        phase : dymos.Phase
            Phase object containing the rate source.

        Returns
        -------
        str
            Path to the rate source.
        ndarray
            Array of source indices.
        """
        try:
            var = phase.state_options[state_name]['rate_source']
        except RuntimeError:
            raise ValueError(f"state '{state_name}' in phase '{phase.name}' was not given a "
                             "rate_source")

        # Note the rate source must be shape-compatible with the state
        var_type = phase.classify_var(var)

        # Determine the path to the variable
        if var_type == 't':
            rate_path = 't'
        elif var_type == 't_phase':
            rate_path = 't_phase'
        elif var_type == 'state':
            rate_path = f'states:{var}'
        elif var_type == 'control':
            rate_path = f'control_values:{var}'
        elif var_type == 'control_rate':
            control_name = var[:-5]
            rate_path = f'control_rates:{control_name}_rate'
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            rate_path = f'control_rates:{control_name}_rate2'
        elif var_type == 'parameter':
            rate_path = f'parameter_vals:{var}'
        else:
            # Failed to find variable, assume it is in the ODE
            rate_path = f'ode_all.{var}'

        return rate_path

    def _get_timeseries_var_source(self, var, output_name, phase):
        """
        Return the source path and indices for a given variable to be connected to a timeseries.

        Parameters
        ----------
        var : str
            Name of the timeseries variable whose source is desired.
        output_name : str
            Name of the timeseries output whose source is desired.
        phase : dymos.Phase
            Phase object containing the variable, either as state, time, control, etc., or as an ODE output.

        Returns
        -------
        meta : dict
            Metadata pertaining to the variable at the given path. This dict contains 'src' (the path to the
            timeseries source), 'src_idxs' (an array of the
            source indices), 'units' (the units of the source variable), and 'shape' (the shape of the variable at
            a given node).
        """
        gd = self.grid_data
        var_type = phase.classify_var(var)
        time_units = phase.time_options['units']

        transcription = phase.options['transcription']
        ode = transcription._get_ode(phase)
        ode_outputs = get_promoted_vars(ode, 'output')

        # The default for node_idxs, applies to everything except states and parameters.
        node_idxs = gd.subset_node_indices['all']

        meta = {}

        # Determine the path to the variable
        if var_type == 't':
            path = 't'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 't_phase':
            path = 't_phase'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 'state':
            path = f'states:{var}'
            src_units = phase.state_options[var]['units']
            src_shape = phase.state_options[var]['shape']

            if gd.transcription == 'radau-ps':
                # Find the state_input indices which occur at segment endpoints, and repeat them twice
                state_input_idxs = gd.subset_node_indices['state_input']
                repeat_idxs = np.ones_like(state_input_idxs)
                if self.options['compressed']:
                    segment_end_idxs = gd.subset_node_indices['segment_ends'][1:-1]
                    # Repeat nodes that are on segment bounds (but not the first or last nodes in the phase)
                    nodes_to_repeat = list(set(state_input_idxs).intersection(set(segment_end_idxs)))
                    # Now find these nodes in the state input indices
                    idxs_of_ntr_in_state_inputs = np.where(np.isin(state_input_idxs, nodes_to_repeat))[0]
                    # All state input nodes are used once, but nodes_to_repeat are used twice
                    repeat_idxs[idxs_of_ntr_in_state_inputs] = 2
                # Now we have a way of mapping the state input indices to all nodes
                map_input_node_idxs_to_all = np.repeat(np.arange(gd.subset_num_nodes['state_input'],
                                                                 dtype=int), repeats=repeat_idxs)
                # Now select the subset of nodes we want to use.
                node_idxs = map_input_node_idxs_to_all[gd.subset_node_indices['all']]
        elif var_type == 'control':
            path = f'control_values:{var}'
            src_units = phase.control_options[var]['units']
            src_shape = phase.control_options[var]['shape']
        elif var_type == 'control_rate':
            control_name = var[:-5]
            path = f'control_rates:{control_name}_rate'
            control_name = var[:-5]
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=1)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            path = f'control_rates:{control_name}_rate2'
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=2)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'parameter':
            path = f'parameter_vals:{var}'
            # Timeseries are never a static_target
            node_idxs = np.zeros(gd.subset_num_nodes['all'], dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            path = f'ode_all.{var}'
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(
                    f'ODE output {var} is tagged with "dymos.static_output" and cannot be a timeseries output.')

        src_idxs = om.slicer[node_idxs, ...]

        meta['src'] = path
        meta['src_idxs'] = src_idxs
        meta['units'] = src_units
        meta['shape'] = src_shape

        return meta

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            Parameter name.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        connection_info = []
        if name in phase.parameter_options:
            options = phase.parameter_options[name]
            for tgt in options['targets']:
                if tgt in options['static_targets']:
                    src_idxs = np.squeeze(get_src_indices_by_row([0], options['shape']), axis=0)
                else:
                    src_idxs_raw = np.zeros(self.grid_data.subset_num_nodes['all'], dtype=int)
                    src_idxs = get_src_indices_by_row(src_idxs_raw, options['shape'])
                    if options['shape'] == (1,):
                        src_idxs = src_idxs.ravel()

                connection_info.append((f'ode_all.{tgt}', (src_idxs,)))

        return connection_info

    def _requires_continuity_constraints(self, phase):
        """
        Tests whether state and/or control and/or control rate continuity are required.

        Parameters
        ----------
        phase : dymos.Phase
            The phase to which this transcription applies.

        Returns
        -------
        any_state_continuity : bool
            True if any state continuity is required to be enforced.
        any_control_continuity : bool
            True if any control value continuity is required to be enforced.
        any_control_rate_continuity : bool
            True if any control rate continuity is required to be enforced.
        """
        num_seg = self.grid_data.num_segments
        compressed = self.grid_data.compressed

        any_state_continuity = False
        any_control_continuity = any([opts['continuity'] for opts in phase.control_options.values()])
        any_control_continuity = any_control_continuity and num_seg > 1 and not compressed
        any_rate_continuity = any([opts['rate_continuity'] or opts['rate2_continuity']
                                   for opts in phase.control_options.values()])
        any_rate_continuity = any_rate_continuity and num_seg > 1

        return any_state_continuity, any_control_continuity, any_rate_continuity

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
        seg_end_idxs = self.grid_data.subset_node_indices['segment_ends']
        seg_initial_idxs = seg_end_idxs[::2]
        seg_final_idxs = seg_end_idxs[1::2]

        input_data = {}
        if np.isscalar(vals):
            input_data[f'states:{name}'] = vals
            input_data[f'initial_states:{name}'] = vals
            input_data[f'final_states:{name}'] = vals

            if phase.state_options[name]['solve_segments'] == 'forward':
                input_data[f'picard_update_comp.seg_initial_states:{name}'] = vals
            else:
                input_data[f'picard_update_comp.seg_final_states:{name}'] = vals
        else:
            interp_vals = phase.interp(name, vals, time_vals,
                                       nodes='all',
                                       kind=interpolation_kind)
            input_data[f'states:{name}'] = interp_vals
            input_data[f'initial_states:{name}'] = interp_vals[0, ...]
            input_data[f'final_states:{name}'] = interp_vals[-1, ...]

            if phase.state_options[name]['solve_segments'] == 'forward':
                input_data[f'picard_update_comp.seg_initial_states:{name}'] = interp_vals[seg_initial_idxs, ...]
            else:
                input_data[f'picard_update_comp.seg_final_states:{name}'] = interp_vals[seg_final_idxs, ...]

        return input_data
