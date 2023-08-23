from collections.abc import Sequence

import numpy as np

import openmdao.api as om

from .common import ControlGroup, PolynomialControlGroup, ParameterComp
from ..utils.constants import INF_BOUND
from ..utils.indexing import get_constraint_flat_idxs
from ..utils.misc import _none_or_unspecified
from ..utils.introspection import configure_states_introspection, get_promoted_vars, \
    configure_states_discovery


class TranscriptionBase(object):
    """
    Base class for all dymos transcriptions.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):

        self._implicit_duration = False
        self.grid_data = None

        self.options = om.OptionsDictionary()

        self.options.declare('num_segments', types=int, desc='Number of segments')
        self.options.declare('segment_ends', default=None, types=(Sequence, np.ndarray),
                             allow_none=True, desc='Locations of segment ends or None for equally '
                             'spaced segments')
        self.options.declare('order', default=3, types=(int, Sequence, np.ndarray),
                             desc='Order of the state transcription. The order of the control '
                                  'transcription is `order - 1`.')
        self.options.declare('compressed', default=True, types=bool,
                             desc='Use compressed transcription, meaning state and control values'
                                  'at segment boundaries are not duplicated on input.  This '
                                  'implicitly enforces value continuity between segments but in '
                                  'some cases may make the problem more difficult to solve.')

        self._declare_options()
        self.initialize()
        self.options.update(kwargs)
        self.init_grid()

        # Where to query var info.
        self._rhs_source = None

    def _declare_options(self):
        pass

    def initialize(self):
        """
        Declare transcription options.
        """
        pass

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method init_grid.')

    def setup_time(self, phase):
        """
        Setup up the time component and time extents for the phase.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_options = phase.time_options
        t_name = time_options['name']
        t_phase_name = f'{t_name}_phase'

        # Warn about invalid options
        phase.check_time_options()

        if phase.time_options['t_duration_balance_options']:
            self._implicit_duration = True

        phase.add_subsystem('param_comp', subsys=ParameterComp(time_options=time_options),
                            promotes_inputs=['*'], promotes_outputs=['*'])

        for ts_name, ts_options in phase._timeseries.items():
            if t_name not in ts_options['outputs']:
                phase.add_timeseries_output(t_name, timeseries=ts_name)
            if t_phase_name not in ts_options['outputs'] and \
                    (phase.timeseries_options['include_t_phase'] or time_options['time_phase_targets']):
                phase.add_timeseries_output(t_phase_name, timeseries=ts_name)

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        time_options = phase.time_options

        # Determine the time unit.
        if time_options['units'] in _none_or_unspecified:
            if time_options['targets']:
                ode = phase._get_subsystem(self._rhs_source)

                _, time_options['units'] = get_target_metadata(ode, name='time',
                                                               user_targets=time_options['targets'],
                                                               user_units=time_options['units'],
                                                               user_shape='')

        if not (time_options['input_initial'] or time_options['fix_initial']):
            lb, ub = time_options['initial_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            phase.add_design_var('t_initial',
                                 lower=lb,
                                 upper=ub,
                                 scaler=time_options['initial_scaler'],
                                 adder=time_options['initial_adder'],
                                 ref0=time_options['initial_ref0'],
                                 ref=time_options['initial_ref'])

        if not (time_options['input_duration'] or time_options['fix_duration']):
            lb, ub = time_options['duration_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            phase.add_design_var('t_duration',
                                 lower=lb,
                                 upper=ub,
                                 scaler=time_options['duration_scaler'],
                                 adder=time_options['duration_adder'],
                                 ref0=time_options['duration_ref0'],
                                 ref=time_options['duration_ref'])

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
            control_group = ControlGroup(control_options=phase.control_options,
                                         time_units=phase.time_options['units'],
                                         grid_data=self.grid_data)

            phase.add_subsystem('control_group',
                                subsys=control_group)

            control_prefix = 'controls:' if phase.timeseries_options['use_prefix'] else ''
            control_rate_prefix = 'control_rates:' if phase.timeseries_options['use_prefix'] else ''

            for name, options in phase.control_options.items():
                for ts_name, ts_options in phase._timeseries.items():
                    if f'{control_prefix}{name}' not in ts_options['outputs']:
                        phase.add_timeseries_output(name, output_name=f'{control_prefix}{name}',
                                                    timeseries=ts_name)
                    if f'{control_rate_prefix}{name}_rate' not in ts_options['outputs'] and \
                            (phase.timeseries_options['include_control_rates'] or options['rate_targets']):
                        phase.add_timeseries_output(f'{name}_rate', output_name=f'{control_rate_prefix}{name}_rate',
                                                    timeseries=ts_name)
                    if f'{control_rate_prefix}{name}_rate2' not in ts_options['outputs'] and \
                            (phase.timeseries_options['include_control_rates'] or options['rate2_targets']):
                        phase.add_timeseries_output(f'{name}_rate2', output_name=f'{control_rate_prefix}{name}_rate2',
                                                    timeseries=ts_name)

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_polynomial_controls(self, phase):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.polynomial_control_options:
            sys = PolynomialControlGroup(grid_data=self.grid_data,
                                         polynomial_control_options=phase.polynomial_control_options,
                                         time_units=phase.time_options['units'])
            phase.add_subsystem('polynomial_control_group', subsys=sys,
                                promotes_inputs=['polynomial_controls:*'],
                                promotes_outputs=['polynomial_control_values:*',
                                                  'polynomial_control_rates:*'])

            phase.connect('t_duration_val', 'polynomial_control_group.t_duration')

            prefix = 'polynomial_controls:' if phase.timeseries_options['use_prefix'] else ''
            rate_prefix = 'polynomial_control_rates:' if phase.timeseries_options['use_prefix'] else ''

            for name, options in phase.polynomial_control_options.items():
                for ts_name, ts_options in phase._timeseries.items():
                    if f'{prefix}{name}' not in ts_options['outputs']:
                        phase.add_timeseries_output(name, output_name=f'{prefix}{name}',
                                                    timeseries=ts_name)
                    if f'{rate_prefix}{name}_rate' not in ts_options['outputs'] and \
                            (phase.timeseries_options['include_control_rates'] or options['rate_targets']):
                        phase.add_timeseries_output(f'{name}_rate', output_name=f'{rate_prefix}{name}_rate',
                                                    timeseries=ts_name)
                    if f'{rate_prefix}{name}_rate2' not in ts_options['outputs'] and \
                            (phase.timeseries_options['include_control_rates'] or options['rate2_targets']):
                        phase.add_timeseries_output(f'{name}_rate2', output_name=f'{rate_prefix}{name}_rate2',
                                                    timeseries=ts_name)

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.polynomial_control_options:
            phase.polynomial_control_group.configure_io()

    def setup_parameters(self, phase):
        """
        Sets input defaults for parameters and optionally adds design variables.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._check_parameter_options()
        param_prefix = 'parameters:' if phase.timeseries_options['use_prefix'] else ''
        include_params = phase.timeseries_options['include_parameters']

        for name, options in phase.parameter_options.items():
            if (options['include_timeseries'] is None and include_params) or options['include_timeseries']:
                for ts_name, ts_options in phase._timeseries.items():
                    if f'{param_prefix}{name}' not in ts_options['outputs']:
                        phase.add_timeseries_output(name, output_name=f'{param_prefix}{name}',
                                                    timeseries=ts_name)

    def configure_parameters(self, phase):
        """
        Configure parameter promotion.

        This method assumes that utils.introspection.configure_parameters_introspection has already populated
        the parameter options with the appropriate targets, units, shape, and static_target fields.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.parameter_options:
            param_comp = phase._get_subsystem('param_comp')

            for name, options in phase.parameter_options.items():
                param_comp.add_parameter(name, val=options['val'], shape=options['shape'], units=options['units'])
                if options['opt']:
                    lb = -INF_BOUND if options['lower'] is None else options['lower']
                    ub = INF_BOUND if options['upper'] is None else options['upper']
                    phase.add_design_var(name=f'parameters:{name}',
                                         lower=lb,
                                         upper=ub,
                                         scaler=options['scaler'],
                                         adder=options['adder'],
                                         ref0=options['ref0'],
                                         ref=options['ref'])

                for tgts, src_idxs in self.get_parameter_connections(name, phase):
                    phase.connect(f'parameter_vals:{name}', tgts, src_indices=src_idxs,
                                  flat_src_indices=True)

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method setup_states.')

    def configure_states_introspection(self, phase):
        """
        Perform introspection on the RHS system for the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ode = self._get_ode(phase)
        try:
            configure_states_introspection(phase.state_options, phase.time_options, phase.control_options,
                                           phase.parameter_options, phase.polynomial_control_options, ode)
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(f'Error during configure_states_introspection in phase {phase.pathname}.') from e

    def configure_states_discovery(self, phase):
        """
        Perform introspection on the RHS system to automatically discover states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        ode = self._get_ode(phase)
        try:
            configure_states_discovery(phase.state_options, ode)
        except (ValueError, RuntimeError) as e:
            raise RuntimeError(f'Error during configure_states_discovery in phase {phase.pathname}.') from e

    def configure_states(self, phase):
        """
        Configure the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        state_prefix = 'states:' if phase.timeseries_options['use_prefix'] else ''
        state_rate_prefix = 'state_rates:' if phase.timeseries_options['use_prefix'] else ''

        for name, options in phase.state_options.items():
            for ts_name, ts_options in phase._timeseries.items():
                if f'{state_prefix}{name}' not in ts_options['outputs']:
                    phase.add_timeseries_output(name, output_name=f'{state_prefix}{name}',
                                                timeseries=ts_name)
                if options['rate_source'] and phase.timeseries_options['include_state_rates']:
                    output_name = f'{state_rate_prefix}{name}' if state_rate_prefix else options['rate_source']
                    if output_name not in ts_options['outputs']:
                        phase.add_timeseries_output(name=options['rate_source'],
                                                    output_name=output_name,
                                                    timeseries=ts_name)

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method setup_ode.')

    def setup_duration_balance(self, phase):
        """
        Setup the implicit computation of the phase duration.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """

        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement'
                                  f' method setup_duration_balance.')

    def configure_duration_balance(self, phase):
        """
        Configure the implicit computation of the phase duration.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement'
                                  f' method setup_duration_balance.')

    def setup_solvers(self, phase):
        """
        Setup the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method setup_solvers.')

    def configure_solvers(self, phase):
        """
        Configure the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method '
                                  f'configure_solvers.')

    def setup_timeseries_outputs(self, phase):
        """
        Setup the timeseries for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method '
                                  f'setup_timeseries_outputs.')

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for timeseries_name, timeseries_options in phase._timeseries.items():
            timeseries_comp = phase._get_subsystem(f'{timeseries_name}.timeseries_comp')

            for ts_output_name, ts_output in timeseries_options['outputs'].items():
                name = ts_output['output_name'] if ts_output['output_name'] is not None else ts_output['name']
                units = ts_output['units']
                shape = ts_output['shape']
                src = ts_output['src']
                is_rate = ts_output['is_rate']

                added_src = timeseries_comp._add_output_configure(name,
                                                                  shape=shape,
                                                                  units=units,
                                                                  desc='',
                                                                  src=src,
                                                                  rate=is_rate)

                if added_src:
                    phase.connect(src_name=src, tgt_name=f'{timeseries_name}.input_values:{name}',
                                  src_indices=ts_output['src_idxs'])

    def configure_boundary_constraints(self, phase):
        """
        Configures the boundary constraints.

        Adds BoundaryConstraintComp for initial and/or final boundary constraints if necessary
        and issues appropriate connections.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """

        for ibc in phase._initial_boundary_constraints:
            con_output, constraint_kwargs = self._get_constraint_kwargs('initial', ibc, phase)
            phase.add_constraint(con_output, **constraint_kwargs)

        for fbc in phase._final_boundary_constraints:
            con_output, constraint_kwargs = self._get_constraint_kwargs('final', fbc, phase)
            phase.add_constraint(con_output, **constraint_kwargs)

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
        con_name = constraint_kwargs.pop('constraint_name')

        # Determine the path to the variable which we will be constraining
        var = con_name if options['is_expr'] else options['name']
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
        else:
            if constraint_type == 'initial':
                constraint_kwargs['indices'] = flat_idxs
            elif constraint_type == 'final':
                constraint_kwargs['indices'] = size * (num_nodes - 1) + flat_idxs
            else:
                # This is a path constraint.
                # Remove any flat indices involved in an initial constraint from the path constraint
                flat_idxs_set = set(flat_idxs)
                idxs_not_in_initial = list(flat_idxs_set - idxs_in_initial)

                # Remove any flat indices involved in the final constraint from the path constraint
                idxs_not_in_final = list(flat_idxs_set - idxs_in_final)
                idxs_not_in_final = (size * (num_nodes - 1) + np.asarray(idxs_not_in_final)).tolist()

                intermediate_idxs = []
                for i in range(1, num_nodes - 1):
                    intermediate_idxs.extend(size * i + flat_idxs)

                constraint_kwargs['indices'] = idxs_not_in_initial + intermediate_idxs + idxs_not_in_final

        alias_map = {'path': 'path_constraint',
                     'initial': 'initial_boundary_constraint',
                     'final': 'final_boundary_constraint'}

        str_idxs = '' if options['indices'] is None else f'{options["indices"]}'

        constraint_kwargs['alias'] = f'{phase.pathname}->{alias_map[constraint_type]}->{con_name}{str_idxs}'
        constraint_kwargs.pop('name')
        con_path = constraint_kwargs.pop('constraint_path')
        constraint_kwargs.pop('shape')
        constraint_kwargs['flat_indices'] = True
        constraint_kwargs.pop('is_expr')

        return con_path, constraint_kwargs

    def configure_path_constraints(self, phase):
        """
        Handle the common operations for configuration of the path constraints.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for pc in phase._path_constraints:
            con_output, constraint_kwargs = self._get_constraint_kwargs('path', pc, phase)
            phase.add_constraint(con_output, **constraint_kwargs)

    def configure_objective(self, phase):
        """
        Find the path of the objective(s) and add them.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for name, options in phase._objectives.items():
            index = options['index']
            loc = options['loc']

            obj_path, shape, _, _ = self._get_objective_src(name, loc, phase)

            shape = options['shape'] if shape is None else shape

            size = int(np.prod(shape))

            if size > 1 and index is None:
                raise ValueError(f'Objective variable is non-scaler {shape} but no index specified for objective')

            idx = 0 if index is None else index
            if idx < 0:
                idx = size + idx

            if idx >= size or idx < -size:
                raise ValueError('Objective index={0}, but the shape of the objective '
                                 'variable is {1}'.format(index, shape))

            if loc == 'final':
                obj_index = -size + idx
            elif loc == 'initial':
                obj_index = idx
            else:
                raise ValueError('Invalid value for objective loc: {0}. Must be '
                                 'one of \'initial\' or \'final\'.'.format(loc))

            from ..phase import Phase
            super(Phase, phase).add_objective(obj_path, ref=options['ref'], ref0=options['ref0'],
                                              index=obj_index, flat_indices=True, adder=options['adder'],
                                              scaler=options['scaler'],
                                              parallel_deriv_color=options['parallel_deriv_color'])

    def _get_objective_src(self, name, loc, phase, ode_outputs=None):
        """
        Return the path to the variable that will be used as the objective.

        Parameters
        ----------
        var : str
            Name of the variable to be used as the objective.
        loc : str
            The location of the objective in the phase ['initial', 'final'].
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
            True if the objective quantity1 is linear.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method '
                                  '_get_objective_src.')

    def _get_rate_source_path(self, name, loc, phase):
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method '
                                  '_get_rate_source_path.')

    def _get_ode(self, phase):
        """
        Returns an instance of the ODE used in the phase that can be interrogated for IO metadata.

        Parameters
        ----------
        phase : dm.Phase
            The Phase instance to which this transcription applies

        Returns
        -------
        ode : om.System
            The OpenMDAO system which serves as the ODE for the given Phase.

        """
        return phase._get_subsystem(self._rhs_source)

    def get_parameter_connections(self, name, phase):
        """
        Returns info about a parameter's target connections in the phase.

        Parameters
        ----------
        name : str
            The name of the parameter for which connection information is desired.
        phase : dymos.Phase
            The phase object to which this transcription applies.

        Returns
        -------
        list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method'
                                  f'get_parameter_connections.')

    def is_static_ode_output(self, var, phase, num_nodes):
        """
        Test whether the given output is a static output of the ODE.

        A variable is considered static if it's first dimension is different than the
        number of nodes in the ODE.

        Parameters
        ----------
        var : str
            The ode-relative path of the variable of interest.
        phase : dymos.Phase or dict
            The phase to which this transcription applies or a dict of the ODE outputs as returned by get_promoted_vars.
        num_nodes : int
            The number of nodes in the ODE.

        Returns
        -------
        bool
            True if the given variable is a static output, otherwise False if it is dynamic.

        Raises
        ------
        KeyError
            KeyError is raised if the given variable isn't present in the ode outputs.
        """
        if isinstance(phase, dict):
            ode_outputs = phase
        else:
            ode_outputs = get_promoted_vars(self._get_ode(phase), 'output')
        ode_shape = ode_outputs[var]['shape']
        return ode_shape[0] != num_nodes

    def _requires_continuity_constraints(self, phase):
        """
        Tests whether state and/or control and/or control rate continuity are required.

        Parameters
        ----------
        phase : dymos.Phase
            The phase to which this transcription applies.

        Returns
        -------
        state_continuity : bool
            True if any state continuity is required to be enforced.
        control_continuity : bool
            True if any control value continuity is required to be enforced.
        control_rate_continuity : bool
            True if any control rate continuity is required to be enforced.
        """
        raise NotImplementedError(f'The transcription {self.__class__} does not provide an '
                                  f'implementation of _requires_continuity_constraints')

    def _get_num_timeseries_nodes(self):
        """
        Returns the number of nodes in the default timeseries for this transcription.

        Returns
        -------
        int
            The number of nodes in the default timeseries for this transcription.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method '
                                  '_get_num_timeseries_nodes.')
