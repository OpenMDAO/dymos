from collections.abc import Sequence

import numpy as np

import openmdao.api as om

from .common import ControlInterpComp, ParameterComp
from ..utils.constants import INF_BOUND
from ..utils.indexing import get_constraint_flat_idxs
from ..utils.introspection import configure_states_introspection, get_promoted_vars, \
    configure_states_discovery, _configure_boundary_balance_introspection
from ..utils.misc import _unspecified, _format_phase_constraint_alias


class TranscriptionBase(object):
    """
    Base class for all dymos transcriptions.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """
    def __init__(self, **kwargs):

        self._implicit_params = False
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

        # Does this transcription have a separate ODE for the phase boundaries?
        self._has_boundary_ode = False

        # Does this transcription include separate variables for the initial and final states?
        self._has_initial_final_states = False

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

        if phase.boundary_balance_options:
            self._implicit_params = True

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
                                           phase.parameter_options, ode)
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
                                                timeseries=ts_name, tags='state')
                if options['rate_source'] and phase.timeseries_options['include_state_rates']:
                    output_name = f'{state_rate_prefix}{name}' if state_rate_prefix else options['rate_source']
                    if output_name not in ts_options['outputs']:
                        phase.add_timeseries_output(name=options['rate_source'],
                                                    output_name=output_name,
                                                    timeseries=ts_name, tags='state_rate')

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method setup_ode.')

    def setup_boundary_balance(self, phase):
        """
        Setup the implicit computation of the phase boundary balance.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        if phase.boundary_balance_options:
            boundary_balance_comp = om.BalanceComp()
            phase.add_subsystem('boundary_balance_comp', boundary_balance_comp, promotes_outputs=['*'])

    def configure_boundary_balance(self, phase):
        """
        Configure the implicit computation of the phase boundary balance.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        param_balance_comp = phase._get_subsystem('boundary_balance_comp')

        _configure_boundary_balance_introspection(phase)

        for param, options in phase.boundary_balance_options.items():
            name = options['name']
            tgt_val = options['tgt_val']
            loc = options['loc']
            index = [options['index']] if np.isscalar(options['index']) else options['index']

            # Get the indices to connect based on loc.
            if loc == 'final':
                src_idxs = om.slicer[(-1, *index)]
            elif loc == 'initial':
                src_idxs = om.slicer[(0, *index)]
            else:
                raise ValueError(f'{phase.msginfo}: Value of `loc` for boundary balance `{param}` '
                                 'must be one of `initial` or `final`, but got `{loc}` instead.')

            # Create the arguments for the balance comp.
            bal_kwargs = {key: options for key, options in options.items()}
            try:
                output_name = bal_kwargs.pop('output_name')
            except KeyError:
                output_name = name.split('.')[-1]
            bal_kwargs.pop('param')
            bal_kwargs.pop('name')
            bal_kwargs.pop('tgt_val')
            bal_kwargs.pop('loc')
            bal_kwargs.pop('index')
            bal_kwargs['rhs_val'] = tgt_val
            bal_kwargs['lhs_name'] = output_name

            prom_param_name = f'parameters:{param}' if param in phase.parameter_options else param

            # Now configure the balance.
            param_balance_comp.add_balance(name=prom_param_name, **bal_kwargs)

            var_type = phase.classify_var(name)
            if var_type == 'ode':
                if name not in phase._timeseries['timeseries']['outputs']:
                    phase.add_timeseries_output(name, output_name=output_name, units=bal_kwargs.get('eq_units', _unspecified))
            if var_type == 'state' and name.startswith('initial_states:') or name.startswith('final_states:'):
                phase.promotes('boundary_balance_comp', inputs=[output_name])
            else:
                phase.connect(f'timeseries.{output_name}', f'boundary_balance_comp.{output_name}',
                              src_indices=src_idxs)

    def setup_solvers(self, phase):
        """
        Setup the solvers for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method setup_solvers.')

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
        if not phase.options['auto_solvers']:
            return

        req_solvers = {'implicit parameters': self._implicit_params}

        if requires_solvers is not None:
            req_solvers.update(requires_solvers)

        reasons_txt = '\n'.join(f'    {key}: {val}' for key, val in req_solvers.items())
        warn = False

        if any(req_solvers.values()):
            msg = (f'{phase.msginfo}\n'
                   f'  Non-default solvers are required\n'
                   f'{reasons_txt}\n')
            if isinstance(phase.nonlinear_solver, om.NonlinearRunOnce):
                msg += (f'  Setting `{phase.pathname}.nonlinear_solver = om.NewtonSolver(iprint=0, '
                        f'solve_subsystems=True, maxiter=1000, stall_limit=3)`\n'
                        f'  Explicitly set {phase.pathname}.nonlinear_solver to override.\n')
                warn = True
                phase.nonlinear_solver = om.NewtonSolver(iprint=0)
                phase.nonlinear_solver.options['solve_subsystems'] = True
                phase.nonlinear_solver.options['maxiter'] = 100
                phase.nonlinear_solver.options['stall_limit'] = 3
                phase.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()

            if isinstance(phase.linear_solver, om.LinearRunOnce):
                warn = True
                msg += (f'  Setting `{phase.pathname}.linear_solver = om.DirectSolver(iprint=2)`\n'
                        f'  Explicitly set {phase.pathname}.linear_solver to override.\n')
                phase.linear_solver = om.DirectSolver(iprint=0)

            if warn:
                msg += f'  Set `{phase.pathname}.options["auto_solvers"] = False` to disable this behavior.'
                om.issue_warning(msg)

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
            timeseries_comp = phase._get_subsystem(timeseries_name)

            for input_name, src, src_idxs in timeseries_comp._configure_io(timeseries_options):
                phase.connect(src_name=src,
                              tgt_name=f'{timeseries_name}.{input_name}',
                              src_indices=src_idxs)

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

        con_path = constraint_kwargs.pop('constraint_path')
        con_name = constraint_kwargs.pop('constraint_name')

        constraint_kwargs['alias'] = _format_phase_constraint_alias(phase, con_name,
                                                                    constraint_type,
                                                                    options['indices'])
        constraint_kwargs.pop('name')
        constraint_kwargs.pop('shape')
        constraint_kwargs['flat_indices'] = True

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

            obj_path, shape, _, _ = self._get_response_src(name, loc, phase)

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

    def _get_response_src(self, name, loc, phase, ode_outputs=None):
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
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method '
                                  '_get_response_src.')

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
        ode = phase._get_subsystem(self._rhs_source)
        if ode is None:
            raise AttributeError(f'Phase ODE subsystem {self._rhs_source} not found.')
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

    def _phase_set_state_val(self, phase, name, vals, times, interpolation_kind):
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
        times : ndarray or Sequence or None
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
        raise NotImplementedError(f'Transcription {self.__class__.__name__} does not implement method '
                                  '_phase_set_val.')

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
