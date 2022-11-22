import numpy as np

import openmdao.api as om

from .explicit_timeseries_comp import ExplicitTimeseriesComp
from .explicit_shooting_continuity_comp import ExplicitShootingContinuityComp
from ..transcription_base import TranscriptionBase
from ..grid_data import GridData
from .rk_integration_comp import RKIntegrationComp, rk_methods
from ...utils.misc import get_rate_units, CoerceDesvar
from ...utils.introspection import get_promoted_vars, get_source_metadata
from ...utils.constants import INF_BOUND
from ..common import TimeseriesOutputGroup


class ExplicitShooting(TranscriptionBase):
    """
    The Transcription class for single explicit shooting.

    This transcription uses an explicit Runge-Kutta method to propagate the states using the
    given ODE through each segment of the phase.  The final value of the states in one
    segment feeds the initial values in a subsequent segment.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of arguments.
    """
    def __init__(self, **kwargs):
        super(ExplicitShooting, self).__init__(**kwargs)
        self._rhs_source = 'ode'

    def initialize(self):
        """
        Declare transcription options.
        """
        self.options.declare('grid', values=['radau-ps', 'gauss-lobatto'],
                             default='gauss-lobatto', desc='The type of transcription used to layout'
                             ' the segments and control discretization nodes.')
        self.options.declare('method', types=(str,), default='rk4',
                             desc='The explicit Runge-Kutta scheme to use. One of' +
                                  str(list(rk_methods.keys())))
        self.options.declare('num_steps_per_segment', types=int,
                             default=10, desc='Number of integration steps in each segment')
        self.options.declare('subprob_reports', default=False,
                             desc='Controls the reports made when running the subproblems for ExplicitShooting')

    def init_grid(self):
        """
        Setup the GridData object for the Transcription.
        """
        self.grid_data = GridData(num_segments=self.options['num_segments'],
                                  transcription=self.options['grid'],
                                  transcription_order=self.options['order'],
                                  segment_ends=self.options['segment_ends'],
                                  compressed=self.options['compressed'])

    def setup_time(self, phase):
        """
        Setup the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase.check_time_options()
        for ts_name, ts_options in phase._timeseries.items():
            if 'time' not in ts_options['outputs']:
                phase.add_timeseries_output('time', timeseries=ts_name)
            if 'time_phase' not in ts_options['outputs']:
                phase.add_timeseries_output('time_phase', timeseries=ts_name)

    def configure_time(self, phase):
        """
        Configure the inputs/outputs on the time component.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_time_io()

        time_options = phase.time_options

        if not time_options['fix_initial']:
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

    def setup_states(self, phase):
        """
        Setup the states for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_states(self, phase):
        """
        Configure state connections post-introspection.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        for name, options in phase.state_options.items():
            for ts_name, ts_options in phase._timeseries.items():
                if f'states:{name}' not in ts_options['outputs']:
                    phase.add_timeseries_output(name, output_name=f'states:{name}',
                                                timeseries=ts_name)
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_states_io()

        # Add the appropriate design parameters
        for state_name, options in phase.state_options.items():
            if options['fix_final']:
                raise ValueError('fix_final is not a valid option for states when using the '
                                 'ExplicitShooting transcription.')
            if options['opt'] and not options['fix_initial']:
                phase.add_design_var(name=f'states:{state_name}',
                                     lower=options['lower'],
                                     upper=options['upper'],
                                     scaler=options['scaler'],
                                     adder=options['adder'],
                                     ref0=options['ref0'],
                                     ref=options['ref'])

    def _get_ode(self, phase):
        integrator = phase._get_subsystem('integrator')
        subprob = integrator._eval_subprob
        ode = subprob.model._get_subsystem('ode_eval.ode')
        return ode

    def setup_ode(self, phase):
        """
        Setup the ode for this transcription.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integrator_comp = RKIntegrationComp(ode_class=phase.options['ode_class'],
                                            time_options=phase.time_options,
                                            state_options=phase.state_options,
                                            parameter_options=phase.parameter_options,
                                            control_options=phase.control_options,
                                            polynomial_control_options=phase.polynomial_control_options,
                                            timeseries_options=phase._timeseries,
                                            method=self.options['method'],
                                            num_steps_per_segment=self.options['num_steps_per_segment'],
                                            grid_data=self.grid_data,
                                            ode_init_kwargs=phase.options['ode_init_kwargs'],
                                            standalone_mode=False,
                                            reports=self.options['subprob_reports'])

        phase.add_subsystem(name='integrator', subsys=integrator_comp, promotes_inputs=['*'])

    def configure_ode(self, phase):
        """
        Create connections to the introspected states.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

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
            for name, options in phase.control_options.items():
                for ts_name, ts_options in phase._timeseries.items():
                    if f'controls:{name}' not in ts_options['outputs']:
                        phase.add_timeseries_output(name, output_name=f'controls:{name}',
                                                    timeseries=ts_name)
                    if f'control_rates:{name}_rate' not in ts_options['outputs']:
                        phase.add_timeseries_output(f'{name}_rate', output_name=f'control_rates:{name}_rate',
                                                    timeseries=ts_name)
                    if f'control_rates:{name}_rate2' not in ts_options['outputs']:
                        phase.add_timeseries_output(f'{name}_rate2', output_name=f'control_rates:{name}_rate2',
                                                    timeseries=ts_name)

    def configure_controls(self, phase):
        """
        Configure the inputs/outputs for the controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_controls(phase)
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_controls_io()

        # Add the appropriate design parameters
        ncin = self.grid_data.subset_num_nodes['control_input']
        for control_name, options in phase.control_options.items():
            if options['opt']:
                coerce_desvar_option = CoerceDesvar(num_input_nodes=ncin, options=options)

                phase.add_design_var(name=f'controls:{control_name}',
                                     lower=coerce_desvar_option('lower'),
                                     upper=coerce_desvar_option('upper'),
                                     scaler=coerce_desvar_option('scaler'),
                                     adder=coerce_desvar_option('adder'),
                                     ref0=coerce_desvar_option('ref0'),
                                     ref=coerce_desvar_option('ref'),
                                     indices=om.slicer[coerce_desvar_option.desvar_indices, ...])

    def setup_polynomial_controls(self, phase):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        phase._check_polynomial_control_options()
        if phase.polynomial_control_options:
            for name, options in phase.polynomial_control_options.items():
                for ts_name, ts_options in phase._timeseries.items():
                    if f'polynomial_controls:{name}' not in ts_options['outputs']:
                        phase.add_timeseries_output(name, output_name=f'polynomial_controls:{name}',
                                                    timeseries=ts_name)
                    if f'polynomial_control_rates:{name}_rate' not in ts_options['outputs']:
                        phase.add_timeseries_output(f'{name}_rate', output_name=f'polynomial_control_rates:{name}_rate',
                                                    timeseries=ts_name)
                    if f'polynomial_control_rates:{name}_rate2' not in ts_options['outputs']:
                        phase.add_timeseries_output(f'{name}_rate2',
                                                    output_name=f'polynomial_control_rates:{name}_rate2',
                                                    timeseries=ts_name)

    def configure_polynomial_controls(self, phase):
        """
        Configure the inputs/outputs for the polynomial controls.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_controls(phase)

        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_polynomial_controls_io()

        # Add the appropriate design parameters
        for name, options in phase.polynomial_control_options.items():
            if options['opt']:
                ncin = options['order'] + 1
                coerce_desvar_option = CoerceDesvar(num_input_nodes=ncin, options=options)

                phase.add_design_var(name=f'polynomial_controls:{name}',
                                     lower=coerce_desvar_option('lower'),
                                     upper=coerce_desvar_option('upper'),
                                     scaler=coerce_desvar_option('scaler'),
                                     adder=coerce_desvar_option('adder'),
                                     ref0=coerce_desvar_option('ref0'),
                                     ref=coerce_desvar_option('ref'),
                                     indices=coerce_desvar_option.desvar_indices)

    def configure_parameters(self, phase):
        """
        Configure parameter promotion.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_parameters(phase)

        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_parameters_io()

        for param, options in phase.parameter_options.items():
            phase.set_input_defaults(name=f'parameters:{param}', units=options['units'],
                                     val=options['val'])

    def setup_defects(self, phase):
        """
        Create the continuity_comp to house the defects.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        state_cont, control_cont, rate_cont = self._requires_continuity_constraints(phase)

        if state_cont or control_cont or rate_cont:
            phase.add_subsystem('continuity_comp',
                                ExplicitShootingContinuityComp(grid_data=self.grid_data,
                                                               state_options=phase.state_options,
                                                               control_options=phase.control_options,
                                                               time_units=phase.time_options['units']))

    def configure_defects(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        any_state_cnty, any_control_cnty, any_rate_cnty = self._requires_continuity_constraints(phase)

        if any((any_state_cnty, any_control_cnty, any_rate_cnty)):
            phase.continuity_comp.configure_io()

        for control_name, options in phase.control_options.items():
            if options['continuity'] and any_control_cnty:
                phase.connect(f'timeseries.controls:{control_name}',
                              f'continuity_comp.controls:{control_name}')
            if options['rate_continuity'] and any_rate_cnty:
                phase.connect(f'timeseries.control_rates:{control_name}_rate',
                              f'continuity_comp.control_rates:{control_name}_rate')
            if options['rate2_continuity'] and any_rate_cnty:
                phase.connect(f'timeseries.control_rates:{control_name}_rate2',
                              f'continuity_comp.control_rates:{control_name}_rate2')

        if any_rate_cnty:
            phase.promotes('continuity_comp', inputs=['t_duration'])

    def setup_solvers(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_solvers(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_storage()

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
            has_expr = False
            for _, output_options in options['outputs'].items():
                if output_options['is_expr']:
                    has_expr = True
                    break
            timeseries_comp = ExplicitTimeseriesComp(input_grid_data=gd,
                                                     output_subset='segment_ends')
            timeseries_group = TimeseriesOutputGroup(has_expr=has_expr, timeseries_output_comp=timeseries_comp)
            phase.add_subsystem(name, subsys=timeseries_group)

    def configure_timeseries_outputs(self, phase):
        """
        Create connections from time series to all post-introspection sources.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_timeseries_outputs()
        for timeseries_name, timeseries_options in phase._timeseries.items():
            timeseries_comp = phase._get_subsystem(f'{timeseries_name}.timeseries_comp')

            for output_name, options in integrator_comp._filtered_timeseries_outputs.items():
                added_src = timeseries_comp._add_output_configure(output_name,
                                                                  shape=options['shape'],
                                                                  units=options['units'],
                                                                  src=options['path'])
                phase.connect(src_name=f'integrator.timeseries:{output_name}',
                              tgt_name=f'{timeseries_name}.input_values:{output_name}')

                if options['path'].split('.')[-1] in timeseries_options['outputs']:
                    timeseries_options['outputs'].pop(options['path'].split('.')[-1])

        super().configure_timeseries_outputs(phase)

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
        return connection_info

    def _get_objective_src(self, var, loc, phase, ode_outputs=None):
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
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

        if var_type == 't':
            shape = (1,)
            units = time_units
            linear = True
            if loc == 'initial':
                obj_path = 't_initial'
            else:
                obj_path = 'integrator.t_final'
        elif var_type == 't_phase':
            shape = (1,)
            units = time_units
            linear = True
            obj_path = 'integrator.t_phase'
        elif var_type == 'state':
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            linear = loc == 'initial'
            obj_path = f'integrator.states_out:{var}'
        elif var_type == 'indep_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = True
            obj_path = f'control_values:{var}'
        elif var_type == 'input_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            obj_path = f'control_values:{var}'
        elif var_type == 'indep_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = True
            obj_path = f'polynomial_control_values:{var}'
        elif var_type == 'input_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = False
            obj_path = f'polynomial_control_values:{var}'
        elif var_type == 'parameter':
            shape = phase.parameter_options[var]['shape']
            units = phase.parameter_options[var]['units']
            linear = True
            obj_path = f'parameter_vals:{var}'
        elif var_type in ('control_rate', 'control_rate2'):
            control_var = var[:-5] if var_type == 'control_rate' else var[:-6]
            shape = phase.control_options[control_var]['shape']
            control_units = phase.control_options[control_var]['units']
            d = 2 if var_type == 'control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            obj_path = f'control_rates:{var}'
        elif var_type in ('polynomial_control_rate', 'polynomial_control_rate2'):
            control_var = var[:-5]
            shape = phase.polynomial_control_options[control_var]['shape']
            control_units = phase.polynomial_control_options[control_var]['units']
            d = 2 if var_type == 'polynomial_control_rate2' else 1
            control_rate_units = get_rate_units(control_units, time_units, deriv=d)
            units = control_rate_units
            linear = False
            obj_path = f'polynomial_control_rates:{var}'
        else:
            # Failed to find variable, assume it is in the ODE. This requires introspection.
            raise NotImplementedError('cannot yet constrain/optimize an ODE output using explicit shooting')
            obj_path = f'{self._rhs_source}.{var}'
            if ode_outputs is None:
                ode = self._get_ode(phase)
            else:
                ode = ode_outputs
            meta = get_source_metadata(ode, var, user_units=None, user_shape=None)
            shape = meta['shape']
            units = meta['units']
            linear = False

        return obj_path, shape, units, linear

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
        num_seg = self.grid_data.num_segments
        compressed = self.grid_data.compressed

        state_continuity = False
        any_control_continuity = any([opts['continuity'] for opts in phase.control_options.values()])
        any_control_continuity = any_control_continuity and num_seg > 1 and not compressed
        any_rate_continuity = any([opts['rate_continuity'] or opts['rate2_continuity']
                                   for opts in phase.control_options.values()])
        any_rate_continuity = any_rate_continuity and num_seg > 1

        return state_continuity, any_control_continuity, any_rate_continuity

    def _get_num_timeseries_nodes(self):
        """
        Returns the number of nodes in the default timeseries for this transcription.

        Returns
        -------
        int
            The number of nodes in the default timeseries for this transcription.
        """
        return self.grid_data.subset_num_nodes['segment_ends']

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
        var_type = phase.classify_var(var)
        time_units = phase.time_options['units']

        transcription = phase.options['transcription']
        ode = transcription._get_ode(phase)
        ode_outputs = get_promoted_vars(ode, 'output')

        # The default for node_idxs, applies to everything except states and parameters.
        node_idxs = None
        meta = {}

        # Determine the path to the variable
        if var_type == 't':
            path = 'integrator.t'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 't_phase':
            path = 'integrator.t_phase'
            src_units = time_units
            src_shape = (1,)
        elif var_type == 'state':
            path = f'integrator.states_out:{var}'
            src_units = phase.state_options[var]['units']
            src_shape = phase.state_options[var]['shape']
        elif var_type in ['indep_control', 'input_control']:
            path = f'integrator.control_values:{var}'
            src_units = phase.control_options[var]['units']
            src_shape = phase.control_options[var]['shape']
        elif var_type == 'control_rate':
            control_name = var[:-5]
            path = f'integrator.control_rates:{control_name}_rate'
            control_name = var[:-5]
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=1)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type == 'control_rate2':
            control_name = var[:-6]
            path = f'integrator.control_rates:{control_name}_rate2'
            src_units = get_rate_units(phase.control_options[control_name]['units'], time_units, deriv=2)
            src_shape = phase.control_options[control_name]['shape']
        elif var_type in ['indep_polynomial_control', 'input_polynomial_control']:
            path = f'integrator.polynomial_control_values:{var}'
            src_units = phase.polynomial_control_options[var]['units']
            src_shape = phase.polynomial_control_options[var]['shape']
        elif var_type == 'polynomial_control_rate':
            control_name = var[:-5]
            path = f'integrator.polynomial_control_rates:{control_name}_rate'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=1)
            src_shape = control['shape']
        elif var_type == 'polynomial_control_rate2':
            control_name = var[:-6]
            path = f'integrator.polynomial_control_rates:{control_name}_rate2'
            control = phase.polynomial_control_options[control_name]
            src_units = get_rate_units(control['units'], time_units, deriv=2)
            src_shape = control['shape']
        elif var_type == 'parameter':
            path = f'parameter_vals:{var}'
            num_seg = self.grid_data.num_segments
            node_idxs = np.zeros(2 * num_seg, dtype=int)
            src_units = phase.parameter_options[var]['units']
            src_shape = phase.parameter_options[var]['shape']
        else:
            # Failed to find variable, assume it is in the ODE
            meta = get_source_metadata(ode_outputs, src=var)
            src_shape = meta['shape']
            src_units = meta['units']
            src_tags = meta['tags']
            if src_tags:
                for tag in src_tags:
                    if 'dymos.state_rate_source' in tag:
                        path = f"integrator.timeseries:state_rates:{tag.split(':')[-1]}"
                        break
                    else:
                        path = f'integrator.timeseries:{var}'
            else:
                path = f'integrator.timeseries:{var}'
            if 'dymos.static_output' in src_tags:
                raise RuntimeError(f'ODE output {var} is tagged with "dymos.static_output" and cannot be a '
                                   f'timeseries output.')

        src_idxs = None if node_idxs is None else om.slicer[node_idxs, ...]

        meta['src'] = path
        meta['src_idxs'] = src_idxs
        meta['units'] = src_units
        meta['shape'] = src_shape

        return meta
