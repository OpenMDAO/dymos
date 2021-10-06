from collections import defaultdict

import numpy as np

import openmdao.api as om

from .explicit_timeseries_comp import ExplicitTimeseriesComp
from ..transcription_base import TranscriptionBase
from ..grid_data import GridData
from .rk_integration_comp import RKIntegrationComp, rk_methods
from ...utils.misc import get_rate_units, CoerceDesvar
from ...utils.introspection import get_source_metadata
from ...utils.constants import INF_BOUND


class ExplicitShooting(TranscriptionBase):
    """
    The Transcription class for explicit shooting methods.

    This transcription uses an explicit general Runge-Kutta method to propagate the states using
    the given ODE.

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
        # phase.add_subsystem('indep_states', om.IndepVarComp(),
        #                     promotes_outputs=['*'])
        pass

    def configure_states(self, phase):
        """
        Configure state connections post-introspection.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
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
        subprob = integrator._prob
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
                                            complex_step_mode=True)

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
        integrator_comp = phase._get_subsystem('integrator')
        integrator_comp._configure_parameters_io()

        for param, options in phase.parameter_options.items():
            phase.set_input_defaults(name=f'parameters:{param}', units=options['units'],
                                     val=options['val'])

    def setup_defects(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_defects(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_objective(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_objective(phase)

    def setup_path_constraints(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def configure_path_constraints(self, phase):
        """
        Not used in ExplicitShooting.

        Parameters
        ----------
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        pass

    def setup_boundary_constraints(self, loc, phase):
        """
        Add necessary structure to support boundary constraints to the given phase.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().setup_boundary_constraints(loc, phase)

    def configure_boundary_constraints(self, loc, phase):
        """
        Configure I/O necessary for boundary constraints in the given phase.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.
        phase : dymos.Phase
            The phase object to which this transcription instance applies.
        """
        super().configure_boundary_constraints(loc, phase)

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
            timeseries_comp = ExplicitTimeseriesComp(input_grid_data=gd,
                                                     output_subset='segment_ends')
            phase.add_subsystem(name, subsys=timeseries_comp)

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

        time_units = phase.time_options['units']

        for timeseries_name in phase._timeseries:
            timeseries_comp = phase._get_subsystem(timeseries_name)

            timeseries_comp._add_output_configure('time',
                                                  shape=(1,),
                                                  units=time_units,
                                                  desc='',
                                                  src='time')

            timeseries_comp._add_output_configure('time_phase',
                                                  shape=(1,),
                                                  units=time_units,
                                                  desc='',
                                                  src='time_phase')

            phase.connect(src_name='integrator.time',
                          tgt_name=f'{timeseries_name}.input_values:time')

            phase.connect(src_name='integrator.time_phase',
                          tgt_name=f'{timeseries_name}.input_values:time_phase')

            for state_name, options in phase.state_options.items():

                timeseries_comp._add_output_configure(f'states:{state_name}',
                                                      shape=options['shape'],
                                                      units=options['units'],
                                                      desc=options['desc'],
                                                      src=f'states:{state_name}')

                phase.connect(src_name=f'integrator.states_out:{state_name}',
                              tgt_name=f'{timeseries_name}.input_values:states:{state_name}')

            for control_name, options in phase.control_options.items():
                control_units = options['units']

                timeseries_comp._add_output_configure(f'controls:{control_name}',
                                                      shape=options['shape'],
                                                      units=control_units,
                                                      desc=options['desc'],
                                                      src=f'control_values:{control_name}')

                phase.connect(src_name=f'integrator.control_values:{control_name}',
                              tgt_name=f'{timeseries_name}.input_values:controls:{control_name}')

                # Control rates
                timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=1),
                                                      desc=f'first time-derivative of {control_name}',
                                                      src=f'control_rates:{control_name}_rate')

                phase.connect(src_name=f'integrator.control_rates:{control_name}_rate',
                              tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate')

                # Control second derivatives
                timeseries_comp._add_output_configure(f'control_rates:{control_name}_rate2',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=2),
                                                      desc=f'second time-derivative of {control_name}',
                                                      src=f'control_rates:{control_name}_rate2')

                phase.connect(src_name=f'integrator.control_rates:{control_name}_rate2',
                              tgt_name=f'{timeseries_name}.input_values:control_rates:{control_name}_rate2')

            for control_name, options in phase.polynomial_control_options.items():
                control_units = options['units']

                # Control values
                timeseries_comp._add_output_configure(f'polynomial_controls:{control_name}',
                                                      shape=options['shape'],
                                                      units=control_units,
                                                      desc=options['desc'],
                                                      src=f'polynomial_control_values:{control_name}')

                phase.connect(src_name=f'integrator.polynomial_control_values:{control_name}',
                              tgt_name=f'{timeseries_name}.input_values:polynomial_controls:{control_name}')

                # Control rates
                timeseries_comp._add_output_configure(f'polynomial_control_rates:{control_name}_rate',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=1),
                                                      desc=f'first time-derivative of {control_name}',
                                                      src=f'polynomial_control_rates:{control_name}_rate')

                phase.connect(src_name=f'integrator.polynomial_control_rates:{control_name}_rate',
                              tgt_name=f'{timeseries_name}.input_values:polynomial_control_rates:{control_name}_rate')

                # Control second derivatives
                timeseries_comp._add_output_configure(f'polynomial_control_rates:{control_name}_rate2',
                                                      shape=options['shape'],
                                                      units=get_rate_units(control_units,
                                                                           time_units, deriv=2),
                                                      desc=f'second time-derivative of {control_name}',
                                                      src=f'polynomial_control_rates:{control_name}_rate2')

                phase.connect(src_name=f'integrator.polynomial_control_rates:{control_name}_rate2',
                              tgt_name=f'{timeseries_name}.input_values:polynomial_control_rates:{control_name}_rate2')

            for param_name, options in phase.parameter_options.items():
                if options['include_timeseries']:
                    prom_name = f'parameters:{param_name}'
                    tgt_name = f'input_values:parameters:{param_name}'

                    # Add output.
                    timeseries_comp = phase._get_subsystem(timeseries_name)
                    timeseries_comp._add_output_configure(prom_name, shape=options['shape'],
                                                          units=options['units'], src=prom_name)

                    src_idxs = np.zeros(self.grid_data.subset_num_nodes['segment_ends'], dtype=int)
                    phase.promotes(timeseries_name, inputs=[(tgt_name, prom_name)],
                                   src_indices=om.slicer[src_idxs, ...], src_shape=options['shape'])

            for var, options in integrator_comp._filtered_timeseries_outputs.items():
                added_src = timeseries_comp._add_output_configure(var,
                                                                  shape=options['shape'],
                                                                  units=options['units'],
                                                                  src=options['path'])

                phase.connect(src_name=f'integrator.timeseries:{var}',
                              tgt_name=f'{timeseries_name}.input_values:{var}')

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

    def _get_boundary_constraint_src(self, var, loc, phase):
        """
        Return the path to the variable that will be  constrained.

        Parameters
        ----------
        var : str
            Name of the state.
        loc : str
            The location of the boundary constraint ['intitial', 'final'].
        phase : dymos.Phase
            Phase object containing the rate source.

        Returns
        -------
        str
            Path to the source.
        shape
            Source shape.
        str
            Source units.
        bool
            True if the constraint is linear.
        """
        time_units = phase.time_options['units']
        var_type = phase.classify_var(var)

        if var_type == 'time':
            shape = (1,)
            units = time_units
            linear = True
            if loc == 'initial':
                constraint_path = 't_initial'
            else:
                constraint_path = 'integrator.t_final'
        elif var_type == 'time_phase':
            shape = (1,)
            units = time_units
            linear = True
            constraint_path = 'integrator.time_phase'
        elif var_type == 'state':
            shape = phase.state_options[var]['shape']
            units = phase.state_options[var]['units']
            linear = loc == 'initial'
            constraint_path = f'integrator.states_out:{var}'
        elif var_type in 'indep_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = True
            constraint_path = f'control_values:{var}'
        elif var_type == 'input_control':
            shape = phase.control_options[var]['shape']
            units = phase.control_options[var]['units']
            linear = False
            constraint_path = f'control_values:{var}'
        elif var_type in 'indep_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = True
            constraint_path = f'polynomial_control_values:{var}'
        elif var_type == 'input_polynomial_control':
            shape = phase.polynomial_control_options[var]['shape']
            units = phase.polynomial_control_options[var]['units']
            linear = False
            constraint_path = f'polynomial_control_values:{var}'
        elif var_type == 'parameter':
            shape = phase.parameter_options[var]['shape']
            units = phase.parameter_options[var]['units']
            linear = True
            constraint_path = f'parameters:{var}'
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
            raise NotImplementedError('cannot yet constrain/optimize an ODE output using explicit shooting')
            constraint_path = f'{self._rhs_source}.{var}'
            ode = phase._get_subsystem(self._rhs_source)
            shape, units = get_source_metadata(ode, var, user_units=None, user_shape=None)
            linear = False

        return constraint_path, shape, units, linear

    def get_rate_source_path(self, state_var, phase):
        """
        Return the rate source location for a given state name.

        Parameters
        ----------
        state_var : str
            Name of the state.
        phase : dymos.Phase
            Phase object containing the rate source.

        Returns
        -------
        str
            Path to the rate source.
        """
        var = phase.state_options[state_var]['rate_source']

        if var == 'time':
            rate_path = 'time'
        elif var == 'time_phase':
            rate_path = 'time_phase'
        elif phase.state_options is not None and var in phase.state_options:
            rate_path = f'state_mux_comp.states:{var}'
        elif phase.control_options is not None and var in phase.control_options:
            rate_path = f'control_values:{var}'
        elif phase.polynomial_control_options is not None and var in phase.polynomial_control_options:
            rate_path = f'polynomial_control_values:{var}'
        elif phase.parameter_options is not None and var in phase.parameter_options:
            rate_path = f'parameters:{var}'
        elif var.endswith('_rate') and phase.control_options is not None and \
                var[:-5] in phase.control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate2') and phase.control_options is not None and \
                var[:-6] in phase.control_options:
            rate_path = f'control_rates:{var}'
        elif var.endswith('_rate') and phase.polynomial_control_options is not None and \
                var[:-5] in phase.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        elif var.endswith('_rate2') and phase.polynomial_control_options is not None and \
                var[:-6] in phase.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
        else:
            rate_path = f'ode.{var}'

        return rate_path
