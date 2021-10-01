import numpy as np
import openmdao.api as om

from ...options import options as dymos_options

from .ode_evaluation_group import ODEEvaluationGroup
from ...utils.misc import get_rate_units
from ...utils.introspection import filter_outputs, classify_var


rk_methods = {'rk4': {'a': np.array([[0.0, 0.0, 0.0, 0.0],
                                     [0.5, 0.0, 0.0, 0.0],
                                     [0.0, 0.5, 0.0, 0.0],
                                     [0.0, 0.0, 1.0, 0.0]]),
                      'b': np.array([1/6, 1/3, 1/3, 1/6]),
                      'c': np.array([0, 1/2, 1/2, 1])},

              '3/8': {'a': np.array([[0.0,  0.0, 0.0, 0.0],
                                     [1/3,  0.0, 0.0, 0.0],
                                     [-1/3, 1.0, 0.0, 0.0],
                                     [1.0, -1.0, 1.0, 0.0]]),
                      'b': np.array([1/8, 3/8, 3/8, 1/8]),
                      'c': np.array([0, 1/3, 2/3, 1])},

              'euler': {'a': np.array([[0.0]]),
                        'b': np.array([1.0]),
                        'c': np.array([0.0])},

              'ralston': {'a': np.array([[0, 0], [2/3, 0]]),
                          'c': np.array([0, 2/3]),
                          'b': np.array([1/4, 3/4])},

              'rkf': {'a': np.array([[0,  0,  0,  0,  0],
                                     [1/4, 0, 0, 0, 0],
                                     [3/32, 9/32, 0, 0, 0],
                                     [1932/2197, -7200/2197, 7296/2197, 0, 0],
                                     [439/216, -8, 3680/513, -845/4104, 0],
                                     [-8/27, 2, -3544/2565, 1859/4104, -11/40]]),
                      'c': np.array([0, 1/4, 3/8, 12/13, 1, 1/2]),
                      'b': np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]),
                      'b_star': np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])},

              'rkck': {'a': np.array([[0,  0,  0,  0,  0],
                                      [1/5, 0, 0, 0, 0],
                                      [3/40, 9/40, 0, 0, 0],
                                      [3/10, -9/10, 6/5, 0, 0],
                                      [-11/54, 5/2, -70/27, 35/27, 0],
                                      [1631/55296, 175/512, 575/13828, 44275/110592, 253/4096]]),
                       'c': np.array([0, 1/5, 3/10, 3/5, 1, 7/8]),
                       'b': np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]),
                       'b_star': np.array([37/378, 0, 250/621, 125/594, 512/1771, 0])},

              'dopri': {'a': np.array([[0,  0,  0,  0,  0, 0],
                                       [1/5, 0, 0, 0, 0, 0],
                                       [3/40, 9/40, 0, 0, 0, 0],
                                       [44/45, -56/15, 32/9, 0, 0, 0],
                                       [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
                                       [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
                                       [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]),
                        'c': np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1]),
                        'b': np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]),
                        'b_star': np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])}
              }


class RKIntegrationComp(om.ExplicitComponent):
    """
    A component to perform explicit integration using a generic Runge-Kutta scheme.

    This component contains a sub-Problem with a component that will be solved over num_nodes
    points instead of creating num_nodes instances of that same component and connecting them
    together.

    Parameters
    ----------
    ode_class : class
        The class of the OpenMDAO system to be used to evaluate the ODE in this Group.
    time_options : OptionsDictionary
        OptionsDictionary of time options.
    state_options : dict of {str: OptionsDictionary}
        For each state variable, a dictionary of its options, keyed by name.
    parameter_options : dict of {str: OptionsDictionary}
        For each parameter, a dictionary of its options, keyed by name.
    control_options : dict of {str: OptionsDictionary}
        For each control variable, a dictionary of its options, keyed by name.
    polynomial_control_options : dict of {str: OptionsDictionary}
        For each polynomial variable, a dictionary of its options, keyed by name.
    timeseries_options : dict
        The timeseries options associated with the parent phase. This is used to access
        requested timeseries outputs.  Some options regarding timeseries are not applicable
        to the RungeKutta integration.
    complex_step_mode : bool
        If True, allocate internal memory as complex to support complex-step differentiation.
    grid_data : GridData
        The GridData instance pertaining to the phase to which this ODEEvaluationGroup belongs.
    standalone_mode : bool
        When True, this component will perform its configuration during setup. This is useful
        for unittesting this component when not embedded in a larger system.
    **kwargs : dict
        Additional keyword arguments passed to Group.
    """
    def __init__(self, ode_class, time_options=None,
                 state_options=None, parameter_options=None, control_options=None,
                 polynomial_control_options=None, timeseries_options=None, complex_step_mode=False,
                 grid_data=None, standalone_mode=True, **kwargs):
        super().__init__(**kwargs)
        self.ode_class = ode_class
        self.time_options = time_options
        self.state_options = state_options
        self.parameter_options = parameter_options or {}
        self.control_options = control_options or {}
        self.polynomial_control_options = polynomial_control_options or {}
        self.timeseries_options = timeseries_options or {}
        self._prob = None
        self._complex_step_mode = complex_step_mode
        self._grid_data = grid_data
        self._TYPE = complex if complex_step_mode else float

        self.x_size = 0
        self.p_size = 0
        self.u_size = 0
        self.up_size = 0
        self.theta_size = 0
        self.Z_size = 0

        self._totals_of_names = []
        self._totals_wrt_names = []

        # If _standalone_mode is True, this component will fully perform all of its setup at setup
        # time.  If False, it will need to have configure_io called on it to properly finish its
        # setup.
        self._standalone_mode = standalone_mode
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare options for the RKIntegrationComp.
        """
        self.options.declare('method', types=(str,), default='rk4',
                             desc='The explicit Runge-Kutta scheme to use. One of' +
                                  str(list(rk_methods.keys())))
        self.options.declare('num_steps_per_segment', types=(int,), default=10)
        self.options.declare('ode_init_kwargs', types=dict, allow_none=True, default=None)

    def _setup_subprob(self):
        self._prob = p = om.Problem(comm=self.comm)
        p.model.add_subsystem('ode_eval',
                              ODEEvaluationGroup(self.ode_class, self.time_options,
                                                 self.state_options,
                                                 self.parameter_options,
                                                 self.control_options,
                                                 self.polynomial_control_options,
                                                 ode_init_kwargs=self.options['ode_init_kwargs'],
                                                 grid_data=self._grid_data),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.setup(force_alloc_complex=self._complex_step_mode)
        p.final_setup()
        self._prob.set_complex_step_mode(self._complex_step_mode)

    def _setup_time(self):
        if self._standalone_mode:
            self._configure_time_io()

    def _configure_time_io(self):
        num_output_rows = self._num_output_rows

        self._totals_of_names.append('time')
        self._totals_wrt_names.extend(['time', 't_initial', 't_duration'])

        self.add_input('t_initial', shape=(1,), units=self.time_options['units'])
        self.add_input('t_duration', shape=(1,), units=self.time_options['units'])
        self.add_output('t_final', shape=(1,), units=self.time_options['units'])
        self.add_output('time', shape=(num_output_rows, 1), units=self.time_options['units'])
        self.add_output('time_phase', shape=(num_output_rows, 1), units=self.time_options['units'])

        self.declare_partials('t_final', 't_initial', val=1.0)
        self.declare_partials('t_final', 't_duration', val=1.0)
        self.declare_partials('time', 't_initial', val=1.0)
        self.declare_partials('time', 't_duration', val=1.0)
        self.declare_partials('time_phase', 't_duration', val=1.0)

    def _setup_states(self):
        if self._standalone_mode:
            self._configure_states_io()

    def _configure_states_io(self):
        num_output_rows = self._num_output_rows

        # The total size of the entire state vector
        self.x_size = 0

        self._state_input_names = {}
        self._state_output_names = {}

        # The indices of each state in x
        self.state_idxs = {}

        # The indices of each state's initial value in Z
        self._state_idxs_in_Z = {}

        for state_name, options in self.state_options.items():
            self._state_input_names[state_name] = f'states:{state_name}'
            self._state_output_names[state_name] = f'states_out:{state_name}'

            self._totals_of_names.append(f'state_rate_collector.state_rates:{state_name}_rate')
            self._totals_wrt_names.append(self._state_input_names[state_name])

            self.add_input(self._state_input_names[state_name],
                           shape=options['shape'],
                           units=options['units'],
                           desc=f'initial value of state {state_name}')
            self.add_output(self._state_output_names[state_name],
                            shape=(num_output_rows,) + options['shape'],
                            units=options['units'],
                            desc=f'final value of state {state_name}')

            state_size = np.prod(options['shape'], dtype=int)

            # The indices of the state in x
            self.state_idxs[state_name] = np.s_[self.x_size:self.x_size + state_size]
            self.x_size += state_size

            self.declare_partials(of=self._state_output_names[state_name],
                                  wrt='t_initial')

            self.declare_partials(of=self._state_output_names[state_name],
                                  wrt='t_duration')

            for state_name_wrt in self.state_options:
                self.declare_partials(of=self._state_output_names[state_name],
                                      wrt=f'states:{state_name_wrt}')

            for param_name_wrt in self.parameter_options:
                self.declare_partials(of=self._state_output_names[state_name],
                                      wrt=f'parameters:{param_name_wrt}')

            for control_name_wrt in self.control_options:
                self.declare_partials(of=self._state_output_names[state_name],
                                      wrt=f'controls:{control_name_wrt}')

            for control_name_wrt in self.polynomial_control_options:
                self.declare_partials(of=self._state_output_names[state_name],
                                      wrt=f'polynomial_controls:{control_name_wrt}')

    def _setup_parameters(self):
        if self._standalone_mode:
            self._configure_parameters_io()

    def _configure_parameters_io(self):
        # The indices of each parameter in p
        self.p_size = 0
        self.parameter_idxs = {}
        self._parameter_idxs_in_theta = {}
        self._parameter_idxs_in_Z = {}
        self._param_input_names = {}

        for param_name, options in self.parameter_options.items():
            self._param_input_names[param_name] = f'parameters:{param_name}'
            self._totals_wrt_names.append(self._param_input_names[param_name])

            self.add_input(self._param_input_names[param_name],
                           shape=options['shape'],
                           val=options['val'],
                           units=options['units'],
                           desc=f'value for parameter {param_name}')

            param_size = np.prod(options['shape'], dtype=int)
            self.parameter_idxs[param_name] = np.s_[self.p_size:self.p_size+param_size]
            self.p_size += param_size

    def _setup_controls(self):
        if self._standalone_mode:
            self._configure_controls_io()

    def _configure_controls_io(self):
        self.u_size = 0
        self.control_idxs = {}
        self._control_idxs_in_theta = {}
        self._control_idxs_in_Z = {}
        self._control_idxs_in_y = {}
        self._control_rate_idxs_in_y = {}
        self._control_rate2_idxs_in_y = {}
        self._control_input_names = {}
        self._control_output_names = {}
        self._control_rate_names = {}
        self._control_rate2_names = {}

        num_output_rows = self._num_output_rows

        if self.control_options:
            time_units = self.time_options['units']
            control_input_node_ptau = self._grid_data.node_ptau[
                self._grid_data.subset_node_indices['control_input']]

        for control_name, options in self.control_options.items():
            control_param_shape = (len(control_input_node_ptau),) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._control_input_names[control_name] = f'controls:{control_name}'
            self._control_output_names[control_name] = f'control_values:{control_name}'
            self._control_rate_names[control_name] = f'control_rates:{control_name}_rate'
            self._control_rate2_names[control_name] = f'control_rates:{control_name}_rate2'

            self._totals_wrt_names.append(self._control_input_names[control_name])
            self._totals_of_names.append(self._control_output_names[control_name])
            self._totals_of_names.append(self._control_rate_names[control_name])
            self._totals_of_names.append(self._control_rate2_names[control_name])

            self.add_input(self._control_input_names[control_name],
                           shape=control_param_shape,
                           units=options['units'],
                           desc=f'values for control {control_name} at input nodes')

            self.add_output(self._control_output_names[control_name],
                            shape=(num_output_rows,) + options['shape'],
                            units=options['units'],
                            desc=f'values for control {control_name} at output nodes')

            self.add_output(self._control_rate_names[control_name],
                            shape=(num_output_rows,) + options['shape'],
                            units=get_rate_units(options['units'], time_units, deriv=1),
                            desc=f'values for rate of control {control_name} at output nodes')

            self.add_output(self._control_rate2_names[control_name],
                            shape=(num_output_rows,) + options['shape'],
                            units=get_rate_units(options['units'], time_units, deriv=2),
                            desc=f'values for second derivative rate of control {control_name} at output nodes')

            self.declare_partials(of=self._control_output_names[control_name],
                                  wrt=self._control_input_names[control_name],
                                  val=1.0)

            self.declare_partials(of=self._control_rate_names[control_name],
                                  wrt=self._control_input_names[control_name],
                                  val=1.0)

            self.declare_partials(of=self._control_rate2_names[control_name],
                                  wrt=self._control_input_names[control_name],
                                  val=1.0)

            self.declare_partials(of=self._control_rate_names[control_name],
                                  wrt='t_duration',
                                  val=1.0)

            self.declare_partials(of=self._control_rate2_names[control_name],
                                  wrt='t_duration',
                                  val=1.0)

            self.control_idxs[control_name] = np.s_[self.u_size:self.u_size+control_param_size]
            self.u_size += control_param_size

    def _configure_polynomial_controls_io(self):
        self.up_size = 0
        self.polynomial_control_idxs = {}
        self._polynomial_control_idxs_in_theta = {}
        self._polynomial_control_idxs_in_Z = {}
        self._polynomial_control_input_names = {}
        self._polynomial_control_output_names = {}
        self._polynomial_control_rate_names = {}
        self._polynomial_control_rate2_names = {}
        self._polynomial_control_idxs_in_y = {}
        self._polynomial_control_rate_idxs_in_y = {}
        self._polynomial_control_rate2_idxs_in_y = {}

        num_output_rows = self._num_output_rows
        time_units = self.time_options['units']

        for name, options in self.polynomial_control_options.items():
            num_input_nodes = options['order'] + 1
            control_param_shape = (num_input_nodes,) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)

            self._polynomial_control_input_names[name] = f'polynomial_controls:{name}'
            self._polynomial_control_output_names[name] = f'polynomial_control_values:{name}'
            self._polynomial_control_rate_names[name] = f'polynomial_control_rates:{name}_rate'
            self._polynomial_control_rate2_names[name] = f'polynomial_control_rates:{name}_rate2'

            self._totals_wrt_names.append(self._polynomial_control_input_names[name])
            self._totals_of_names.append(self._polynomial_control_output_names[name])
            self._totals_of_names.append(self._polynomial_control_rate_names[name])
            self._totals_of_names.append(self._polynomial_control_rate2_names[name])

            self.add_input(self._polynomial_control_input_names[name],
                           shape=control_param_shape,
                           units=options['units'],
                           desc=f'values for control {name} at input nodes')

            self.add_output(self._polynomial_control_output_names[name],
                            shape=(num_output_rows,) + options['shape'],
                            units=options['units'],
                            desc=f'values for control {name} at output nodes')

            self.add_output(self._polynomial_control_rate_names[name],
                            shape=(num_output_rows,) + options['shape'],
                            units=get_rate_units(options['units'], time_units, deriv=1),
                            desc=f'values for rate of control {name} at output nodes')

            self.add_output(self._polynomial_control_rate2_names[name],
                            shape=(num_output_rows,) + options['shape'],
                            units=get_rate_units(options['units'], time_units, deriv=2),
                            desc=f'values for second derivative rate of control {name} at output nodes')

            self.declare_partials(of=self._polynomial_control_output_names[name],
                                  wrt=self._polynomial_control_input_names[name],
                                  val=1.0)

            self.declare_partials(of=self._polynomial_control_rate_names[name],
                                  wrt=self._polynomial_control_input_names[name],
                                  val=1.0)

            self.declare_partials(of=self._polynomial_control_rate2_names[name],
                                  wrt=self._polynomial_control_input_names[name],
                                  val=1.0)

            self.declare_partials(of=self._polynomial_control_rate_names[name],
                                  wrt='t_duration',
                                  val=1.0)

            self.declare_partials(of=self._polynomial_control_rate2_names[name],
                                  wrt='t_duration',
                                  val=1.0)

            self.polynomial_control_idxs[name] = np.s_[self.up_size:self.up_size+control_param_size]
            self.up_size += control_param_size

    def _setup_timeseries(self):
        if self._standalone_mode:
            self._configure_timeseries_outputs()

    def _configure_timeseries_outputs(self):
        """
        Creates a mapping of {output_name : {'path': str, 'units': str, 'shape': tuple, 'idxs_in_y': numpy.Indexer}.

        This mapping is used to determine which variables of the ODE need to be saved in the
        algebratic outputs (y) due to being requested as timeseries outputs.
        """
        num_output_rows = self._num_output_rows
        ode_eval = self._prob.model._get_subsystem('ode_eval.ode')

        self._timeseries_output_names = {}
        self._timeseries_idxs_in_y = {}
        self._filtered_timeseries_outputs = {}

        for ts_name, ts_opts in self.timeseries_options.items():
            patterns = list(ts_opts['outputs'].keys())
            matching_outputs = filter_outputs(patterns, ode_eval)

            explicit_requests = set([key for key in
                                     self.timeseries_options[ts_name]['outputs'].keys()
                                     if '*' not in key])

            unmatched_requests = sorted(list(set(explicit_requests) - set(matching_outputs.keys())))

            if unmatched_requests:
                om.issue_warning(msg='The following timeseries outputs were requested but '
                                     f'not found in the ODE: {", ".join(unmatched_requests)}',
                                 category=om.OpenMDAOWarning)

            for var, var_meta in matching_outputs.items():
                if var in self.timeseries_options[ts_name]['outputs']:
                    ts_var_options = self.timeseries_options[ts_name]['outputs'][var]
                    # var explicitly matched
                    output_name = ts_var_options['output_name'] if ts_var_options['output_name'] else var.split('.')[-1]
                    units = ts_var_options.get('units', None) or var_meta.get('units', None)
                    shape = var_meta['shape']
                else:
                    # var matched via wildcard
                    output_name = var.split('.')[-1]
                    units = var_meta['units']
                    shape = var_meta['shape']

                if output_name in self._filtered_timeseries_outputs:
                    raise ValueError(f"Requested timeseries output {var} matches multiple output names "
                                     f"within the ODE. Use `<phase>.add_timeseries_output({var}, "
                                     f"output_name=<new_name>)' to disambiguate the timeseries name.")

                self._filtered_timeseries_outputs[output_name] = {'path': f'ode_eval.ode.{var}',
                                                                  'units': units,
                                                                  'shape': shape}

                ode_eval.add_constraint(var)

                self._timeseries_output_names[output_name] = f'timeseries:{output_name}'
                self._totals_of_names.append(self._filtered_timeseries_outputs[output_name]['path'])

                self.add_output(self._timeseries_output_names[output_name],
                                shape=(num_output_rows,) + shape,
                                units=units,
                                desc=f'values for timeseries output {output_name} at output nodes')

                self.declare_partials(of=self._timeseries_output_names[output_name],
                                      wrt='t_initial')

                self.declare_partials(of=self._timeseries_output_names[output_name],
                                      wrt='t_duration')

                for state_name_wrt in self.state_options:
                    self.declare_partials(of=self._timeseries_output_names[output_name],
                                          wrt=self._state_input_names[state_name_wrt])

                for param_name_wrt in self.parameter_options:
                    self.declare_partials(of=self._timeseries_output_names[output_name],
                                          wrt=self._param_input_names[param_name_wrt])

                for control_name_wrt in self.control_options:
                    self.declare_partials(of=self._timeseries_output_names[output_name],
                                          wrt=self._control_input_names[control_name_wrt])

                for control_name_wrt in self.polynomial_control_options:
                    self.declare_partials(of=self._timeseries_output_names[output_name],
                                          wrt=self._polynomial_control_input_names[control_name_wrt])

    def _setup_storage(self):
        if self._standalone_mode:
            self._configure_storage()

    def _configure_storage(self):
        gd = self._grid_data
        control_input_node_ptau = gd.node_ptau[gd.subset_node_indices['control_input']]

        # allocate the ODE parameter vector
        self.theta_size = 2 + self.p_size + self.u_size + self.up_size

        # allocate the integration parameter vector
        self.Z_size = self.x_size + self.theta_size

        # allocate the algebraic outputs vector
        self.y_size = 3 * self.u_size

        start_Z = 0
        for state_name, options in self.state_options.items():
            state_size = np.prod(options['shape'], dtype=int)
            self._state_idxs_in_Z[state_name] = np.s_[start_Z: start_Z+state_size]
            start_Z += state_size

        start_Z = self.x_size + 2
        start_theta = 2
        for param_name, options in self.parameter_options.items():
            param_size = np.prod(options['shape'], dtype=int)
            self._parameter_idxs_in_Z[param_name] = np.s_[start_Z: start_Z+param_size]
            self._parameter_idxs_in_theta[param_name] = np.s_[start_theta: start_theta+param_size]
            start_Z += param_size
            start_theta += param_size

        start_Z = self.x_size + 2 + self.p_size
        start_theta = 2 + self.p_size
        start_y = 0
        for control_name, options in self.control_options.items():
            control_size = np.prod(options['shape'], dtype=int)
            control_param_shape = (len(control_input_node_ptau),) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._control_idxs_in_Z[control_name] = np.s_[start_Z:start_Z+control_param_size]
            self._control_idxs_in_theta[control_name] = np.s_[start_theta:start_theta+control_param_size]
            self._control_idxs_in_y[control_name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._control_rate_idxs_in_y[control_name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._control_rate2_idxs_in_y[control_name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            start_Z += control_param_size
            start_theta += control_param_size

        start_Z = self.x_size + 2 + self.p_size + self.u_size
        start_theta = 2 + self.p_size + self.u_size
        for name, options in self.polynomial_control_options.items():
            control_size = np.prod(options['shape'], dtype=int)
            num_input_nodes = options['order'] + 1
            control_param_shape = (num_input_nodes,) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._polynomial_control_idxs_in_Z[name] = np.s_[start_Z:start_Z+control_param_size]
            self._polynomial_control_idxs_in_theta[name] = np.s_[start_theta:start_theta+control_param_size]
            self._polynomial_control_idxs_in_y[name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._polynomial_control_rate_idxs_in_y[name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._polynomial_control_rate2_idxs_in_y[name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            start_Z += control_param_size
            start_theta += control_param_size

        for output_name, options in self._filtered_timeseries_outputs.items():
            size = np.prod(options['shape'], dtype=int)
            self._timeseries_idxs_in_y[output_name] = np.s_[start_y:start_y+size]
            start_y += size

        N = self.options['num_steps_per_segment']
        rk = rk_methods[self.options['method']]
        num_rows = self._num_rows
        num_stages = len(rk['b'])
        num_x = self.x_size
        num_theta = self.theta_size
        num_z = num_x + num_theta
        num_y = self.y_size = start_y

        # The contiguous vector of state values
        self._x = np.zeros((num_rows, self.x_size, 1), dtype=self._TYPE)

        # The contiguous vector of time values
        self._t = np.zeros((num_rows, 1), dtype=self._TYPE)

        # The contiguous vector of ODE parameter values
        self._theta = np.zeros((self.theta_size, 1), dtype=self._TYPE)

        # The contiguous vector of state rates
        self._f = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The contiguous vector of ODE algebraic outputs
        self._y = np.zeros((num_rows, num_y, 1), dtype=self._TYPE)

        # The derivatives of the state rates wrt the current time
        self._f_t = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The derivatives of the state rates wrt the current state
        self._f_x = np.zeros((self.x_size, self.x_size), dtype=self._TYPE)

        # The derivatives of the state rates wrt the parameters
        self._f_theta = np.zeros((self.x_size, self.theta_size), dtype=self._TYPE)

        # The derivatives of the state rates wrt the current time
        self._y_t = np.zeros((self.y_size, 1), dtype=self._TYPE)

        # The derivatives of the state rates wrt the current state
        self._y_x = np.zeros((self.y_size, self.x_size), dtype=self._TYPE)

        # The derivatives of the state rates wrt the parameters
        self._y_theta = np.zeros((self.y_size, self.theta_size), dtype=self._TYPE)

        # Intermediate state rate storage
        self._k_q = np.zeros((num_stages, self.x_size, 1), dtype=self._TYPE)

        # The partial derivative of the final state vector wrt time from state update equation.
        self.px_pt = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The partial derivative of the final state vector wrt the initial state vector from the
        # state update equation.
        self.px_px = np.zeros((self.x_size, self.x_size), dtype=self._TYPE)

        # Derivatives pertaining to the stage ODE evaluations
        self._dTi_dZ = np.zeros((1, num_z), dtype=self._TYPE)
        self._dXi_dZ = np.zeros((num_x, num_z), dtype=self._TYPE)
        self._dkq_dZ = np.zeros((num_stages, num_x, num_z), dtype=self._TYPE)

        # The ODE parameter derivatives wrt the integration parameters
        self._dtheta_dZ = np.zeros((num_theta, num_z), dtype=self._TYPE)
        self._dtheta_dZ[:, num_x:] = np.eye(num_theta, dtype=self._TYPE)

        # Total derivatives of evolving quantities (x, t, h) wrt the integration parameters.
        # Let Z be [x0.ravel() t0 tp p.ravel() u.ravel()]
        self._dx_dZ = np.zeros((num_rows, num_x, num_z), dtype=self._TYPE)
        self._dx_dZ[:, :, :num_x] = np.eye(num_x, dtype=self._TYPE)
        self._dt_dZ = np.zeros((num_rows, 1, num_z), dtype=self._TYPE)
        self._dt_dZ[:, 0, num_x] = 1.0
        self._dh_dZ = np.zeros((num_rows, 1, num_z), dtype=self._TYPE)
        self._dh_dZ[:, 0, num_x+1] = 1.0 / N

        # Total derivatives of ODE outputs (y) wrt the integration parameters.
        self._dy_dZ = np.zeros((num_rows, num_y, num_z), dtype=self._TYPE)

    def setup(self):
        """
        Add the necessary I/O and storage for the RKIntegrationComp.
        """
        gd = self._grid_data
        N = self.options['num_steps_per_segment']

        # Indices to map the rows to output rows
        temp = np.zeros((gd.num_segments, N+1))
        temp[:, 0] = 1
        temp[:, -1] = 1
        self._output_src_idxs = np.where(temp.ravel() == 1)[0]

        self._num_output_rows = gd.num_segments * 2
        self._num_rows = gd.num_segments * (N + 1)

        self._totals_of_names = []
        self._totals_wrt_names = []

        self._setup_subprob()
        self._setup_time()
        self._setup_parameters()
        self._setup_controls()
        self._setup_states()
        self._setup_timeseries()
        self._setup_storage()

    def _reset_derivs(self):
        """
        Reset the value of total derivatives prior to propagation.
        """
        N = self.options['num_steps_per_segment']
        num_x = self.x_size
        num_theta = self.theta_size

        # Let Z be [x0.ravel() t0 tp p.ravel() u.ravel()]
        self._dx_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._TYPE)
        self._dx_dZ[0, :, :num_x] = np.eye(num_x, dtype=self._TYPE)
        self._dt_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._TYPE)
        self._dt_dZ[0, 0, num_x] = 1.0
        self._dh_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._TYPE)
        self._dh_dZ[:, 0, num_x+1] = 1.0 / N
        self._dTi_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._TYPE)
        self._dXi_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._TYPE)
        self._dkq_dZ[...] = 0.0  # np.zeros((num_stages, num_x, num_z), dtype=self._TYPE)
        self._dtheta_dZ[...] = 0.0  # np.zeros((num_theta, num_z), dtype=self._TYPE)
        self._dtheta_dZ[:, num_x:] = np.eye(num_theta, dtype=self._TYPE)

    def _initialize_segment(self, row, inputs=None, derivs=False):
        """
        Set the derivatives at the current row to those of the previous row.\

        This is used to continue the value of derivatives over a segment boundary.
        """
        if row == 0:
            # start x, t, and h
            for state_name in self.state_options:
                i_name = self._state_input_names[state_name]
                self._x[0, self.state_idxs[state_name], 0] = inputs[i_name].ravel()
            self._t[0, 0] = inputs['t_initial'].copy()

            if derivs:
                self._reset_derivs()
        else:
            # copy last x, t, h
            self._x[row, ...] = self._x[row-1, ...]
            self._t[row, ...] = self._t[row-1, ...]

            if derivs:
                # The 3 arrays of propagated derivatives need to copy over previous values
                self._dx_dZ[row, ...] = self._dx_dZ[row-1, ...]
                self._dt_dZ[row, ...] = self._dt_dZ[row-1, ...]
                self._dh_dZ[row, ...] = self._dh_dZ[row-1, ...]

                # Derivatives of the internal calls are just reset
                self._dTi_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._TYPE)
                self._dXi_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._TYPE)
                self._dkq_dZ[...] = 0.0  # np.zeros((num_stages, num_x, num_z), dtype=self._TYPE)

                # dtheta_dZ remains constant across segments
                # self._dtheta_dZ[...] = 0.0  # np.zeros((num_theta, num_z), dtype=self._TYPE)
                # self._dtheta_dZ[:, num_x:] = np.eye(num_theta, dtype=self._TYPE)

    def eval_f(self, x, t, theta, f, y=None):
        """
        Evaluate the ODE which provides the state rates for integration.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.
        theta : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.
        f : np.ndarray
            A flattened, contiguous vector of the state rates.
        y : np.ndarray or None
            A flattened, contiguous vector of the auxiliary ODE outputs, if desired.
            If present, the first positions are reserved for the contiguous control values, rates,
            and second derivatives, respectively. The remaining elements are the requested ODE-based
            timeseries outputs.
        """
        # transcribe time
        self._prob.set_val('time', t, units=self.time_options['units'])
        self._prob.set_val('t_initial', theta[0], units=self.time_options['units'])
        self._prob.set_val('t_duration', theta[1], units=self.time_options['units'])

        # transcribe states
        for name in self.state_options:
            self._prob.set_val(self._state_input_names[name], x[self.state_idxs[name], 0])

        # transcribe parameters
        for name in self.parameter_options:
            self._prob.set_val(self._param_input_names[name], theta[self._parameter_idxs_in_theta[name]])

        # transcribe controls
        for name in self.control_options:
            self._prob.set_val(self._control_input_names[name], theta[self._control_idxs_in_theta[name]])

        # transcribe polynomial controls
        for name in self.polynomial_control_options:
            self._prob.set_val(self._polynomial_control_input_names[name],
                               theta[self._polynomial_control_idxs_in_theta[name]])

        # execute the ODE
        self._prob.run_model()

        # pack the resulting array
        for name in self.state_options:
            f[self.state_idxs[name]] = self._prob.get_val(f'state_rate_collector.state_rates:{name}_rate').ravel()

        if y is not None:
            # pack any control values and rates into y
            for name in self.control_options:
                output_name = self._control_output_names[name]
                rate_name = self._control_rate_names[name]
                rate2_name = self._control_rate2_names[name]
                y[self._control_idxs_in_y[name]] = self._prob.get_val(output_name).ravel()
                y[self._control_rate_idxs_in_y[name]] = self._prob.get_val(rate_name).ravel()
                y[self._control_rate2_idxs_in_y[name]] = self._prob.get_val(rate2_name).ravel()

            # pack any polynomial control values and rates into y
            for name in self.polynomial_control_options:
                output_name = self._polynomial_control_output_names[name]
                rate_name = self._polynomial_control_rate_names[name]
                rate2_name = self._polynomial_control_rate2_names[name]
                y[self._polynomial_control_idxs_in_y[name]] = self._prob.get_val(output_name).ravel()
                y[self._polynomial_control_rate_idxs_in_y[name]] = self._prob.get_val(rate_name).ravel()
                y[self._polynomial_control_rate2_idxs_in_y[name]] = self._prob.get_val(rate2_name).ravel()

            # pack any polynomial control values and rates into y

            for output_name, options in self._filtered_timeseries_outputs.items():
                path = options['path']
                y[self._timeseries_idxs_in_y[output_name]] = self._prob.get_val(path).ravel()

    def eval_f_derivs(self, x, t, theta, f_x, f_t, f_theta, y_x=None, y_t=None, y_theta=None):
        """
        Evaluate the derivative of the ODE output rates wrt the inputs.

        Note that the control parameterization `u` undergoes an interpolation to provide the
        control values at any given time.  The ODE is then a function of these interpolated control
        values, we'll call them `u_hat`.  Technically, the derivatives wrt to `u` need to be chained
        together, but in this implementation the interpolation is part of the execution of the ODE
        and the chained derivatives are captured correctly there.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.
        theta : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.
        f_x : np.ndarray
            A matrix of the derivative of each element of the rates `f` wrt each value in `x`.
        f_t : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt `time`.
        f_theta : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt the parameters `theta`.
        y_x : np.ndarray
            A matrix of the derivative of each element of the rates `y` wrt each value in `x`.
        y_t : np.ndarray
            A matrix of the derivatives of each element of the rates `y` wrt `time`.
        y_theta : np.ndarray
            A matrix of the derivatives of each element of the rates `y` wrt the parameters `theta`.
        """
        # transcribe time
        self._prob.set_val('time', t, units=self.time_options['units'])
        self._prob.set_val('t_initial', theta[0, 0], units=self.time_options['units'])
        self._prob.set_val('t_duration', theta[1, 0], units=self.time_options['units'])

        # transcribe states
        for name in self.state_options:
            input_name = self._state_input_names[name]
            self._prob.set_val(input_name, x[self.state_idxs[name], 0])

        # transcribe parameters
        for name in self.parameter_options:
            input_name = self._param_input_names[name]
            self._prob.set_val(input_name, theta[self._parameter_idxs_in_theta[name], 0])

        # transcribe controls
        for name in self.control_options:
            input_name = self._control_input_names[name]
            self._prob.set_val(input_name, theta[self._control_idxs_in_theta[name], 0])

        for name in self.polynomial_control_options:
            input_name = self._polynomial_control_input_names[name]
            self._prob.set_val(input_name, theta[self._polynomial_control_idxs_in_theta[name], 0])

        # Re-run in case the inputs have changed.
        self._prob.run_model()

        totals = self._prob.compute_totals(of=self._totals_of_names, wrt=self._totals_wrt_names,
                                           use_abs_names=False)

        for state_name in self.state_options:
            of_name = f'state_rate_collector.state_rates:{state_name}_rate'
            idxs = self.state_idxs[state_name]
            f_t[self.state_idxs[state_name]] = totals[of_name, 'time']

            for state_name_wrt in self.state_options:
                idxs_wrt = self.state_idxs[state_name_wrt]
                px_px = totals[of_name, self._state_input_names[state_name_wrt]]
                f_x[idxs, idxs_wrt] = px_px.ravel()

            f_theta[idxs, 0] = totals[of_name, 't_initial']
            f_theta[idxs, 1] = totals[of_name, 't_duration']

            for param_name_wrt in self.parameter_options:
                idxs_wrt = self._parameter_idxs_in_theta[param_name_wrt]
                px_pp = totals[of_name, self._param_input_names[param_name_wrt]]
                f_theta[idxs, idxs_wrt] = px_pp.ravel()

            for control_name_wrt in self.control_options:
                idxs_wrt = self._control_idxs_in_theta[control_name_wrt]
                px_pu = totals[of_name, self._control_input_names[control_name_wrt]]
                f_theta[idxs, idxs_wrt] = px_pu.ravel()

            for pc_name_wrt in self.polynomial_control_options:
                idxs_wrt = self._polynomial_control_idxs_in_theta[pc_name_wrt]
                px_pu = totals[of_name, self._polynomial_control_input_names[pc_name_wrt]]
                f_theta[idxs, idxs_wrt] = px_pu.ravel()

        if y_x is not None and y_t is not None and y_theta is not None:
            for control_name in self.control_options:
                wrt_name = self._control_input_names[control_name]
                idxs_wrt = self._control_idxs_in_theta[control_name]
                of_name = self._control_output_names[control_name]
                of_rate_name = self._control_rate_names[control_name]
                of_rate2_name = self._control_rate2_names[control_name]

                of_idxs = self._control_idxs_in_y[control_name]
                of_rate_idxs = self._control_rate_idxs_in_y[control_name]
                of_rate2_idxs = self._control_rate2_idxs_in_y[control_name]

                y_t[of_idxs, 0] = totals[of_name, 'time']
                y_t[of_rate_idxs, 0] = totals[of_rate_name, 'time']
                y_t[of_rate2_idxs, 0] = totals[of_rate2_name, 'time']

                y_theta[of_idxs, 1] = totals[of_name, 't_duration']
                y_theta[of_rate_idxs, 1] = totals[of_rate_name, 't_duration']
                y_theta[of_rate2_idxs, 1] = totals[of_rate2_name, 't_duration']

                y_theta[of_idxs, idxs_wrt] = totals[of_name, wrt_name]
                y_theta[of_rate_idxs, idxs_wrt] = totals[of_rate_name, wrt_name]
                y_theta[of_rate2_idxs, idxs_wrt] = totals[of_rate2_name, wrt_name]

            for polynomial_control_name in self.polynomial_control_options:
                wrt_name = self._polynomial_control_input_names[polynomial_control_name]
                idxs_wrt = self._polynomial_control_idxs_in_theta[polynomial_control_name]
                of_name = self._polynomial_control_output_names[polynomial_control_name]
                of_rate_name = self._polynomial_control_rate_names[polynomial_control_name]
                of_rate2_name = self._polynomial_control_rate2_names[polynomial_control_name]

                of_idxs = self._polynomial_control_idxs_in_y[polynomial_control_name]
                of_rate_idxs = self._polynomial_control_rate_idxs_in_y[polynomial_control_name]
                of_rate2_idxs = self._polynomial_control_rate2_idxs_in_y[polynomial_control_name]

                y_t[of_idxs, 0] = totals[of_name, 'time']
                y_t[of_rate_idxs, 0] = totals[of_rate_name, 'time']
                y_t[of_rate2_idxs, 0] = totals[of_rate2_name, 'time']

                y_theta[of_idxs, 1] = totals[of_name, 't_duration']
                y_theta[of_rate_idxs, 1] = totals[of_rate_name, 't_duration']
                y_theta[of_rate2_idxs, 1] = totals[of_rate2_name, 't_duration']

                y_theta[of_idxs, idxs_wrt] = totals[of_name, wrt_name]
                y_theta[of_rate_idxs, idxs_wrt] = totals[of_rate_name, wrt_name]
                y_theta[of_rate2_idxs, idxs_wrt] = totals[of_rate2_name, wrt_name]

            for name, options in self._filtered_timeseries_outputs.items():
                idxs_of = self._timeseries_idxs_in_y[name]
                of_name = options['path']

                y_t[idxs_of, 0] = totals[options['path'], 'time']

                y_theta[idxs_of, 0] = totals[of_name, 't_initial']
                y_theta[idxs_of, 1] = totals[of_name, 't_duration']

                for state_name_wrt in self.state_options:
                    idxs_wrt = self.state_idxs[state_name_wrt]
                    py_px = totals[of_name, self._state_input_names[state_name_wrt]]
                    y_x[idxs_of, idxs_wrt] = py_px.ravel()

                for param_name_wrt in self.parameter_options:
                    idxs_wrt = self._parameter_idxs_in_theta[param_name_wrt]
                    py_pp = totals[of_name, self._param_input_names[param_name_wrt]]
                    y_theta[idxs_of, idxs_wrt] = py_pp.ravel()

                for control_name_wrt in self.control_options:
                    idxs_wrt = self._control_idxs_in_theta[control_name_wrt]
                    py_puhat = totals[of_name, self._control_input_names[control_name_wrt]]
                    y_theta[idxs_of, idxs_wrt] = py_puhat.ravel()

                for pc_name_wrt in self.polynomial_control_options:
                    idxs_wrt = self._polynomial_control_idxs_in_theta[pc_name_wrt]
                    py_puhat = totals[of_name, self._polynomial_control_input_names[pc_name_wrt]]
                    y_theta[idxs_of, idxs_wrt] = py_puhat.ravel()

    def _propagate(self, inputs, outputs, derivs=None):
        """
        Propagate the states from t_initial to t_initial + t_duration, optionally computing
        the derivatives along the way and caching the current time and state values.

        Parameters
        ----------
        inputs : vector
            The inputs from the compute call to the RKIntegrationComp.
        outputs : vector
            The outputs from the compute call to the RKIntegrationComp.
        derivs : vector or None
            If derivatives are to be calculated in a forward mode, this is the vector of partials
            from the compute_partials call to this component.  If derivatives are not to be
            computed, this should be None.
        """
        gd = self._grid_data
        N = self.options['num_steps_per_segment']

        # RK Constants
        rk = rk_methods[self.options['method']]
        a = rk['a']
        b = rk['b']
        c = rk['c']
        num_stages = len(b)

        # Initialize states
        x = self._x
        t = self._t
        theta = self._theta

        # Make t_initial and t_duration the first two elements of the ODE parameter vector.
        theta[0] = inputs['t_initial'].copy()
        theta[1] = inputs['t_duration'].copy()

        f_x = self._f_x
        f_t = self._f_t
        f_theta = self._f_theta

        y_x = self._y_x
        y_t = self._y_t
        y_theta = self._y_theta

        # Initialize parameters
        for name in self.parameter_options:
            theta[self._parameter_idxs_in_theta[name], 0] = inputs[f'parameters:{name}'].ravel()

        # Initialize controls
        for name in self.control_options:
            theta[self._control_idxs_in_theta[name], 0] = inputs[f'controls:{name}'].ravel()

        # Initialize polynomial controls
        for name in self.polynomial_control_options:
            theta[self._polynomial_control_idxs_in_theta[name], 0] = \
                inputs[f'polynomial_controls:{name}'].ravel()

        seg_durations = theta[1] * np.diff(gd.segment_ends) / 2.0

        # step counter
        row = 0

        for seg_i in range(gd.num_segments):
            self._prob.model._get_subsystem('ode_eval').set_segment_index(seg_i)

            # Initialize, t, x, h, and derivatives for the start of the current segment
            self._initialize_segment(row, inputs, derivs=derivs)

            h = np.asarray(seg_durations[seg_i] / N, dtype=self._TYPE)
            # On each segment, the total derivative of the stepsize h is a function of
            # the duration of the phase (the second element of the parameter vector after states)
            if derivs:
                self._dh_dZ[row:row+N+1, 0, self.x_size+1] = seg_durations[seg_i] / theta[1] / N

            rm1 = row
            row = row + 1

            for q in range(N):
                # Compute the state rates and their partials at the start of the step
                self.eval_f(x[rm1, ...], t[rm1, 0], theta, self._k_q[0, ...], y=self._y[rm1, ...])

                if derivs:
                    # Compute the state rate derivatives
                    self.eval_f_derivs(x[rm1, ...], t[rm1, 0], theta, f_x, f_t, f_theta,
                                       y_x, y_t, y_theta)

                    self._dkq_dZ[0, ...] = \
                        f_t @ self._dt_dZ[rm1, ...] + f_x @ self._dx_dZ[rm1, ...] + \
                        f_theta @ self._dtheta_dZ

                    self._dy_dZ[rm1, ...] = \
                        y_x @ self._dx_dZ[rm1, ...] + y_t @ self._dt_dZ[rm1, ...] + \
                        y_theta @ self._dtheta_dZ

                for i in range(1, num_stages):
                    T_i = t[rm1, ...] + c[i] * h
                    a_tdot_k = np.tensordot(a[i, :i], self._k_q[:i, ...], axes=(0, 0))
                    # a_tdot_k = np.einsum('i,ijk->jk', a[i, :i], self._k_q[:i, ...])
                    X_i = x[rm1, ...] + h * a_tdot_k
                    self.eval_f(X_i, T_i, theta, self._k_q[i, ...])

                    if derivs:
                        self.eval_f_derivs(X_i, T_i, theta, f_x, f_t, f_theta)
                        self._dTi_dZ[...] = self._dt_dZ[row - 1, ...] + c[i] * self._dh_dZ[rm1, ...]
                        a_tdot_dkqdz = np.tensordot(a[i, :i], self._dkq_dZ[:i, ...], axes=(0, 0))
                        # a_tdot_dkqdz = np.einsum('i,ijk->jk', a[i, :i], self._dkq_dZ[:i, ...])
                        self._dXi_dZ[...] = self._dx_dZ[rm1, ...] + a_tdot_k @ self._dh_dZ[rm1, ...] + h * a_tdot_dkqdz
                        self._dkq_dZ[i, ...] = f_t @ self._dTi_dZ + f_x @ self._dXi_dZ + f_theta @ self._dtheta_dZ

                b_tdot_kq = np.tensordot(b, self._k_q, axes=(0, 0))
                # b_tdot_kq = np.einsum('i,ijk->jk', b, self._k_q)
                x[row, ...] = x[rm1, ...] + h * b_tdot_kq
                t[row, 0] = t[rm1, 0] + h

                if derivs:
                    b_tdot_dkqdz = np.tensordot(b, self._dkq_dZ, axes=(0, 0))
                    # b_tdot_dkqdz = np.einsum('i,ijk->jk', b, self._dkq_dZ)
                    self._dx_dZ[row, ...] = \
                        self._dx_dZ[rm1, ...] + b_tdot_kq @ self._dh_dZ[rm1, ...] + h * b_tdot_dkqdz
                    self._dt_dZ[row, ...] = self._dt_dZ[rm1, ...] + self._dh_dZ[rm1, ...]

                rm1 = row
                row = row + 1

            # Evaluate the ODE at the last point in the segment (with the final times and states)
            self.eval_f(x[rm1, ...], t[rm1, 0], theta, self._k_q[0, ...], y=self._y[rm1, ...])

            if derivs:
                self.eval_f_derivs(x[rm1, ...], t[rm1, 0], theta, f_x, f_t, f_theta, y_x, y_t, y_theta)
                self._dy_dZ[rm1, ...] = y_x @ self._dx_dZ[rm1, ...] + y_t @ self._dt_dZ[rm1, ...] + y_theta @ self._dtheta_dZ

    def compute(self, inputs, outputs):
        """
        Compute propagated state values.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        self._propagate(inputs, outputs)

        # Unpack the outputs
        idxs = self._output_src_idxs
        outputs['t_final'] = self._t[-1, ...]

        # Extract time
        outputs['time'] = self._t[idxs, ...]
        outputs['time_phase'] = self._t[idxs, ...] - inputs['t_initial']

        # Extract the state values
        for state_name, options in self.state_options.items():
            of = self._state_output_names[state_name]
            outputs[of] = self._x[idxs, self.state_idxs[state_name]]

        # Extract the control values and rates
        for control_name, options in self.control_options.items():
            oname = self._control_output_names[control_name]
            rate_name = self._control_rate_names[control_name]
            rate2_name = self._control_rate2_names[control_name]
            outputs[oname] = self._y[idxs, self._control_idxs_in_y[control_name]]
            outputs[rate_name] = self._y[idxs, self._control_rate_idxs_in_y[control_name]]
            outputs[rate2_name] = self._y[idxs, self._control_rate2_idxs_in_y[control_name]]

        # Extract the control values and rates
        for control_name, options in self.polynomial_control_options.items():
            oname = self._polynomial_control_output_names[control_name]
            rate_name = self._polynomial_control_rate_names[control_name]
            rate2_name = self._polynomial_control_rate2_names[control_name]
            outputs[oname] = self._y[idxs, self._polynomial_control_idxs_in_y[control_name]]
            outputs[rate_name] = self._y[idxs, self._polynomial_control_rate_idxs_in_y[control_name]]
            outputs[rate2_name] = self._y[idxs, self._polynomial_control_rate2_idxs_in_y[control_name]]

        # Extract the timeseries outputs
        for name, options in self._filtered_timeseries_outputs.items():
            oname = self._timeseries_output_names[name]
            outputs[oname] = self._y[idxs, self._timeseries_idxs_in_y[name]]

    def compute_partials(self, inputs, partials):
        """
        Compute derivatives of propagated states wrt the inputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        self._propagate(inputs, outputs=False, derivs=True)

        idxs = self._output_src_idxs
        partials['time', 't_duration'] = self._dt_dZ[idxs, 0, self.x_size+1]
        partials['time_phase', 't_duration'] = self._dt_dZ[idxs, 0, self.x_size+1]

        for state_name in self.state_options:
            of = self._state_output_names[state_name]

            # Unpack the derivatives
            of_rows = self.state_idxs[state_name]

            partials[of, 't_initial'] = self._dx_dZ[idxs, of_rows, self.x_size]
            partials[of, 't_duration'] = self._dx_dZ[idxs, of_rows, self.x_size+1]

            for wrt_state_name in self.state_options:
                wrt = self._state_input_names[wrt_state_name]
                wrt_cols = self._state_idxs_in_Z[wrt_state_name]
                partials[of, wrt] = self._dx_dZ[idxs, of_rows, wrt_cols]

            for wrt_param_name in self.parameter_options:
                wrt = self._param_input_names[wrt_param_name]
                wrt_cols = self._parameter_idxs_in_Z[wrt_param_name]
                partials[of, wrt] = self._dx_dZ[idxs, of_rows, wrt_cols]

            for wrt_control_name in self.control_options:
                wrt = self._control_input_names[wrt_control_name]
                wrt_cols = self._control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = self._dx_dZ[idxs, of_rows, wrt_cols]

            for wrt_pc_name in self.polynomial_control_options:
                wrt = self._polynomial_control_input_names[wrt_pc_name]
                wrt_cols = self._polynomial_control_idxs_in_Z[wrt_pc_name]
                partials[of, wrt] = self._dx_dZ[idxs, of_rows, wrt_cols]

        for control_name in self.control_options:
            of = self._control_output_names[control_name]
            of_rate = self._control_rate_names[control_name]
            of_rate2 = self._control_rate2_names[control_name]

            # Unpack the derivatives
            of_rows = self._control_idxs_in_y[control_name]
            of_rate_rows = self._control_rate_idxs_in_y[control_name]
            of_rate2_rows = self._control_rate2_idxs_in_y[control_name]

            wrt_cols = self.x_size + 1
            partials[of_rate, 't_duration'] = self._dy_dZ[idxs, of_rate_rows, wrt_cols]
            partials[of_rate2, 't_duration'] = self._dy_dZ[idxs, of_rate2_rows, wrt_cols]

            for wrt_control_name in self.control_options:
                wrt = self._control_input_names[wrt_control_name]
                wrt_cols = self._control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = self._dy_dZ[idxs, of_rows, wrt_cols]
                partials[of_rate, wrt] = self._dy_dZ[idxs, of_rate_rows, wrt_cols]
                partials[of_rate2, wrt] = self._dy_dZ[idxs, of_rate2_rows, wrt_cols]

        for name in self.polynomial_control_options:
            of = self._polynomial_control_output_names[name]
            of_rate = self._polynomial_control_rate_names[name]
            of_rate2 = self._polynomial_control_rate2_names[name]

            # Unpack the derivatives
            of_rows = self._polynomial_control_idxs_in_y[name]
            of_rate_rows = self._polynomial_control_rate_idxs_in_y[name]
            of_rate2_rows = self._polynomial_control_rate2_idxs_in_y[name]

            wrt_cols = self.x_size + 1
            partials[of_rate, 't_duration'] = self._dy_dZ[idxs, of_rate_rows, wrt_cols]
            partials[of_rate2, 't_duration'] = self._dy_dZ[idxs, of_rate2_rows, wrt_cols]

            for wrt_control_name in self.polynomial_control_options:
                wrt = self._polynomial_control_input_names[wrt_control_name]
                wrt_cols = self._polynomial_control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = self._dy_dZ[idxs, of_rows, wrt_cols]
                partials[of_rate, wrt] = self._dy_dZ[idxs, of_rate_rows, wrt_cols]
                partials[of_rate2, wrt] = self._dy_dZ[idxs, of_rate2_rows, wrt_cols]

        for name, options in self._filtered_timeseries_outputs.items():
            of = self._timeseries_output_names[name]
            of_rows = self._timeseries_idxs_in_y[name]

            partials[of, 't_initial'] = self._dy_dZ[idxs, of_rows, self.x_size]
            partials[of, 't_duration'] = self._dy_dZ[idxs, of_rows, self.x_size+1]

            for wrt_state_name in self.state_options:
                wrt = self._state_input_names[wrt_state_name]
                wrt_cols = self._state_idxs_in_Z[wrt_state_name]
                partials[of, wrt] = self._dy_dZ[idxs, of_rows, wrt_cols]

            for wrt_param_name in self.parameter_options:
                wrt = self._param_input_names[wrt_param_name]
                wrt_cols = self._parameter_idxs_in_Z[wrt_param_name]
                partials[of, wrt] = self._dy_dZ[idxs, of_rows, wrt_cols]

            for wrt_control_name in self.control_options:
                wrt = self._control_input_names[wrt_control_name]
                wrt_cols = self._control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = self._dy_dZ[idxs, of_rows, wrt_cols]

            for wrt_pc_name in self.polynomial_control_options:
                wrt = self._polynomial_control_input_names[wrt_pc_name]
                wrt_cols = self._polynomial_control_idxs_in_Z[wrt_pc_name]
                partials[of, wrt] = self._dy_dZ[idxs, of_rows, wrt_cols]
