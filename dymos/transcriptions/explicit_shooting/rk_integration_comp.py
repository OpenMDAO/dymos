import numpy as np
import openmdao.api as om

from ...options import options as dymos_options

from .ode_evaluation_group import ODEEvaluationGroup
from ...utils.misc import get_rate_units
from ...utils.introspection import filter_outputs


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
    grid_data : GridData
        The GridData instance pertaining to the phase to which this ODEEvaluationGroup belongs.
    standalone_mode : bool
        When True, this component will perform its configuration during setup. This is useful
        for unittesting this component when not embedded in a larger system.
    **kwargs : dict
        Additional keyword arguments passed to Group.

    Notes
    -----
    This code includes the following unicode symbols:
    θ:  U+03B8
    """
    def __init__(self, ode_class, time_options=None,
                 state_options=None, parameter_options=None, control_options=None,
                 polynomial_control_options=None, timeseries_options=None,
                 grid_data=None, standalone_mode=True, **kwargs):
        super().__init__(**kwargs)
        self.ode_class = ode_class
        self.time_options = time_options
        self.state_options = state_options
        self.parameter_options = parameter_options or {}
        self.control_options = control_options or {}
        self.polynomial_control_options = polynomial_control_options or {}
        self.timeseries_options = timeseries_options or {}
        self._eval_subprob = None
        self._deriv_subprob = None
        self._grid_data = grid_data
        self._DTYPE = float

        self._inputs_cache = None

        self.x_size = 0
        self.p_size = 0
        self.u_size = 0
        self.up_size = 0
        self.θ_size = 0
        self.Z_size = 0

        self._state_rate_of_names = []
        self._totals_of_names = []
        self._totals_wrt_names = []

        # If _standalone_mode is True, this component will fully perform all of its setup at setup
        # time.  If False, it will need to have configure_io called on it to properly finish its
        # setup.
        self._standalone_mode = standalone_mode
        self._no_check_partials = not dymos_options['include_check_partials']
        self._num_control_input_nodes = grid_data.subset_num_nodes['control_input']

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
        rk = rk_methods[self.options['method']]
        num_stages = len(rk['b'])

        self._eval_subprob = p = om.Problem(comm=self.comm)
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

        p.setup(force_alloc_complex=True if self._DTYPE is complex else False)
        p.final_setup()

        self._deriv_subprob = p = om.Problem(comm=self.comm)
        p.model.add_subsystem('ode_eval',
                              ODEEvaluationGroup(self.ode_class, self.time_options,
                                                 self.state_options,
                                                 self.parameter_options,
                                                 self.control_options,
                                                 self.polynomial_control_options,
                                                 ode_init_kwargs=self.options['ode_init_kwargs'],
                                                 grid_data=self._grid_data,
                                                 vec_size=num_stages),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.driver.declare_coloring()
        p.model.ode_eval.linear_solver = om.DirectSolver()
        p.setup(force_alloc_complex=False)
        p.final_setup()
        # self._eval_subprob.set_complex_step_mode(self._complex_step_mode)

    def _set_complex_step_mode(self, active):
        """
        Sets complex step mode on this component, adjust the complex step mode of the evaluation
        subproblem, and reallocates storage with the appropriate dtype.

        Parameters
        ----------
        active : bool
            True if complex-step mode is being enabled, else False

        """
        super()._set_complex_step_mode(active)
        self._DTYPE = complex if active else float

        self._eval_subprob.setup(force_alloc_complex=True if self._DTYPE is complex else False)
        self._eval_subprob.final_setup()
        self._eval_subprob.set_complex_step_mode(active)

        self._allocate_storage()

    def _allocate_storage(self):
        N = self.options['num_steps_per_segment']
        rk = rk_methods[self.options['method']]
        num_rows = self._num_rows
        num_stages = len(rk['b'])
        num_x = self.x_size
        num_θ = self.θ_size
        num_z = num_x + num_θ
        num_y = self.y_size

        # The contiguous vector of state values
        self._x = np.zeros((num_rows, self.x_size, 1), dtype=self._DTYPE)

        # The contiguous vector of time values
        self._t = np.zeros((num_rows, 1), dtype=self._DTYPE)

        # The contiguous vector of ODE parameter values
        self._θ = np.zeros((self.θ_size, 1), dtype=self._DTYPE)

        # The contiguous vector of state rates
        self._f = np.zeros((self.x_size, 1), dtype=self._DTYPE)

        # The contiguous vector of ODE algebraic outputs
        self._y = np.zeros((num_rows, num_y, 1), dtype=self._DTYPE)

        # The derivatives of the state rates wrt the current time
        self._f_t = np.zeros((self.x_size, 1), dtype=self._DTYPE)
        self._f_t_vec = np.zeros((num_stages, self.x_size, 1), dtype=self._DTYPE)

        # The derivatives of the state rates wrt the current state
        self._f_x = np.zeros((self.x_size, self.x_size), dtype=self._DTYPE)
        self._f_x_vec = np.zeros((num_stages, self.x_size, self.x_size), dtype=self._DTYPE)

        # The derivatives of the state rates wrt the parameters
        self._f_θ = np.zeros((self.x_size, self.θ_size), dtype=self._DTYPE)
        self._f_θ_vec = np.zeros((num_stages, self.x_size, self.θ_size), dtype=self._DTYPE)

        # The derivatives of the state rates wrt the current time
        self._y_t = np.zeros((self.y_size, 1), dtype=self._DTYPE)
        self._y_t_vec = np.zeros((num_stages, self.y_size, 1), dtype=self._DTYPE)

        # The derivatives of the state rates wrt the current state
        self._y_x = np.zeros((self.y_size, self.x_size), dtype=self._DTYPE)
        self._y_x_vec = np.zeros((num_stages, self.y_size, self.x_size), dtype=self._DTYPE)

        # The derivatives of the state rates wrt the parameters
        self._y_θ = np.zeros((self.y_size, self.θ_size), dtype=self._DTYPE)
        self._y_θ_vec = np.zeros((num_stages, self.y_size, self.θ_size), dtype=self._DTYPE)

        # Intermediate state rate storage
        self._k_q = np.zeros((num_stages, self.x_size, 1), dtype=self._DTYPE)

        # Intermediate time and states
        self._T_i = np.zeros((num_stages, 1), dtype=self._DTYPE)
        self._X_i = np.zeros((num_stages, self.x_size, 1), dtype=self._DTYPE)

        # The partial derivative of the final state vector wrt time from state update equation.
        self.px_pt = np.zeros((self.x_size, 1), dtype=self._DTYPE)

        # The partial derivative of the final state vector wrt the initial state vector from the
        # state update equation.
        self.px_px = np.zeros((self.x_size, self.x_size), dtype=self._DTYPE)

        # Derivatives pertaining to the stage ODE evaluations
        self._dTi_dZ = np.zeros((1, num_z), dtype=self._DTYPE)
        self._dXi_dZ = np.zeros((num_x, num_z), dtype=self._DTYPE)
        self._dkq_dZ = np.zeros((num_stages, num_x, num_z), dtype=self._DTYPE)

        # The ODE parameter derivatives wrt the integration parameters
        self._dθ_dZ = np.zeros((num_θ, num_z), dtype=self._DTYPE)
        self._dθ_dZ[:, num_x:] = np.eye(num_θ, dtype=self._DTYPE)

        # Total derivatives of evolving quantities (x, t, h) wrt the integration parameters.
        # Let Z be [x0.ravel() t0 tp p.ravel() u.ravel()]
        self._dx_dZ = np.zeros((num_rows, num_x, num_z), dtype=self._DTYPE)
        self._dx_dZ[:, :, :num_x] = np.eye(num_x, dtype=self._DTYPE)
        self._dt_dZ = np.zeros((num_rows, 1, num_z), dtype=self._DTYPE)
        self._dt_dZ[:, 0, num_x] = 1.0
        self._dh_dZ = np.zeros((num_rows, 1, num_z), dtype=self._DTYPE)
        self._dh_dZ[:, 0, num_x+1] = 1.0 / N

        # Total derivatives of ODE outputs (y) wrt the integration parameters.
        self._dy_dZ = np.zeros((num_rows, num_y, num_z), dtype=self._DTYPE)

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

            # Keep track of the derivative "of" names for state rates separately, so we don't
            # request them when they're not necessary.
            self._state_rate_of_names.append(f'state_rate_collector.state_rates:{state_name}_rate')
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
        self._parameter_idxs_in_θ = {}
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
        self._control_idxs_in_θ = {}
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

        for control_name, options in self.control_options.items():
            control_param_shape = (self._num_control_input_nodes,) + options['shape']
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
        self._polynomial_control_idxs_in_θ = {}
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
        ode_eval = self._eval_subprob.model._get_subsystem('ode_eval.ode')

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
        self.θ_size = 2 + self.p_size + self.u_size + self.up_size

        # allocate the integration parameter vector
        self.Z_size = self.x_size + self.θ_size

        # allocate the algebraic outputs vector
        self.y_size = 3 * self.u_size

        start_Z = 0
        for state_name, options in self.state_options.items():
            state_size = np.prod(options['shape'], dtype=int)
            self._state_idxs_in_Z[state_name] = np.s_[start_Z: start_Z+state_size]
            start_Z += state_size

        start_Z = self.x_size + 2
        start_θ = 2
        for param_name, options in self.parameter_options.items():
            param_size = np.prod(options['shape'], dtype=int)
            self._parameter_idxs_in_Z[param_name] = np.s_[start_Z: start_Z+param_size]
            self._parameter_idxs_in_θ[param_name] = np.s_[start_θ: start_θ+param_size]
            start_Z += param_size
            start_θ += param_size

        start_Z = self.x_size + 2 + self.p_size
        start_θ = 2 + self.p_size
        start_y = 0
        for control_name, options in self.control_options.items():
            control_size = np.prod(options['shape'], dtype=int)
            control_param_shape = (len(control_input_node_ptau),) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._control_idxs_in_Z[control_name] = np.s_[start_Z:start_Z+control_param_size]
            self._control_idxs_in_θ[control_name] = np.s_[start_θ:start_θ+control_param_size]
            self._control_idxs_in_y[control_name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._control_rate_idxs_in_y[control_name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._control_rate2_idxs_in_y[control_name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            start_Z += control_param_size
            start_θ += control_param_size

        start_Z = self.x_size + 2 + self.p_size + self.u_size
        start_θ = 2 + self.p_size + self.u_size
        for name, options in self.polynomial_control_options.items():
            control_size = np.prod(options['shape'], dtype=int)
            num_input_nodes = options['order'] + 1
            control_param_shape = (num_input_nodes,) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._polynomial_control_idxs_in_Z[name] = np.s_[start_Z:start_Z+control_param_size]
            self._polynomial_control_idxs_in_θ[name] = np.s_[start_θ:start_θ+control_param_size]
            self._polynomial_control_idxs_in_y[name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._polynomial_control_rate_idxs_in_y[name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            self._polynomial_control_rate2_idxs_in_y[name] = np.s_[start_y:start_y+control_size]
            start_y += control_size
            start_Z += control_param_size
            start_θ += control_param_size

        for output_name, options in self._filtered_timeseries_outputs.items():
            size = np.prod(options['shape'], dtype=int)
            self._timeseries_idxs_in_y[output_name] = np.s_[start_y:start_y+size]
            start_y += size
        self.y_size = start_y

        self._allocate_storage()

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
        num_θ = self.θ_size

        # Let Z be [x0.ravel() t0 tp p.ravel() u.ravel()]
        self._dx_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._DTYPE)
        self._dx_dZ[0, :, :num_x] = np.eye(num_x, dtype=self._DTYPE)
        self._dt_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._DTYPE)
        self._dt_dZ[0, 0, num_x] = 1.0
        self._dh_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._DTYPE)
        self._dh_dZ[:, 0, num_x+1] = 1.0 / N
        self._dTi_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._DTYPE)
        self._dXi_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._DTYPE)
        self._dkq_dZ[...] = 0.0  # np.zeros((num_stages, num_x, num_z), dtype=self._DTYPE)
        self._dθ_dZ[...] = 0.0  # np.zeros((num_θ, num_z), dtype=self._DTYPE)
        self._dθ_dZ[:, num_x:] = np.eye(num_θ, dtype=self._DTYPE)

    def _initialize_segment(self, row, inputs=None, derivs=False):
        """
        Set the derivatives at the current row to those of the previous row.

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
                self._dTi_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._DTYPE)
                self._dXi_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._DTYPE)
                self._dkq_dZ[...] = 0.0  # np.zeros((num_stages, num_x, num_z), dtype=self._DTYPE)

                # dθ_dZ remains constant across segments

    def _subprob_run_model(self, x, t, θ, linearize=True, subprob=None):
        """
        Set inputs to the model given x, t, and θ, evaluate the model, and linearize if requested.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.
        θ : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.
        linearize : bool
            If True, linearize the model after calling run_model.
        subprob : om.Problem
            The instance of the subproblem to be run.  If None, run self._deriv_subprob.

        Returns
        -------

        """
        if subprob is None:
            subprob = self._deriv_subprob

        # transcribe time
        subprob.set_val('time', t, units=self.time_options['units'])
        subprob.set_val('t_initial', θ[0, 0], units=self.time_options['units'])
        subprob.set_val('t_duration', θ[1, 0], units=self.time_options['units'])

        # transcribe states
        for name in self.state_options:
            input_name = self._state_input_names[name]
            if subprob is self._deriv_subprob:
                subprob.set_val(input_name, x[:, self.state_idxs[name], :])
            else:
                subprob.set_val(input_name, x[self.state_idxs[name], :])

        # transcribe parameters
        for name in self.parameter_options:
            input_name = self._param_input_names[name]
            subprob.set_val(input_name, θ[self._parameter_idxs_in_θ[name], 0])

        # transcribe controls
        for name in self.control_options:
            input_name = self._control_input_names[name]
            subprob.set_val(input_name, θ[self._control_idxs_in_θ[name], 0])

        for name in self.polynomial_control_options:
            input_name = self._polynomial_control_input_names[name]
            subprob.set_val(input_name, θ[self._polynomial_control_idxs_in_θ[name], 0])

        # Re-run in case the inputs have changed.
        subprob.run_model()

        if linearize:
            subprob.model._linearize(None)

    def eval_f(self, x, t, θ, f, y=None):
        """
        Evaluate the ODE which provides the state rates for integration.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.
        θ : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.
        f : np.ndarray
            A flattened, contiguous vector of the state rates.
        y : np.ndarray or None
            A flattened, contiguous vector of the auxiliary ODE outputs, if desired.
            If present, the first positions are reserved for the contiguous control values, rates,
            and second derivatives, respectively. The remaining elements are the requested ODE-based
            timeseries outputs.
        """
        self._subprob_run_model(x, t, θ, linearize=False, subprob=self._eval_subprob)

        # pack the resulting array
        for name in self.state_options:
            f[self.state_idxs[name]] = self._eval_subprob.get_val(f'state_rate_collector.state_rates:{name}_rate').ravel()

        if y is not None:
            # pack any control values and rates into y
            for name in self.control_options:
                output_name = self._control_output_names[name]
                rate_name = self._control_rate_names[name]
                rate2_name = self._control_rate2_names[name]
                y[self._control_idxs_in_y[name]] = self._eval_subprob.get_val(output_name).ravel()
                y[self._control_rate_idxs_in_y[name]] = self._eval_subprob.get_val(rate_name).ravel()
                y[self._control_rate2_idxs_in_y[name]] = self._eval_subprob.get_val(rate2_name).ravel()

            # pack any polynomial control values and rates into y
            for name in self.polynomial_control_options:
                output_name = self._polynomial_control_output_names[name]
                rate_name = self._polynomial_control_rate_names[name]
                rate2_name = self._polynomial_control_rate2_names[name]
                y[self._polynomial_control_idxs_in_y[name]] = self._eval_subprob.get_val(output_name).ravel()
                y[self._polynomial_control_rate_idxs_in_y[name]] = self._eval_subprob.get_val(rate_name).ravel()
                y[self._polynomial_control_rate2_idxs_in_y[name]] = self._eval_subprob.get_val(rate2_name).ravel()

            # pack any polynomial control values and rates into y

            for output_name, options in self._filtered_timeseries_outputs.items():
                path = options['path']
                y[self._timeseries_idxs_in_y[output_name]] = self._eval_subprob.get_val(path).ravel()

    def eval_f_derivs(self, x, t, θ, f_x=None, f_t=None, f_θ=None, y_x=None, y_t=None, y_θ=None):
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
        θ : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.
        f_x : np.ndarray
            A matrix of the derivative of each element of the rates `f` wrt each value in `x`.
        f_t : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt `time`.
        f_θ : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt the parameters `θ`.
        y_x : np.ndarray
            A matrix of the derivative of each element of the rates `y` wrt each value in `x`.
        y_t : np.ndarray
            A matrix of the derivatives of each element of the rates `y` wrt `time`.
        y_θ : np.ndarray
            A matrix of the derivatives of each element of the rates `y` wrt the parameters `θ`.
        """
        self._subprob_run_model(x, t, θ, linearize=False, subprob=self._eval_subprob)

        eval_state_rate_derivs = f_x is not None or f_t is not None or f_θ is not None

        totals_of_names = self._totals_of_names
        if eval_state_rate_derivs:
            totals_of_names += self._state_rate_of_names

        totals = self._eval_subprob.compute_totals(of=totals_of_names,
                                                   wrt=self._totals_wrt_names,
                                                   use_abs_names=False)
        if eval_state_rate_derivs:
            for state_name in self.state_options:
                of_name = f'state_rate_collector.state_rates:{state_name}_rate'
                idxs = self.state_idxs[state_name]

                if f_t is not None:
                    f_t[self.state_idxs[state_name]] = totals[of_name, 'time']

                if f_x is not None:
                    for state_name_wrt in self.state_options:
                        idxs_wrt = self.state_idxs[state_name_wrt]
                        px_px = totals[of_name, self._state_input_names[state_name_wrt]]
                        f_x[idxs, idxs_wrt] = px_px.ravel()

                if f_θ is not None:
                    f_θ[idxs, 0] = totals[of_name, 't_initial']
                    f_θ[idxs, 1] = totals[of_name, 't_duration']

                    for param_name_wrt in self.parameter_options:
                        idxs_wrt = self._parameter_idxs_in_θ[param_name_wrt]
                        px_pp = totals[of_name, self._param_input_names[param_name_wrt]]
                        f_θ[idxs, idxs_wrt] = px_pp.ravel()

                    for control_name_wrt in self.control_options:
                        idxs_wrt = self._control_idxs_in_θ[control_name_wrt]
                        px_pu = totals[of_name, self._control_input_names[control_name_wrt]]
                        f_θ[idxs, idxs_wrt] = px_pu.ravel()

                    for pc_name_wrt in self.polynomial_control_options:
                        idxs_wrt = self._polynomial_control_idxs_in_θ[pc_name_wrt]
                        px_pu = totals[of_name, self._polynomial_control_input_names[pc_name_wrt]]
                        f_θ[idxs, idxs_wrt] = px_pu.ravel()

        if y_x is not None and y_t is not None and y_θ is not None:
            for control_name in self.control_options:
                wrt_name = self._control_input_names[control_name]
                idxs_wrt = self._control_idxs_in_θ[control_name]
                of_name = self._control_output_names[control_name]
                of_rate_name = self._control_rate_names[control_name]
                of_rate2_name = self._control_rate2_names[control_name]

                of_idxs = self._control_idxs_in_y[control_name]
                of_rate_idxs = self._control_rate_idxs_in_y[control_name]
                of_rate2_idxs = self._control_rate2_idxs_in_y[control_name]

                y_t[of_idxs, 0] = totals[of_name, 'time']
                y_t[of_rate_idxs, 0] = totals[of_rate_name, 'time']
                y_t[of_rate2_idxs, 0] = totals[of_rate2_name, 'time']

                y_θ[of_idxs, 1] = totals[of_name, 't_duration']
                y_θ[of_rate_idxs, 1] = totals[of_rate_name, 't_duration']
                y_θ[of_rate2_idxs, 1] = totals[of_rate2_name, 't_duration']

                y_θ[of_idxs, idxs_wrt] = totals[of_name, wrt_name]
                y_θ[of_rate_idxs, idxs_wrt] = totals[of_rate_name, wrt_name]
                y_θ[of_rate2_idxs, idxs_wrt] = totals[of_rate2_name, wrt_name]

            for polynomial_control_name in self.polynomial_control_options:
                wrt_name = self._polynomial_control_input_names[polynomial_control_name]
                idxs_wrt = self._polynomial_control_idxs_in_θ[polynomial_control_name]
                of_name = self._polynomial_control_output_names[polynomial_control_name]
                of_rate_name = self._polynomial_control_rate_names[polynomial_control_name]
                of_rate2_name = self._polynomial_control_rate2_names[polynomial_control_name]

                of_idxs = self._polynomial_control_idxs_in_y[polynomial_control_name]
                of_rate_idxs = self._polynomial_control_rate_idxs_in_y[polynomial_control_name]
                of_rate2_idxs = self._polynomial_control_rate2_idxs_in_y[polynomial_control_name]

                y_t[of_idxs, 0] = totals[of_name, 'time']
                y_t[of_rate_idxs, 0] = totals[of_rate_name, 'time']
                y_t[of_rate2_idxs, 0] = totals[of_rate2_name, 'time']

                y_θ[of_idxs, 1] = totals[of_name, 't_duration']
                y_θ[of_rate_idxs, 1] = totals[of_rate_name, 't_duration']
                y_θ[of_rate2_idxs, 1] = totals[of_rate2_name, 't_duration']

                y_θ[of_idxs, idxs_wrt] = totals[of_name, wrt_name]
                y_θ[of_rate_idxs, idxs_wrt] = totals[of_rate_name, wrt_name]
                y_θ[of_rate2_idxs, idxs_wrt] = totals[of_rate2_name, wrt_name]

            for name, options in self._filtered_timeseries_outputs.items():
                idxs_of = self._timeseries_idxs_in_y[name]
                of_name = options['path']

                y_t[idxs_of, 0] = totals[options['path'], 'time']

                y_θ[idxs_of, 0] = totals[of_name, 't_initial']
                y_θ[idxs_of, 1] = totals[of_name, 't_duration']

                for state_name_wrt in self.state_options:
                    idxs_wrt = self.state_idxs[state_name_wrt]
                    py_px = totals[of_name, self._state_input_names[state_name_wrt]]
                    y_x[idxs_of, idxs_wrt] = py_px.ravel()

                for param_name_wrt in self.parameter_options:
                    idxs_wrt = self._parameter_idxs_in_θ[param_name_wrt]
                    py_pp = totals[of_name, self._param_input_names[param_name_wrt]]
                    y_θ[idxs_of, idxs_wrt] = py_pp.ravel()

                for control_name_wrt in self.control_options:
                    idxs_wrt = self._control_idxs_in_θ[control_name_wrt]
                    py_puhat = totals[of_name, self._control_input_names[control_name_wrt]]
                    y_θ[idxs_of, idxs_wrt] = py_puhat.ravel()

                for pc_name_wrt in self.polynomial_control_options:
                    idxs_wrt = self._polynomial_control_idxs_in_θ[pc_name_wrt]
                    py_puhat = totals[of_name, self._polynomial_control_input_names[pc_name_wrt]]
                    y_θ[idxs_of, idxs_wrt] = py_puhat.ravel()

    def eval_f_derivs_vectorized(self, x, t, θ, f_x, f_t, f_θ, y_x=None, y_t=None, y_θ=None):
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
        θ : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.
        f_x : np.ndarray (num_stages, num_states, num_states)
            A matrix of the derivative of each element of the rates `f` wrt each value in `x`.
        f_t : np.ndarray (num_stages, num_states, 1)
            A matrix of the derivatives of each element of the rates `f` wrt `time`.
        f_θ : np.ndarray (num_stages, num_states, num_θ)
            A matrix of the derivatives of each element of the rates `f` wrt the parameters `θ`.
        y_x : np.ndarray (num_stages, num_y, num_states)
            A matrix of the derivative of each element of the outputs `y` wrt each value in `x`.
        y_t : np.ndarray (num_stages, num_y, 1)
            A matrix of the derivatives of each element of the outputs `y` wrt `time`.
        y_θ : np.ndarray (num_stages, num_y, num_θ)
            A matrix of the derivatives of each element of the outputs `y` wrt the parameters `θ`.
        """
        rk = rk_methods[self.options['method']]
        num_stages = len(rk['b'])
        subprob = self._deriv_subprob
        ncin = self._num_control_input_nodes

        self._subprob_run_model(x, t, θ, linearize=False)

        totals = subprob.compute_totals(of=self._totals_of_names + self._state_rate_of_names,
                                        wrt=self._totals_wrt_names,
                                        use_abs_names=False)

        for state_name, options in self.state_options.items():
            size = np.prod(options['shape'])
            name_of = f'state_rate_collector.state_rates:{state_name}_rate'
            idxs_of = self.state_idxs[state_name]
            f_t[:, idxs_of, 0] = np.diagonal(totals[name_of, 'time']).reshape((num_stages, size))

            for state_name_wrt, options_wrt in self.state_options.items():
                size_wrt = np.prod(options_wrt['shape'])
                idxs_wrt = self.state_idxs[state_name_wrt]
                px_px = totals[name_of, self._state_input_names[state_name_wrt]]
                f_x[:, idxs_of, idxs_wrt] = np.diagonal(px_px).reshape((num_stages, size, size_wrt))

            f_θ[:, idxs_of, 0] = totals[name_of, 't_initial']
            f_θ[:, idxs_of, 1] = totals[name_of, 't_duration']

            for param_name_wrt, options_wrt in self.parameter_options.items():
                size_wrt = np.prod(options_wrt['shape'])
                idxs_wrt = self._parameter_idxs_in_θ[param_name_wrt]
                px_pp = totals[name_of, self._param_input_names[param_name_wrt]]
                f_θ[:, idxs_of, idxs_wrt] = px_pp.reshape((num_stages, size, size_wrt))

            for control_name_wrt, options_wrt in self.control_options.items():
                size_wrt = np.prod(options_wrt['shape']) * ncin
                idxs_wrt = self._control_idxs_in_θ[control_name_wrt]
                px_puhat = totals[name_of, self._control_input_names[control_name_wrt]]
                f_θ[:, idxs_of, idxs_wrt] = px_puhat.reshape((num_stages, size, size_wrt))

            for pc_name_wrt, options_wrt in self.polynomial_control_options.items():
                size_wrt = np.prod(options_wrt['shape']) * (options_wrt['order'] + 1)
                idxs_wrt = self._polynomial_control_idxs_in_θ[pc_name_wrt]
                px_puhat = totals[name_of, self._polynomial_control_input_names[pc_name_wrt]]
                f_θ[:, idxs_of, idxs_wrt] = px_puhat.reshape((num_stages, size, size_wrt))

        if y_x is not None and y_t is not None and y_θ is not None:
            for control_name, options in self.control_options.items():
                wrt_name = self._control_input_names[control_name]
                idxs_wrt = self._control_idxs_in_θ[control_name]
                name_of = self._control_output_names[control_name]
                of_rate_name = self._control_rate_names[control_name]
                of_rate2_name = self._control_rate2_names[control_name]
                size_of = np.prod(options['shape'])

                idxs_of = self._control_idxs_in_y[control_name]
                idxs_of_rate = self._control_rate_idxs_in_y[control_name]
                idxs_of_rate2 = self._control_rate2_idxs_in_y[control_name]

                y_t[:, idxs_of, 0] = np.diagonal(totals[name_of, 'time']).reshape((num_stages, size_of))
                y_t[:, idxs_of_rate, 0] = np.diagonal(totals[of_rate_name, 'time']).reshape((num_stages, size_of))
                y_t[:, idxs_of_rate2, 0] = np.diagonal(totals[of_rate2_name, 'time']).reshape((num_stages, size_of))

                y_θ[:, idxs_of, 1] = np.diagonal(totals[name_of, 't_duration'])
                y_θ[:, idxs_of_rate, 1] = np.diagonal(totals[of_rate_name, 't_duration'])
                y_θ[:, idxs_of_rate2, 1] = np.diagonal(totals[of_rate2_name, 't_duration'])

                y_θ[:, idxs_of, idxs_wrt] = totals[name_of, wrt_name].reshape((num_stages, size_of, ncin))
                y_θ[:, idxs_of_rate, idxs_wrt] = totals[of_rate_name, wrt_name].reshape((num_stages, size_of, ncin))
                y_θ[:, idxs_of_rate2, idxs_wrt] = totals[of_rate2_name, wrt_name].reshape((num_stages, size_of, ncin))

            for polynomial_control_name, options in self.polynomial_control_options.items():
                wrt_name = self._polynomial_control_input_names[polynomial_control_name]
                idxs_wrt = self._polynomial_control_idxs_in_θ[polynomial_control_name]
                name_of = self._polynomial_control_output_names[polynomial_control_name]
                of_rate_name = self._polynomial_control_rate_names[polynomial_control_name]
                of_rate2_name = self._polynomial_control_rate2_names[polynomial_control_name]
                size_of = np.prod(options['shape'])
                order = options['order']

                idxs_of = self._polynomial_control_idxs_in_y[polynomial_control_name]
                idxs_of_rate = self._polynomial_control_rate_idxs_in_y[polynomial_control_name]
                idxs_of_rate2 = self._polynomial_control_rate2_idxs_in_y[polynomial_control_name]

                y_t[:, idxs_of, 0] = np.diagonal(totals[name_of, 'time']).reshape((num_stages, size_of))
                y_t[:, idxs_of_rate, 0] = np.diagonal(totals[of_rate_name, 'time']).reshape((num_stages, size_of))
                y_t[:, idxs_of_rate2, 0] = np.diagonal(totals[of_rate2_name, 'time']).reshape((num_stages, size_of))

                y_θ[:, idxs_of, 1] = totals[name_of, 't_duration']
                y_θ[:, idxs_of_rate, 1] = totals[of_rate_name, 't_duration']
                y_θ[:, idxs_of_rate2, 1] = totals[of_rate2_name, 't_duration']

                y_θ[:, idxs_of, idxs_wrt] = totals[name_of, wrt_name].reshape((num_stages, size_of, order + 1))
                y_θ[:, idxs_of_rate, idxs_wrt] = totals[of_rate_name, wrt_name].reshape((num_stages, size_of, order + 1))
                y_θ[:, idxs_of_rate2, idxs_wrt] = totals[of_rate2_name, wrt_name].reshape((num_stages, size_of, order + 1))

            for name, options in self._filtered_timeseries_outputs.items():
                idxs_of = self._timeseries_idxs_in_y[name]
                name_of = options['path']
                size_of = np.prod(options['shape'], dtype=int)

                y_t[:, idxs_of, 0] = np.diagonal(totals[options['path'], 'time']).reshape((num_stages, size_of))

                y_θ[:, idxs_of, 0] = totals[name_of, 't_initial']
                y_θ[:, idxs_of, 1] = totals[name_of, 't_duration']

                for state_name_wrt, wrt_options in self.state_options.items():
                    idxs_wrt = self.state_idxs[state_name_wrt]
                    size_wrt = np.prod(wrt_options['shape'], dtype=int)
                    py_px = totals[name_of, self._state_input_names[state_name_wrt]]
                    y_x[:, idxs_of, idxs_wrt] = np.diagonal(py_px).reshape((num_stages, size_of, size_wrt))

                for param_name_wrt, wrt_options in self.parameter_options.items():
                    idxs_wrt = self._parameter_idxs_in_θ[param_name_wrt]
                    size_wrt = np.prod(wrt_options['shape'], dtype=int)
                    py_pp = totals[name_of, self._param_input_names[param_name_wrt]]
                    y_θ[:, idxs_of, idxs_wrt] = py_pp.reshape((num_stages, size_of, size_wrt))

                for control_name_wrt, wrt_options in self.control_options.items():
                    size_wrt = np.prod(wrt_options['shape']) * ncin
                    idxs_wrt = self._control_idxs_in_θ[control_name_wrt]
                    py_puhat = totals[name_of, self._control_input_names[control_name_wrt]]
                    y_θ[:, idxs_of, idxs_wrt] = py_puhat.reshape((num_stages, size_of, size_wrt))

                for pc_name_wrt, wrt_options in self.polynomial_control_options.items():
                    idxs_wrt = self._polynomial_control_idxs_in_θ[pc_name_wrt]
                    order = wrt_options['order']
                    size_wrt = np.prod(options_wrt['shape']) * (order + 1)
                    py_puhat = totals[name_of, self._polynomial_control_input_names[pc_name_wrt]]
                    y_θ[:, idxs_of, idxs_wrt] = py_puhat.reshape((num_stages, size_of, order + 1))

    def _propagate(self, inputs, derivs=None):
        """
        Propagate the states from t_initial to t_initial + t_duration, optionally computing
        the derivatives along the way and caching the current time and state values.

        Notes
        -----
        This function is deprecated in favor of _propagate_vectorized_derivs, but is still included
        for benchmarking purposes.

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
        θ = self._θ

        # Make t_initial and t_duration the first two elements of the ODE parameter vector.
        θ[0] = inputs['t_initial'].copy()
        θ[1] = inputs['t_duration'].copy()

        f_x = self._f_x
        f_t = self._f_t
        f_θ = self._f_θ

        y_x = self._y_x
        y_t = self._y_t
        y_θ = self._y_θ

        dx_dZ = self._dx_dZ
        dt_dZ = self._dt_dZ
        dθ_dZ = self._dθ_dZ
        dkq_dZ = self._dkq_dZ
        dy_dZ = self._dy_dZ
        dh_dZ = self._dh_dZ
        dXi_dZ = self._dXi_dZ
        dTi_dZ = self._dTi_dZ

        k_q = self._k_q

        T_i = self._T_i
        X_i = self._X_i

        # Initialize parameters
        for name in self.parameter_options:
            θ[self._parameter_idxs_in_θ[name], 0] = inputs[f'parameters:{name}'].ravel()

        # Initialize controls
        for name in self.control_options:
            θ[self._control_idxs_in_θ[name], 0] = inputs[f'controls:{name}'].ravel()

        # Initialize polynomial controls
        for name in self.polynomial_control_options:
            θ[self._polynomial_control_idxs_in_θ[name], 0] = inputs[f'polynomial_controls:{name}'].ravel()

        seg_durations = θ[1] * np.diff(gd.segment_ends) / 2.0

        # step counter
        row = 0

        for seg_i in range(gd.num_segments):
            self._eval_subprob.model._get_subsystem('ode_eval').set_segment_index(seg_i)
            self._deriv_subprob.model._get_subsystem('ode_eval').set_segment_index(seg_i)

            # Initialize, t, x, h, and derivatives for the start of the current segment
            self._initialize_segment(row, inputs, derivs=derivs)

            h = np.asarray(seg_durations[seg_i] / N, dtype=self._DTYPE)
            # On each segment, the total derivative of the stepsize h is a function of
            # the duration of the phase (the second element of the parameter vector after states)
            if derivs:
                dh_dZ[row:row+N+1, 0, self.x_size+1] = seg_durations[seg_i] / θ[1] / N

            rm1 = row
            row = row + 1

            for q in range(N):
                # Compute the state rates and their partials at the start of the step
                self.eval_f(x[rm1, ...], t[rm1, 0], θ, k_q[0, ...], y=self._y[rm1, ...])

                if derivs:
                    # Compute the state rate derivatives
                    self.eval_f_derivs(x[rm1, ...], t[rm1, 0], θ,
                                       f_x, f_t, f_θ,
                                       y_x, y_t, y_θ)

                    dkq_dZ[0, ...] = f_t @ dt_dZ[rm1, ...] + f_x @ dx_dZ[rm1, ...] + f_θ @ dθ_dZ
                    dy_dZ[rm1, ...] = y_x @ dx_dZ[rm1, ...] + y_t @ dt_dZ[rm1, ...] + y_θ @ dθ_dZ

                for i in range(1, num_stages):
                    T_i = t[rm1, ...] + c[i] * h
                    a_tdot_k = np.tensordot(a[i, :i], k_q[:i, ...], axes=(0, 0))
                    # a_tdot_k = np.einsum('i,ijk->jk', a[i, :i], k_q[:i, ...])
                    X_i = x[rm1, ...] + h * a_tdot_k
                    self.eval_f(X_i, T_i, θ, k_q[i, ...])

                    if derivs:
                        self.eval_f_derivs(X_i, T_i, θ, f_x, f_t, f_θ)
                        dTi_dZ[...] = dt_dZ[row - 1, ...] + c[i] * dh_dZ[rm1, ...]
                        a_tdot_dkqdz = np.tensordot(a[i, :i], dkq_dZ[:i, ...], axes=(0, 0))
                        # a_tdot_dkqdz = np.einsum('i,ijk->jk', a[i, :i], dkq_dZ[:i, ...])
                        dXi_dZ[...] = dx_dZ[rm1, ...] + a_tdot_k @ dh_dZ[rm1, ...] + h * a_tdot_dkqdz
                        dkq_dZ[i, ...] = f_t @ dTi_dZ + f_x @ dXi_dZ + f_θ @ dθ_dZ

                b_tdot_kq = np.tensordot(b, k_q, axes=(0, 0))
                # b_tdot_kq = np.einsum('i,ijk->jk', b, k_q)
                x[row, ...] = x[rm1, ...] + h * b_tdot_kq
                t[row, 0] = t[rm1, 0] + h

                if derivs:
                    b_tdot_dkqdz = np.tensordot(b, dkq_dZ, axes=(0, 0))
                    # b_tdot_dkqdz = np.einsum('i,ijk->jk', b, dkq_dZ)
                    dx_dZ[row, ...] = dx_dZ[rm1, ...] + b_tdot_kq @ dh_dZ[rm1, ...] + h * b_tdot_dkqdz
                    dt_dZ[row, ...] = dt_dZ[rm1, ...] + dh_dZ[rm1, ...]

                rm1 = row
                row = row + 1

            # Evaluate the ODE at the last point in the segment (with the final times and states)
            self.eval_f(x[rm1, ...], t[rm1, 0], θ, k_q[0, ...], y=y[rm1, ...])

            if derivs:
                self.eval_f_derivs(x[rm1, ...], t[rm1, 0], θ, f_x, f_t, f_θ, y_x, y_t, y_θ)
                dy_dZ[rm1, ...] = y_x @ dx_dZ[rm1, ...] + y_t @ dt_dZ[rm1, ...] + y_θ @ dθ_dZ

    def _propagate_vectorized_derivs(self, inputs):
        """
        Propagate the states from t_initial to t_initial + t_duration, optionally computing
        the derivatives along the way and caching the current time and state values.

        The evaluation of the derivatives of the ODE is vectorized to remove overhead costs of
        compute totals.

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
        θ = self._θ
        y = self._y
        num_y = np.prod(y.shape)

        # Make t_initial and t_duration the first two elements of the ODE parameter vector.
        θ[0] = inputs['t_initial'].copy()
        θ[1] = inputs['t_duration'].copy()

        # Alias cached arrays
        f_x = self._f_x_vec
        f_t = self._f_t_vec
        f_θ = self._f_θ_vec

        y_x = self._y_x_vec
        y_t = self._y_t_vec
        y_θ = self._y_θ_vec

        dx_dZ = self._dx_dZ
        dt_dZ = self._dt_dZ
        dθ_dZ = self._dθ_dZ
        dkq_dZ = self._dkq_dZ
        dy_dZ = self._dy_dZ
        dh_dZ = self._dh_dZ
        dXi_dZ = self._dXi_dZ
        dTi_dZ = self._dTi_dZ

        k_q = self._k_q

        T_i = self._T_i
        X_i = self._X_i

        # Cache the tensordot product of a and k_q when integrating through each stage.
        a_tdot_k = {}

        # Initialize parameters
        for name in self.parameter_options:
            θ[self._parameter_idxs_in_θ[name], 0] = inputs[f'parameters:{name}'].ravel()

        # Initialize controls
        for name in self.control_options:
            θ[self._control_idxs_in_θ[name], 0] = inputs[f'controls:{name}'].ravel()

        # Initialize polynomial controls
        for name in self.polynomial_control_options:
            θ[self._polynomial_control_idxs_in_θ[name], 0] = inputs[f'polynomial_controls:{name}'].ravel()

        seg_durations = θ[1] * np.diff(gd.segment_ends) / 2.0

        # step counter
        row = 0

        for seg_i in range(gd.num_segments):
            self._eval_subprob.model._get_subsystem('ode_eval').set_segment_index(seg_i)
            self._deriv_subprob.model._get_subsystem('ode_eval').set_segment_index(seg_i)

            # Initialize, t, x, h, and derivatives for the start of the current segment
            self._initialize_segment(row, inputs, derivs=True)

            h = np.asarray(seg_durations[seg_i] / N, dtype=self._DTYPE)
            # On each segment, the total derivative of the stepsize h is a function of
            # the duration of the phase (the second element of the parameter vector after states)
            dh_dZ[row:row+N+1, 0, self.x_size+1] = seg_durations[seg_i] / θ[1] / N

            rm1 = row
            row = row + 1

            for q in range(N):
                # Compute the state rates and their partials at the start of the step
                T_i[0, 0] = t[rm1, 0]
                X_i[0, ...] = x[rm1, ...]

                self.eval_f(X_i[0, ...], T_i[0, 0], θ, k_q[0, ...], y=y[rm1, ...])

                # Now evaluate the ODE at each stage of the step.
                # States in subsequent ODE calls depend on the results of prior calls, so
                # the state values cannot be vectorized.
                for i in range(1, num_stages):
                    T_i[i, ...] = t[rm1, ...] + c[i] * h

                    a_tdot_k[i] = np.tensordot(a[i, :i], k_q[:i, ...], axes=(0, 0))
                    # a_tdot_k = np.einsum('i,ijk->jk', a[i, :i], self._k_q[:i, ...])
                    X_i[i, ...] = x[rm1, ...] + h * a_tdot_k[i]

                    self.eval_f(X_i[i, ...], T_i[i, 0], θ, k_q[i, ...])

                # Now make a single vectorized derivs call to evaluate the derivatives at all stages
                self.eval_f_derivs_vectorized(X_i, T_i, θ,
                                              f_x, f_t, f_θ,
                                              y_x, y_t, y_θ)

                # Accumulate the derivatives through the stages
                dkq_dZ[0, ...] = f_t[0, ...] @ dt_dZ[rm1, ...] + f_x[0, ...] @ dx_dZ[rm1, ...] + f_θ[0, ...] @ dθ_dZ

                if num_y > 0:
                    dy_dZ[rm1, ...] = y_x[0, ...] @ dx_dZ[rm1, ...] + y_t[0, ...] @ dt_dZ[rm1, ...] + y_θ[0, ...] @ dθ_dZ

                for i in range(1, num_stages):
                    dTi_dZ[...] = dt_dZ[rm1, ...] + c[i] * dh_dZ[rm1, ...]
                    a_tdot_dkqdz = np.tensordot(a[i, :i], dkq_dZ[:i, ...], axes=(0, 0))
                    # a_tdot_dkqdz = np.einsum('i,ijk->jk', a[i, :i], self._dkq_dZ[:i, ...])
                    dXi_dZ[...] = dx_dZ[rm1, ...] + a_tdot_k[i] @ dh_dZ[rm1, ...] + h * a_tdot_dkqdz
                    dkq_dZ[i, ...] = f_t[i, ...] @ dTi_dZ + f_x[i, ...] @ dXi_dZ + f_θ[i, ...] @ dθ_dZ

                # Compute x and t at the end of the step.
                b_tdot_kq = np.tensordot(b, k_q, axes=(0, 0))
                # b_tdot_kq = np.einsum('i,ijk->jk', b, self._k_q)
                x[row, ...] = x[rm1, ...] + h * b_tdot_kq
                t[row, 0] = t[rm1, 0] + h

                # Compute the derivatives of x and t wrt Z at the end of the step.
                b_tdot_dkqdz = np.tensordot(b, self._dkq_dZ, axes=(0, 0))
                # b_tdot_dkqdz = np.einsum('i,ijk->jk', b, self._dkq_dZ)
                dx_dZ[row, ...] = dx_dZ[rm1, ...] + b_tdot_kq @ dh_dZ[rm1, ...] + h * b_tdot_dkqdz
                dt_dZ[row, ...] = dt_dZ[rm1, ...] + dh_dZ[rm1, ...]

                rm1 = row
                row = row + 1

            # Evaluate the ODE at the last point in the segment (with the final times and states)
            self.eval_f(x[rm1, ...], t[rm1, 0], θ, k_q[0, ...], y=y[rm1, ...])
            self.eval_f_derivs(x[rm1, ...], t[rm1, 0], θ,
                               f_x=None, f_t=None, f_θ=None,
                               y_x=y_x[0, ...], y_t=y_t[0, ...], y_θ=y_θ[0, ...])
            dy_dZ[rm1, ...] = y_x[0, ...] @ dx_dZ[rm1, ...] + y_t[0, ...] @ dt_dZ[rm1, ...] + y_θ[0, ...] @ dθ_dZ

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
        self._inputs_cache = inputs.asarray()
        # self._propagate(inputs)
        self._propagate_vectorized_derivs(inputs)

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
        dt_dZ = self._dt_dZ
        dx_dZ = self._dx_dZ
        dy_dZ = self._dy_dZ

        if np.max(np.abs(self._inputs_cache - inputs.asarray())) > 1.0E-16:
            self._propagate_vectorized_derivs(inputs)

        idxs = self._output_src_idxs
        partials['time', 't_duration'] = dt_dZ[idxs, 0, self.x_size+1]
        partials['time_phase', 't_duration'] = dt_dZ[idxs, 0, self.x_size+1]

        for state_name in self.state_options:
            of = self._state_output_names[state_name]

            # Unpack the derivatives
            of_rows = self.state_idxs[state_name]

            partials[of, 't_initial'] = dx_dZ[idxs, of_rows, self.x_size]
            partials[of, 't_duration'] = dx_dZ[idxs, of_rows, self.x_size+1]

            for wrt_state_name in self.state_options:
                wrt = self._state_input_names[wrt_state_name]
                wrt_cols = self._state_idxs_in_Z[wrt_state_name]
                partials[of, wrt] = dx_dZ[idxs, of_rows, wrt_cols]

            for wrt_param_name in self.parameter_options:
                wrt = self._param_input_names[wrt_param_name]
                wrt_cols = self._parameter_idxs_in_Z[wrt_param_name]
                partials[of, wrt] = dx_dZ[idxs, of_rows, wrt_cols]

            for wrt_control_name in self.control_options:
                wrt = self._control_input_names[wrt_control_name]
                wrt_cols = self._control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = dx_dZ[idxs, of_rows, wrt_cols]

            for wrt_pc_name in self.polynomial_control_options:
                wrt = self._polynomial_control_input_names[wrt_pc_name]
                wrt_cols = self._polynomial_control_idxs_in_Z[wrt_pc_name]
                partials[of, wrt] = dx_dZ[idxs, of_rows, wrt_cols]

        for control_name in self.control_options:
            of = self._control_output_names[control_name]
            of_rate = self._control_rate_names[control_name]
            of_rate2 = self._control_rate2_names[control_name]

            # Unpack the derivatives
            of_rows = self._control_idxs_in_y[control_name]
            of_rate_rows = self._control_rate_idxs_in_y[control_name]
            of_rate2_rows = self._control_rate2_idxs_in_y[control_name]

            wrt_cols = self.x_size + 1
            partials[of_rate, 't_duration'] = dy_dZ[idxs, of_rate_rows, wrt_cols]
            partials[of_rate2, 't_duration'] = dy_dZ[idxs, of_rate2_rows, wrt_cols]

            for wrt_control_name in self.control_options:
                wrt = self._control_input_names[wrt_control_name]
                wrt_cols = self._control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = dy_dZ[idxs, of_rows, wrt_cols]
                partials[of_rate, wrt] = dy_dZ[idxs, of_rate_rows, wrt_cols]
                partials[of_rate2, wrt] = dy_dZ[idxs, of_rate2_rows, wrt_cols]

        for name in self.polynomial_control_options:
            of = self._polynomial_control_output_names[name]
            of_rate = self._polynomial_control_rate_names[name]
            of_rate2 = self._polynomial_control_rate2_names[name]

            # Unpack the derivatives
            of_rows = self._polynomial_control_idxs_in_y[name]
            of_rate_rows = self._polynomial_control_rate_idxs_in_y[name]
            of_rate2_rows = self._polynomial_control_rate2_idxs_in_y[name]

            wrt_cols = self.x_size + 1
            partials[of_rate, 't_duration'] = dy_dZ[idxs, of_rate_rows, wrt_cols]
            partials[of_rate2, 't_duration'] = dy_dZ[idxs, of_rate2_rows, wrt_cols]

            for wrt_control_name in self.polynomial_control_options:
                wrt = self._polynomial_control_input_names[wrt_control_name]
                wrt_cols = self._polynomial_control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = dy_dZ[idxs, of_rows, wrt_cols]
                partials[of_rate, wrt] = dy_dZ[idxs, of_rate_rows, wrt_cols]
                partials[of_rate2, wrt] = dy_dZ[idxs, of_rate2_rows, wrt_cols]

        for name, options in self._filtered_timeseries_outputs.items():
            of = self._timeseries_output_names[name]
            of_rows = self._timeseries_idxs_in_y[name]

            partials[of, 't_initial'] = dy_dZ[idxs, of_rows, self.x_size]
            partials[of, 't_duration'] = dy_dZ[idxs, of_rows, self.x_size+1]

            for wrt_state_name in self.state_options:
                wrt = self._state_input_names[wrt_state_name]
                wrt_cols = self._state_idxs_in_Z[wrt_state_name]
                partials[of, wrt] = dy_dZ[idxs, of_rows, wrt_cols]

            for wrt_param_name in self.parameter_options:
                wrt = self._param_input_names[wrt_param_name]
                wrt_cols = self._parameter_idxs_in_Z[wrt_param_name]
                partials[of, wrt] = dy_dZ[idxs, of_rows, wrt_cols]

            for wrt_control_name in self.control_options:
                wrt = self._control_input_names[wrt_control_name]
                wrt_cols = self._control_idxs_in_Z[wrt_control_name]
                partials[of, wrt] = dy_dZ[idxs, of_rows, wrt_cols]

            for wrt_pc_name in self.polynomial_control_options:
                wrt = self._polynomial_control_input_names[wrt_pc_name]
                wrt_cols = self._polynomial_control_idxs_in_Z[wrt_pc_name]
                partials[of, wrt] = dy_dZ[idxs, of_rows, wrt_cols]
