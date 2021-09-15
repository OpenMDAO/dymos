import collections

import numpy as np
import openmdao.api as om

from .ode_evaluation_group import ODEEvaluationGroup


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
                 polynomial_control_options=None, complex_step_mode=False,
                 grid_data=None, standalone_mode=True, **kwargs):
        super().__init__(**kwargs)
        self.ode_class = ode_class
        self.time_options = time_options
        self.state_options = state_options
        self.parameter_options = parameter_options
        self.control_options = control_options
        self.polynomial_control_options = polynomial_control_options
        self._prob = None
        self._complex_step_mode = complex_step_mode
        self._grid_data = grid_data
        self._TYPE = complex if complex_step_mode else float

        self.x_size = 0
        self.p_size = 0
        self.u_size = 0
        self.up_size = 0
        self.phi_size = 0
        self.Z_size = 0

        # If _standalone_mode is True, this component will fully perform all of its setup at setup
        # time.  If False, it will need to have configure_io called on it to properly finish its
        # setup.
        self._standalone_mode = standalone_mode

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
        self.add_input('t_initial', shape=(1,), units=self.time_options['units'])
        self.add_input('t_duration', shape=(1,), units=self.time_options['units'])
        self.add_output('t_final', shape=(1,), units=self.time_options['units'])

        self.declare_partials('t_final', 't_initial', val=1.0)
        self.declare_partials('t_final', 't_duration', val=1.0)

    def _setup_states(self):
        if self._standalone_mode:
            self._configure_states_io()

    def _configure_states_io(self):
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
            self.add_input(self._state_input_names[state_name],
                           shape=options['shape'],
                           desc=f'initial value of state {state_name}')
            self.add_output(self._state_output_names[state_name],
                            shape=options['shape'],
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
        self._parameter_idxs_in_phi = {}
        self._parameter_idxs_in_Z = {}
        self._param_input_names = {}

        for param_name, options in self.parameter_options.items():
            self._param_input_names[param_name] = f'parameters:{param_name}'
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
        self._control_idxs_in_phi = {}
        self._control_idxs_in_Z = {}
        self._control_input_names = {}

        if self.control_options:
            control_input_node_ptau = self._grid_data.node_ptau[
                self._grid_data.subset_node_indices['control_input']]

        for control_name, options in self.control_options.items():
            control_param_shape = (len(control_input_node_ptau),) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._control_input_names[control_name] = f'controls:{control_name}'
            self.add_input(self._control_input_names[control_name],
                           shape=control_param_shape,
                           units=options['units'],
                           desc=f'values for control {control_name} at input nodes')

            self.control_idxs[control_name] = np.s_[self.u_size:self.u_size+control_param_size]
            self.u_size += control_param_size

    def _configure_polynomial_controls_io(self):
        self.up_size = 0
        self.polynomial_control_idxs = {}
        self._polynomial_control_idxs_in_phi = {}
        self._polynomial_control_idxs_in_Z = {}
        self._polynomial_control_input_names = {}

        for name, options in self.polynomial_control_options.items():
            num_input_nodes = options['order'] + 1
            control_param_shape = (num_input_nodes,) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._polynomial_control_input_names[name] = f'polynomial_controls:{name}'
            self.add_input(self._polynomial_control_input_names[name],
                           shape=control_param_shape,
                           units=options['units'],
                           desc=f'values for control {name} at input nodes')

            self.polynomial_control_idxs[name] = np.s_[self.up_size:self.up_size+control_param_size]
            self.up_size += control_param_size

    def _setup_storage(self):
        if self._standalone_mode:
            self._configure_storage()

    def _configure_storage(self):
        control_input_node_ptau = self._grid_data.node_ptau[self._grid_data.subset_node_indices['control_input']]
        rk = rk_methods[self.options['method']]
        num_stages = len(rk['b'])

        # allocate the ODE parameter vector
        self.phi_size = 2 + self.p_size + self.u_size + self.up_size

        # allocate the integration parameter vector
        self.Z_size = self.x_size + self.phi_size + self.up_size

        start_Z = 0
        for state_name, options in self.state_options.items():
            state_size = np.prod(options['shape'], dtype=int)
            self._state_idxs_in_Z[state_name] = np.s_[start_Z: start_Z+state_size]
            start_Z += state_size

        start_Z = self.x_size + 2
        start_phi = 2
        for param_name, options in self.parameter_options.items():
            param_size = np.prod(options['shape'], dtype=int)
            self._parameter_idxs_in_Z[param_name] = np.s_[start_Z: start_Z+param_size]
            self._parameter_idxs_in_phi[param_name] = np.s_[start_phi: start_phi+param_size]
            start_Z += param_size
            start_phi += param_size

        start_Z = self.x_size + 2 + self.p_size
        start_phi = 2 + self.p_size
        for control_name, options in self.control_options.items():
            control_param_shape = (len(control_input_node_ptau),) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._control_idxs_in_Z[control_name] = np.s_[start_Z:start_Z+control_param_size]
            self._control_idxs_in_phi[control_name] = np.s_[start_phi:start_phi+control_param_size]
            start_Z += control_param_size
            start_phi += control_param_size

        start_Z = self.x_size + 2 + self.p_size + self.u_size
        start_phi = 2 + self.p_size + self.u_size
        for name, options in self.polynomial_control_options.items():
            num_input_nodes = options['order'] + 1
            control_param_shape = (num_input_nodes,) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._polynomial_control_idxs_in_Z[name] = np.s_[start_Z:start_Z+control_param_size]
            self._polynomial_control_idxs_in_phi[name] = np.s_[start_phi:start_phi+control_param_size]
            start_Z += control_param_size
            start_phi += control_param_size

        # The contiguous vector of state values
        self._x = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The contiguous vector of ODE parameter values
        self._phi = np.zeros((self.phi_size, 1), dtype=self._TYPE)

        # The contiguous vector of state rates
        self._f = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The derivatives of the state rates wrt the current time
        self._f_t = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The derivatives of the state rates wrt the current state
        self._f_x = np.zeros((self.x_size, self.x_size), dtype=self._TYPE)

        # The derivatives of the state rates wrt the parameters
        self._f_phi = np.zeros((self.x_size, self.phi_size), dtype=self._TYPE)

        # Intermediate state rate storage
        self._k_q = np.zeros((num_stages, self.x_size, 1), dtype=self._TYPE)

        # The partial derivative of the final state vector wrt time from state update equation.
        self.px_pt = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The partial derivative of the final state vector wrt the initial state vector from the
        # state update equation.
        self.px_px = np.zeros((self.x_size, self.x_size), dtype=self._TYPE)

        # The total derivative of the current value of x wrt the initial value of x
        self.dx_dx0 = np.eye(self.x_size, dtype=self._TYPE)

        # The total derivative of the current value of x wrt the parameter values
        self.dx_dp = np.zeros((self.x_size, self.p_size), dtype=self._TYPE)

        # The total derivative of the current value of time wrt the parameter values
        self.dt_dp = np.zeros((1, self.p_size), dtype=self._TYPE)

        N = self.options['num_steps_per_segment']
        rk = rk_methods[self.options['method']]
        num_stages = len(rk['b'])
        num_x = self.x_size
        num_phi = self.phi_size
        num_z = num_x + num_phi

        # Let Z be [x0.ravel() t0 tp p.ravel() u.ravel()]
        self._dx_dZ = np.zeros((num_x, num_z), dtype=self._TYPE)
        self._dx_dZ[:, :num_x] = np.eye(num_x, dtype=self._TYPE)
        self._dt_dZ = np.zeros((1, num_z), dtype=self._TYPE)
        self._dt_dZ[0, num_x] = 1.0
        self._dh_dZ = np.zeros((1, num_z), dtype=self._TYPE)
        self._dh_dZ[0, num_x+1] = 1.0 / N
        self._dTi_dZ = np.zeros((1, num_z), dtype=self._TYPE)
        self._dXi_dZ = np.zeros((num_x, num_z), dtype=self._TYPE)
        self._dkq_dZ = np.zeros((num_stages, num_x, num_z), dtype=self._TYPE)
        self._dphi_dZ = np.zeros((num_phi, num_z), dtype=self._TYPE)
        self._dphi_dZ[:, num_x:] = np.eye(num_phi, dtype=self._TYPE)

    def setup(self):
        """
        Add the necessary I/O and storage for the RKIntegrationComp.
        """
        self._setup_subprob()
        self._setup_time()
        self._setup_parameters()
        self._setup_controls()
        self._setup_states()
        self._setup_storage()

    def _reset_derivs(self):
        """
        Reset the value of total derivatives prior to propagation.
        """
        N = self.options['num_steps_per_segment']
        num_x = self.x_size
        num_phi = self.phi_size

        # Let Z be [x0.ravel() t0 tp p.ravel() u.ravel()]
        self._dx_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._TYPE)
        self._dx_dZ[:, :num_x] = np.eye(num_x, dtype=self._TYPE)
        self._dt_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._TYPE)
        self._dt_dZ[0, num_x] = 1.0
        self._dh_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._TYPE)
        self._dh_dZ[0, num_x+1] = 1.0 / N
        self._dTi_dZ[...] = 0.0  # np.zeros((1, num_z), dtype=self._TYPE)
        self._dXi_dZ[...] = 0.0  # np.zeros((num_x, num_z), dtype=self._TYPE)
        self._dkq_dZ[...] = 0.0  # np.zeros((num_stages, num_x, num_z), dtype=self._TYPE)
        self._dphi_dZ[...] = 0.0  # np.zeros((num_phi, num_z), dtype=self._TYPE)
        self._dphi_dZ[:, num_x:] = np.eye(num_phi, dtype=self._TYPE)

    def eval_f(self, x, t, phi):
        """
        Evaluate the ODE which provides the state rates for integration.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.
        phi : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.

        Returns
        -------
        np.ndarray
            A flattened, contiguous vector of the state rates.
        """
        # transcribe time
        self._prob.set_val('time', t, units=self.time_options['units'])
        self._prob.set_val('t_initial', phi[0], units=self.time_options['units'])
        self._prob.set_val('t_duration', phi[1], units=self.time_options['units'])

        # transcribe states
        for name in self.state_options:
            self._prob.set_val(f'states:{name}', x[self.state_idxs[name], 0])

        # transcribe parameters
        for name in self.parameter_options:
            self._prob.set_val(f'parameters:{name}', phi[self._parameter_idxs_in_phi[name]])

        # transcribe controls
        for name in self.control_options:
            self._prob.set_val(f'controls:{name}', phi[self._control_idxs_in_phi[name]])

        # transcribe polynomial controls
        for name in self.polynomial_control_options:
            self._prob.set_val(f'polynomial_controls:{name}', phi[self._polynomial_control_idxs_in_phi[name]])

        # execute the ODE
        self._prob.run_model()

        # pack the resulting array
        for name in self.state_options:
            self._f[self.state_idxs[name]] = \
                self._prob.get_val(f'state_rate_collector.state_rates:{name}_rate').ravel()

        return self._f

    def eval_f_derivs(self, x, t, phi):
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
        phi : np.ndarray
            A flattened, contiguous vector of the ODE parameter values.

        Returns
        -------
        f_x : np.ndarray
            A matrix of the derivative of each element of the rates `f` wrt each value in `x`.
        f_t : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt `time`.
        f_phi : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt the parameters `phi`.
        """
        of_names = []
        wrt_names = ['time', 't_initial', 't_duration']

        # transcribe time
        self._prob.set_val('time', t, units=self.time_options['units'])
        self._prob.set_val('t_initial', phi[0, 0], units=self.time_options['units'])
        self._prob.set_val('t_duration', phi[1, 0], units=self.time_options['units'])

        # transcribe states
        for name in self.state_options:
            input_name = f'states:{name}'
            self._prob.set_val(input_name, x[self.state_idxs[name], 0])
            of_names.append(f'state_rate_collector.state_rates:{name}_rate')
            wrt_names.append(input_name)

        # transcribe parameters
        for name in self.parameter_options:
            input_name = f'parameters:{name}'
            self._prob.set_val(input_name, phi[self._parameter_idxs_in_phi[name], 0])
            wrt_names.append(input_name)

        # transcribe controls
        for name in self.control_options:
            input_name = f'controls:{name}'
            self._prob.set_val(input_name, phi[self._control_idxs_in_phi[name], 0])
            wrt_names.append(input_name)

        for name in self.polynomial_control_options:
            input_name = f'polynomial_controls:{name}'
            self._prob.set_val(input_name, phi[self._polynomial_control_idxs_in_phi[name], 0])
            wrt_names.append(input_name)

        # Re-run in case the inputs have changed.
        self._prob.run_model()

        totals = self._prob.compute_totals(of=of_names, wrt=wrt_names, use_abs_names=False)

        for state_name in self.state_options:
            of_name = f'state_rate_collector.state_rates:{state_name}_rate'
            idxs = self.state_idxs[state_name]
            self._f_t[self.state_idxs[state_name]] = totals[of_name, 'time']

            for state_name_wrt in self.state_options:
                idxs_wrt = self.state_idxs[state_name_wrt]
                px_px = totals[of_name, f'states:{state_name_wrt}']
                self._f_x[idxs, idxs_wrt] = px_px.ravel()

            self._f_phi[idxs, 0] = totals[of_name, 't_initial']
            self._f_phi[idxs, 1] = totals[of_name, 't_duration']

            for param_name_wrt in self.parameter_options:
                idxs_wrt = self._parameter_idxs_in_phi[param_name_wrt]
                px_pp = totals[of_name, f'parameters:{param_name_wrt}']
                self._f_phi[idxs, idxs_wrt] = px_pp.ravel()

            for control_name_wrt in self.control_options:
                idxs_wrt = self._control_idxs_in_phi[control_name_wrt]
                px_pu = totals[of_name, f'controls:{control_name_wrt}']
                self._f_phi[idxs, idxs_wrt] = px_pu.ravel()

            for pc_name_wrt in self.polynomial_control_options:
                idxs_wrt = self._polynomial_control_idxs_in_phi[pc_name_wrt]
                px_pu = totals[of_name, f'polynomial_controls:{pc_name_wrt}']
                self._f_phi[idxs, idxs_wrt] = px_pu.ravel()

        return self._f_x, self._f_t, self._f_phi

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
        for state_name in self.state_options:
            x[self.state_idxs[state_name], 0] = inputs[f'states:{state_name}'].ravel()

        # Initialize time
        phi = self._phi

        t = np.asarray(inputs['t_initial'], dtype=self._TYPE)
        t_initial = inputs['t_initial'].copy()
        t_duration = inputs['t_duration'].copy()

        # Make t_initial and t_duration the first two elements of the parameter vector.
        phi[0] = t_initial
        phi[1] = t_duration

        # Initialize parameters
        for name in self.parameter_options:
            phi[self._parameter_idxs_in_phi[name], 0] = inputs[f'parameters:{name}'].ravel()

        # Initialize controls
        for name in self.control_options:
            phi[self._control_idxs_in_phi[name], 0] = inputs[f'controls:{name}'].ravel()

        # Initialize polynomial controls
        for name in self.polynomial_control_options:
            phi[self._polynomial_control_idxs_in_phi[name], 0] = \
                inputs[f'polynomial_controls:{name}'].ravel()

        if derivs:
            # Initialize the total derivatives
            self._reset_derivs()

        seg_durations = t_duration * np.diff(gd.segment_ends) / 2.0

        for seg_i in range(gd.num_segments):

            if N > 0:
                h = np.asarray(seg_durations[seg_i] / N, dtype=self._TYPE)
            else:
                h = np.zeros(1, dtype=self._TYPE)

            # On each segment, the total derivative of the stepsize h is a function of
            # the duration of the phase (the second element of the parameter vector)
            if derivs:
                self._dh_dZ[0, self.x_size+1] = np.diff(gd.segment_ends)[seg_i] / 2.0 / N

            for q in range(N):
                # Compute the state rates and their partials
                self._k_q[0, ...] = self.eval_f(x, t, phi)

                if derivs:
                    # Compute the state rate derivatives
                    f_x, f_t, f_phi = self.eval_f_derivs(x, t, phi)
                    self._dkq_dZ[0, ...] = f_t @ self._dt_dZ + f_x @ self._dx_dZ + f_phi @ self._dphi_dZ

                for i in range(1, num_stages):
                    T_i = t + c[i] * h
                    # a_dot_k = np.tensordot(a[i, :i], self._k_q[:i, ...], axes=(0, 0))
                    a_tdot_k = np.einsum('i,ijk->jk', a[i, :i], self._k_q[:i, ...])
                    X_i = x + h * a_tdot_k
                    self._k_q[i, ...] = self.eval_f(X_i, T_i, phi)

                    if derivs:
                        f_x, f_t, f_phi = self.eval_f_derivs(X_i, T_i, phi)
                        self._dTi_dZ[...] = self._dt_dZ + c[i] * self._dh_dZ
                        a_tdot_dkqdz = np.tensordot(a[i, :i], self._dkq_dZ[:i, ...], axes=(0, 0))
                        # a_tdot_dkqdz = np.einsum('i,ijk->jk', a[i, :i], self._dkq_dZ[:i, ...])
                        self._dXi_dZ[...] = self._dx_dZ + a_tdot_k @ self._dh_dZ + h * a_tdot_dkqdz
                        self._dkq_dZ[i, ...] = f_t @ self._dTi_dZ + f_x @ self._dXi_dZ + f_phi @ self._dphi_dZ

                b_tdot_kq = px_ph = np.tensordot(b, self._k_q, axes=(0, 0))
                # b_tdot_kq = np.einsum('i,ijk->jk', b, self._k_q)
                x = x + h * b_tdot_kq
                t = t + h

                if derivs:
                    b_tdot_dkqdz = np.tensordot(b, self._dkq_dZ, axes=(0, 0))
                    # b_tdot_dkqdz = np.einsum('i,ijk->jk', b, self._dkq_dZ)
                    self._dx_dZ[...] = self._dx_dZ + b_tdot_kq @ self._dh_dZ + h * b_tdot_dkqdz
                    self._dt_dZ[...] = self._dt_dZ + self._dh_dZ

        # Unpack the final values
        if outputs:
            outputs['t_final'] = t

            for state_name, options in self.state_options.items():
                of = f'states_out:{state_name}'
                outputs[of] = x[self.state_idxs[state_name]].reshape(options['shape'])

        if derivs:
            derivs['t_final', 't_initial'] = self._dt_dZ[0, self.x_size]
            derivs['t_final', 't_duration'] = self._dt_dZ[0, self.x_size+1]

            for state_name in self.state_options:
                of = f'states_out:{state_name}'

                # Unpack the derivatives
                of_rows = self.state_idxs[state_name]

                derivs[of, 't_initial'] = self._dx_dZ[of_rows, self.x_size]
                derivs[of, 't_duration'] = self._dx_dZ[of_rows, self.x_size+1]

                for wrt_state_name in self.state_options:
                    wrt = f'states:{wrt_state_name}'
                    wrt_cols = self._state_idxs_in_Z[wrt_state_name]
                    derivs[of, wrt] = self._dx_dZ[of_rows, wrt_cols]

                for wrt_param_name in self.parameter_options:
                    wrt = self._param_input_names[wrt_param_name]
                    wrt_cols = self._parameter_idxs_in_Z[wrt_param_name]
                    derivs[of, wrt] = self._dx_dZ[of_rows, wrt_cols]

                for wrt_control_name in self.control_options:
                    wrt = self._control_input_names[wrt_control_name]
                    wrt_cols = self._control_idxs_in_Z[wrt_control_name]
                    derivs[of, wrt] = self._dx_dZ[of_rows, wrt_cols]

                for wrt_pc_name in self.polynomial_control_options:
                    wrt = self._polynomial_control_input_names[wrt_pc_name]
                    wrt_cols = self._polynomial_control_idxs_in_Z[wrt_pc_name]
                    derivs[of, wrt] = self._dx_dZ[of_rows, wrt_cols]

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
        self._propagate(inputs, outputs=False, derivs=partials)
