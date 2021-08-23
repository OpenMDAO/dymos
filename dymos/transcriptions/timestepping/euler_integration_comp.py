import collections

import numpy as np
import openmdao.api as om

from .ode_evaluation_group import ODEEvaluationGroup


class EulerIntegrationComp(om.ExplicitComponent):
    """
    This component contains a sub-Problem with a component that will be solved over num_nodes
    points instead of creating num_nodes instances of that same component and connecting them
    together.
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
        self.grid_data = grid_data
        self._TYPE = complex if complex_step_mode else float

        # If standalone_mode is True, this component will fully perform all of its setup at setup
        # time.  If False, it will need to have configure_io called on it to properly finish its
        # setup.
        self.standalone_mode = standalone_mode

    def initialize(self):
        self.options.declare('num_steps_per_segment', types=(int,), default=10)
        self.options.declare('ode_init_kwargs', types=dict, allow_none=True, default=None)

    def _setup_subprob(self):
        print('EulerIntegrtionComp: setup_subprob')
        self._prob = p = om.Problem(comm=self.comm)
        p.model.add_subsystem('ode_eval',
                              ODEEvaluationGroup(self.ode_class, self.time_options,
                                                 self.state_options, self.control_options,
                                                 self.polynomial_control_options,
                                                 self.parameter_options,
                                                 ode_init_kwargs=self.options['ode_init_kwargs'],
                                                 grid_data=self.grid_data),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.setup(force_alloc_complex=self._complex_step_mode)
        p.final_setup()
        self._prob.set_complex_step_mode(self._complex_step_mode)

    def _setup_time(self):
        if self.standalone_mode:
            self.configure_time_io()

    def configure_time_io(self):
        print('Euler integrator: configure time io')
        # First two elements of p are t_initial and t_duration
        self.p_size = 2

        self.add_input('t_initial', shape=(1,), units=self.time_options['units'])
        self.add_input('t_duration', shape=(1,), units=self.time_options['units'])
        self.add_output('t_final', shape=(1,), units=self.time_options['units'])

        self.declare_partials('t_final', 't_initial', val=1.0)
        self.declare_partials('t_final', 't_duration', val=1.0)

    def _setup_states(self):
        if self.standalone_mode:
            self.configure_states_io()

    def configure_states_io(self):
        print('EulerIntegrationComp: configure_states_io')
        # The total size of the entire state vector
        self.x_size = 0

        self._state_input_names = {}
        self._state_output_names = {}

        # The indices of each state
        self.state_idxs = {}

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

    def _setup_parameters(self):
        if self.standalone_mode:
            self.configure_parameters_io()

    def configure_parameters_io(self):
        print('EulerIntegrationComp: configure_parameters_io')
        # The indices of each parameter
        self.parameter_idxs = {}
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
        if self.standalone_mode:
            self.configure_controls_io()

    def configure_controls_io(self):
        print('Euler integration comp: configure controls io')
        # The indices of each control
        self.control_idxs = {}
        self._control_input_names = {}

        if self.control_options:
            control_input_node_ptau = self.grid_data.node_ptau[
                self.grid_data.subset_node_indices['control_input']]

        for control_name, options in self.control_options.items():
            control_param_shape = (len(control_input_node_ptau),) + options['shape']
            control_param_size = np.prod(control_param_shape, dtype=int)
            self._control_input_names[control_name] = f'controls:{control_name}'
            self.add_input(self._control_input_names[control_name],
                           shape=control_param_shape,
                           units=options['units'],
                           desc=f'values for control {control_name} at input nodes')

            self.control_idxs[control_name] = np.s_[self.p_size:self.p_size+control_param_size]
            self.p_size += control_param_size

    def _setup_storage(self):
        if self.standalone_mode:
            self._configure_storage()

    def _configure_storage(self):
        N = self.options['num_steps_per_segment']

        # The contiguous vector of state values
        self._x = np.zeros(self.x_size, dtype=self._TYPE)

        # The contiguous vector of parameter values
        self._p = np.zeros(self.p_size, dtype=self._TYPE)

        # The contiguous vector of state rates
        self._f = np.zeros(self.x_size, dtype=self._TYPE)

        # The derivatives of the state rates wrt the current time
        self._f_t = np.zeros((self.x_size, 1), dtype=self._TYPE)

        # The derivatives of the state rates wrt the current state
        self._f_x = np.zeros((self.x_size, self.x_size), dtype=self._TYPE)

        # The derivatives of the state rates wrt the parameters
        self._f_p = np.zeros((self.x_size, self.p_size), dtype=self._TYPE)

        # An identity matrix of the size of x
        self.I_x = np.eye(self.x_size, dtype=self._TYPE)

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

        self.dh_dt0 = np.zeros((1, 1), dtype=self._TYPE)
        self.dh_dtd = np.ones((1, 1), dtype=self._TYPE) / N

        self.dh_dp = np.zeros((1, self.p_size), dtype=self._TYPE)
        self.dh_dp[0, 1] = 0.0

        self.dt_dt = np.ones((1, 1), dtype=self._TYPE)
        self.dt_dh = np.ones((1, 1), dtype=self._TYPE)

    def setup(self):
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

        # Initialize the total derivatives of/wrt the states
        # self.dx_dt0[...] = 0.0
        # self.dx_dtd[...] = 0.0
        self.dx_dx0[...] = 0.0
        np.fill_diagonal(self.dx_dx0, 1.0)
        self.dx_dp[...] = 0.0

        # Initialize total derivatives of/wrt time
        self.dt_dp[...] = 0.0
        self.dt_dp[0, 0] = 1.0
        self.dh_dp[...] = 0.0
        self.dh_dp[0, 1] = 1.0 / N

    def eval_f(self, x, t, p):
        """
        Evaluate the ODE which provides the state rates for integration.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.
        p : np.ndarray
            A flattened, contiguous vector of the parameter values.

        Returns
        -------
        f : np.ndarray
            A flattened, contiguous vector of the state rates.

        """
        # transcribe time
        self._prob.set_val('time', t, units=self.time_options['units'])
        self._prob.set_val('t_initial', p[0], units=self.time_options['units'])
        self._prob.set_val('t_duration', p[1], units=self.time_options['units'])

        # transcribe states
        for state_name in self.state_options:
            self._prob.set_val(f'states:{state_name}', x[self.state_idxs[state_name]])

        # transcribe parameters
        for param_name in self.parameter_options:
            self._prob.set_val(f'parameters:{param_name}', p[self.parameter_idxs[param_name]])

        # transcribe controls
        for control_name in self.control_options:
            self._prob.set_val(f'controls:{control_name}', p[self.control_idxs[control_name]])

        # execute the ODE
        self._prob.run_model()

        # pack the resulting array
        for state_name in self.state_options:
            self._f[self.state_idxs[state_name]] = self._prob.get_val(f'state_rate_collector.state_rates:{state_name}_rate').ravel()

        return self._f

    def eval_f_derivs(self, x, t, p):
        """
        Evaluate the derivative of the ODE output rates wrt the inputs.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.
        p : np.ndarray
            A flattened, contiguous vector of the parameter values.

        Note that the control parameterization `u` undergoes an interpolation to provide the
        control values at any given time.  The ODE is then a function of these interpolated control
        values, we'll call them `u_hat`.  Technically, the derivatives wrt to `u` need to be chained
        together, but in this implementation the interpolation is part of the execution of the ODE
        and the chained derivatives are captured correctly there.

        Returns
        -------
        f_x : np.ndarray
            A matrix of the derivative of each element of the rates `f` wrt each value in `x`.
        f_t : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt `time`.
        f_p : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt the parameters `p`.
        f_u : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt the control parameters `u`.
        pu_pt : np.ndarray
            A matrix of the derivatives of each element of the interpolated state values `u` wrt `time`.

        """
        # transcribe time
        self._prob.set_val('time', t, units=self.time_options['units'])
        self._prob.set_val('t_initial', p[0], units=self.time_options['units'])
        self._prob.set_val('t_duration', p[1], units=self.time_options['units'])

        # transcribe states
        for state_name in self.state_options:
            self._prob.set_val(f'states:{state_name}', x[self.state_idxs[state_name]])

        # transcribe parameters
        for param_name in self.parameter_options:
            self._prob.set_val(f'parameters:{param_name}', p[self.parameter_idxs[param_name]])

        # transcribe controls
        for control_name in self.control_options:
            self._prob.set_val(f'controls:{control_name}', p[self.control_idxs[control_name]])

        self._prob.run_model()

        for state_name in self.state_options:
            idxs = self.state_idxs[state_name]
            self._f_t[self.state_idxs[state_name]] = self._prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                                               wrt='time', return_format='array', use_abs_names=False)

            for state_name_wrt in self.state_options:
                idxs_wrt = self.state_idxs[state_name_wrt]

                px_px = self._prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                 wrt=f'states:{state_name_wrt}', return_format='array', use_abs_names=False)

                self._f_x[idxs, idxs_wrt] = px_px.ravel()

            self._f_p[idxs, 0] = self._prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                           wrt='t_initial', return_format='array', use_abs_names=False)

            self._f_p[idxs, 1] = self._prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                           wrt='t_duration', return_format='array', use_abs_names=False)

            for param_name_wrt in self.parameter_options:
                idxs_wrt = self.parameter_idxs[param_name_wrt]

                px_pp = self._prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                 wrt=f'parameters:{param_name_wrt}', return_format='array', use_abs_names=False)

                self._f_p[idxs, idxs_wrt] = px_pp.ravel()

            for control_name_wrt in self.control_options:
                idxs_wrt = self.control_idxs[control_name_wrt]

                px_pu = self._prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                 wrt=f'controls:{control_name_wrt}', return_format='array', use_abs_names=False)

                self._f_p[idxs, idxs_wrt] = px_pu.ravel()

            for pc_name_wrt in self.polynomial_control_options:
                idxs_wrt = self.polynomial_control_idxs[pc_name_wrt]

                px_pu = self._prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                 wrt=f'polynomial_controls:{pc_name_wrt}', return_format='array', use_abs_names=False)

                self._f_p[idxs, idxs_wrt] = px_pu.ravel()

        return self._f_x, self._f_t, self._f_p

    def _update_derivs_fwd(self, pt_pt, pt_ph, px_pt, px_px, px_ph, px_pp):
        """
        Update the total derivatives being integrated across the propagation in forward mode.

        Parameters
        ----------
        pt_pt : np.array
            The partial derivative of time after the time update wrt time before the update.
        pt_ph : np.array
            The partial derivative of time after the time update wrt the step size.
        px_pt : np.array
            The partial derivative of the state vector after the state update wrt the time before
            the update.
        px_px : np.array
            The partial derivative of the state vector after the state update wrt the state vector
            before the update.
        px_ph : np.array
            The partial derivative of the state vector after the state update wrt the step size.
        """
        self.dx_dx0[...] = px_px @ self.dx_dx0

        self.dx_dp[...] = px_px @ self.dx_dp + \
                          px_pt @ self.dt_dp + \
                          px_ph @ self.dh_dp + \
                          px_pp

        self.dt_dp[...] = pt_pt @ self.dt_dp + \
                          pt_ph @ self.dh_dp

    def _update_derivs_rev(self, pt_pt, pt_ph, px_pt, px_px, px_ph, px_pp, px_pu):
        """
        Update the total derivatives being integrated across the propagation in reverse mode.

        Parameters
        ----------
        pt_pt : np.array
            The partial derivative of time after the time update wrt time before the update.
        pt_ph : np.array
            The partial derivative of time after the time update wrt the step size.
        px_pt : np.array
            The partial derivative of the state vector after the state update wrt the time before
            the update.
        px_px : np.array
            The partial derivative of the state vector after the state update wrt the state vector
            before the update.
        px_ph : np.array
            The partial derivative of the state vector after the state update wrt the step size.
        """
        raise NotImplementedError('Reverse-mode derivatives across integration are not yet implemented.')

    def _propagate(self, inputs, outputs, derivs=None, time_stack=None, state_stack=None):
        """
        Propagate the states from t_initial to t_initial + t_duration, optionally computing
        the derivatives along the way and caching the current time and state values.

        Parameters
        ----------
        inputs : vector
            The inputs from the compute call to the EulerIntegrationComp.
        outputs : vector
            The outputs from the compute call to the EulerIntegrationComp.
        derivs : vector or None
            If derivatives are to be calculated in a forward mode, this is the vector of partials
            from the compute_partials call to this component.  If derivatives are not to be
            computed, this should be None.
        time_stack : collections.deque
            A stack into which the time of each evaluation should be stored. This stack is cleared
            by this method before any values are added.
        state_stack : collections.deque
            A stack into which the state vector at each evaluation should be stored. This stack is
            cleared by this method before any values are added.
        """
        print('propagating')
        gd = self.grid_data
        N = self.options['num_steps_per_segment']

        # Initialize states
        x = self._x
        for state_name in self.state_options:
            x[self.state_idxs[state_name]] = inputs[f'states:{state_name}'].ravel()

        # Initialize time
        p = self._p

        t = inputs['t_initial']
        t_initial = inputs['t_initial']
        t_duration = inputs['t_duration']

        # Make t_initial and t_duration the first two elements of the parameter vector.
        p[0] = t_initial
        p[1] = t_duration

        # Initialize parameters
        for param_name in self.parameter_options:
            p[self.parameter_idxs[param_name]] = inputs[f'parameters:{param_name}'].ravel()

        # Initialize controls
        for control_name in self.control_options:
            p[self.control_idxs[control_name]] = inputs[f'controls:{control_name}'].ravel()

        if derivs:
            # Initialize the total derivatives
            self._reset_derivs()
            # From the time update equation, the partial of the new time wrt the previous time and
            # the partial wrt the stepsize are both [1].
            pt_pt = np.ones((1, 1), dtype=self._TYPE)
            pt_ph = np.ones((1, 1), dtype=self._TYPE)
            px_ph = np.ones((self.x_size, 1), dtype=self._TYPE)

        # if state_stack is not None:
        #     state_stack.clear()
        #     state_stack.append(x)
        # if time_stack is not None:
        #     time_stack.clear()
        #     time_stack.append(t)

        seg_durations = t_duration * np.diff(gd.segment_ends) / 2.0

        for seg_i in range(gd.num_segments):

            if N > 0:
                h = seg_durations[seg_i] / N
            else:
                h = 0

            # On each segment, the total derivative of the stepsize h is a function of
            # the duration of the phase (the second element of the parameter vector)
            self.dh_dp[0, 1] = np.diff(gd.segment_ends)[seg_i] / 2.0 / N

            for step_j in range(N):
                # Compute the state rates
                f = self.eval_f(x, t, p)

                if derivs:
                    # Compute the state rate derivatives
                    f_x, f_t, f_p = self.eval_f_derivs(x, t, p)

                    # The partials of the state update equation for Euler's method.
                    px_px = self.I_x + h * f_x
                    px_pt = h * f_t
                    px_ph[:, 0] = f
                    px_pp = h * f_p

                    # Accumulate the totals
                    self._update_derivs_fwd(pt_pt, pt_ph, px_pt, px_px, px_ph, px_pp)
                    # else:
                    #     self._update_derivs_rev(pt_pt, pt_ph, px_pt, px_px, px_ph, px_pp)

                t = t + h
                x = x + h * f

                # # The ODE is not evaluated at the final time, so don't append its value here.
                # if i < N - 1:
                #     if state_stack:
                #         state_stack.append(x)
                #     if time_stack:
                #         time_stack.append(t)

        # Unpack the final values
        if outputs:
            outputs['t_final'] = t

            for state_name, options in self.state_options.items():
                of = f'states_out:{state_name}'
                outputs[of] = x[self.state_idxs[state_name]].reshape(options['shape'])

        if derivs:
            derivs['t_final', 't_initial'] = self.dt_dp[0, 0]
            derivs['t_final', 't_duration'] = self.dt_dp[0, 1]

            for state_name in self.state_options:
                of = f'states_out:{state_name}'

                # Unpack the derivatives
                of_rows = self.state_idxs[state_name]

                derivs[of, 't_initial'] = self.dx_dp[of_rows, 0]
                derivs[of, 't_duration'] = self.dx_dp[of_rows, 1]

                for wrt_state_name in self.state_options:
                    wrt = f'states:{wrt_state_name}'
                    wrt_cols = self.state_idxs[wrt_state_name]
                    derivs[of, wrt] = self.dx_dx0[of_rows, wrt_cols]

                for wrt_param_name in self.parameter_options:
                    wrt = self._param_input_names[wrt_param_name]
                    wrt_cols = self.parameter_idxs[wrt_param_name]
                    derivs[of, wrt] = self.dx_dp[of_rows, wrt_cols]

                for wrt_control_name in self.control_options:
                    wrt = self._control_input_names[wrt_control_name]
                    wrt_cols = self.control_idxs[wrt_control_name]
                    derivs[of, wrt] = self.dx_dp[of_rows, wrt_cols]

    def _backpropagate_derivs(self, inputs, derivs, time_stack, state_stack):
        """
        Use backward propagation to compute the derivatives in reverse mode.

        Parameters
        ----------
        inputs : vector
            The inputs from the compute call to the EulerIntegrationComp.
        derivs : vector or None
            The vector of partials from the compute_partials call to this component.
        time_stack : collections.deque
            A stack into which the time of each evaluation are stored. This stack is emptied
            in the process of executing this method.
        state_stack : collections.deque
            A stack into which the state vector at each evaluation are stored. This stack is emptied
            in the process of executing this method.
        """
        N = self.options['num_steps_per_segment']

        if N > 0:
            h = inputs['t_duration'] / N
        else:
            h = 0

        self._reset_derivs()

        # From the time update equation, the partial of the new time wrt the previous time and
        # the partial wrt the stepsize are both [1].
        pt_pt = np.ones((1, 1), dtype=self._TYPE)
        pt_ph = np.ones((1, 1), dtype=self._TYPE)

        for i in range(N):
            # Extract the saved integrated quantities
            t = time_stack.pop()
            x = state_stack.pop()

            # Compute the state rates
            f = self.eval_f(x, t, p)

            # Compute the state rate derivatives
            f_x, f_t, f_p = self.eval_f_derivs(x, t, p)

            # The partials of the state update equation for Euler's method.
            px_px = self.I_x + h * f_x
            px_pt = h * f_t
            px_ph = f

            self._update_derivs_rev(pt_pt, pt_ph, px_pt, px_px, px_ph)

        derivs['t_final', 't_initial'] = self.dt_dt0_bar.T
        derivs['t_final', 't_duration'] = self.dt_dtd_bar.T

        for state_name in self.state_options:
            of = f'state_final_values:{state_name}'

            # Unpack the derivatives
            of_rows = self.state_idxs[state_name]

            derivs[of, 't_initial'] = self.dx_dt0_bar.T[of_rows]
            derivs[of, 't_duration'] = self.dx_dtd_bar.T[of_rows]

            for wrt_state_name in self.state_options:
                wrt = f'state_initial_values:{wrt_state_name}'
                wrt_cols = self.state_idxs[wrt_state_name]
                derivs[of, wrt] = self.dx_dx0_bar[wrt_cols, of_rows]

    def compute(self, inputs, outputs):
        self._propagate(inputs, outputs)

    def compute_partials(self, inputs, partials):
        # note that typically you would only have to define partials for one direction,
        # either fwd OR rev, not both.
        # if self._mode == 'fwd':
        self._propagate(inputs, outputs=False, derivs=partials)
        # else:
        #     time_stack = collections.deque()
        #     state_stack = collections.deque()
        #
        #     self._propagate(inputs, outputs=False, derivs=None, time_stack=time_stack, state_stack=state_stack)
        #     self._backpropagate_derivs(inputs, partials, time_stack=time_stack, state_stack=state_stack)
