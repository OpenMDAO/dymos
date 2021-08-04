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
                 polynomial_control_options=None, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.ode_class = ode_class
        self.time_options = time_options
        self.state_options = state_options
        self.parameter_options = parameter_options
        self.control_options = control_options
        self.polynomial_control_options = polynomial_control_options
        self.mode = mode
        self.prob = None

    def initialize(self):
        self.options.declare('num_steps', types=(int,), default=10)
        self.options.declare('ode_init_kwargs', types=dict, allow_none=True, default=None)

    def _setup_subprob(self):
        self.prob = p = om.Problem(comm=self.comm)
        p.model.add_subsystem('ode_eval',
                              ODEEvaluationGroup(self.ode_class, self.time_options,
                                                 self.state_options, self.control_options,
                                                 self.polynomial_control_options,
                                                 self.parameter_options, ode_init_kwargs=None),
                              promotes_inputs=['*'],
                              promotes_outputs=['*'])

        p.setup()
        p.final_setup()

    def _setup_time(self):
        self.add_input('time', shape=(1,), units=self.time_options['units'])
        self.add_input('time_phase', shape=(1,), units=self.time_options['units'])
        self.add_input('t_initial', shape=(1,), units=self.time_options['units'])
        self.add_input('t_duration', shape=(1,), units=self.time_options['units'])
        self.add_output('t_final', shape=(1,), units=self.time_options['units'])

        self.declare_partials('t_final', 't_initial', val=1.0)
        self.declare_partials('t_final', 't_duration', val=1.0)

    def _setup_states(self):
        N = self.options['num_steps']

        if self.mode == 'fwd':
            self.dt_dt0 = np.zeros((1, 1), dtype=complex)
            self.dt_dtd = np.zeros((1, 1), dtype=complex)

            self.dh_dt0 = np.zeros((1, 1), dtype=complex)
            self.dh_dtd = np.ones((1, 1), dtype=complex) / N
            self.dt_dt = np.ones((1, 1), dtype=complex)
            self.dt_dh = np.ones((1, 1), dtype=complex)
        else:
            # These are all 1x1 matrices but we transpose them here just to avoid confusion.
            self.dt_dt0_bar = np.zeros((1, 1), dtype=complex).T
            self.dt_dtd_bar = np.zeros((1, 1), dtype=complex).T

            self.dh_dt0_bar = np.zeros((1, 1), dtype=complex).T
            self.dh_dtd_bar = np.ones((1, 1), dtype=complex).T / N
            self.dt_dt_bar = np.ones((1, 1), dtype=complex).T
            self.dt_dh_bar = np.ones((1, 1), dtype=complex).T


        # The total size of the entire state vector
        self.x_size = 0

        # The indices of each state
        self.state_idxs = {}

        for state_name, options in self.state_options.items():
            self.add_input(f'state_initial_value:{state_name}',
                           shape=options['shape'],
                           desc=f'initial value of state {state_name}')
            self.add_output(f'state_final_value:{state_name}',
                            shape=options['shape'],
                            desc=f'final value of state {state_name}')

            # self.state_rates[state_name] = np.zeros(options['shape'])

            state_size = np.prod(options['shape'], dtype=int)
            self.state_idxs[state_name] = np.s_[self.x_size:state_size]
            self.x_size += state_size

            self.declare_partials(of=f'state_final_value:{state_name}',
                                  wrt='t_initial')

            self.declare_partials(of=f'state_final_value:{state_name}',
                                  wrt='t_duration')

            for state_name_wrt, options_wrt in self.state_options.items():
                self.declare_partials(of=f'state_final_value:{state_name}',
                                      wrt=f'state_initial_value:{state_name_wrt}')

        # The contiguous vector of state values
        self._x = np.zeros(self.x_size, dtype=complex)

        # The contiguous vector of state rates
        self._f = np.zeros(self.x_size, dtype=complex)

        self._f_t = np.zeros((self.x_size, 1), dtype=complex)
        self._f_x = np.zeros((self.x_size, self.x_size), dtype=complex)

        # An identity matrix of the size of x
        self.I_x = np.eye(self.x_size, dtype=complex)

        # The partial derivative of the final state vector wrt time from state update equation.
        self.px_pt = np.zeros((self.x_size, 1), dtype=complex)

        # The partial derivative of the final state vector wrt the initial state vector from the
        # state update equation.
        self.px_px = np.zeros((self.x_size, self.x_size), dtype=complex)

        if self.mode == 'fwd':
            # The total derivative of the current value of x wrt the initial time of the propagation.
            self.dx_dt0 = np.zeros((self.x_size, 1), dtype=complex)

            # The total derivative of the current value of x wrt the time duration of the propagation.
            self.dx_dtd = np.zeros((self.x_size, 1), dtype=complex)

            # The total derivative of the current value of x wrt the initial value of x
            self.dx_dx0 = np.eye(self.x_size, dtype=complex)
        else:
            # The total derivative of the current value of x wrt the initial time of the propagation.
            self.dx_dt0_bar = np.zeros((self.x_size, 1), dtype=complex).T

            # The total derivative of the current value of x wrt the time duration of the propagation.
            self.dx_dtd_bar = np.zeros((self.x_size, 1), dtype=complex).T

            # The total derivative of the current value of x wrt the initial value of x
            self.dx_dx0_bar = np.eye(self.x_size, dtype=complex).T

    def setup(self):
        self._setup_subprob()
        self._setup_time()
        self._setup_states()

    def _initialize_derivs(self):
        """
        Reset the value of total derivatives prior to propagation.
        """
        N = self.options['num_steps']
        if self.mode == 'fwd':
            # Initialize the total derivatives of/wrt the states
            self.dx_dt0[...] = 0.0
            self.dx_dtd[...] = 0.0
            self.dx_dx0[...] = 0.0
            np.fill_diagonal(self.dx_dx0, 1.0)

            # Initialize total derivatives of/wrt time
            self.dt_dtd[...] = 0.0
            self.dt_dt0[...] = 1.0
            self.dh_dt0[...] = 0.0
            self.dh_dtd[...] = 1.0 / N
        else:
            # Initialize the total derivatives of/wrt the states
            self.dx_dt0_bar[...] = 0.0
            self.dx_dtd_bar[...] = 0.0
            self.dx_dx0_bar[...] = 0.0
            np.fill_diagonal(self.dx_dx0_bar, 1.0)

            # Initialize total derivatives of/wrt time
            self.dt_dtd_bar[...] = 0.0
            self.dt_dt0_bar[...] = 1.0
            self.dh_dt0_bar[...] = 0.0
            self.dh_dtd_bar[...] = 1.0 / N

    def eval_f(self, x, t):
        """
        Evaluate the ODE which provides the state rates for integration.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.

        Returns
        -------
        f : np.ndarray
            A flattened, contiguous vector of the state rates.

        """
        # transcribe time
        self.prob.set_val('time', t, units=self.time_options['units'])

        # transcribe states
        for state_name in self.state_options:
            self.prob.set_val(f'states:{state_name}', x[self.state_idxs[state_name]])

        # execute the ODE
        self.prob.run_model()

        # pack the resulting array
        for state_name in self.state_options:
            self._f[self.state_idxs[state_name]] = self.prob.get_val(f'state_rate_collector.state_rates:{state_name}_rate').ravel()

        return self._f

    def eval_f_derivs(self, x, t):
        """
        Evaluate the derivative of the ODE output rates wrt the inputs.

        Parameters
        ----------
        x : np.ndarray
            A flattened, contiguous vector of the state values.
        t : float
            The current time of the integration.

        Returns
        -------
        f_x : np.ndarray
            A matrix of the derivative of each element of the rates `f` wrt each value in `x`.
        f_t : np.ndarray
            A matrix of the derivatives of each element of the rates `f` wrt time.

        """
        # transcribe time
        self.prob.set_val('time', t, units=self.time_options['units'])

        # transcribe states
        for state_name in self.state_options:
            self.prob.set_val(f'states:{state_name}', x[self.state_idxs[state_name]])

            idxs = self.state_idxs[state_name]

            self._f_t[self.state_idxs[state_name]] = self.prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                                              wrt='time', return_format='array', use_abs_names=False)

            for state_name_wrt in self.state_options:
                idxs_wrt = self.state_idxs[state_name]

                px_px = self.prob.compute_totals(of=f'state_rate_collector.state_rates:{state_name}_rate',
                                                 wrt=f'states:{state_name_wrt}', return_format='array', use_abs_names=False)

                self._f_x[idxs, idxs_wrt] = px_px.ravel()

        return self._f_x, self._f_t

    def _update_derivs_fwd(self, pt_pt, pt_ph, px_pt, px_px, px_ph):
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
        self.dx_dtd[...] = px_px @ self.dx_dtd + \
                           px_pt @ self.dt_dtd + \
                           px_ph @ self.dh_dtd

        self.dx_dt0[...] = px_px @ self.dx_dt0 + \
                           px_pt @ self.dt_dt0 + \
                           px_ph @ self.dh_dt0

        self.dx_dx0[...] = px_px @ self.dx_dx0

        self.dt_dtd[...] = pt_pt @ self.dt_dtd + \
                           pt_ph @ self.dh_dtd

        self.dt_dt0[...] = pt_pt @ self.dt_dt0

    def _update_derivs_rev(self, pt_pt, pt_ph, px_pt, px_px, px_ph):
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
        self.dx_dtd_bar = self.dx_dtd_bar @ px_px.T + \
                          self.dt_dtd_bar @ px_pt.T + \
                          self.dh_dtd_bar @ px_ph.T

        self.dx_dt0_bar = self.dx_dt0_bar @ px_px.T + \
                          self.dt_dt0_bar @ px_pt.T + \
                          self.dh_dt0_bar @ px_ph.T

        self.dx_dx0_bar = self.dx_dx0_bar @ px_px.T

        self.dt_dtd_bar = self.dt_dtd_bar @ pt_pt.T + \
                          self.dh_dtd_bar @ pt_ph.T

        self.dt_dt0_bar = self.dt_dt0_bar @ pt_pt.T

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
        N = self.options['num_steps']

        if N > 0:
            h = inputs['t_duration'] / N
        else:
            h = 0

        # Initialize states
        x = self._x
        for state_name, options in self.state_options.items():
            x[self.state_idxs[state_name]] = inputs[f'state_initial_value:{state_name}'].ravel()

        # Initialize time
        t = inputs['t_initial']

        if derivs:
            # Initialize the total derivatives
            self._initialize_derivs()
            # From the time update equation, the partial of the new time wrt the previous time and
            # the partial wrt the stepsize are both [1].
            pt_pt = np.ones((1, 1), dtype=complex)
            pt_ph = np.ones((1, 1), dtype=complex)

        if state_stack is not None:
            state_stack.clear()
            state_stack.append(x)
        if time_stack is not None:
            time_stack.clear()
            time_stack.append(t)

        for i in range(N):
            # Compute the state rates
            f = self.eval_f(x, t)

            if derivs:
                # Compute the state rate derivatives
                f_x, f_t = self.eval_f_derivs(x, t)

                # The partials of the state update equation for Euler's method.
                px_px = self.I_x + h * f_x
                px_pt = h * f_t
                px_ph = f

                # Accumulate the totals
                if self.mode == 'fwd':
                    self._update_derivs_fwd(pt_pt, pt_ph, px_pt, px_px, px_ph)

            t = t + h
            x = x + h * f

            # The ODE is not evaluated at the final time, so don't append its value here.
            if i < N - 1:
                if state_stack:
                    state_stack.append(x)
                if time_stack:
                    time_stack.append(t)

        # Unpack the final values
        if outputs:
            outputs['t_final'] = t

            for state_name in self.state_options:
                of = f'state_final_value:{state_name}'
                outputs[of] = x[self.state_idxs[state_name]].reshape(options['shape'])

        if derivs:
            derivs['t_final', 't_initial'] = self.dt_dt0
            derivs['t_final', 't_duration'] = self.dt_dtd

            for state_name in self.state_options:
                of = f'state_final_value:{state_name}'

                # Unpack the derivatives
                of_rows = self.state_idxs[state_name]

                derivs[of, 't_initial'] = self.dx_dt0[of_rows]
                derivs[of, 't_duration'] = self.dx_dtd[of_rows]

                for wrt_state_name in self.state_options:
                    wrt = f'state_initial_value:{wrt_state_name}'
                    wrt_cols = self.state_idxs[wrt_state_name]
                    derivs[of, wrt] = self.dx_dx0[of_rows, wrt_cols]

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
        N = self.options['num_steps']

        if N > 0:
            h = inputs['t_duration'] / N
        else:
            h = 0

        self._initialize_derivs()

        # From the time update equation, the partial of the new time wrt the previous time and
        # the partial wrt the stepsize are both [1].
        pt_pt = np.ones((1, 1), dtype=complex)
        pt_ph = np.ones((1, 1), dtype=complex)

        for i in range(N):
            # Extract the saved integrated quantities
            t = time_stack.pop()
            x = state_stack.pop()

            # Compute the state rates
            f = self.eval_f(x, t)

            # Compute the state rate derivatives
            f_x, f_t = self.eval_f_derivs(x, t)

            # The partials of the state update equation for Euler's method.
            px_px = self.I_x + h * f_x
            px_pt = h * f_t
            px_ph = f

            self._update_derivs_rev(pt_pt, pt_ph, px_pt, px_px, px_ph)

        derivs['t_final', 't_initial'] = self.dt_dt0_bar.T
        derivs['t_final', 't_duration'] = self.dt_dtd_bar.T

        for state_name in self.state_options:
            of = f'state_final_value:{state_name}'

            # Unpack the derivatives
            of_rows = self.state_idxs[state_name]

            derivs[of, 't_initial'] = self.dx_dt0_bar.T[of_rows]
            derivs[of, 't_duration'] = self.dx_dtd_bar.T[of_rows]

            for wrt_state_name in self.state_options:
                wrt = f'state_initial_value:{wrt_state_name}'
                wrt_cols = self.state_idxs[wrt_state_name]
                derivs[of, wrt] = self.dx_dx0_bar[wrt_cols, of_rows]

    def compute(self, inputs, outputs):
        self._propagate(inputs, outputs)

    def compute_partials(self, inputs, partials):
        # note that typically you would only have to define partials for one direction,
        # either fwd OR rev, not both.
        if self.mode == 'fwd':
            self._propagate(inputs, outputs=False, derivs=partials)
        else:
            time_stack = collections.deque()
            state_stack = collections.deque()

            self._propagate(inputs, outputs=False, derivs=None, time_stack=time_stack, state_stack=state_stack)
            self._backpropagate_derivs(inputs, partials, time_stack=time_stack, state_stack=state_stack)
