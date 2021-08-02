import collections

import numpy as np
import openmdao.api as om
import dymos as dm

class SimpleODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s**2')
        self.add_input('t', shape=(nn,), units='s')

        self.add_output('x_dot', shape=(nn,), units='s')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        x = inputs['x']
        t = inputs['t']
        outputs['x_dot'] = x - t**2 + 1

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['x_dot', 't'] = -2*t


class ODEEvaluationGroup(om.Group):
    """
    A special group whose purpose is to evaluate the ODE and return the computed
    state weights.
    """

    def __init__(self, ode_class, time_options, state_options, control_options,
                 polynomial_control_options, parameter_options, ode_init_kwargs=None):
        super().__init__()

        # Get the state vector.  This isn't necessarily ordered
        # so just pick the default ordering and go with it.
        self.state_options = state_options
        self.time_options = time_options
        self.control_options = control_options
        self.polynomial_control_options = polynomial_control_options
        self.parameter_options = parameter_options
        self.control_interpolants = {}
        self.polynomial_control_interpolants = {}
        self.ode_class = ode_class
        self.ode_init_kwargs = {} if ode_init_kwargs is None else ode_init_kwargs

    def setup(self):
        if self.control_options:
            # Add control interpolant
            raise NotImplementedError('dynamic controls not yet implemented')
        if self.polynomial_control_options:
            # Add polynomial control interpolant
            raise NotImplementedError('polynomial controls not yet implemented')

        self.add_subsystem('ode', self.ode_class(num_nodes=1, **self.ode_init_kwargs))

        self.add_subsystem('state_rate_collector',
                           StateRateCollectorComp(state_options=self.state_options,
                                                  time_units=self.time_options['units']))

    def configure(self):
        self._configure_time()
        self._configure_states()

    def _configure_time(self):
        targets = self.time_options['targets']
        time_phase_targets = self.time_options['time_phase_targets']
        t_initial_targets = self.time_options['t_initial_targets']
        t_duration_targets = self.time_options['t_duration_targets']
        units = self.time_options['units']

        for tgts, var in [(targets, 'time'), (time_phase_targets, 'time_phase'),
                          (t_initial_targets, 't_initial'), (t_duration_targets, 't_duration')]:
            for t in tgts:
                self.promotes('ode', inputs=[(t, var)])
                print(f'promoted {t} to {var}')
            if tgts:
                self.set_input_defaults(name=var,
                                        val=np.ones((1,)),
                                        units=units)

    def _configure_states(self):
        for name, options in self.state_options.items():
            shape = options['shape']
            targets = options['targets']
            rate_src = options['rate_source']
            rate_path, rate_io = self._get_rate_source_path(name)

            # Promote targets from the ODE
            for tgt in targets:
                self.promotes('ode', inputs=[(tgt, f'states:{name}')])
            if targets:
                self.set_input_defaults(name=f'states:{name}',
                                        val=np.ones(shape),
                                        units=options['units'])

            # If the state rate source is an output, connect it, otherwise
            # promote it to the appropriate name
            if rate_io == 'output':
                self.connect(rate_path, f'state_rate_collector.state_rates_in:{name}_rate')
            else:
                self.promotes('state_rate_collector',
                              inputs=[(f'state_rates_in:{name}_rate', rate_path)])

            self.add_design_var(f'states:{name}')
            self.add_constraint(f'state_rate_collector.state_rates:{name}_rate')

    def _get_rate_source_path(self, state_var):
        """
        Get path of the rate source variable so that we can connect it to the
        outputs when we're done.

        Parameters
        ----------
        state_var : str
            The name of the state variable whose path is desired.

        Returns
        -------
        path : str
            The path to the rate source of the state variable.
        io : str
            A string indicating whether the variable in the path is an 'input'
            or an 'output'.
        """
        var = self.state_options[state_var]['rate_source']

        if var == 'time':
            rate_path = 'time'
            io = 'input'
        elif var == 'time_phase':
            rate_path = 'time_phase'
            io = 'input'
        elif self.state_options is not None and var in self.state_options:
            rate_path = f'states:{var}'
            io = 'input'
        elif self.control_options is not None and var in self.control_options:
            rate_path = f'controls:{var}'
            io = 'output'
        elif self.polynomial_control_options is not None and var in self.polynomial_control_options:
            rate_path = f'polynomial_controls:{var}'
            io = 'output'
        elif self.parameter_options is not None and var in self.parameter_options:
            rate_path = f'parameters:{var}'
            io = 'input'
        elif var.endswith('_rate') and self.control_options is not None and \
                var[:-5] in self.control_options:
            rate_path = f'control_rates:{var}'
            io = 'output'
        elif var.endswith('_rate2') and self.control_options is not None and \
                var[:-6] in self.control_options:
            rate_path = f'control_rates:{var}'
            io = 'output'
        elif var.endswith('_rate') and self.polynomial_control_options is not None and \
                var[:-5] in self.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
            io = 'output'
        elif var.endswith('_rate2') and self.polynomial_control_options is not None and \
                var[:-6] in self.polynomial_control_options:
            rate_path = f'polynomial_control_rates:{var}'
            io = 'output'
        else:
            rate_path = f'ode.{var}'
            io = 'output'
        return rate_path, io


from dymos.utils.misc import get_rate_units
from dymos.options import options as dymos_options


class StateRateCollectorComp(om.ExplicitComponent):
    """
    Class definition for StateRateCollectorComp.

    Collects the state rates and outputs them in the units specified in the state options.
    For explicit integration this is necessary when the output providing the state rate has
    different units than those defined in the state_options/time_options.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of options for the ODE state variables.')
        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')

        # Save the names of the dynamic controls/parameters
        self._input_names = {}
        self._output_names = {}

        self._no_check_partials = not dymos_options['include_check_partials']

    def setup(self):
        """
        Create inputs/outputs on this component.
        """
        state_options = self.options['state_options']
        time_units = self.options['time_units']

        for name, options in state_options.items():
            self._input_names[name] = f'state_rates_in:{name}_rate'
            self._output_names[name] = f'state_rates:{name}_rate'
            shape = options['shape']
            size = np.prod(shape, dtype=int)
            units = options['units']

            rate_units = get_rate_units(units, time_units)

            self.add_input(self._input_names[name], shape=shape, units=rate_units)
            self.add_output(self._output_names[name], shape=shape, units=rate_units)

            ar = np.arange(size, dtype=int)
            self.declare_partials(of=self._output_names[name],
                                  wrt=self._input_names[name],
                                  rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        state_options = self.options['state_options']

        for name, options in state_options.items():
            outputs[self._output_names[name]] = inputs[self._input_names[name]]


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
        # Accumulate the totals
        # Compute this with the initial values of dx_dx and dt_dtd before they're updated
        self.dx_dtd = px_px @ self.dx_dtd + \
                      px_pt @ self.dt_dtd + \
                      px_ph @ self.dh_dtd

        self.dx_dt0 = px_px @ self.dx_dt0 + \
                      px_pt @ self.dt_dt0 + \
                      px_ph @ self.dh_dt0

        self.dx_dx0 = px_px @ self.dx_dx0

        self.dt_dtd = pt_pt @ self.dt_dtd + \
                      pt_ph @ self.dh_dtd

        self.dt_dt0 = pt_pt @ self.dt_dt0

    def _update_derivs_rev(self, pt_pt, pt_ph, px_pt, px_px, px_ph):
        # Accumulate the totals in reverse
        # Compute this with the initial values of dx_dx and dt_dtd before they're updated
        self.dx_dtd_bar = self.dx_dtd_bar @ px_px.T  + \
                          self.dt_dtd_bar @ px_pt.T + \
                          self.dh_dtd_bar @ px_ph.T

        self.dx_dt0_bar = self.dx_dt0_bar @ px_px.T + \
                          self.dt_dt0_bar @ px_pt.T + \
                          self.dh_dt0_bar @ px_ph.T

        self.dx_dx0_bar = self.dx_dx0_bar @ px_px.T

        self.dt_dtd_bar = self.dt_dtd_bar @ pt_pt.T + \
                          self.dh_dtd_bar @ pt_ph.T

        self.dt_dt0_bar = self.dt_dt0.T @ pt_pt.T

    def _propagate(self, inputs, outputs, derivs=None, time_stack=None, state_stack=None):
        """
        Propagate the states from t_initial to t_initial + t_duration, optionally computing
        the derivatives along the way and caching the current time and state values.

        Parameters
        ----------
        inputs
        outputs
        derivs
        time_stack
        state_stack

        Returns
        -------

        """
        N = self.options['num_steps']

        if N > 0:
            h = inputs['t_duration'] / N
        else:
            h = 0

        # Initialize the total derivatives
        self._initialize_derivs()

        # Initialize states
        x = self._x
        for state_name, options in self.state_options.items():
            x[self.state_idxs[state_name]] = inputs[f'state_initial_value:{state_name}'].ravel()

        # Initialize time
        t = inputs['t_initial']

        if derivs:
            # From the time update equation, the partial of the new time wrt the previous time and
            # the partial wrt the stepsize are both [1].
            pt_pt = np.ones((1, 1), dtype=complex)
            pt_ph = np.ones((1, 1), dtype=complex)

        if state_stack is not None:
            state_stack.clear()
            state_stack.append(x)
        if time_stack is not None:
            time_stack.clear()
            time_stack.append(x)

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
        inputs
        derivs
        time_stack
        state_stack

        Returns
        -------

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

            self.dx_dtd_bar = self.dx_dtd_bar @ px_px.T + \
                              self.dt_dtd_bar @ px_pt.T + \
                              self.dh_dtd_bar @ px_ph.T

            print(self.dx_dt0_bar)

            self.dx_dt0_bar = self.dx_dt0_bar @ px_px.T + \
                              self.dt_dt0_bar @ px_pt.T + \
                              self.dh_dt0_bar @ px_ph.T

            self.dx_dx0_bar = self.dx_dx0_bar @ px_px.T

            self.dt_dtd_bar = self.dt_dtd_bar @ pt_pt.T + \
                              self.dh_dtd_bar @ pt_ph.T

            self.dt_dt0_bar = self.dt_dt0_bar @ pt_pt.T

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

def test_eval():
    ode_class = SimpleODE
    time_options = dm.phase.options.TimeOptionsDictionary()

    time_options['targets'] = 't'
    time_options['units'] = 's'

    state_options = {'x': dm.phase.options.StateOptionsDictionary()}

    state_options['x']['shape'] = (1,)
    state_options['x']['units'] = 's**2'
    state_options['x']['rate_source'] = 'x_dot'
    state_options['x']['targets'] = ['x']

    control_options = {}
    polynomial_control_options = {}
    parameter_options = {}

    p = om.Problem()
    p.model.add_subsystem('ode_eval', ODEEvaluationGroup(ode_class, time_options, state_options,
                                                         control_options,
                                                         polynomial_control_options,
                                                         parameter_options, ode_init_kwargs=None))
    p.setup()

    p.set_val('ode_eval.states:x', [1.25])
    p.set_val('ode_eval.time', [2.2])

    p.run_model()

    p.model.list_inputs()
    p.model.list_outputs()
    # p.check_partials()



def test_fwd():
    ode_class = SimpleODE
    time_options = dm.phase.options.TimeOptionsDictionary()

    time_options['targets'] = 't'
    time_options['units'] = 's'

    state_options = {'x': dm.phase.options.StateOptionsDictionary()}

    state_options['x']['shape'] = (1,)
    state_options['x']['units'] = 's**2'
    state_options['x']['rate_source'] = 'x_dot'
    state_options['x']['targets'] = ['x']

    control_options = {}
    polynomial_control_options = {}
    parameter_options = {}

    p = om.Problem()

    p.model.add_subsystem('fixed_step_integrator', EulerIntegrationComp(ode_class, time_options, state_options,
                                                                        parameter_options, control_options,
                                                                        polynomial_control_options, mode='fwd',
                                                                        num_steps=100,
                                                                        ode_init_kwargs=None))
    p.setup(mode='fwd', force_alloc_complex=True)

    p.set_val('fixed_step_integrator.state_initial_value:x', 0.5)
    p.set_val('fixed_step_integrator.t_initial', 0.0)
    p.set_val('fixed_step_integrator.t_duration', 2.0)

    p.run_model()

    print('t_f', p.get_val('fixed_step_integrator.t_final'))
    print('x_f', p.get_val('fixed_step_integrator.state_final_value:x'))

    p.check_partials(method='fd', form='central')


def test_rev():
    ode_class = SimpleODE
    time_options = dm.phase.options.TimeOptionsDictionary()

    time_options['targets'] = 't'
    time_options['units'] = 's'

    state_options = {'x': dm.phase.options.StateOptionsDictionary()}

    state_options['x']['shape'] = (1,)
    state_options['x']['units'] = 's**2'
    state_options['x']['rate_source'] = 'x_dot'
    state_options['x']['targets'] = ['x']

    control_options = {}
    polynomial_control_options = {}
    parameter_options = {}

    p = om.Problem()

    p.model.add_subsystem('fixed_step_integrator', EulerIntegrationComp(ode_class, time_options, state_options,
                                                                        parameter_options, control_options,
                                                                        polynomial_control_options, mode='rev',
                                                                        num_steps=20,
                                                                        ode_init_kwargs=None))
    p.setup(mode='rev', force_alloc_complex=True)

    p.set_val('fixed_step_integrator.state_initial_value:x', 0.5)
    p.set_val('fixed_step_integrator.t_initial', 0.0)
    p.set_val('fixed_step_integrator.t_duration', 2.0)

    p.run_model()

    print('t_f', p.get_val('fixed_step_integrator.t_final'))
    print('x_f', p.get_val('fixed_step_integrator.state_final_value:x'))

    p.check_partials(method='fd', form='central')



if __name__ == '__main__':
    # test_eval()
    # test_fwd()
    test_rev()
