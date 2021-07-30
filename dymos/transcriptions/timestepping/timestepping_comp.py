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
        self.ode_init_kwargs = {} if ode_init_kwargs is None else ode_init_kwargs

    def setup(self):
        if self.control_options:
            # Add control interpolant
            raise NotImplementedError('dynamic controls not yet implemented')
        if self.polynomial_control_options:
            # Add polynomial control interpolant
            raise NotImplementedError('polynomial controls not yet implemented')

        self.add_subsystem('ode', ode_class(num_nodes=1, **self.ode_init_kwargs))

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
            print(shape)
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
    def __init__(self, ode_class, ode_init_kwargs=None, time_options=None,
                 state_options=None, parameter_options=None, control_options=None,
                 polynomial_control_options=None, mode=None, **kwargs):
        super().__init__(**kwargs)
        self.ode_class = ode_class
        self.ode_init_kwargs = ode_init_kwargs
        self.time_options = time_options
        self.state_options = state_options
        self.parameter_options = parameter_options
        self.control_options = control_options
        self.polynomial_control_options = polynomial_control_options
        self.mode = mode
        self.prob = None

    def initialize(self):
        self.options.declare('num_steps', types=(int,), default=10)

    def _setup_subprob(self):
        self.prob = p = om.Problem(comm=self.comm)
        p.model.add_subsystem('ode_eval',
                              ODEEvaluationGroup(self.ode_class, self.time_options,
                                                 self.state_options, self.control_options,
                                                 self.polynomial_control_options,
                                                 self.parameter_options, ode_init_kwargs=None))
        p.setup()
        p.final_setup()

    def _setup_time(self):
        self.add_input('time', shape=(1,), units=self.time_options['units'])
        self.add_input('time_phase', shape=(1,), units=self.time_options['units'])
        self.add_input('t_initial', shape=(1,), units=self.time_options['units'])
        self.add_input('t_duration', shape=(1,), units=self.time_options['units'])

    def _setup_states(self):
        N = self.options['num_steps']

        self.state_rates = {}

        self.dx_dx0 = {}
        self.dx_dt0 = {}
        self.dx_dtd = {}

        self.dt_dt0 = np.zeros((1, 1), dtype=complex)
        self.dt_dtd = np.zeros((1, 1), dtype=complex)

        self.dh_dt0 = np.zeros((1, 1), dtype=complex) / N
        self.dh_dtd = np.ones((1, 1), dtype=complex) / N
        self.dt_dt = np.ones((1, 1), dtype=complex)
        self.dt_dh = np.ones((1, 1), dtype=complex)

        for state_name, options in self.state_options.items():
            self.add_input(f'state_initial_value:{state_name}',
                           shape=options['shape'],
                           desc=f'initial value of state {state_name}')
            self.add_output(f'state_final_value:{state_name}',
                            shape=options['shape'],
                            desc=f'final value of state {state_name}')

            self.state_rates[state_name] = np.zeros(options['shape'])

            x_size = np.prod(options['shape'], dtype=int)

            self.dx_dt0[state_name] = np.zeros((x_size, 1), dtype=complex)
            self.dx_dtd[state_name] = np.zeros((x_size, 1), dtype=complex)

            for state_name_wrt, options_wrt in self.state_options.items():
                x_size_wrt = np.prod(options_wrt['shape'], dtype=int)
                self.dx_dx0[state_name, state_name_wrt] = np.zeros((x_size, x_size_wrt))

    def setup(self):
        self._setup_subprob()
        self._setup_time()
        self._setup_states()

    def eval_ode(self, ode_inputs, ode_outputs):
        # transcribe inputs

        self.prob.run_model()

        # extract state rates

        return self.state_rates

    def eval_derivs(self, ode_inputs, ode_derivs):
        pass

    def _propagate(self, inputs, outputs, store_history=False):
        N = self.options['num_steps']

        t0 = inputs['t_initial']
        td = inputs['t_duration']

        p = self.prob

        h = td / N
        t = t0

        ode_inputs = {}
        ode_outputs = self.state_rates

        if N > 0:
            h = td / N
        else:
            h = 0

        # x = x0.copy()
        # x_size = np.prod(x.shape, dtype=int)
        # t = t0

        # Initialize derivatives
        for state_name, options in self.state_options.items():
            self.dx_dt0[state_name][...] = 0.0
            self.dx_dtd[state_name][...] = 0.0

            for wrt_state_name, wrt_options in self.state_options.items():
                # The sensitivity of a state wrt its own value starts as identity
                # The sensitivity of a state wrt other state values starts at 0
                self.dx_dx0[state_name, wrt_state_name] = 0.0
                if wrt_state_name == state_name:
                    np.fill_diagonal(self.dx_dx0, 1.0)

        self.dt_dtd[...] = 0.0  # np.zeros((1, 1), dtype=complex)
        self.dt_dt0[...] = 1.0  # np.ones((1, 1), dtype=complex)
        self.dh_dt0[...] = 0.0  # np.zeros((1, 1), dtype=complex) / N
        self.dh_dtd[...] = 1.0 / N  # np.ones((1, 1), dtype=complex) / N
        self.dt_dt[...] = 1.0  # np.ones((1, 1), dtype=complex)
        self.dt_dh[...] = 1.0  # np.ones((1, 1), dtype=complex)

        pt_pt = np.eye(1, dtype=complex)

        # I_x = np.eye(x_size, dtype=complex)
        # I_t = np.eye(1, dtype=complex)

        for i in range(N):
            ode_outputs = self.eval_ode(ode_inputs)
            ode_derivs = self.eval_ode_derivs(ode_inputs)
            # f = ode.eval(x, t)
            # f_x = ode.dx(x, t)
            # f_t = ode.dt(x, t)

            for state_name, options in self.state_options.items():
                px_px = I_x + h * f_x
                px_pt = h * f_t
                px_ph = f

                # Compute this with the initial values of dx_dx and dt_dtd before they're updated
                dx_dtd = px_px @ self.dx_dtd + \
                         px_pt @ self.dt_dtd + \
                         px_ph @ self.dh_dtd

                dx_dt0 = px_px @ self.dx_dt0 + \
                         px_pt @ self.dt_dt0 + \
                         px_ph @ self.dh_dt0

                dx_dx0 = px_px @ self.dx_dx0

                # State update
                ode_inputs[f'states:{state_name}'] = ode_inputs[f'states:{state_name}'] + \
                                                     h * ode_outputs[f'state_rates:{state_name}']
                #x = x + h * f

            dt_dtd = self.dt_dt @ self.dt_dtd + \
                     self.dt_dh @ self.dh_dtd

            dt_dt0 = pt_pt @ dt_dt0

            # Time update
            ode_inputs['time'] = ode_inputs['time] + h']

        return x, t, dx_dx0, dt_dtd, dx_dtd, dt_dt0, dx_dt0

    def compute(self, inputs, outputs):
        self._propagate(inputs, outputs, store_history=False)

        # p.set_val('ode_eval')
        # p['comp.x'] = inputs['x']
        # p['comp.inp'] = inputs['inp']
        # inp = inputs['inp']
        # for i in range(self.num_nodes):
        #     p['comp.inp'] = inp
        #     p.run_model()
        #     inp = p['comp.out']
        #
        # outputs['out'] = p['comp.out']

    def _compute_partials_fwd(self, inputs, partials):
        p = self.prob
        x = inputs['x']
        p['comp.x'] = x
        p['comp.inp'] = inputs['inp']

        seed = {'comp.x':np.zeros(x.size), 'comp.inp': np.zeros(1)}
        p.run_model()
        p.model._linearize(None)
        for rhsname in seed:
            for rhs_i in range(seed[rhsname].size):
                seed['comp.x'][:] = 0.0
                seed['comp.inp'][:] = 0.0
                seed[rhsname][rhs_i] = 1.0
                for i in range(self.num_nodes):
                    p.model._vectors['output']['linear'].set_val(0.0)
                    p.model._vectors['residual']['linear'].set_val(0.0)
                    jvp = p.compute_jacvec_product(of=['comp.out'], wrt=['comp.x','comp.inp'], mode='fwd', seed=seed)
                    seed['comp.inp'][:] = jvp['comp.out']

                if rhsname == 'comp.x':
                    partials[self.pathname + '.out', self.pathname +'.x'][0, rhs_i] = jvp[self.pathname + '.out']
                else:
                    partials[self.pathname + '.out', self.pathname + '.inp'][0, 0] = jvp[self.pathname + '.out']

    def _compute_partials_rev(self, inputs, partials):
        p = self.prob
        p['comp.x'] = inputs['x']
        p['comp.inp'] = inputs['inp']
        seed = {'comp.out': np.ones(1)}

        stack = []
        comp = p.model.comp
        comp._inputs['inp'] = inputs['inp']
        # store the inputs to each comp (the comp at each node point) by doing nonlinear solves
        # and storing what the inputs are for each node point.  We'll set these inputs back
        # later when we linearize about each node point.
        for i in range(self.num_nodes):
            stack.append(comp._inputs['inp'][0])
            comp._inputs['x'] = inputs['x']
            comp._solve_nonlinear()
            comp._inputs['inp'] = comp._outputs['out']

        for i in range(self.num_nodes):
            p.model._vectors['output']['linear'].set_val(0.0)
            p.model._vectors['residual']['linear'].set_val(0.0)
            comp._inputs['inp'] = stack.pop()
            comp._inputs['x'] = inputs['x']
            p.model._linearize(None)
            jvp = p.compute_jacvec_product(of=['comp.out'], wrt=['comp.x','comp.inp'], mode='rev', seed=seed)
            seed['comp.out'][:] = jvp['comp.inp']

            # all of the comp.x's are connected to the same indepvarcomp, so we have
            # to accumulate their contributions together
            partials[self.pathname + '.out', self.pathname + '.x'] += jvp['comp.x']

            # this one doesn't get accumulated because each comp.inp contributes to the
            # previous comp's .out (or to comp.inp in the case of the first comp) only.
            # Note that we have to handle this explicitly here because normally in OpenMDAO
            # we accumulate derivatives when we do reverse transfers.  We can't do that
            # here because we only have one instance of our component, so instead of
            # accumulating into separate 'comp.out' variables for each comp instance,
            # we would be accumulating into a single comp.out variable, which would make
            # our derivative too big.
            partials[self.pathname + '.out', self.pathname + '.inp'] = jvp['comp.inp']

    def compute_partials(self, inputs, partials):
        # note that typically you would only have to define partials for one direction,
        # either fwd OR rev, not both.
        if self.mode == 'fwd':
            self._compute_partials_fwd(inputs, partials)
        else:
            self._compute_partials_rev(inputs, partials)


if __name__ == '__main__':
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
