import numpy as np

import openmdao.api as om
import dymos as dm
# from ...phase.options import TimeOptionsDictionary


def f_ode(t, x, u=None, d=None):
    # dy_dt = np.zeros(1)
    dy_dt = x - t**2 + 1
    # dy_dt[1] = 0.5 * x[1] - t + 1
    return dy_dt


class TestODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('t', val=np.ones(nn), units='s')
        self.add_input('x0', val=np.ones(nn), units='m')
        self.add_input('x1', val=np.ones(nn), units='m')

        self.add_output('x0_dot', val=np.ones(nn), units='m/s')
        self.add_output('x1_dot', val=np.ones(nn), units='m/s')

        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='x0_dot', wrt='x0', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x0_dot', wrt='t', rows=ar, cols=ar)

        self.declare_partials(of='x1_dot', wrt='x1', rows=ar, cols=ar, val=0.5)
        self.declare_partials(of='x1_dot', wrt='t', rows=ar, cols=ar, val=-1.0)

    def compute(self, inputs, outputs):
        x0 = inputs['x0']
        x1 = inputs['x1']
        t = inputs['t']

        outputs['x0_dot'] = x0 - t**2 + 1
        outputs['x1_dot'] = 0.5 * x1 - t + 1

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials['x0_dot', 't'] = -2 * inputs['t']


def lotka_volterra(t, x, u=None, d=None):
    dx_dt = np.empty(np.asarray(x).shape)

    a = alpha = 1
    b = beta = 0.5
    g = gamma = 0.5
    s = sigma = 2

    x0, x1 = x

    print(x0)
    print(x1)

    dx_dt[0] = a * x0 - b * x0 * x1
    dx_dt[1] = g * x0 * x1 - s * x1

    print(dx_dt)

    return dx_dt

euler = {'a': np.array([[0]]),
         'c': np.array([0]),
         'b': np.array([1])}

rk4 = {'a': np.array([[0, 0, 0, 0],
                      [1/2, 0, 0, 0],
                      [0, 1/2, 0, 0],
                      [0, 0, 1, 0]]),
       'c': np.array([0, 1/2, 1/2, 1]),
       'b': np.array([1/6, 1/3, 1/3, 1/6])}

ralston = {'a': np.array([[0, 0], [2/3, 0]]),
           'c': np.array([0, 2/3]),
           'b': np.array([1/4, 3/4]),
           'order': 2}

rkf = {'a': np.array([[0,  0,  0,  0,  0],
                      [1/4, 0, 0, 0, 0],
                      [3/32, 9/32, 0, 0, 0],
                      [1932/2197, -7200/2197, 7296/2197, 0, 0],
                      [439/216, -8, 3680/513, -845/4104, 0],
                      [-8/27, 2, -3544/2565, 1859/4104, -11/40]]),
       'c': np.array([0, 1/4, 3/8, 12/13, 1, 1/2]),
       'b': np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55]),
       'b_star': np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])}

rkck = {'a': np.array([[0,  0,  0,  0,  0],
                      [1/5, 0, 0, 0, 0],
                      [3/40, 9/40, 0, 0, 0],
                      [3/10, -9/10, 6/5, 0, 0],
                      [-11/54, 5/2, -70/27, 35/27, 0],
                      [1631/55296, 175/512, 575/13828, 44275/110592, 253/4096]]),
       'c': np.array([0, 1/5, 3/10, 3/5, 1, 7/8]),
       'b': np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]),
       'b_star': np.array([37/378, 0, 250/621, 125/594, 512/1771, 0])}

dopri = {'a': np.array([[0,  0,  0,  0,  0, 0],
                        [1/5, 0, 0, 0, 0, 0],
                        [3/40, 9/40, 0, 0, 0, 0],
                        [44/45, -56/15, 32/9, 0, 0, 0],
                        [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0],
                        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0],
                        [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]]),
         'c': np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1]),
         'b': np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]),
         'b_star': np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])}


class RKIntegrationComp(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prob = None
        self._state_rates = {}
        self._state_rates_derivs = {}

    def initialize(self):
        self.options.declare('ode_class', desc='The system providing the ODE for the integration component.')
        self.options.declare('ode_init_kwargs', types=dict, default={})

        self.options.declare('time_options', types=dm.phase.options.TimeOptionsDictionary,
                             desc='Time options for the phase')

        self.options.declare('state_options', types=dict,
                             desc='Dictionary of state names/options for the segments parent Phase')

        self.options.declare('control_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of control names/options for the segments parent Phase.')

        self.options.declare('polynomial_control_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of polynomial control names/options for the segments '
                                  'parent Phase.')

        self.options.declare('parameter_options', default=None, types=dict, allow_none=True,
                             desc='Dictionary of parameter names/options for the segments '
                                  'parent Phase.')

        self.options.declare('tableau', default=rk4, types=dict,
                             desc='Dictionary containing parameters for the Runge Kutta tableau.')

        # self.options.declare('h', default=0.5, types=float,
        #                      desc='stepsize for the integration.  this is only the initial '
        #                           'stepsize if a variable step method is used.')

        self.options.declare('num_steps', default=10, types=int, desc='number of fixed steps')


    def _setup_subprob(self):
        self.prob = p = om.Problem(comm=self.comm)

        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        p.model.add_subsystem('ode', ode_class(num_nodes=1, **ode_init_kwargs))

        ivc.add_output('time', units=time_options['units'])
        #TODO: Hook these up
        # ivc.add_output('time_phase', units=time_options['units'])
        # ivc.add_output('t_initial', units=time_options['units'])
        # ivc.add_output('t_duration', units=time_options['units'])
        for tgt in self.options['time_options']['targets']:
            p.model.connect(f'time', f'ode.{tgt}')

        for state_name, options in self.options['state_options'].items():
            ivc.add_output(f'states:{state_name}',
                           shape=options['shape'],
                           units=options['units'],
                           desc=f'value of state {state_name}')

            for tgt in options['targets']:
                p.model.connect(f'states:{state_name}', f'ode.{tgt}')

        p.setup()
        p.final_setup()

    def setup(self):
        self._setup_subprob()

        self.add_input('t_initial', val=0.0)
        self.add_input('t_duration', val=2.0)

        if self.options['time_options']['t_initial_targets']:
            self.add_input('t_initial', val=0.0, desc='time at the start of the integration interval')
        if self.options['time_options']['t_duration_targets']:
            self.add_input('t_duration', val=1.0, desc='duration of the integration interval')
        for state_name, options in self.options['state_options'].items():
            self.add_input(f'state_initial_value:{state_name}',
                           shape=options['shape'],
                           desc=f'initial value of state {state_name}')
            self.add_output(f'state_final_value:{state_name}',
                            shape=options['shape'],
                            desc=f'final value of state {state_name}')

        self.declare_partials('*', '*')

    def _eval_f(self, stage_inputs, state_rates, rate_partials=None):
        """
        Evaluate the ODE sub-problem with the given inputs.

        Parameters
        ----------
        stage_inputs : dict of {str : np.array}
            The inputs to be evaluated at the given point.
        state_rates : dict of {str : np.array}
            The time derivative of each state variable for the given stage inputs.
        rate_partials : dict of { (str, str) : np.array}
            The partial of the time derivatives wrt each input variable.

        Returns
        -------
        dict of (str: np.array)
            The state rates from the given evaluation.
        """
        p = self.prob

        p.set_val('time', stage_inputs['time'])

        # for tgt in self.options['time_options']['t_initial_targets']:
        #     p[f'ode.{tgt}'] = stage_inputs['t_initial']
        #
        # for tgt in self.options['time_options']['t_duration_targets']:
        #     p[f'ode.{tgt}'] = stage_inputs['t_duration']

        wrt = ['time'] + \
              [f'states:{state_name}' for state_name in self.options['state_options'].keys()]

        for name, options in self.options['state_options'].items():
            p.set_val(f'states:{name}', stage_inputs[f'states:{name}'])

        p.run_model()

        for name, options in self.options['state_options'].items():
            rate_src = options['rate_source']
            state_rates[name][...] = p[f'ode.{rate_src}']
            derivs = p.compute_totals(of=f'ode.{rate_src}', wrt=wrt)
            for var in wrt:
                rate_partials[name, var] = derivs[f'ode.{rate_src}', var]

    def _rk_step(self, t0, y0, h, inputs):
        """
        Evaluate a single step of the given RK method.

        Parameters
        ----------
        t0
        y0
        h
        inputs

        Returns
        -------

        """
        #todo: Remove instantiation of numpy arrays from this method and use pre-created ones instead.

        a = self.options['tableau']['a']
        b = self.options['tableau']['b']
        b_star = self.options['tableau'].get('b_star', None)
        c = self.options['tableau']['c']
        num_stages = len(c)

        k = {}
        stage_inputs = {'time': np.zeros(1)}
        step_outputs = {}
        step_errors = {}
        step_derivs = {}

        for state_name, options in self.options['state_options'].items():
            k[state_name] = np.zeros((num_stages,) + options['shape'])
            size = np.prod(options['shape'])
            # y0[state_name] = inputs[f'state_initial_value:{state_name}']
            step_errors[state_name] = np.zeros((num_stages,) + options['shape'])
            stage_inputs[f'states:{state_name}'] = np.zeros(shape=options['shape'])
            self._state_rates[state_name] = np.zeros(shape=options['shape'])
            for wrt_state, wrt_state_options in self.options['state_options'].items():
                wrt_size = np.prod(wrt_state_options['shape'])
                self._state_rates_derivs[state_name, wrt_state] = {i: np.zeros((size, wrt_size)) for i in range(num_stages)}

        # For each stage in the RK method used, set the stage inputs and compute the k
        # coefficients of the RK method.
        for i in range(num_stages):
            t_s = t0 + c[i] * h
            stage_inputs['time'][:] = t_s

            for state_name in self.options['state_options']:
                stage_inputs[f'states:{state_name}'][...] = y0[state_name]
                for j in range(i):
                    stage_inputs[f'states:{state_name}'][...] += a[i, j] * k[state_name][j, ...]

            self._eval_f(stage_inputs, self._state_rates, self._state_rates_derivs)

            for state_name in self.options['state_options']:
                k[state_name][i, ...] = h * self._state_rates[state_name]

        # Apply the step for the RK method
        for state_name, options in self.options['state_options'].items():
            step_outputs[state_name] = y0[state_name] + np.tensordot(b, k[state_name], axes=(0, 0))

            # if b_star is not None:
            #     yf_star = y0[state_name] + np.tensordot(b, k[state_name], axes=(0, 0))
            #     step_errors[state_name][...] = step_outputs[state_name] - yf_star
            # else:
            #     step_errors[state_name][...] = 0.0

        return step_outputs, step_errors, step_derivs

    def compute(self, inputs, outputs):

        t0_interval = inputs['t_initial']
        t_step = t0_interval
        y_step = {}
        h = inputs['t_duration'] / self.options['num_steps']

        # print(inputs)

        for state_name in self.options['state_options']:
            y_step[state_name] = inputs[f'state_initial_value:{state_name}']

        for i in range(self.options['num_steps']):
            yf, yerr, yderivs = self._rk_step(t_step, y_step, h, inputs)
            t_step += h
            y_step = yf

        for state_name in self.options['state_options']:
            outputs[f'state_final_value:{state_name}'] = yf[state_name]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        t0_interval = inputs['t_initial']
        t_step = t0_interval
        y_step = {}
        h = inputs['t_duration'] / self.options['num_steps']

        derivs_per_step = []

        for state_name in self.options['state_options']:
            y_step[state_name] = inputs[f'state_initial_value:{state_name}']

        for i in range(self.options['num_steps']):
            step_y, step_err, step_derivs = self._rk_step(t_step, y_step, h, inputs)
            t_step += h
            y_step = step_y
            derivs_per_step.append(step_derivs)

        for state_name, options in self.options['state_options'].items():
            state_size = np.prod(options['shape'])
            for input_name in inputs:
                input_size = np.prod(inputs[input_name].shape)
                partials[f'state_final_value:{state_name}', input_name] = np.zeros((state_size, input_size))


def validate(h=0.2, num_steps=10):
    t = 0
    y = 0.5
    print(f'Step 0: t = {t:6.2f}, y= {y:12.8f}')
    for i in range(num_steps):
        k1 = h * f_ode(t, y)
        k2 = h * f_ode(t + h / 2, y + k1 / 2)
        k3 = h * f_ode(t + h / 2, y + k2 / 2)
        k4 = h * f_ode(t + h, y + k3)
        y = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t = t + h
        print(f'Step {i:d}: t = {t:6.2f}, y= {y:12.8f}')


if __name__ == '__main__':
    h = 0.5
    y = [0.5, 0.5]
    t = 0
    tf = 2.0
    yscal = np.ones_like(y)

    f = f_ode

    # yf = rk_integrate(f_ode, y0=y, t0=t, tf=tf, h=h, tableau=rk4)

    # y = [2, 2]
    # f = lotka_volterra

    # print(tf, yf)

    p = om.Problem()

    time_options = dm.phase.options.TimeOptionsDictionary()

    time_options['targets'] = ['t']

    state_options = {'x0': dm.phase.options.StateOptionsDictionary(),
                     'x1': dm.phase.options.StateOptionsDictionary()}

    state_options['x0']['name'] = 'x0'
    state_options['x0']['targets'] = ['x0']
    state_options['x0']['rate_source'] = 'x0_dot'
    state_options['x0']['shape'] = (1,)
    state_options['x0']['units'] = 'm'

    state_options['x1']['name'] = 'x1'
    state_options['x1']['targets'] = ['x1']
    state_options['x1']['rate_source'] = 'x1_dot'
    state_options['x1']['shape'] = (1,)
    state_options['x1']['units'] = 'm'

    p.model.add_subsystem('rk', RKIntegrationComp(ode_class=TestODE, time_options=time_options,
                                                  state_options=state_options, num_steps=100,
                                                  tableau=euler))

    p.model.add_design_var('rk.state_initial_value:x0')
    p.model.add_design_var('rk.t_initial')
    p.model.add_design_var('rk.t_duration')

    p.model.add_objective('rk.state_final_value:x0')

    p.setup(force_alloc_complex=True)

    p.set_val('rk.state_initial_value:x0', 0.5)
    p.set_val('rk.state_initial_value:x1', 0.5)
    p.set_val('rk.t_initial', 0.0)
    p.set_val('rk.t_duration', 2.0)

    p.run_model()

    p.model.list_outputs()

    p.compute_totals(of=['rk.state_final_value:x0'], wrt=['rk.state_initial_value:x0'])

    p.check_totals(method='fd', form='central')
    # validate()
