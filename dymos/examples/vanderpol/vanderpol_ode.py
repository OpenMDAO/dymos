import numpy as np
import openmdao.api as om
import time


class vanderpol_ode(om.ExplicitComponent):
    """ODE for optimal control of a Van der Pol oscillator

    objective J:
        minimize integral of (x0**2 + x1**2 + u**2) for 0.0 <= t <= 15

    subject to:
        x0dot = (1 - x1^2) * x0 - x1 + u
        x1dot = x0
        -0.75 <= u <= 1.0

    initial conditions:
        x0(0) = 1.0   x1(0) = 1.0

    final conditions:
        x0(15) = 0.0  x1(15) = 0.0
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs: 2 states and a control
        self.add_input('x0', val=np.zeros(nn), desc='derivative of Output', units='V/s')
        self.add_input('x1', val=np.ones(nn), desc='Output', units='V')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.zeros(nn), desc='second derivative of Output', units='V/s**2')
        self.add_output('x1dot', val=np.zeros(nn), desc='derivative of Output', units='V/s')
        self.add_output('Jdot', val=np.zeros(nn), desc='derivative of objective', units='1.0/s')

        # partials
        r = c = np.arange(self.options['num_nodes'])

        self.declare_partials(of='x0dot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='u',   rows=r, cols=c, val=1.0)

        self.declare_partials(of='x1dot', wrt='x0',  rows=r, cols=c, val=1.0)
        self.declare_partials(of='x1dot', wrt='x1',  rows=r, cols=c, val=0.0)
        self.declare_partials(of='x1dot', wrt='u',   rows=r, cols=c, val=0.0)

        self.declare_partials(of='Jdot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='u',   rows=r, cols=c)

    def compute(self, inputs, outputs):
        x0 = inputs['x0']
        x1 = inputs['x1']
        u = inputs['u']

        outputs['x0dot'] = (1.0 - x1**2) * x0 - x1 + u
        outputs['x1dot'] = x0
        outputs['Jdot'] = x0**2 + x1**2 + u**2

    def compute_partials(self, inputs, jacobian):
        # partials declared with 'val' above do not need to be computed
        x0 = inputs['x0']
        x1 = inputs['x1']
        u = inputs['u']

        jacobian['x0dot', 'x0'] = 1.0 - x1 * x1
        jacobian['x0dot', 'x1'] = -2.0 * x1 * x0 - 1.0

        jacobian['Jdot', 'x0'] = 2.0 * x0
        jacobian['Jdot', 'x1'] = 2.0 * x1
        jacobian['Jdot', 'u'] = 2.0 * u


class vanderpol_ode_delay(vanderpol_ode):
    """slow version of vanderpol_ode for demonstrating distributed component calculations"""

    def __init__(self, *args, **kwargs):
        self.delay_time = 0.050
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs: 2 states and a control
        self.add_input('x0', val=np.zeros(nn), desc='derivative of Output', units='V/s')
        self.add_input('x1', val=np.ones(nn), desc='Output', units='V')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.zeros(nn), desc='second derivative of Output', units='V/s**2')
        self.add_output('x1dot', val=np.zeros(nn), desc='derivative of Output', units='V/s')
        self.add_output('Jdot', val=np.zeros(nn), desc='derivative of objective', units='1.0/s')

        # partials
        r = c = np.arange(self.options['num_nodes'])

        self.declare_partials(of='x0dot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='u',   rows=r, cols=c, val=1.0)

        self.declare_partials(of='x1dot', wrt='x0',  rows=r, cols=c, val=1.0)
        self.declare_partials(of='x1dot', wrt='x1',  rows=r, cols=c, val=0.0)
        self.declare_partials(of='x1dot', wrt='u',   rows=r, cols=c, val=0.0)

        self.declare_partials(of='Jdot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='u',   rows=r, cols=c)

    def compute(self, inputs, outputs):
        sizes = (len(inputs['x0']), len(inputs['x1']), len(inputs['u']))
        print('in vanderpol_ode_delay.compute', sizes)  # TODO delete print
        time.sleep(self.delay_time)  # make this method slow to test MPI

        x0 = inputs['x0']
        x1 = inputs['x1']
        u = inputs['u']

        outputs['x0dot'] = (1.0 - x1**2) * x0 - x1 + u
        outputs['x1dot'] = x0
        outputs['Jdot'] = x0**2 + x1**2 + u**2

    def compute_partials(self, inputs, jacobian):
        sizes = (len(inputs['x0']), len(inputs['x1']), len(inputs['u']))
        print('in vanderpol_ode_delay.compute_partials', sizes)  # TODO delete print
        time.sleep(self.delay_time)  # make this method slow to test MPI

        # partials declared with 'val' above do not need to be computed
        x0 = inputs['x0']
        x1 = inputs['x1']
        u = inputs['u']

        jacobian['x0dot', 'x0'] = 1.0 - x1 * x1
        jacobian['x0dot', 'x1'] = -2.0 * x1 * x0 - 1.0

        jacobian['Jdot', 'x0'] = 2.0 * x0
        jacobian['Jdot', 'x1'] = 2.0 * x1
        jacobian['Jdot', 'u'] = 2.0 * u
