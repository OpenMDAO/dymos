import numpy as np
import openmdao.api as om
import time
from openmdao.utils.array_utils import evenly_distrib_idxs


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
        self.add_input('x0', val=np.ones(nn), desc='derivative of Output', units='V/s')
        self.add_input('x1', val=np.ones(nn), desc='Output', units='V')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.ones(nn), desc='second derivative of Output', units='V/s**2')
        self.add_output('x1dot', val=np.ones(nn), desc='derivative of Output', units='V/s')
        self.add_output('Jdot', val=np.ones(nn), desc='derivative of objective', units='1.0/s')

        # partials
        r = c = np.arange(nn)

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


class vanderpol_ode_group(om.Group):
    """Group containing distributed vanderpol_ode pass through and calculation"""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='vanderpol_ode_passthrough',
                           subsys=vanderpol_ode_passthrough(num_nodes=nn),
                           promotes_inputs=['x0', 'x1', 'u'])

        self.add_subsystem(name='vanderpol_ode_delay',
                           subsys=vanderpol_ode_delay(num_nodes=nn),
                           promotes_outputs=['x0dot', 'x1dot', 'Jdot'])

        # connect collect_comp (pass through) output to distributed ODE input
        self.connect('vanderpol_ode_passthrough.x0pass', 'vanderpol_ode_delay.x0')
        self.connect('vanderpol_ode_passthrough.x1pass', 'vanderpol_ode_delay.x1')
        self.connect('vanderpol_ode_passthrough.upass', 'vanderpol_ode_delay.u')


class vanderpol_ode_passthrough(om.ExplicitComponent):
    """Pass through component that just copies control and state from input to output

    if you just use a plain old passthrough (non-distributed with a full sized input and output) component to connect
    to the distributed output, the framework will do the MPI allgathering for you"""

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']  # total number of inputs and outputs over all processes

        # inputs: 2 states and a control
        self.add_input('x0', val=np.ones(nn), desc='derivative of Output', units='V/s')
        self.add_input('x1', val=np.ones(nn), desc='Output', units='V')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)

        # outputs: same as inputs
        self.add_output('x0pass', val=np.ones(nn), desc='derivative of Output', units='V/s')
        self.add_output('x1pass', val=np.ones(nn), desc='Output', units='V')
        self.add_output('upass', val=np.ones(nn), desc='control', units=None)

        # partials
        row_col = np.arange(nn)
        self.declare_partials(of='x0pass', wrt='x0', rows=row_col, cols=row_col, val=1.0)
        self.declare_partials(of='x1pass', wrt='x1', rows=row_col, cols=row_col, val=1.0)
        self.declare_partials(of='upass', wrt='u', rows=row_col, cols=row_col, val=1.0)

    def compute(self, inputs, outputs):
        outputs['x0pass'] = inputs['x0']
        outputs['x1pass'] = inputs['x1']
        outputs['upass'] = inputs['u']


class vanderpol_ode_delay(om.ExplicitComponent):
    """intentionally slow version of vanderpol_ode for effects of demonstrating distributed component calculations

    MPI can run this component in multiple processes, distributing the calculation of derivatives.
    This code has a delay in it to simulate a longer computation. It should run faster with more processes.
    """

    def __init__(self, *args, **kwargs):
        self.delay_time = 0.005
        self.progress_prints = False
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options['distributed'] = True
        self.options.declare('size', types=int, default=1, desc="Size of input and output vectors.")

    def setup(self):
        nn = self.options['num_nodes']
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, nn)  # (#cpus, #inputs) -> (size array, offset array)
        start = offsets[rank]
        self.io_size = sizes[rank]  # number of inputs and outputs managed by this distributed process
        end = start + self.io_size

        if self.progress_prints:
            print('in vanderpol_ode_delay.setup', self.io_size, self.comm.rank)

        # inputs: 2 states and a control
        self.add_input('x0', val=np.ones(self.io_size), desc='derivative of Output', units='V/s',
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('x1', val=np.ones(self.io_size), desc='Output', units='V',
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('u', val=np.ones(self.io_size), desc='control', units=None,
                       src_indices=np.arange(start, end, dtype=int))

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.ones(self.io_size), desc='second derivative of Output', units='V/s**2')
        self.add_output('x1dot', val=np.ones(self.io_size), desc='derivative of Output', units='V/s')
        self.add_output('Jdot', val=np.ones(self.io_size), desc='derivative of objective', units='1.0/s')

        # partials
        r = c = np.arange(self.io_size)

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
        if self.progress_prints:
            sizes = (len(inputs['x0']), len(inputs['x1']), len(inputs['u']))
            print('in vanderpol_ode_delay.compute', sizes, self.comm.rank)

        time.sleep(self.delay_time * self.io_size)  # introduce slowness proportional to size of computation

        x0 = inputs['x0']
        x1 = inputs['x1']
        u = inputs['u']

        outputs['x0dot'] = (1.0 - x1**2) * x0 - x1 + u
        outputs['x1dot'] = x0
        outputs['Jdot'] = x0**2 + x1**2 + u**2

    def compute_partials(self, inputs, jacobian):
        if self.progress_prints:
            sizes = (len(inputs['x0']), len(inputs['x1']), len(inputs['u']))
            print('in vanderpol_ode_delay.compute_partials', sizes)

        time.sleep(self.delay_time * self.io_size)  # introduce slowness proportional to size of computation

        # partials declared with 'val' above do not need to be computed
        x0 = inputs['x0']
        x1 = inputs['x1']
        u = inputs['u']

        jacobian['x0dot', 'x0'] = 1.0 - x1 * x1
        jacobian['x0dot', 'x1'] = -2.0 * x1 * x0 - 1.0

        jacobian['Jdot', 'x0'] = 2.0 * x0
        jacobian['Jdot', 'x1'] = 2.0 * x1
        jacobian['Jdot', 'u'] = 2.0 * u
