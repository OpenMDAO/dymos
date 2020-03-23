import numpy as np
import openmdao.api as om
import time
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI


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
    """Group containing distributed vanderpol_ode calculation and collection"""
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(name='vanderpol_ode_collect_comp',
                           subsys=vanderpol_ode_collect_comp(),
                           promotes_inputs=['x0', 'x1', 'u'])

        self.add_subsystem(name='vanderpol_ode_delay',
                           subsys=vanderpol_ode_delay(num_nodes=nn),
                           promotes_inputs=['x0', 'x1', 'u'])

        self.add_subsystem(name='vanderpol_ode_rate_collect',
                           subsys=vanderpol_ode_rate_collect(num_nodes=nn),
                           promotes_outputs=['x0dot', 'x1dot', 'Jdot'])

        self.connect('vanderpol_ode_collect_comp.x0pass', 'vanderpol_ode_delay.x0')
        self.connect('vanderpol_ode_collect_comp.x1pass', 'vanderpol_ode_delay.x1')
        self.connect('vanderpol_ode_collect_comp.upass', 'vanderpol_ode_delay.u')

        self.connect('vanderpol_ode_delay.x0dot', 'vanderpol_ode_rate_collect.partx0dot')
        self.connect('vanderpol_ode_delay.x1dot', 'vanderpol_ode_rate_collect.partx1dot')
        self.connect('vanderpol_ode_delay.Jdot', 'vanderpol_ode_rate_collect.partJdot')


class vanderpol_ode_collect_comp(om.ExplicitComponent):
    """Pass through component that just copies control and state from input to output so that Dymos can connect"""

    def initialize(self):
        self.options.declare('size', types=int, default=1, desc="Size of input and output vectors.")

    def setup(self):
        size = self.options['size']  # total number of inputs and outputs over all processes

        # inputs: 2 states and a control
        self.add_input('x0', val=np.ones(size), desc='derivative of Output', units='V/s')
        self.add_input('x1', val=np.ones(size), desc='Output', units='V')
        self.add_input('u', val=np.ones(size), desc='control', units=None)

        self.add_output('x0pass', val=np.ones(size), desc='derivative of Output', units='V/s')
        self.add_output('x1pass', val=np.ones(size), desc='Output', units='V')
        self.add_output('upass', val=np.ones(size), desc='control', units=None)

    def compute(self, inputs, outputs):
        outputs['x0pass'] = inputs['x0']
        outputs['x1pass'] = inputs['x1']
        outputs['upass'] = inputs['u']


class vanderpol_ode_delay(om.ExplicitComponent):
    """intentionally slow version of vanderpol_ode for effects of demonstrating distributed component calculations"""

    def __init__(self, *args, **kwargs):
        self.delay_time = 0.050
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options['distributed'] = True
        self.options.declare('size', types=int, default=1, desc="Size of input and output vectors.")

    def setup(self):
        nn = self.options['num_nodes']
        comm = self.comm
        rank = comm.rank
        size = self.options['size']  # total number of inputs and outputs over all processes

        sizes, offsets = evenly_distrib_idxs(comm.size, size)  # (#cpus, #inputs) -> (size array, offset array)
        start = offsets[rank]
        end = start + sizes[rank]
        # sizes[rank] is the number of inputs and output in this process

        # inputs: 2 states and a control
        self.add_input('x0', val=np.ones(sizes[rank]), desc='derivative of Output', units='V/s',
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('x1', val=np.ones(sizes[rank]), desc='Output', units='V',
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('u', val=np.ones(sizes[rank]), desc='control', units=None,
                       src_indices=np.arange(start, end, dtype=int))

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.ones(sizes[rank]), desc='second derivative of Output', units='V/s**2')
        self.add_output('x1dot', val=np.ones(sizes[rank]), desc='derivative of Output', units='V/s')
        self.add_output('Jdot', val=np.ones(sizes[rank]), desc='derivative of objective', units='1.0/s')

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
        sizes = (len(inputs['x0']), len(inputs['x1']), len(inputs['u']))
        print('in vanderpol_ode_delay.compute', sizes, self.comm.rank)  # TODO delete print
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


class vanderpol_ode_rate_collect(om.ExplicitComponent):
    """Collects and combines the parts of a distributed ODE computation input."""

    def initialize(self):
        self.options.declare('size', types=int, default=1, desc="Size of input and output vectors.")
        self.options.declare('num_nodes', types=int)

    def setup(self):
        comm = self.comm
        self.rank = comm.rank
        self.size = self.options['size']

        self.sizes, self.offsets = evenly_distrib_idxs(comm.size, self.size)
        start = self.offsets[self.rank]
        end = start + self.sizes[self.rank]

        # inputs are partial vectors of output to be combined
        self.add_input('partx0dot', val=np.ones(self.sizes[self.rank]), desc='second derivative of Output', units='V/s**2',
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('partx1dot', val=np.ones(self.sizes[self.rank]), desc='derivative of Output', units='V/s',
                       src_indices=np.arange(start, end, dtype=int))
        self.add_input('partJdot', val=np.ones(self.sizes[self.rank]), desc='derivative of objective', units='1.0/s',
                       src_indices=np.arange(start, end, dtype=int))

        # outputs: derivative of states, total size of combined outputs
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.ones(self.options['size']), desc='second derivative of Output', units='V/s**2')
        self.add_output('x1dot', val=np.ones(self.options['size']), desc='derivative of Output', units='V/s')
        self.add_output('Jdot', val=np.ones(self.options['size']), desc='derivative of objective', units='1.0/s')

    def compute(self, inputs, outputs):
        # gathers results from here with other MPI results
        self.comm.Gatherv(inputs['partx0dot'], [outputs['oux0dot'],
                          self.sizes, self.offsets, MPI.DOUBLE], root=self.rank)
        self.comm.Gatherv(inputs['partx1dot'], [outputs['oux1dot'],
                          self.sizes, self.offsets, MPI.DOUBLE], root=self.rank)
        self.comm.Gatherv(inputs['partJdot'], [outputs['Jdot'],
                          self.sizes, self.offsets, MPI.DOUBLE], root=self.rank)


        print(self.comm.rank, 'vanderpol_ode_rate_collect inputs', inputs['partx0dot'], 'outputs', outputs['oux0dot'])

# BUG: RuntimeError: Phase (traj.phases.phase0):
#      src_indices has been defined in both connect('control_values:u', 'rhs_disc.u') and add_input('rhs_disc.u', ...)

# TODO: this needs to be implemented as two components:
# 1 - a distributed computation component
#     (vanderpol_ode_delay)
# 2 - a distributed collect component that gets distributed inputs and merges them into a single array output
#     (vanderpol_ode_rate_collect)
# 3 - a Group that add the two new components, connects them, and can be used instead of the original ODE
#     (vanderpol_ode_group)
