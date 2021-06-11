import numpy as np
import openmdao.api as om
import time
from openmdao.utils.array_utils import evenly_distrib_idxs


class VanderpolODE(om.ExplicitComponent):
    """intentionally slow version of vanderpol_ode for effects of demonstrating distributed component calculations

    MPI can run this component in multiple processes, distributing the calculation of derivatives.
    This code has a delay in it to simulate a longer computation. It should run faster with more processes.
    """

    def __init__(self, *args, **kwargs):
        self.progress_prints = False
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('distrib', types=bool, default=False)
        self.options.declare('delay', types=(float,), default=0.0)

    def setup(self):
        nn = self.options['num_nodes']
        comm = self.comm
        rank = comm.rank

        sizes, offsets = evenly_distrib_idxs(comm.size, nn)  # (#cpus, #inputs) -> (size array, offset array)
        self.start_idx = offsets[rank]
        self.io_size = sizes[rank]  # number of inputs and outputs managed by this distributed process
        self.end_idx = self.start_idx + self.io_size

        # inputs: 2 states and a control
        self.add_input('x0', val=np.ones(nn), desc='derivative of Output', units='V/s')
        self.add_input('x1', val=np.ones(nn), desc='Output', units='V')
        self.add_input('u', val=np.ones(nn), desc='control', units=None)

        # outputs: derivative of states
        # the objective function will be treated as a state for computation, so its derivative is an output
        self.add_output('x0dot', val=np.ones(self.io_size), desc='second derivative of Output',
                        units='V/s**2', distributed=self.options['distrib'])
        self.add_output('x1dot', val=np.ones(self.io_size), desc='derivative of Output',
                        units='V/s', distributed=self.options['distrib'])
        self.add_output('Jdot', val=np.ones(self.io_size), desc='derivative of objective',
                        units='1.0/s', distributed=self.options['distrib'])

        # self.declare_coloring(method='cs')
        # # partials
        r = np.arange(self.io_size, dtype=int)
        c = r + self.start_idx

        self.declare_partials(of='x0dot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='x0dot', wrt='u',   rows=r, cols=c, val=1.0)

        self.declare_partials(of='x1dot', wrt='x0',  rows=r, cols=c, val=1.0)

        self.declare_partials(of='Jdot', wrt='x0',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='x1',  rows=r, cols=c)
        self.declare_partials(of='Jdot', wrt='u',   rows=r, cols=c)

    def compute(self, inputs, outputs):
        # introduce slowness proportional to size of computation
        time.sleep(self.options['delay'] * self.io_size)

        # The inputs contain the entire vector, be each rank will only operate on a portion of it.
        x0 = inputs['x0'][self.start_idx:self.end_idx]
        x1 = inputs['x1'][self.start_idx:self.end_idx]
        u = inputs['u'][self.start_idx:self.end_idx]

        outputs['x0dot'] = (1.0 - x1**2) * x0 - x1 + u
        outputs['x1dot'] = x0
        outputs['Jdot'] = x0**2 + x1**2 + u**2

    def compute_partials(self, inputs, jacobian):
        time.sleep(self.options['delay'] * self.io_size)

        x0 = inputs['x0'][self.start_idx:self.end_idx]
        x1 = inputs['x1'][self.start_idx:self.end_idx]
        u = inputs['u'][self.start_idx:self.end_idx]

        jacobian['x0dot', 'x0'] = 1.0 - x1 * x1
        jacobian['x0dot', 'x1'] = -2.0 * x1 * x0 - 1.0

        jacobian['Jdot', 'x0'] = 2.0 * x0
        jacobian['Jdot', 'x1'] = 2.0 * x1
        jacobian['Jdot', 'u'] = 2.0 * u
