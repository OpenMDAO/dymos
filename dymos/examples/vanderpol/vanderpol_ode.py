import numpy as np
import openmdao.api as om
import time
from openmdao.utils.array_utils import evenly_distrib_idxs
from openmdao.utils.mpi import MPI


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

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # it's necessary to make this component matrix free because the inputs are non-distributed
        # and the outputs are distributed, and the framework doesn't know how to populate the full
        # nondistributed inputs on each rank in reverse mode.

        # FIXME: this delay will be applied on every call to compute_jacvec_product, which may be
        #        more often than was originally intended before this component was converted to
        #        matrix free (it was originally done in compute_partials).
        time.sleep(self.options['delay'])

        myslice = slice(self.start_idx, self.end_idx)

        locx0 = inputs['x0'][myslice]
        locx1 = inputs['x1'][myslice]
        locu = inputs['u'][myslice]

        locdx0 = d_inputs['x0'][myslice]
        locdx1 = d_inputs['x1'][myslice]
        locdu = d_inputs['u'][myslice]

        if mode == 'fwd':
            if 'x0dot' in d_outputs:
                if 'x0' in d_inputs:
                    d_outputs['x0dot'] += (1.0 - locx1**2) * locdx0
                if 'x1' in d_inputs:
                    d_outputs['x0dot'] += (-2.0 * locx1 * locx0 - 1.) * locdx1
                if 'u' in d_inputs:
                    d_outputs['x0dot'] += locdu
            if 'x1dot' in d_outputs:
                if 'x0' in d_inputs:
                    d_outputs['x1dot'] += locdx0
            if 'Jdot' in d_outputs:
                if 'x0' in d_inputs:
                    d_outputs['Jdot'] += 2.0 * locx0 * locdx0
                if 'x1' in d_inputs:
                    d_outputs['Jdot'] += 2.0 * locx1 * locdx1
                if 'u' in d_inputs:
                    d_outputs['Jdot'] += 2.0 * locu * locdu

        elif mode == 'rev':
            if 'x0dot' in d_outputs:
                if 'x0' in d_inputs:
                    d_inputs['x0'][myslice] += (1.0 - locx1**2) * d_outputs['x0dot']
                if 'x1' in d_inputs:
                    d_inputs['x1'][myslice] += (-2.0 * locx1 * locx0 - 1.) * d_outputs['x0dot']
                if 'u' in d_inputs:
                    d_inputs['u'][myslice] += d_outputs['x0dot']
            if 'x1dot' in d_outputs:
                if 'x0' in d_inputs:
                    d_inputs['x0'][myslice] += d_outputs['x1dot']
            if 'Jdot' in d_outputs:
                if 'x0' in d_inputs:
                    d_inputs['x0'][myslice] += 2.0 * locx0 * d_outputs['Jdot']
                if 'x1' in d_inputs:
                    d_inputs['x1'][myslice] += 2.0 * locx1 * d_outputs['Jdot']
                if 'u' in d_inputs:
                    d_inputs['u'][myslice] += 2.0 * locu * d_outputs['Jdot']

            if self.comm.size > 1 and self.options['distrib']:
                d_inputs['x0'] = self.comm.allreduce(d_inputs['x0'], op=MPI.SUM)
                d_inputs['x1'] = self.comm.allreduce(d_inputs['x1'], op=MPI.SUM)
                d_inputs['u'] = self.comm.allreduce(d_inputs['u'], op=MPI.SUM)
