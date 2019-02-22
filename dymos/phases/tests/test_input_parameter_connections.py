import numpy as np

import unittest
from openmdao.api import ExplicitComponent, Group, Problem
from dymos import Phase, ODEOptions


n_traj = 4


class MyComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_traj', types=int)
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        n_traj = self.options['n_traj']
        self.add_input('time', val=np.zeros(nn))
        self.add_input('alpha', shape=np.zeros((n_traj, 2)).shape)

        self.add_output('y', val=np.zeros(nn))

    def compute(self, inputs, outputs):
        pass

    def compute_partials(self, inputs, partials):
        pass


class MyODE(Group):
    ode_options = ODEOptions()
    ode_options.declare_time(units='s', targets=['comp.time'])
    ode_options.declare_state(name='F', rate_source='comp.y')
    ode_options.declare_parameter(name='alpha', shape=(n_traj, 2), targets='comp.alpha',
                                  dynamic=False)

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('n_traj', default=2, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        n_traj = self.options['n_traj']

        self.add_subsystem(name='comp',
                           subsys=MyComp(num_nodes=nn,
                                         n_traj=n_traj))


class TestStaticInputParameters(unittest.TestCase):

    def test_radau(self):

        p = Problem(model=Group())

        phase = Phase(transcription='radau-ps',
                      ode_class=MyODE,
                      ode_init_kwargs={'n_traj': n_traj},
                      num_segments=25,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.add_input_parameter('alpha', val=np.ones((n_traj, 2)), units='m')

        try:
            p.setup()
        except:
            self.fail('Exception encountered in setup')

    def test_gauss_lobatto(self):

        p = Problem(model=Group())

        phase = Phase(transcription='gauss-lobatto',
                      ode_class=MyODE,
                      ode_init_kwargs={'n_traj': n_traj},
                      num_segments=25,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.add_input_parameter('alpha', val=np.ones((n_traj, 2)), units='m')

        try:
            p.setup()
        except:
            self.fail('Exception encountered in setup')
