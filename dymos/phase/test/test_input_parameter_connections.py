import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


n_traj = 4


class MyComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('n_traj', types=int)
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        n_traj = self.options['n_traj']
        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_input('alpha', shape=np.zeros((n_traj, 2)).shape, units='m')
        self.add_output('y', val=np.zeros(nn), units='1/s')

    def compute(self, inputs, outputs):
        pass

    def compute_partials(self, inputs, partials):
        pass


class MyODE(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('n_traj', default=2, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        n_traj = self.options['n_traj']

        self.add_subsystem(name='comp',
                           subsys=MyComp(num_nodes=nn,
                                         n_traj=n_traj))


@use_tempdirs
class TestStaticParameters(unittest.TestCase):

    def test_radau(self):

        p = om.Problem(model=om.Group())

        phase = dm.Phase(ode_class=MyODE,
                         ode_init_kwargs={'n_traj': n_traj},
                         transcription=dm.Radau(num_segments=25,
                                                order=3,
                                                compressed=True))

        phase.set_time_options(units='s', targets=['comp.time'])
        phase.add_state(name='F', rate_source='comp.y')
        phase.add_parameter('alpha', val=np.ones((n_traj, 2)), units='m',
                            targets='comp.alpha', dynamic=False)

        p.model.add_subsystem('phase0', phase)

        try:
            p.setup()
        except Exception as e:
            self.fail('Exception encountered in setup:\n' + str(e))

    def test_gauss_lobatto(self):

        p = om.Problem(model=om.Group())

        phase = dm.Phase(ode_class=MyODE,
                         ode_init_kwargs={'n_traj': n_traj},
                         transcription=dm.GaussLobatto(num_segments=25,
                                                       order=3,
                                                       compressed=True))

        phase.set_time_options(units='s', targets=['comp.time'])
        phase.add_state(name='F', rate_source='comp.y')
        phase.add_parameter('alpha', val=np.ones((n_traj, 2)), units='m',
                            targets='comp.alpha', dynamic=False)

        p.model.add_subsystem('phase0', phase)

        try:
            p.setup()
        except Exception as e:
            self.fail('Exception encountered in setup:\n' + str(e))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
