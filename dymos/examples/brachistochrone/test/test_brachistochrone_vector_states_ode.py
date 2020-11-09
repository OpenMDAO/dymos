import unittest

import numpy as np

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

from dymos.examples.brachistochrone.brachistochrone_vector_states_ode import \
    BrachistochroneVectorStatesODE


class TestBrachistochroneVectorStatesODE(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 5

        p = cls.p = om.Problem(model=om.Group())

        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('v', val=np.ones((nn,)), units='m/s')
        ivc.add_output('g', val=np.zeros((nn,)), units='m/s**2')
        ivc.add_output('theta', val=np.zeros((nn)), units='rad')

        p.model.add_subsystem('eom', BrachistochroneVectorStatesODE(num_nodes=nn),
                              promotes_inputs=['*'], promotes_outputs=['*'])
        p.setup(check=True, force_alloc_complex=True)

        p['v'] = np.random.rand(nn)
        p['g'] = np.random.rand(nn)
        p['theta'] = np.random.rand(nn)

        p.run_model()

    def test_results(self):
        pass

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)
