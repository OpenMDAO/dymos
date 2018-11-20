from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.examples.spacecraft_reorientation.spacecraft_reorientation_eom import \
    SpacecraftReorientationODE


class TestSpacecraftReorientationEOM(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        nn = 5

        p = cls.p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('I', val=np.ones(3), units='kg*m**2')
        ivc.add_output('q', val=np.zeros((nn, 4)), units=None)
        ivc.add_output('w', val=np.zeros((nn, 3)), units='rad/s')
        ivc.add_output('u', val=np.zeros((nn, 3)), units='N*m')

        p.model.add_subsystem('eom', SpacecraftReorientationODE(num_nodes=nn),
                              promotes_inputs=['*'], promotes_outputs=['*'])
        p.setup(check=True, force_alloc_complex=True)

        p['I'] = [5621., 4547., 2364.]
        p['q'] = np.random.rand(nn, 4)
        p['w'] = np.random.rand(nn, 3)

        p.run_model()

    def test_results(self):
        pass

    def test_partials(self):
        np.set_printoptions(linewidth=1024, edgeitems=1000)
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)
