from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import Problem, Group, IndepVarComp

from openmdoc.examples.min_time_climb.prop.bryson_thrust_comp import BrysonThrustComp

SHOW_PLOTS = True


class TestBrysonThrustComp(unittest.TestCase):

    @unittest.skipIf(not SHOW_PLOTS, 'this test is for visual confirmation, requires plotting')
    def test_other_values(self):
        n = 5

        p = Problem(model=Group())

        ivc = IndepVarComp()
        ivc.add_output(name='h', val=np.zeros(n), units='ft')
        ivc.add_output(name='mach', val=np.zeros(n), units=None)

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['h', 'mach'])
        p.model.add_subsystem(name='tcomp', subsys=BrysonThrustComp(num_nodes=n))

        p.model.connect('h', 'tcomp.h')
        p.model.connect('mach', 'tcomp.mach')

        p.setup()

        # Values of alt and mach at our test points
        h = [0, 100, 1000, 2000.0, 65000.0]
        M = [0.29386358, 0.29386358, .1,  0.2, 1.0]

        p['h'] = h
        p['mach'] = M

        p.run_model()

        # TODO: asserts


if __name__ == '__main__':
    unittest.main()
