import unittest

import numpy as np
from numpy.testing import assert_almost_equal

import openmdao.api as om

from dymos.examples.min_time_climb.prop.max_thrust_comp import MaxThrustComp, THR_DATA, _LBF2N


class TestBrysonThrustComp(unittest.TestCase):

    def test_grid_values(self):
        n = 10

        p = om.Problem(model=om.Group())

        ivc = om.IndepVarComp()
        ivc.add_output(name='h', val=np.zeros(n), units='ft')
        ivc.add_output(name='mach', val=np.zeros(n), units=None)

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['h', 'mach'])
        p.model.add_subsystem(name='tcomp',
                              subsys=MaxThrustComp(vec_size=n, extrapolate=True, method='cubic'))

        p.model.connect('h', 'tcomp.h')
        p.model.connect('mach', 'tcomp.mach')

        p.setup()

        p['mach'] = THR_DATA['mach']
        for i in range(10):
            p['h'] = THR_DATA['h'][i] * np.ones(n)
            p.run_model()
            thrust_N = p['tcomp.max_thrust']
            assert_almost_equal(thrust_N, THR_DATA['thrust'][i, :])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
