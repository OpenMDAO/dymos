import unittest

import numpy as np

import openmdao.api as om
from dymos.utils.testing_utils import assert_check_partials

from dymos.examples.min_time_climb.aero.cd0_comp import CD0Comp

assert_almost_equal = np.testing.assert_almost_equal

import matplotlib
matplotlib.use('Agg')

SHOW_PLOTS = True


class TestCD0Comp(unittest.TestCase):
    @unittest.skipIf(not SHOW_PLOTS, 'this test is for visual confirmation, requires plotting')
    def test_visual_inspection(self):
        n = 500

        p = om.Problem(model=om.Group())

        ivc = om.IndepVarComp()

        ivc.add_output(name='mach', units=None, val=np.zeros(n))

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem(name='cd0_comp',
                              subsys=CD0Comp(num_nodes=n))

        p.model.connect('mach', 'cd0_comp.mach')

        p.setup()

        p['mach'] = np.linspace(0, 1.8, n)
        p.run_model()

        import matplotlib.pyplot as plt
        plt.plot(p['mach'], p['cd0_comp.CD0'], 'ro', ms=2)
        plt.show()

    def test_partials(self):

        n = 10

        p = om.Problem(model=om.Group())

        ivc = om.IndepVarComp()

        ivc.add_output(name='mach', units=None, val=np.zeros(n))

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem(name='cd0_comp',
                              subsys=CD0Comp(num_nodes=n))

        p.model.connect('mach', 'cd0_comp.mach')

        p.setup()

        p['mach'] = np.linspace(0, 1.14, n)

        p.run_model()

        cpd = p.check_partials(compact_print=False, out_stream=None)
        assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
