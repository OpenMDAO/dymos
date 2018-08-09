from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_check_partials

from dymos.examples.min_time_climb.aero.cla_comp import CLaComp

assert_almost_equal = np.testing.assert_almost_equal

import matplotlib
matplotlib.use('Agg')

SHOW_PLOTS = True


class TestCLaComp(unittest.TestCase):

    @unittest.skipIf(not SHOW_PLOTS, 'this test is for visual confirmation, requires plotting')
    def test_visual_inspection(self):
        n = 500

        p = Problem(model=Group())

        ivc = IndepVarComp()

        ivc.add_output(name='mach', units=None, val=np.zeros(n))

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem(name='cla_comp',
                                   subsys=CLaComp(num_nodes=n))

        p.model.connect('mach', 'cla_comp.mach')

        p.setup()

        p['mach'] = np.linspace(0, 1.8, n)
        p.run_model()

        import matplotlib.pyplot as plt
        plt.plot(p['mach'], p['cla_comp.CLa'], 'ro', ms=2)
        plt.show()

    def test_partials(self):

        n = 10

        p = Problem(model=Group())

        ivc = IndepVarComp()

        ivc.add_output(name='mach', units=None, val=np.zeros(n))

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem(name='cla_comp',
                                   subsys=CLaComp(num_nodes=n))

        p.model.connect('mach', 'cla_comp.mach')

        p.setup()

        p['mach'] = np.linspace(0, 1.8, n)

        p.run_model()

        cpd = p.check_partials(compact_print=False, out_stream=None)
        assert_check_partials(cpd, atol=1.0E-3, rtol=1.0E-4)


if __name__ == '__main__':
    unittest.main()
