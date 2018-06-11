from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from dymos.examples.min_time_climb.aero.cla_comp import CLaComp

assert_almost_equal = np.testing.assert_almost_equal

import matplotlib
matplotlib.use('Agg')

SHOW_PLOTS = False


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
        np.set_printoptions(linewidth=1024)
        cpd = p.check_partials(compact_print=False)
        for comp in cpd:
            for (var, wrt) in cpd[comp]:
                assert_almost_equal(cpd[comp][var, wrt]['rel error'], 0.0, decimal=4)


if __name__ == '__main__':
    unittest.main()
