from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_rel_error

from dymos.examples.min_time_climb.aero.kappa_comp import KappaComp


class TestKappaComp(unittest.TestCase):

    def test_value(self):
        n = 500

        p = om.Problem(model=om.Group())

        ivc = om.IndepVarComp()

        ivc.add_output(name='mach', units=None, val=np.zeros(n))

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem(name='kappa_comp',
                              subsys=KappaComp(num_nodes=n))

        p.model.connect('mach', 'kappa_comp.mach')

        p.setup()

        p['mach'] = np.linspace(0, 1.8, n)
        p.run_model()

        M = p.get_val('mach')
        kappa = p.get_val('kappa_comp.kappa')

        idxs_0 = np.where(M <= 1.15)[0]
        idxs_1 = np.where(M > 1.15)[0]

        kappa_analtic_0 = 0.54 + 0.15 * (1.0 + np.tanh((M[idxs_0] - 0.9)/0.06))
        kappa_analtic_1 = 0.54 + 0.15 * (1.0 + np.tanh(0.25/0.06)) + 0.14 * (M[idxs_1] - 1.15)

        assert_rel_error(self, kappa[idxs_0], kappa_analtic_0)
        assert_rel_error(self, kappa[idxs_1], kappa_analtic_1)

    def test_partials(self):

        n = 10

        p = om.Problem(model=om.Group())

        ivc = om.IndepVarComp()

        ivc.add_output(name='mach', units=None, val=np.zeros(n))

        p.model.add_subsystem(name='ivc', subsys=ivc, promotes_outputs=['*'])

        p.model.add_subsystem(name='kappa_comp',
                              subsys=KappaComp(num_nodes=n))

        p.model.connect('mach', 'kappa_comp.mach')

        p.setup()

        p['mach'] = np.linspace(0, 1.8, n)

        p.run_model()

        cpd = p.check_partials(compact_print=False, out_stream=None)
        assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-4)


if __name__ == '__main__':
    unittest.main()
