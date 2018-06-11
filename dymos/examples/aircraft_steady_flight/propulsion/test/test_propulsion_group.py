from __future__ import print_function, division, absolute_import

import unittest

import numpy as np
from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.examples.aircraft_steady_flight.propulsion.propulsion_group import PropulsionGroup


class TestPropulsionComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 10

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('alt', val=np.zeros(cls.n), units='m', desc='altitude above MSL')

        ivc.add_output('pres', val=101325.0*np.ones(cls.n), units='Pa', desc='atmospheric_pressure')

        ivc.add_output('CT', val=np.linspace(0, 1.0E4, cls.n), units=None,
                       desc='coefficient of thrust')

        cls.p.model.add_subsystem('propulsion', PropulsionGroup(num_nodes=cls.n))

        cls.p.model.connect('alt', 'propulsion.alt')
        cls.p.model.connect('pres', 'propulsion.pres')
        cls.p.model.connect('CT', 'propulsion.CT')

        cls.p.setup(mode='fwd', force_alloc_complex=True)

        cls.p.run_model()

    @unittest.skip('temporaryily skipped until CT is worked out analytically')
    def test_results(self):

        assert_rel_error(self,
                         self.p['propulsion.tsfc'],
                         2 * 8.951e-6 * 9.80665 * np.ones(self.n))

        print(self.p['propulsion.tau'])

        # assert_rel_error(self,
        #                  self.p['propulsion.tau'],
        #                  np.linspace(0, 1.0, self.n))
        #
        # assert_rel_error(self,
        #                  self.p['propulsion.dXdt:mass_fuel'],
        #                  np.linspace(0, -1.02E6 * 2 * 8.951E-6, self.n))

    def test_partials(self):
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
