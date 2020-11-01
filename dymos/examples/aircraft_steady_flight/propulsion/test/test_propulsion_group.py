import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from dymos.utils.testing_utils import assert_check_partials

from dymos.examples.aircraft_steady_flight.propulsion.propulsion_group import PropulsionGroup


class TestPropulsionComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.n = 10

        cls.p = om.Problem(model=om.Group())

        ivc = cls.p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('alt', val=np.zeros(cls.n), units='m', desc='altitude above MSL')

        ivc.add_output('pres', val=101325.0*np.ones(cls.n), units='Pa', desc='atmospheric_pressure')

        ivc.add_output('CT', val=np.linspace(0, 1.0E4, cls.n), units=None,
                       desc='coefficient of thrust')

        cls.p.model.add_subsystem('propulsion', PropulsionGroup(num_nodes=cls.n))

        cls.p.model.connect('alt', 'propulsion.alt')
        cls.p.model.connect('pres', 'propulsion.pres')
        cls.p.model.connect('CT', 'propulsion.CT')

        cls.p.setup(force_alloc_complex=True)

        cls.p.run_model()

    def test_results(self):

        assert_near_equal(self.p['propulsion.tsfc'],
                          2 * 8.951e-6 * 9.80665 * np.ones(self.n))

    def test_partials(self):
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
