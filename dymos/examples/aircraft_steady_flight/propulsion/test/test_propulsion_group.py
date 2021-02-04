import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.testing_utils import assert_check_partials
from dymos.examples.aircraft_steady_flight.propulsion.propulsion_group import PropulsionGroup


@use_tempdirs
class TestPropulsionComp(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.n = 10

        self.p = om.Problem(model=om.Group())

        ivc = self.p.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('alt', val=np.zeros(self.n), units='m', desc='altitude above MSL')

        ivc.add_output('pres', val=101325.0*np.ones(self.n), units='Pa', desc='atmospheric_pressure')

        ivc.add_output('CT', val=np.linspace(0, 1.0E4, self.n), units=None,
                       desc='coefficient of thrust')

        self.p.model.add_subsystem('propulsion', PropulsionGroup(num_nodes=self.n))

        self.p.model.connect('alt', 'propulsion.alt')
        self.p.model.connect('pres', 'propulsion.pres')
        self.p.model.connect('CT', 'propulsion.CT')

        self.p.setup(force_alloc_complex=True)

        self.p.run_model()

    def test_results(self):

        assert_near_equal(self.p['propulsion.tsfc'],
                          2 * 8.951e-6 * 9.80665 * np.ones(self.n))

    def test_partials(self):
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)
