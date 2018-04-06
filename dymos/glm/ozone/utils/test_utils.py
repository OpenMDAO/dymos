import numpy as np
import unittest
from six import iteritems

import matplotlib
matplotlib.use('Agg')

from parameterized import parameterized
from itertools import product

from dymos.glm.ozone.ode_integrator import ODEIntegrator

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_check_partials


class OzoneODETestCase(unittest.TestCase):

    @parameterized.expand(product(
        ['optimizer-based', 'solver-based', 'time-marching'],
        ['ForwardEuler', 'BackwardEuler'],
    ))
    def test_partials(self, glm_formulation, glm_integrator):
        p = Problem(model=Group())
        phase = Phase('glm',
                      ode_class=BrachistochroneODE,
                      num_segments=2,
                      formulation=glm_formulation,
                      method_name=glm_integrator)
        p.model.add_subsystem('phase0', phase)

        phase.add_control('theta', units='deg', dynamic=True,
                          rate_continuity=None, lower=0.01, upper=179.9)

        p.setup(force_alloc_complex=True)
        p.final_setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0
        phase.set_values('x', [0, 10])
        phase.set_values('y', [10, 5])
        phase.set_values('v', [0, 9.9])
        phase.set_values('theta', [5, 100.5])

        p.run_model()
        jac = p.check_partials()
        assert_check_partials(jac)
