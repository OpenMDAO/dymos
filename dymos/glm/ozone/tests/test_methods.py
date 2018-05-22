import numpy as np
import unittest
from six import iteritems

import matplotlib
matplotlib.use('Agg')

from parameterized import parameterized
from itertools import product

from dymos.glm.ozone.ode_integrator import ODEIntegrator
from dymos.glm.ozone.methods_list import method_classes

from dymos import Phase, declare_state, declare_time
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp, ScipyOptimizeDriver


@declare_time(targets='t')
@declare_state('y', 'dy_dt', targets='y')
class SimpleODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        self.options.declare('a', default=1., types=(int, float))

    def setup(self):
        num = self.options['num_nodes']

        self.add_input('y', shape=(num, 1))
        self.add_input('t', shape=num)
        self.add_output('dy_dt', shape=(num, 1))

        self.declare_partials('dy_dt', 'y', val=self.options['a'] * np.eye(num))

        self.eye = np.eye(num)

    def compute(self, inputs, outputs):
        outputs['dy_dt'] = self.options['a'] * inputs['y']

    def get_test_parameters(self):
        t0 = 0.
        t1 = 1.
        initial_conditions = {'y': 1.}
        return initial_conditions, t0, t1

    def get_exact_solution(self, initial_conditions, t0, t):
        a = self.options['a']
        y0 = initial_conditions['y']
        C = y0 / np.exp(a * t0)
        return {'y': C * np.exp(a * t)}


class OzoneODETestCase(unittest.TestCase):

    @parameterized.expand(product(
        ['optimizer-based', 'solver-based', 'time-marching'],
        method_classes.keys(),
    ))
    def test(self, glm_formulation, glm_integrator):
        tf = 1e-2

        p = Problem(model=Group())
        phase = Phase('glm',
                      ode_class=SimpleODE,
                      num_segments=10,
                      formulation=glm_formulation,
                      method_name=glm_integrator,
                      compressed=False)
        p.model.add_subsystem('phase0', phase)

        p.model.add_subsystem('dummy_comp', IndepVarComp('dummy_var'))
        p.model.add_objective('dummy_comp.dummy_var')

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(tf, tf))
        phase.set_state_options('y', fix_initial=True)

        if glm_formulation == 'optimizer-based':
            p.driver = ScipyOptimizeDriver()
            p.driver.options['optimizer'] = 'SLSQP'
            p.driver.options['tol'] = 1e-9
            p.driver.options['disp'] = True

        p.setup()
        p.final_setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = tf
        phase.set_values('y', [1, 1])

        if glm_formulation == 'optimizer-based':
            p.run_driver()
        else:
            p.run_model()

        np.testing.assert_almost_equal(phase.get_values('time')[-1, 0], tf, decimal=5)
        np.testing.assert_almost_equal(phase.get_values('y')[-1, 0], np.exp(tf), decimal=5)
