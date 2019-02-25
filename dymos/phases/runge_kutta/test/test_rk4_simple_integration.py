from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, NonlinearBlockGS, NonlinearRunOnce
from openmdao.utils.assert_utils import assert_rel_error

from dymos import RungeKuttaPhase, declare_time, declare_state
from dymos.phases.runge_kutta.test.rk_test_ode import TestODE, _test_ode_solution
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestRK4SimpleIntegration(unittest.TestCase):

    def test_simple_integration_forward(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=200,
                            method='rk4',
                            ode_class=TestODE,
                            k_solver_options={'iprint': 2},
                            continuity_solver_options={'iprint': 2, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, fix_duration=True)
        phase.set_state_options('y', fix_initial=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        expected = _test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])
        assert_rel_error(self, p['phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_simple_integration_backward(self):

        p = Problem(model=Group())
        phase = p.model.add_subsystem(
            'phase0',
            RungeKuttaPhase(num_segments=4,
                            method='rk4',
                            ode_class=TestODE,
                            direction='backward',
                            k_solver_options={'iprint': 2},
                            continuity_solver_options={'iprint': 2, 'solve_subsystems': True}))

        phase.set_time_options(fix_initial=True, fix_duration=True)

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p.final_setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])).T
        assert_rel_error(self, p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)


if __name__ == '__main__':
    unittest.main()
