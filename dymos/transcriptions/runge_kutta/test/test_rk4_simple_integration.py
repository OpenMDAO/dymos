import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.transcriptions.runge_kutta.test.rk_test_ode import TestODE, _test_ode_solution


@use_tempdirs
class TestRK4SimpleIntegration(unittest.TestCase):

    def test_simple_integration_forward(self):

        p = om.Problem(model=om.Group())
        phase = dm.Phase(ode_class=TestODE, transcription=dm.RungeKutta(num_segments=200, method='RK4'))
        p.model.add_subsystem('phase0', subsys=phase)

        phase.set_time_options(fix_initial=True, fix_duration=True, targets=['t'])
        phase.add_state('y', fix_initial=True, targets=['y'], rate_source='ydot', units='m')

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:y'] = 0.5

        p.run_model()

        expected = _test_ode_solution(p['phase0.ode.y'], p['phase0.ode.t'])
        assert_near_equal(p['phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_simple_integration_forward_connected_initial(self):

        p = om.Problem(model=om.Group())

        traj = p.model.add_subsystem('traj', subsys=dm.Trajectory())
        phase = dm.Phase(ode_class=TestODE, transcription=dm.RungeKutta(num_segments=200, method='RK4'))
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, fix_duration=True, targets=['t'])
        phase.add_state('y', fix_initial=False, connected_initial=True, targets=['y'], rate_source='ydot', units='m')

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 2.0

        # The initial guess of states at the segment boundaries
        p['traj.phase0.states:y'] = 0.0

        # The initial value of the states from which the integration proceeds
        p['traj.phase0.initial_states:y'] = 0.5

        p.run_model()

        expected = _test_ode_solution(p['traj.phase0.ode.y'], p['traj.phase0.ode.t'])
        assert_near_equal(p['traj.phase0.ode.y'], expected, tolerance=1.0E-3)

    def test_simple_integration_forward_connected_initial_fixed_initial(self):

        p = om.Problem(model=om.Group())

        phase = dm.Phase(ode_class=TestODE, transcription=dm.RungeKutta(num_segments=200, method='RK4'))
        p.model.add_subsystem('phase0', subsys=phase)

        phase.set_time_options(fix_initial=True, fix_duration=True, targets=['t'])
        phase.add_state('y', fix_initial=True, connected_initial=True, targets=['y'], rate_source='ydot', units='m')

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        with self.assertRaises(ValueError) as e:
            p.setup(check=True, force_alloc_complex=True)
        expected = "Cannot specify 'fix_initial=True' and specify 'connected_initial=True' for " \
                   "state y in phase phase0."
        self.assertEqual(str(e.exception), expected)

    def test_simple_integration_backward(self):

        p = om.Problem(model=om.Group())

        phase = dm.Phase(ode_class=TestODE, transcription=dm.RungeKutta(num_segments=200, method='RK4'))
        p.model.add_subsystem('phase0', subsys=phase)

        phase.set_time_options(fix_initial=True, fix_duration=True, targets=['t'])
        phase.add_state('y', fix_initial=True, targets=['y'], rate_source='ydot', units='m')

        phase.add_timeseries_output('ydot', output_name='state_rate:y', units='m/s')

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 2.0
        p['phase0.t_duration'] = -2.0

        p['phase0.states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.timeseries.time']))
        assert_near_equal(p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)

    def test_simple_integration_backward_connected_initial(self):

        p = om.Problem(model=om.Group())

        phase = dm.Phase(ode_class=TestODE, transcription=dm.RungeKutta(num_segments=200, method='RK4'))
        p.model.add_subsystem('phase0', subsys=phase)

        phase.set_time_options(fix_initial=True, fix_duration=True, targets=['t'])
        phase.add_state('y', connected_initial=True, targets=['y'], rate_source='ydot', units='m')

        phase.add_timeseries_output('ydot', output_name='state_rate:y')

        p.setup(check=True, force_alloc_complex=True)

        p['phase0.t_initial'] = 2.0
        p['phase0.t_duration'] = -2.0

        # The initial guess of state values at the segment boundaries
        p['phase0.states:y'] = 0

        # The initial value of the states from which the integration proceeds
        p['phase0.initial_states:y'] = 5.305471950534675

        p.run_model()

        expected = np.atleast_2d(_test_ode_solution(p['phase0.ode.y'], p['phase0.timeseries.time']))
        assert_near_equal(p['phase0.timeseries.states:y'], expected, tolerance=1.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
