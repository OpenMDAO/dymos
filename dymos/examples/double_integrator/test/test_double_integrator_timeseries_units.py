import itertools
import os
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE
from dymos.utils.testing_utils import assert_timeseries_near_equal


def double_integrator_direct_collocation(transcription=dm.GaussLobatto, compressed=True):

    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()

    t = transcription(num_segments=30, order=3)

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=DoubleIntegratorODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

    phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')
    phase.add_state('x', fix_initial=True, rate_source='v', units='ft', shape=(1, ))

    phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                      rate2_continuity=False, shape=(1, ), lower=-1.0, upper=1.0)

    phase.set_simulate_options(rtol=1.0E-9, atol=1.0E-9)

    # Maximize distance travelled in one second.
    phase.add_objective('x', loc='final', scaler=-1)

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True)

    p.set_val('traj.phase0.t_initial', 0.0)
    p.set_val('traj.phase0.t_duration', 1.0)

    p.set_val('traj.phase0.states:x', phase.interp('x', [0, 0.25]), units='m')
    p.set_val('traj.phase0.states:v', phase.interp('v', [0, 0]), units='m/s')
    p.set_val('traj.phase0.controls:u', phase.interp('u', [1, -1]), units='m/s**2')

    dm.run_problem(p, simulate=True, make_plots=True)

    return p


@use_tempdirs
class TestDoubleIntegratorExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def _assert_results(self, p, traj=True, tol=1.0E-4):
        if traj:
            x = p.get_val('traj.phase0.timeseries.states:x')
            v = p.get_val('traj.phase0.timeseries.states:v')
        else:
            x = p.get_val('phase0.timeseries.states:x')
            v = p.get_val('phase0.timeseries.states:v')

        assert_near_equal(x[0], 0.0, tolerance=tol)
        assert_near_equal(x[-1], 0.25, tolerance=tol)

        assert_near_equal(v[0], 0.0, tolerance=tol)
        assert_near_equal(v[-1], 0.0, tolerance=tol)

    def test_timeseries_units_gl(self):
        double_integrator_direct_collocation(dm.GaussLobatto, compressed=True)

        sol_case = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case = om.CaseReader('dymos_simulation.db').get_case('final')

        t_sol = sol_case.get_val('traj.phase0.timeseries.time')
        t_sim = sim_case.get_val('traj.phase0.timeseries.time')

        for var in ['states:x', 'states:v', 'state_rates:x', 'state_rates:v', 'controls:u']:
            sol = sol_case.get_val(f'traj.phase0.timeseries.{var}')
            sim = sim_case.get_val(f'traj.phase0.timeseries.{var}')
            assert_timeseries_near_equal(t_sol, sol, t_sim, sim, tolerance=1.0E-3)

    def test_timeseries_units_radau(self):
        double_integrator_direct_collocation(dm.Radau, compressed=True)

        sol_case = om.CaseReader('dymos_solution.db').get_case('final')
        sim_case = om.CaseReader('dymos_simulation.db').get_case('final')

        t_sol = sol_case.get_val('traj.phase0.timeseries.time')
        t_sim = sim_case.get_val('traj.phase0.timeseries.time')

        for var in ['states:x', 'states:v', 'state_rates:x']:
            sol = sol_case.get_val(f'traj.phase0.timeseries.{var}')
            sim = sim_case.get_val(f'traj.phase0.timeseries.{var}')
            assert_timeseries_near_equal(t_sol, sol, t_sim, sim, tolerance=1.0E-3)
