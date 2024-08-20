import openmdao.api as om
import dymos as dm
import unittest
import os

try:
    import matplotlib  # noqa: F401
    SHOW_PLOTS = True
except ImportError:
    SHOW_PLOTS = False

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.goddard_rocket_problem import RocketODE


@require_pyoptsparse(optimizer='IPOPT')
def goddard_rocket_direct_collocation(grid_type='lgl'):

    optimizer = 'IPOPT'
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-3
        p.driver.opt_settings['iSumm'] = 6

    t = dm.Birkhoff(num_nodes=150, grid_type=grid_type)

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=RocketODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(20, 80), units='s',
                           duration_ref=100)

    phase.add_state('h', fix_initial=True, fix_final=False, rate_source='h_dot', units='ft', lower=0, ref=20000)
    phase.add_state('v', fix_initial=True, fix_final=False, rate_source='v_dot', units='ft/s', ref=1000, lower=0)
    phase.add_state('m', fix_initial=True, fix_final=True, rate_source='m_dot', units='slug',
                    lower=1e-3, ref=3, upper=3.0)

    phase.add_control('T', units='lbf', lower=0, upper=200, ref=200)

    # Minimize the control effort
    phase.add_objective('h', loc='final', scaler=-1)

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=True)

    phase.set_time_val(initial=0.0, duration=40.0)
    phase.set_state_val('h', [0.0, 10000.0])
    phase.set_state_val('v', [0.0, 0.0])
    phase.set_state_val('m', [3.0, 1.0])
    phase.set_control_val('T', [0.0, 0.0])

    return p


@use_tempdirs
class TestGoddardRocketProblem(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @staticmethod
    def _assert_results(p, tol=1.0E-4):
        h = p.get_val('traj.phase0.timeseries.h')
        assert_near_equal(h[-1], 18745.0, tolerance=tol)

    def test_goddard_rocket_problem_lgl(self):
        p = goddard_rocket_direct_collocation(grid_type='lgl')
        dm.run_problem(p)
        if SHOW_PLOTS:
            t = p.get_val('traj.phase0.timeseries.time')
            h = p.get_val('traj.phase0.timeseries.h')
            v = p.get_val('traj.phase0.timeseries.v')
            m = p.get_val('traj.phase0.timeseries.m')
            T = p.get_val('traj.phase0.timeseries.T')

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(t, h)
            axs[0, 0].set_xlabel('time (s)')
            axs[0, 0].set_ylabel('altitude (ft)')
            axs[0, 1].plot(t, v)
            axs[0, 1].set_xlabel('time (s)')
            axs[0, 1].set_ylabel('velocity (ft/s)')
            axs[1, 0].plot(t, m)
            axs[1, 0].set_xlabel('time (s)')
            axs[1, 0].set_ylabel('mass (slug)')
            axs[1, 1].plot(t, T)
            axs[1, 1].set_xlabel('time (s)')
            axs[1, 1].set_ylabel('thrust (lb)')
            plt.show()

        self._assert_results(p)

    def test_goddard_rocket_problem_cgl(self):
        p = goddard_rocket_direct_collocation(grid_type='cgl')
        dm.run_problem(p)
        if SHOW_PLOTS:
            t = p.get_val('traj.phase0.timeseries.time')
            h = p.get_val('traj.phase0.timeseries.h')
            v = p.get_val('traj.phase0.timeseries.v')
            m = p.get_val('traj.phase0.timeseries.m')
            T = p.get_val('traj.phase0.timeseries.T')

            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].plot(t, h)
            axs[0, 1].plot(t, v)
            axs[1, 0].plot(t, m)
            axs[1, 1].plot(t, T)
            plt.show()
        self._assert_results(p)

    def test_check_partials(self):
        p = goddard_rocket_direct_collocation(grid_type='cgl')
        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)


if __name__ == "__main__":
    unittest.main()
