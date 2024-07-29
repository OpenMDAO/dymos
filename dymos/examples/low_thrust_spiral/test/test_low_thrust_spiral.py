import openmdao.api as om
import dymos as dm
import unittest
import numpy as np

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from dymos.examples.low_thrust_spiral import LowThrustODE

show_plots = False


def low_thrust_spiral_direct_collocation(grid_type='lgl'):

    optimizer = 'SNOPT'
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 5.0E-4
        p.driver.opt_settings['iSumm'] = 6

    t = dm.Birkhoff(num_nodes=101, grid_type=grid_type)

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=LowThrustODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=False, units='s', duration_bounds=(100, 500))

    phase.add_state('r', fix_initial=True, fix_final=True, rate_source='vr', lower=0.5, upper=6.5,
                    defect_ref=1e2)
    phase.add_state('theta', fix_initial=True, fix_final=False, rate_source='theta_dot', units='rad', defect_ref=1e2)
    phase.add_state('vr', fix_initial=True, fix_final=True, rate_source='vr_dot', units='1/s', defect_ref=1e2)
    phase.add_state('vt', fix_initial=True, fix_final=True, rate_source='vt_dot', units='1/s', defect_ref=1e2)

    phase.add_control('alpha', units='rad')

    # Minimize the control effort
    phase.add_objective('time', loc='final')

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=True)

    phase.set_time_val(initial=0.0, duration=300.0)
    phase.set_state_val('r', [1.0, 6.0])
    phase.set_state_val('theta', [0.0, 0.0])
    phase.set_state_val('vr', [0.0, 0.0])
    phase.set_state_val('vt', [1.0, 1/np.sqrt(6)])
    phase.set_control_val('alpha', [0.0, 0.0])

    return p


@require_pyoptsparse(optimizer='SNOPT')
@use_tempdirs
class TestLowThrustSpiral(unittest.TestCase):

    @staticmethod
    def _assert_results(p, tol=0.05):
        t = p.get_val('traj.phase0.timeseries.time')
        assert_near_equal(t[-1], 228, tolerance=tol)
        return

    @unittest.skip('Long running test skipped on CI.')
    def test_low_thrust_spiral_lgl(self):
        p = low_thrust_spiral_direct_collocation(grid_type='lgl')
        dm.run_problem(p)
        theta = p.get_val('traj.phase0.timeseries.theta')
        r = p.get_val('traj.phase0.timeseries.r')

        if show_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(theta, r)
            ax.grid(True)
            plt.show()

        self._assert_results(p)

    @unittest.skip('Long running test skipped on CI.')
    def test_low_thrust_spiral_cgl(self):
        p = low_thrust_spiral_direct_collocation(grid_type='cgl')
        dm.run_problem(p)
        theta = p.get_val('traj.phase0.timeseries.theta')
        r = p.get_val('traj.phase0.timeseries.r')

        if show_plots:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(theta, r)
            ax.grid(True)
            plt.show()

        self._assert_results(p)

    def test_check_partials(self):
        p = low_thrust_spiral_direct_collocation(grid_type='cgl')
        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)


if __name__ == "__main__":
    unittest.main()
