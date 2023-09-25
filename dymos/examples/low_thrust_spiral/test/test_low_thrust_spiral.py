import openmdao.api as om
import dymos as dm
import unittest
import numpy as np

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.general_utils import printoptions

from dymos.examples.low_thrust_spiral import LowThrustODE


@require_pyoptsparse(optimizer='SLSQP')
def low_thrust_spiral_direct_collocation(grid_type='lgl'):

    optimizer = 'SNOPT'
    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 1000
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-3
        p.driver.opt_settings['iSumm'] = 6

    t = dm.Birkhoff(grid=dm.BirkhoffGrid(num_segments=1, nodes_per_seg=200, grid_type=grid_type))

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=LowThrustODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=False, units='s', duration_bounds=(10, 1200),
                           duration_ref=1500)

    phase.add_state('r', fix_initial=True, fix_final=True, rate_source='vr')
    phase.add_state('theta', fix_initial=True, fix_final=False, rate_source='theta_dot', units='rad')
    phase.add_state('vr', fix_initial=True, fix_final=True, rate_source='vr_dot', units='1/s')
    phase.add_state('vt', fix_initial=True, fix_final=True, rate_source='vt_dot', units='1/s')

    phase.add_control('alpha', units='rad', lower=-np.pi, upper=np.pi, ref0=-np.pi, ref=np.pi)

    # Minimize the control effort
    phase.add_objective('time', loc='final')

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True, force_alloc_complex=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 500.0

    p['traj.phase0.initial_states:r'] = 1.0
    p['traj.phase0.initial_states:theta'] = 0.0
    p['traj.phase0.initial_states:vr'] = 0.0
    p['traj.phase0.initial_states:vt'] = 1.0
    p['traj.phase0.final_states:r'] = 6.0
    p['traj.phase0.final_states:vr'] = 0.0
    p['traj.phase0.final_states:vt'] = 1/np.sqrt(6)

    p['traj.phase0.states:r'] = phase.interp('r', [1.0, 3.0])
    p['traj.phase0.states:vr'] = phase.interp('vr', [0.0, 0.0])
    p['traj.phase0.states:vt'] = phase.interp('vt', [1.0, 1.0])
    p['traj.phase0.controls:alpha'] = phase.interp('alpha', [0.0, 0.0])

    return p


# @use_tempdirs
class TestLowThrustSpiral(unittest.TestCase):

    # @classmethod
    # def tearDownClass(cls):
    #     for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
    #         if os.path.exists(filename):
    #             os.remove(filename)

    @staticmethod
    def _assert_results(p, tol=1.0E-4):
        h = p.get_val('traj.phase0.timeseries.r')

        # assert_near_equal(h[-1], 18550.9, tolerance=tol)

    def test_low_thrust_spiral_lgl(self):
        p = low_thrust_spiral_direct_collocation(grid_type='lgl')
        dm.run_problem(p)
        theta = p.get_val('traj.phase0.timeseries.theta')
        r = p.get_val('traj.phase0.timeseries.r')

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(theta, r)
        ax.grid(True)
        plt.show()

        self._assert_results(p)

    def test_low_thrust_spiral_cgl(self):
        p = low_thrust_spiral_direct_collocation(grid_type='cgl')
        dm.run_problem(p)
        self._assert_results(p)

    def test_check_partials(self):
        p = low_thrust_spiral_direct_collocation(grid_type='cgl')
        p.run_model()
        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)


if __name__ == "__main__":

    unittest.main()


