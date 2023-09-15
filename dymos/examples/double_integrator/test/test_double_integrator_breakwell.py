import itertools
import os
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.double_integrator.double_integrator_breakwell_ode import DoubleIntegratorBreakwellODE


@require_pyoptsparse(optimizer='IPOPT')
def double_integrator_direct_collocation(transcription='gauss-lobatto', compressed=True, optimizer='IPOPT'):

    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()
    p.driver.options['optimizer'] = optimizer

    if optimizer == 'IPOPT':
        p.driver.opt_settings['max_iter'] = 500
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['tol'] = 1.0E-7

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=30, order=3, compressed=compressed)
    elif transcription == "radau-ps":
        t = dm.Radau(num_segments=30, order=3, compressed=compressed)
    elif transcription == 'birkhoff':
        grid = dm.BirkhoffGrid(num_segments=1, nodes_per_seg=101)
        t = dm.Birkhoff(grid=grid)
    else:
        raise ValueError('invalid transcription')

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=DoubleIntegratorBreakwellODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

    phase.add_state('v', fix_initial=True, rate_source='u', units='m/s')
    phase.add_state('x', fix_initial=True, rate_source='v', units='m', shape=(1, ))
    phase.add_state('J', fix_initial=True, rate_source='J_dot')

    phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                      rate2_continuity=False, shape=(1, ), lower=-1.0, upper=1.0)

    # Maximize distance travelled in one second.
    phase.add_objective('J', loc='final')

    phase.add_path_constraint('x', upper=0.1)
    phase.add_boundary_constraint('x', loc='final', equals=0.0)
    phase.add_boundary_constraint('v', loc='final', equals=-1.0)

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 1.0
    p['traj.phase0.states:x'] = phase.interp('x', [0, 0.25])
    p['traj.phase0.states:v'] = phase.interp('v', [1, 0])
    p['traj.phase0.controls:u'] = phase.interp('u', [1, -1])
    if transcription == 'birkhoff':
        p['traj.phase0.initial_states:x'] = 0.0
        p['traj.phase0.initial_states:v'] = 1.0

    p.run_driver()

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
            x = p.get_val('traj.phase0.timeseries.x')
            v = p.get_val('traj.phase0.timeseries.v')
        else:
            x = p.get_val('phase0.timeseries.x')
            v = p.get_val('phase0.timeseries.v')

        assert_near_equal(x[0], 0.0, tolerance=tol)
        assert_near_equal(x[-1], 0.25, tolerance=tol)

        assert_near_equal(v[0], 0.0, tolerance=tol)
        assert_near_equal(v[-1], 0.0, tolerance=tol)

    def test_ex_double_integrator_gl_compressed(self):
        p = double_integrator_direct_collocation('gauss-lobatto',
                                                 compressed=True)

        x = p.get_val('traj.phase0.timeseries.x')
        v = p.get_val('traj.phase0.timeseries.v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_gl_uncompressed(self):
        p = double_integrator_direct_collocation('gauss-lobatto',
                                                 compressed=False)

        x = p.get_val('traj.phase0.timeseries.x')
        v = p.get_val('traj.phase0.timeseries.v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_radau_compressed(self):
        p = double_integrator_direct_collocation('radau-ps',
                                                 compressed=True)

        x = p.get_val('traj.phase0.timeseries.x')
        v = p.get_val('traj.phase0.timeseries.v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_radau_uncompressed(self):
        p = double_integrator_direct_collocation('radau-ps',
                                                 compressed=False)

        x = p.get_val('traj.phase0.timeseries.x')
        v = p.get_val('traj.phase0.timeseries.v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_birkhoff(self):
        p = double_integrator_direct_collocation('birkhoff')

        t = p.get_val('traj.phase0.timeseries.time')
        x = p.get_val('traj.phase0.timeseries.x')
        v = p.get_val('traj.phase0.timeseries.v')
        u = p.get_val('traj.phase0.timeseries.u')

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(t, x)
        plt.plot(t, v)

        plt.figure()
        plt.plot(t, u)

        plt.show()


if __name__ == "__main__":

    unittest.main()
