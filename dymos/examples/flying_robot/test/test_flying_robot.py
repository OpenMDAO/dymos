import itertools
import os
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.flying_robot.flying_robot_ode import FlyingRobotODE


def flying_robot_direct_collocation(transcription='gauss-lobatto', compressed=True):

    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    # p.driver.opt_settings['iSumm'] = 6
    p.driver.declare_coloring()

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=10, order=5, compressed=compressed)
    elif transcription == "radau-ps":
        t = dm.Radau(num_segments=10, order=5, compressed=compressed)
    else:
        raise ValueError('invalid transcription')

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=FlyingRobotODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=False, duration_bounds=(0.1, 1E4), units='s')

    phase.add_state('x', shape=(2,), fix_initial=True, fix_final=True, rate_source='v', units='m')
    phase.add_state('v', shape=(2,), fix_initial=True, fix_final=True, rate_source='u', units='m/s')
    phase.add_state('J', fix_initial=True, fix_final=False, rate_source='u_mag2', units='m**2/s**3')

    phase.add_control('u', units='m/s**2', shape=(2,), scaler=0.01, continuity=False, rate_continuity=False,
                      rate2_continuity=False, lower=-1, upper=1)

    # Minimize the control effort
    phase.add_objective('time', loc='final')

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 1.0

    p['traj.phase0.states:x'] = phase.interpolate(ys=[[0.0, 0.0], [-100.0, 100.0]], nodes='state_input')
    p['traj.phase0.states:v'] = phase.interpolate(ys=[[0.0, 0.0], [0.0, 0.0]], nodes='state_input')
    p['traj.phase0.controls:u'] = phase.interpolate(ys=[[1, 1], [-1, -1]], nodes='control_input')

    p.run_driver()

    return p


@use_tempdirs
class TestFlyingRobot(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def _assert_results(self, p, tol=1.0E-4):
        t = p.get_val('traj.phase0.timeseries.time')
        x = p.get_val('traj.phase0.timeseries.states:x')
        v = p.get_val('traj.phase0.timeseries.states:v')
        u = p.get_val('traj.phase0.timeseries.controls:u')

        assert_near_equal(t[-1], 20.0, tolerance=tol)
        assert_near_equal(x[-1, ...], [-100, 100], tolerance=tol)
        assert_near_equal(v[-1, ...], [0, 0], tolerance=tol)

    def test_flying_robot_gl_compressed(self):
        p = flying_robot_direct_collocation('gauss-lobatto',
                                            compressed=True)
        self._assert_results(p)

    def test_flying_robot_gl_uncompressed(self):
        p = flying_robot_direct_collocation('gauss-lobatto',
                                            compressed=False)
        self._assert_results(p)

    def test_flying_robot_radau_compressed(self):
        p = flying_robot_direct_collocation('radau-ps',
                                            compressed=True)
        self._assert_results(p)

    def test_flying_robot_radau_uncompressed(self):
        p = flying_robot_direct_collocation('radau-ps',
                                            compressed=False)
        self._assert_results(p)


if __name__ == "__main__":

    unittest.main()
