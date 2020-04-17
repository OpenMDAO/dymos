from __future__ import print_function, division, absolute_import

import os
import unittest
from openmdao.api import Problem, Group, pyOptSparseDriver
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
from openmdao.utils.general_utils import set_pyoptsparse_opt, printoptions
from dymos import Trajectory, GaussLobatto, Phase, Radau
from dymos.examples.robot_arm.robot_arm_ode import RobotArmODE
import numpy as np
import dymos as dm


class TestRobotArm(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def make_problem(self, transcription=Radau, optimizer='SLSQP', numseg=30):
        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()
        OPT, OPTIMIZER = set_pyoptsparse_opt(optimizer, fallback=False)
        p.driver.options['optimizer'] = OPTIMIZER

        traj = p.model.add_subsystem('traj', Trajectory())
        phase = traj.add_phase('phase', Phase(ode_class=RobotArmODE,
                                              transcription=transcription(num_segments=numseg, order=3)))
        phase.set_time_options(fix_initial=True, fix_duration=False)

        phase.add_state('x0', fix_initial=True, fix_final=True, rate_source='x0_dot', targets=['x0'], units='m')
        phase.add_state('x1', fix_initial=True, fix_final=True, rate_source='x1_dot', targets=['x1'], units='rad')
        phase.add_state('x2', fix_initial=True, fix_final=True, rate_source='x2_dot', targets=['x2'], units='rad')
        phase.add_state('x3', fix_initial=True, fix_final=True, rate_source='x3_dot', targets=['x3'], units='m/s')
        phase.add_state('x4', fix_initial=True, fix_final=True, rate_source='x4_dot', targets=['x4'], units='rad/s')
        phase.add_state('x5', fix_initial=True, fix_final=True, rate_source='x5_dot', targets=['x5'], units='rad/s')

        phase.add_control('u0', opt=True, targets=['u0'], units='m**2/s**2')
        phase.add_control('u1', opt=True, targets=['u1'], units='m**3*rad/s**2')
        phase.add_control('u2', opt=True, targets=['u2'], units='m**3*rad/s**2')

        phase.add_objective('time', index=-1, scaler=0.01)

        p.setup(check=True, force_alloc_complex=True)
        p.set_val('traj.phase.states:x0', phase.interpolate(ys=[4.5, 4.5], nodes='state_input'))
        p.set_val('traj.phase.states:x1', phase.interpolate(ys=[0.0, 2*np.pi/3], nodes='state_input'))
        p.set_val('traj.phase.states:x2', phase.interpolate(ys=[np.pi/4, np.pi/4], nodes='state_input'))
        p.set_val('traj.phase.states:x3', phase.interpolate(ys=[0.0, 0.0], nodes='state_input'))
        p.set_val('traj.phase.states:x4', phase.interpolate(ys=[0.0, 0.0], nodes='state_input'))
        p.set_val('traj.phase.states:x5', phase.interpolate(ys=[0.0, 0.0], nodes='state_input'))

        return p

    def test_partials(self):
        p = self.make_problem(transcription=Radau, optimizer='SLSQP')
        p.run_model()
        with printoptions(linewidth=1024, edgeitems=100):
            cpd = p.check_partials(method='fd', compact_print=True, out_stream=None)

    def test_robot_arm_radau(self):
        p = self.make_problem(transcription=Radau, optimizer='SNOPT', numseg=100)
        dm.run_problem(p, refine=True)

        t = p.get_val('traj.phase.timeseries.time')
        assert_rel_error(self, t[-1], 9.14138)

    def test_robot_arm_gl(self):
        p = self.make_problem(transcription=GaussLobatto, optimizer='SNOPT', numseg=100)
        dm.run_problem(p, refine=True)

        t = p.get_val('traj.phase.timeseries.time')

        assert_rel_error(self, t[-1], 9.14138)
