import os
import unittest

import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import printoptions
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.robot_arm.robot_arm_ode import RobotArmODE


@use_tempdirs
class TestRobotArm(unittest.TestCase):

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @require_pyoptsparse(optimizer='SLSQP')
    def make_problem(self, tx, optimizer='SLSQP', numseg=30):
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()
        p.driver.options['optimizer'] = optimizer
        if optimizer == 'SNOPT':
            p.driver.opt_settings['iSumm'] = 6
            p.driver.opt_settings['Verify level'] = 3
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
            p.driver.opt_settings['max_iter'] = 500
            p.driver.opt_settings['print_level'] = 0
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase', dm.Phase(ode_class=RobotArmODE, transcription=tx))
        phase.set_time_options(fix_initial=True, fix_duration=False)

        phase.add_state('x0', fix_initial=True, fix_final=True, rate_source='x0_dot', units='m')
        phase.add_state('x1', fix_initial=True, fix_final=True, rate_source='x1_dot', units='rad')
        phase.add_state('x2', fix_initial=True, fix_final=True, rate_source='x2_dot', units='rad')
        phase.add_state('x3', fix_initial=True, fix_final=True, rate_source='x3_dot', units='m/s')
        phase.add_state('x4', fix_initial=True, fix_final=True, rate_source='x4_dot', units='rad/s')
        phase.add_state('x5', fix_initial=True, fix_final=True, rate_source='x5_dot', units='rad/s')

        phase.add_control('u0', opt=True, lower=-1, upper=1, scaler=0.1, units='m**2/s**2',
                          continuity=False, rate_continuity=False)
        phase.add_control('u1', opt=True, lower=-1, upper=1, scaler=0.1, units='m**3*rad/s**2',
                          continuity=False, rate_continuity=False)
        phase.add_control('u2', opt=True, lower=-1, upper=1, scaler=0.1, units='m**3*rad/s**2',
                          continuity=False, rate_continuity=False)

        phase.add_path_constraint('u0', lower=-1, upper=1, scaler=0.1)
        phase.add_path_constraint('u1', lower=-1, upper=1, scaler=0.1)
        phase.add_path_constraint('u2', lower=-1, upper=1, scaler=0.1)

        phase.add_objective('time', ref=0.1)

        phase.set_refine_options(refine=True, tol=1e-5, smoothness_factor=1.2)

        p.setup(check=True, force_alloc_complex=False, mode='auto')

        p.set_val('traj.phase.t_initial', 0)
        p.set_val('traj.phase.t_duration', 10)

        if isinstance(tx, dm.Birkhoff):
            p.set_val('traj.phase.initial_states:x0', 4.5)
            p.set_val('traj.phase.initial_states:x1', 0.0)
            p.set_val('traj.phase.initial_states:x2', np.pi / 4)
            p.set_val('traj.phase.initial_states:x3', 0.0)
            p.set_val('traj.phase.initial_states:x4', 0.0)
            p.set_val('traj.phase.initial_states:x5', 0.0)
            p.set_val('traj.phase.final_states:x0', 4.5)
            p.set_val('traj.phase.final_states:x1', 2 * np.pi / 3)
            p.set_val('traj.phase.final_states:x2', np.pi / 4)
            p.set_val('traj.phase.final_states:x3', 0.0)
            p.set_val('traj.phase.final_states:x4', 0.0)
            p.set_val('traj.phase.final_states:x5', 0.0)
        else:
            p.set_val('traj.phase.states:x0', phase.interp('x0', [4.5, 4.5]))
            p.set_val('traj.phase.states:x1', phase.interp('x1', [0.0, 2 * np.pi / 3]))
            p.set_val('traj.phase.states:x2', phase.interp('x2', [np.pi / 4, np.pi / 4]))
            p.set_val('traj.phase.states:x3', phase.interp('x3', [0.0, 0.0]))
            p.set_val('traj.phase.states:x4', phase.interp('x4', [0.0, 0.0]))
            p.set_val('traj.phase.states:x5', phase.interp('x5', [0.0, 0.0]))

        return p

    def test_partials(self):
        p = self.make_problem(tx=dm.Radau(num_segments=30, order=3), optimizer='SLSQP')
        p.run_model()
        with printoptions(linewidth=1024, edgeitems=100):
            cpd = p.check_partials(method='fd', compact_print=True, out_stream=None)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_radau(self):
        p = self.make_problem(tx=dm.Radau(num_segments=12, order=3), optimizer='IPOPT', numseg=12)
        dm.run_problem(p, refine_iteration_limit=5)

        t = p.get_val('traj.phase.timeseries.time')
        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_gl(self):
        p = self.make_problem(tx=dm.GaussLobatto(num_segments=20, order=3), optimizer='IPOPT')
        dm.run_problem(p, refine_iteration_limit=3)

        t = p.get_val('traj.phase.timeseries.time')
        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_birkhoff_lgl(self):
        tx = dm.Birkhoff(grid=dm.BirkhoffGrid(num_segments=1, nodes_per_seg=30, grid_type='lgl'))
        p = self.make_problem(tx=tx, optimizer='IPOPT')
        dm.run_problem(p)

        t = p.get_val('traj.phase.timeseries.time')
        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_birkhoff_cgl(self):
        tx = dm.Birkhoff(grid=dm.BirkhoffGrid(num_segments=1, nodes_per_seg=30, grid_type='cgl'))
        p = self.make_problem(tx=tx, optimizer='IPOPT')
        dm.run_problem(p)

        t = p.get_val('traj.phase.timeseries.time')

        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)
