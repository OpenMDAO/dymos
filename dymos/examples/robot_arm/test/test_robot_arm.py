import os
import unittest

import numpy as np

try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    show_plots = True
except ImportError:
    show_plots = False

import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import printoptions
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.robot_arm.robot_arm_ode import RobotArmODE


@use_tempdirs
@require_pyoptsparse(optimizer='SLSQP')
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

        phase.set_time_val(initial=0, duration=10)
        phase.set_state_val('x0', [4.5, 4.5])
        phase.set_state_val('x1', [0.0, 2 * np.pi / 3])
        phase.set_state_val('x2', [np.pi / 4, np.pi / 4])
        phase.set_state_val('x3', [0.0, 0.0])
        phase.set_state_val('x4', [0.0, 0.0])
        phase.set_state_val('x5', [0.0, 0.0])

        return p

    def test_partials(self):
        p = self.make_problem(tx=dm.Radau(num_segments=30, order=3), optimizer='SLSQP')
        p.run_model()
        with printoptions(linewidth=1024, edgeitems=100):
            p.check_partials(method='fd', compact_print=True, out_stream=None)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_radau(self):
        p = self.make_problem(tx=dm.Radau(num_segments=12, order=3), optimizer='IPOPT', numseg=12)
        dm.run_problem(p)

        t = p.get_val('traj.phase.timeseries.time')
        rho = p.get_val('traj.phase.timeseries.x0')
        theta = p.get_val('traj.phase.timeseries.x1')
        phi = p.get_val('traj.phase.timeseries.x2')
        u0 = p.get_val('traj.phase.timeseries.u0')
        u1 = p.get_val('traj.phase.timeseries.u1')
        u2 = p.get_val('traj.phase.timeseries.u2')

        exp_out = p.model.traj.simulate()
        t_exp = exp_out.get_val('traj.phase.timeseries.time')
        rho_exp = exp_out.get_val('traj.phase.timeseries.x0')
        theta_exp = exp_out.get_val('traj.phase.timeseries.x1')
        phi_exp = exp_out.get_val('traj.phase.timeseries.x2')
        u0_exp = exp_out.get_val('traj.phase.timeseries.u0')
        u1_exp = exp_out.get_val('traj.phase.timeseries.u1')
        u2_exp = exp_out.get_val('traj.phase.timeseries.u2')

        if show_plots:
            fig, axs = plt.subplots(2, 3)
            axs[0, 0].plot(t, rho, marker='o', linestyle='')
            axs[0, 1].plot(t, theta, marker='o', linestyle='')
            axs[0, 2].plot(t, phi, marker='o', linestyle='')
            axs[1, 0].plot(t, u0, marker='o', linestyle='')
            axs[1, 1].plot(t, u1, marker='o', linestyle='')
            axs[1, 2].plot(t, u2, marker='o', linestyle='')

            axs[0, 0].plot(t_exp, rho_exp, marker='', linestyle='-')
            axs[0, 1].plot(t_exp, theta_exp, marker='', linestyle='-')
            axs[0, 2].plot(t_exp, phi_exp, marker='', linestyle='-')
            axs[1, 0].plot(t_exp, u0_exp, marker='', linestyle='-')
            axs[1, 1].plot(t_exp, u1_exp, marker='', linestyle='-')
            axs[1, 2].plot(t_exp, u2_exp, marker='', linestyle='-')
            plt.show()

        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_gl(self):
        p = self.make_problem(tx=dm.GaussLobatto(num_segments=20, order=3), optimizer='IPOPT')
        dm.run_problem(p)

        t = p.get_val('traj.phase.timeseries.time')
        rho = p.get_val('traj.phase.timeseries.x0')
        theta = p.get_val('traj.phase.timeseries.x1')
        phi = p.get_val('traj.phase.timeseries.x2')
        u0 = p.get_val('traj.phase.timeseries.u0')
        u1 = p.get_val('traj.phase.timeseries.u1')
        u2 = p.get_val('traj.phase.timeseries.u2')

        if show_plots:
            fig, axs = plt.subplots(2, 3)
            axs[0, 0].plot(t, rho)
            axs[0, 1].plot(t, theta)
            axs[0, 2].plot(t, phi)
            axs[1, 0].plot(t, u0)
            axs[1, 1].plot(t, u1)
            axs[1, 2].plot(t, u2)
            plt.show()

        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_birkhoff_lgl(self):
        tx = dm.Birkhoff(num_nodes=30, grid_type='lgl')
        p = self.make_problem(tx=tx, optimizer='IPOPT')
        dm.run_problem(p)

        t = p.get_val('traj.phase.timeseries.time')
        rho = p.get_val('traj.phase.timeseries.x0')
        theta = p.get_val('traj.phase.timeseries.x1')
        phi = p.get_val('traj.phase.timeseries.x2')
        u0 = p.get_val('traj.phase.timeseries.u0')
        u1 = p.get_val('traj.phase.timeseries.u1')
        u2 = p.get_val('traj.phase.timeseries.u2')

        if show_plots:
            fig, axs = plt.subplots(2, 3)
            axs[0, 0].plot(t, rho)
            axs[0, 1].plot(t, theta)
            axs[0, 2].plot(t, phi)
            axs[1, 0].plot(t, u0)
            axs[1, 1].plot(t, u1)
            axs[1, 2].plot(t, u2)
            plt.show()

        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_robot_arm_birkhoff_cgl(self):
        tx = dm.Birkhoff(num_nodes=30, grid_type='cgl')
        p = self.make_problem(tx=tx, optimizer='IPOPT')
        dm.run_problem(p)

        t = p.get_val('traj.phase.timeseries.time')
        rho = p.get_val('traj.phase.timeseries.x0')
        theta = p.get_val('traj.phase.timeseries.x1')
        phi = p.get_val('traj.phase.timeseries.x2')
        u0 = p.get_val('traj.phase.timeseries.u0')
        u1 = p.get_val('traj.phase.timeseries.u1')
        u2 = p.get_val('traj.phase.timeseries.u2')

        if show_plots:
            fig, axs = plt.subplots(2, 3)
            axs[0, 0].plot(t, rho)
            axs[0, 1].plot(t, theta)
            axs[0, 2].plot(t, phi)
            axs[1, 0].plot(t, u0)
            axs[1, 1].plot(t, u1)
            axs[1, 2].plot(t, u2)
            plt.show()

        assert_near_equal(t[-1], 9.14138, tolerance=1e-3)
