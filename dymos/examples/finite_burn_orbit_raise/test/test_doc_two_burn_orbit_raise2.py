from __future__ import print_function, division, absolute_import

import unittest

import matplotlib
# matplotlib.use('Agg')


class TestTwoBurnOrbitRaiseForDocs(unittest.TestCase):

    def test_two_burn_orbit_raise_for_docs(self):
        import numpy as np

        import matplotlib.pyplot as plt

        from openmdao.api import Problem, pyOptSparseDriver, DirectSolver, ScipyOptimizeDriver,\
            SqliteRecorder
        from openmdao.utils.assert_utils import assert_rel_error

        from dymos import Phase, Trajectory
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        OPTIMIZER = 'SNOPT'

        traj = Trajectory()
        p = Problem(model=traj)

        if OPTIMIZER == 'SNOPT':
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = OPTIMIZER
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.opt_settings['Major iterations limit'] = 100
            p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
            p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
            p.driver.opt_settings['iSumm'] = 6
        else:
            p.driver = pyOptSparseDriver()
            p.driver.options['optimizer'] = OPTIMIZER
            p.driver.options['dynamic_simul_derivs'] = True
            p.driver.opt_settings['ACC'] = 1.0E-9

        # First Phase (burn)

        burn1 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=4,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('burn1', burn1)


        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10))
        burn1.set_state_options('r', fix_initial=True, fix_final=False)
        burn1.set_state_options('theta', fix_initial=True, fix_final=False)
        burn1.set_state_options('vr', fix_initial=True, fix_final=False, defect_scaler=0.1)
        burn1.set_state_options('vt', fix_initial=True, fix_final=False, defect_scaler=0.1)
        burn1.set_state_options('accel', fix_initial=True, fix_final=False)
        burn1.set_state_options('deltav', fix_initial=True, fix_final=False)
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg')
        burn1.add_design_parameter('c', opt=False, val=1.5)

        # Second Phase (Coast)

        coast = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=10,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
        coast.set_state_options('r', fix_initial=False, fix_final=False)
        coast.set_state_options('theta', fix_initial=False, fix_final=False)
        coast.set_state_options('vr', fix_initial=False, fix_final=False)
        coast.set_state_options('vt', fix_initial=False, fix_final=False)
        coast.set_state_options('accel', fix_initial=True, fix_final=True)
        coast.set_state_options('deltav', fix_initial=False, fix_final=False)
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_design_parameter('c', opt=False, val=1.5)

        # Third Phase (burn)

        burn2 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=3,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
        burn2.set_state_options('r', fix_initial=False, fix_final=True, defect_scaler=1.0)
        burn2.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.set_state_options('vr', fix_initial=False, fix_final=True, defect_scaler=0.1)
        burn2.set_state_options('vt', fix_initial=False, fix_final=True, defect_scaler=0.1)
        burn2.set_state_options('accel', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.set_state_options('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          ref0=0, ref=10)
        burn2.add_design_parameter('c', opt=False, val=1.5)

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.driver.add_recorder(SqliteRecorder('two_burn_orbit_raise_example.db'))

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interpolate(ys=[1, 1.5], nodes='state_input'))
        p.set_val('burn1.states:theta', value=burn1.interpolate(ys=[0, 1.7], nodes='state_input'))
        p.set_val('burn1.states:vr', value=burn1.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('burn1.states:vt', value=burn1.interpolate(ys=[1, 1], nodes='state_input'))
        p.set_val('burn1.states:accel', value=burn1.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1], nodes='state_input'))
        p.set_val('burn1.controls:u1',
                  value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))
        p.set_val('burn1.design_parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interpolate(ys=[1.3, 1.5], nodes='state_input'))
        p.set_val('coast.states:theta',
                  value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))
        p.set_val('coast.states:vr', value=coast.interpolate(ys=[0.3285, 0], nodes='state_input'))
        p.set_val('coast.states:vt', value=coast.interpolate(ys=[0.97, 1], nodes='state_input'))
        p.set_val('coast.states:accel', value=coast.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('coast.controls:u1', value=coast.interpolate(ys=[0, 0], nodes='control_input'))
        p.set_val('coast.design_parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interpolate(ys=[1, 3], nodes='state_input'))
        p.set_val('burn2.states:theta', value=burn2.interpolate(ys=[0, 4.0], nodes='state_input'))
        p.set_val('burn2.states:vr', value=burn2.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('burn2.states:vt',
                  value=burn2.interpolate(ys=[1, np.sqrt(1 / 3)], nodes='state_input'))
        p.set_val('burn2.states:accel', value=burn2.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('burn2.states:deltav',
                  value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
        p.set_val('burn2.controls:u1', value=burn2.interpolate(ys=[1, 1], nodes='control_input'))
        p.set_val('burn2.design_parameters:c', value=1.5)

        p.run_driver()

        assert_rel_error(self, p.get_val('burn2.states:deltav')[-1], 0.3995, tolerance=1.0E-3)

        # Plot results
        exp_out = traj.simulate2(times=50)

        exit(0)

        fig_xy, ax_xy = plt.subplots()
        fig_xy.suptitle('Two Burn Orbit Raise Solution')
        ax_xy.set_aspect('equal', 'datalim')
        theta = np.linspace(0, 2 * np.pi, 100)
        ax_xy.plot(1 * np.cos(theta), 1 * np.sin(theta), ls='-', color='gray',
                   label='initial orbit')
        ax_xy.plot(3 * np.cos(theta), 3 * np.sin(theta), ls='--', color='gray',
                   label='final orbit')

        fig_at, ax_at = plt.subplots()
        fig_at.suptitle('Thrust/Mass History')

        fig_u1, ax_u1 = plt.subplots()
        fig_u1.suptitle('Control History')

        fig_deltav, ax_deltav = plt.subplots()
        fig_deltav.suptitle('Delta-V History')

        for (phase, phase_exp_out) in [(burn1, exp_out['burn1']),
                                       (coast, exp_out['coast']),
                                       (burn2, exp_out['burn2'])]:
            x_imp = phase.get_values('pos_x', nodes='all')
            y_imp = phase.get_values('pos_y', nodes='all')

            x_exp = phase_exp_out.get_values('pos_x')
            y_exp = phase_exp_out.get_values('pos_y')

            ax_xy.plot(x_imp, y_imp, 'ro', label='implicit' if phase == burn1 else None)
            ax_xy.plot(x_exp, y_exp, 'b-', label='explicit' if phase == burn1 else None)

            if phase is not coast:
                theta_imp = phase.get_values('theta', nodes='all', units='rad')
                u_imp = phase.get_values('u1', nodes='all', units='rad')
                at_imp = phase.get_values('accel', nodes='all')
                a_x = at_imp * np.cos(theta_imp + u_imp + np.radians(90))
                a_y = at_imp * np.sin(theta_imp + u_imp + np.radians(90))
                ax_xy.quiver(x_imp, y_imp, 10 * a_x, 10 * a_y, scale=1, angles='xy',
                             scale_units='xy',
                             width=0.002, headwidth=0.1)

            x_imp = phase.get_values('time', nodes='all')
            y_imp = phase.get_values('accel', nodes='all')

            x_exp = phase_exp_out.get_values('time')
            y_exp = phase_exp_out.get_values('accel')

            ax_at.plot(x_imp, y_imp, 'ro', label='implicit')
            ax_at.plot(x_exp, y_exp, 'b-', label='explicit')

            x_imp = phase.get_values('time', nodes='all')
            y_imp = phase.get_values('u1', nodes='all')

            x_exp = phase_exp_out.get_values('time')
            y_exp = phase_exp_out.get_values('u1')

            ax_u1.plot(x_imp, y_imp, 'ro', label='implicit')
            ax_u1.plot(x_exp, y_exp, 'b-', label='explicit')

            x_imp = phase.get_values('time', nodes='all')
            y_imp = phase.get_values('deltav', nodes='all')

            x_exp = phase_exp_out.get_values('time')
            y_exp = phase_exp_out.get_values('deltav')

            ax_deltav.plot(x_imp, y_imp, 'ro', label='implicit')
            ax_deltav.plot(x_exp, y_exp, 'b-', label='explicit')

        ax_xy.set_xlim(-4.5, 4.5)
        ax_xy.set_ylim(-4.5, 4.5)
        ax_xy.set_xlabel('x (DU)')
        ax_xy.set_ylabel('y (DU)')
        ax_xy.grid(True)
        ax_xy.legend(loc='upper right')

        plt.show()