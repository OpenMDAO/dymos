from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, SqliteRecorder

from dymos import Phase
from dymos.phases.components.phase_linkage_comp import PhaseLinkageComp
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

OPTIMIZER = 'SLSQP'
SHOW_PLOTS = True


def two_burn_orbit_raise_problem(transcription='gauss-lobatto',
                                 transcription_order=3, compressed=True):
    p = Problem(model=Group())

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

    burn1 = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=4,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('burn1', burn1)

    burn1.set_time_options(opt_initial=False, duration_bounds=(.5, 10))
    burn1.set_state_options('r', fix_initial=True, fix_final=False)
    burn1.set_state_options('theta', fix_initial=True, fix_final=False)
    burn1.set_state_options('vr', fix_initial=True, fix_final=False, defect_scaler=0.1)
    burn1.set_state_options('vt', fix_initial=True, fix_final=False, defect_scaler=0.1)
    burn1.set_state_options('at', fix_initial=True, fix_final=False)
    burn1.set_state_options('deltav', fix_initial=True, fix_final=False)
    burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg')
    burn1.add_design_parameter('c', opt=False, val=1.5)

    # Second Phase (Coast)

    coast = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=10,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('coast', coast)

    coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
    coast.set_state_options('r', fix_initial=False, fix_final=False)
    coast.set_state_options('theta', fix_initial=False, fix_final=False)
    coast.set_state_options('vr', fix_initial=False, fix_final=False)
    coast.set_state_options('vt', fix_initial=False, fix_final=False)
    coast.set_state_options('at', fix_initial=True, fix_final=True)
    coast.set_state_options('deltav', fix_initial=False, fix_final=False)
    coast.add_control('u1', opt=False, val=0.0, units='deg')
    coast.add_design_parameter('c', opt=False, val=1.5)

    # Third Phase (burn)

    burn2 = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=3,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('burn2', burn2)

    burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
    burn2.set_state_options('r', fix_initial=False, fix_final=True, defect_scaler=1.0)
    burn2.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=1.0)
    burn2.set_state_options('vr', fix_initial=False, fix_final=True, defect_scaler=0.1)
    burn2.set_state_options('vt', fix_initial=False, fix_final=True, defect_scaler=0.1)
    burn2.set_state_options('at', fix_initial=False, fix_final=False, defect_scaler=1.0)
    burn2.set_state_options('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0)
    burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                      ref0=-10, ref=10)
    burn2.add_design_parameter('c', opt=False, val=1.5)

    burn2.add_objective('deltav', loc='final')

    # Link Phases
    linkage_comp = p.model.add_subsystem('linkages', subsys=PhaseLinkageComp())
    linkage_comp.add_linkage(name='L01', vars=['t', 'r', 'theta', 'vr', 'vt', 'deltav'], linear=True)
    linkage_comp.add_linkage(name='L12', vars=['t', 'r', 'theta', 'vr', 'vt', 'deltav'], linear=True)
    linkage_comp.add_linkage(name='L02', vars=['at'], linear=True)

    # Time Continuity
    p.model.connect('burn1.time++', 'linkages.L01_t:lhs')
    p.model.connect('coast.time--', 'linkages.L01_t:rhs')

    p.model.connect('coast.time++', 'linkages.L12_t:lhs')
    p.model.connect('burn2.time--', 'linkages.L12_t:rhs')

    # Position and velocity continuity
    for state in ['r', 'theta', 'vr', 'vt', 'deltav']:
        p.model.connect('burn1.states:{0}++'.format(state), 'linkages.L01_{0}:lhs'.format(state))
        p.model.connect('coast.states:{0}--'.format(state), 'linkages.L01_{0}:rhs'.format(state))

        p.model.connect('coast.states:{0}++'.format(state), 'linkages.L12_{0}:lhs'.format(state))
        p.model.connect('burn2.states:{0}--'.format(state), 'linkages.L12_{0}:rhs'.format(state))

    # Thrust/weight continuity between the burn phases
    p.model.connect('burn1.states:at++', 'linkages.L02_at:lhs')
    p.model.connect('burn2.states:at--', 'linkages.L02_at:rhs')

    # Finish Problem Setup

    p.model.options['assembled_jac_type'] = 'csc'
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(check=True)

    # Set Initial Guesses

    p.set_val('burn1.t_initial', value=0.0)
    p.set_val('burn1.t_duration', value=2.25)

    p.set_val('burn1.states:r', value=burn1.interpolate(ys=[1, 1.5], nodes='state_input'))
    p.set_val('burn1.states:theta', value=burn1.interpolate(ys=[0, 1.7], nodes='state_input'))
    p.set_val('burn1.states:vr', value=burn1.interpolate(ys=[0, 0], nodes='state_input'))
    p.set_val('burn1.states:vt', value=burn1.interpolate(ys=[1, 1], nodes='state_input'))
    p.set_val('burn1.states:at', value=burn1.interpolate(ys=[0.1, 0], nodes='state_input'))
    p.set_val('burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1], nodes='state_input'))
    p.set_val('burn1.controls:u1', value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))
    p.set_val('burn1.design_parameters:c', value=1.5)

    p.set_val('coast.t_initial', value=2.25)
    p.set_val('coast.t_duration', value=3.0)

    p.set_val('coast.states:r', value=coast.interpolate(ys=[1.3, 1.5], nodes='state_input'))
    p.set_val('coast.states:theta', value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))
    p.set_val('coast.states:vr', value=coast.interpolate(ys=[0.3285, 0], nodes='state_input'))
    p.set_val('coast.states:vt', value=coast.interpolate(ys=[0.97, 1], nodes='state_input'))
    p.set_val('coast.states:at', value=coast.interpolate(ys=[0, 0], nodes='state_input'))
    p.set_val('coast.controls:u1', value=coast.interpolate(ys=[0, 0], nodes='control_input'))
    p.set_val('coast.design_parameters:c', value=1.5)

    p.set_val('burn2.t_initial', value=5.25)
    p.set_val('burn2.t_duration', value=1.75)

    p.set_val('burn2.states:r', value=burn2.interpolate(ys=[1, 3], nodes='state_input'))
    p.set_val('burn2.states:theta', value=burn2.interpolate(ys=[0, 4.0], nodes='state_input'))
    p.set_val('burn2.states:vr', value=burn2.interpolate(ys=[0, 0], nodes='state_input'))
    p.set_val('burn2.states:vt', value=burn2.interpolate(ys=[1, np.sqrt(1/3)], nodes='state_input'))
    p.set_val('burn2.states:at', value=burn2.interpolate(ys=[0.1, 0], nodes='state_input'))
    p.set_val('burn2.states:deltav', value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
    p.set_val('burn2.controls:u1', value=burn2.interpolate(ys=[1, 1], nodes='control_input'))
    p.set_val('burn2.design_parameters:c', value=1.5)

    p.run_model()
    p.run_driver()

    # Plot results
    if SHOW_PLOTS:
        burn1_exp_out = burn1.simulate(times=np.linspace(
            p['burn1.t_initial'], p['burn1.t_initial'] + p['burn1.t_duration'], 50))

        coast_exp_out = coast.simulate(times=np.linspace(
            p['coast.t_initial'], p['coast.t_initial'] + p['coast.t_duration'], 50))

        burn2_exp_out = burn2.simulate(times=np.linspace(
            p['burn2.t_initial'], p['burn2.t_initial'] + p['burn2.t_duration'], 50))

        fig_xy, ax_xy = plt.subplots()
        fig_xy.suptitle('Two Burn Orbit Raise Solution')
        ax_xy.set_aspect('equal', 'datalim')
        theta = np.linspace(0, 2*np.pi, 100)
        ax_xy.plot(1*np.cos(theta), 1*np.sin(theta), ls='-', color='gray', label='initial orbit')
        ax_xy.plot(3*np.cos(theta), 3*np.sin(theta), ls='--', color='gray', label='final orbit')

        fig_at, ax_at = plt.subplots()
        fig_at.suptitle('Thrust/Mass History')

        fig_u1, ax_u1 = plt.subplots()
        fig_u1.suptitle('Control History')

        fig_deltav, ax_deltav = plt.subplots()
        fig_deltav.suptitle('Delta-V History')

        for (phase, phase_exp_out) in [(burn1, burn1_exp_out),
                                       (coast, coast_exp_out),
                                       (burn2, burn2_exp_out)]:
            x_imp = phase.get_values('pos_x', nodes='all')
            y_imp = phase.get_values('pos_y', nodes='all')

            x_exp = phase_exp_out.get_values('pos_x')
            y_exp = phase_exp_out.get_values('pos_y')

            ax_xy.plot(x_imp, y_imp, 'ro', label='implicit' if phase == burn1 else None)
            ax_xy.plot(x_exp, y_exp, 'b-', label='explicit' if phase == burn1 else None)

            if phase is not coast:
                theta_imp = phase.get_values('theta', nodes='all', units='rad')
                u_imp = phase.get_values('u1', nodes='all', units='rad')
                at_imp = phase.get_values('at', nodes='all')
                a_x = at_imp * np.cos(theta_imp + u_imp + np.radians(90))
                a_y = at_imp * np.sin(theta_imp + u_imp + np.radians(90))
                ax_xy.quiver(x_imp, y_imp, 10*a_x, 10*a_y, scale=1, angles='xy', scale_units='xy',
                             width=0.002, headwidth=0.1)

            x_imp = phase.get_values('time', nodes='all')
            y_imp = phase.get_values('at', nodes='all')

            x_exp = phase_exp_out.get_values('time')
            y_exp = phase_exp_out.get_values('at')

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

    return p


if __name__ == '__main__':
    two_burn_orbit_raise_problem(transcription='gauss-lobatto',
                                 transcription_order=3, compressed=True)
