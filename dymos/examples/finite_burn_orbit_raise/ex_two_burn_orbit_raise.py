from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, pyOptSparseDriver, DirectSolver, SqliteRecorder, \
    ScipyOptimizeDriver

from dymos import Phase, Trajectory
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE


def two_burn_orbit_raise_problem(transcription='gauss-lobatto', optimizer='SLSQP',
                                 transcription_order=3, compressed=True, show_plots=True):

    traj = Trajectory()
    p = Problem(model=traj)

    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['iSumm'] = 6
    else:
        p.driver = pyOptSparseDriver()
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.opt_settings['ACC'] = 1.0E-6

    # First Phase (burn)

    burn1 = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=10,
                  transcription_order=transcription_order,
                  compressed=compressed)

    traj.add_phase('burn1', burn1)

    burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10))
    burn1.set_state_options('r', fix_initial=True, fix_final=False)
    burn1.set_state_options('theta', fix_initial=True, fix_final=False)
    burn1.set_state_options('vr', fix_initial=True, fix_final=False, defect_scaler=1.0)
    burn1.set_state_options('vt', fix_initial=True, fix_final=False, defect_scaler=1.0)
    burn1.set_state_options('accel', fix_initial=True, fix_final=False)
    burn1.set_state_options('deltav', fix_initial=True, fix_final=False)
    burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg')
    burn1.add_design_parameter('c', opt=False, val=1.5)

    # Second Phase (Coast)

    coast = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=10,
                  transcription_order=transcription_order,
                  compressed=compressed)

    traj.add_phase('coast', coast)

    coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10)
    coast.set_state_options('r', fix_initial=False, fix_final=False)
    coast.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=100.0)
    coast.set_state_options('vr', fix_initial=False, fix_final=False)
    coast.set_state_options('vt', fix_initial=False, fix_final=False)
    coast.set_state_options('accel', fix_initial=True, fix_final=True)
    coast.set_state_options('deltav', fix_initial=False, fix_final=False)
    coast.add_control('u1', opt=False, val=0.0, units='deg')
    coast.add_design_parameter('c', opt=False, val=1.5)

    # Third Phase (burn)

    burn2 = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=10,
                  transcription_order=transcription_order,
                  compressed=compressed)

    traj.add_phase('burn2', burn2)

    burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10)
    burn2.set_state_options('r', fix_initial=False, fix_final=True, defect_scaler=1000.0)
    burn2.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=1.0)
    burn2.set_state_options('vr', fix_initial=False, fix_final=True, defect_scaler=1.0)
    burn2.set_state_options('vt', fix_initial=False, fix_final=True, defect_scaler=1.0)
    burn2.set_state_options('accel', fix_initial=False, fix_final=False, defect_scaler=1.0)
    burn2.set_state_options('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0)
    burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                      ref0=0, ref=10)
    burn2.add_design_parameter('c', opt=False, val=1.5)

    burn2.add_objective('deltav', loc='final', scaler=10.0)

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
    p.set_val('burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1], nodes='state_input'),)
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

    # Plot results
    exp_out = traj.simulate(times=50, num_procs=3)

    fig = plt.figure(figsize=(8, 4))
    fig.suptitle('Two Burn Orbit Raise Solution')
    ax_u1 = plt.subplot2grid((2, 2), (0, 0))
    ax_deltav = plt.subplot2grid((2, 2), (1, 0))
    ax_xy = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

    span = np.linspace(0, 2 * np.pi, 100)
    ax_xy.plot(np.cos(span), np.sin(span), 'k--', lw=1)
    ax_xy.plot(3 * np.cos(span), 3 * np.sin(span), 'k--', lw=1)
    ax_xy.set_xlim(-4.5, 4.5)
    ax_xy.set_ylim(-4.5, 4.5)

    ax_xy.set_xlabel('x ($R_e$)')
    ax_xy.set_ylabel('y ($R_e$)')

    ax_u1.set_xlabel('time ($TU$)')
    ax_u1.set_ylabel('$u_1$ ($deg$)')
    ax_u1.grid(True)

    ax_deltav.set_xlabel('time ($TU$)')
    ax_deltav.set_ylabel('${\Delta}v$ ($DU/TU$)')
    ax_deltav.grid(True)

    t_sol = traj.get_values('time')
    x_sol = traj.get_values('pos_x')
    y_sol = traj.get_values('pos_y')
    dv_sol = traj.get_values('deltav')
    u1_sol = traj.get_values('u1', units='deg')

    t_exp = exp_out.get_values('time')
    x_exp = exp_out.get_values('pos_x')
    y_exp = exp_out.get_values('pos_y')
    dv_exp = exp_out.get_values('deltav')
    u1_exp = exp_out.get_values('u1', units='deg')

    for phase_name in ['burn1', 'coast', 'burn2']:
        ax_u1.plot(t_sol[phase_name], u1_sol[phase_name], 'ro', ms=3)
        ax_u1.plot(t_exp[phase_name], u1_exp[phase_name], 'b-')

        ax_deltav.plot(t_sol[phase_name], dv_sol[phase_name], 'ro', ms=3)
        ax_deltav.plot(t_exp[phase_name], dv_exp[phase_name], 'b-')

        ax_xy.plot(x_sol[phase_name], y_sol[phase_name], 'ro', ms=3,
                   label='implicit' if phase_name == 'burn1' else None)
        ax_xy.plot(x_exp[phase_name], y_exp[phase_name], 'b-',
                   label='explicit' if phase_name == 'burn1' else None)

    if show_plots:
        plt.show()

    return p


if __name__ == '__main__':  # pragma: no cover
    p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', optimizer='SLSQP',
                                     transcription_order=3, compressed=True, show_plots=True)

    # p.check_totals(compact_print=True)

    totes_mgotes = p.compute_totals()

    from six import iteritems
    np.set_printoptions(linewidth=1024)

    for pair, jac in iteritems(totes_mgotes):
        of, wrt = pair
        # jac = tote[1]
        if np.linalg.norm(jac) > 0:
            print(of, wrt, np.linalg.norm(jac))


