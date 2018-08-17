from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, SqliteRecorder,\
    ScipyOptimizeDriver, pyOptSparseDriver

from dymos import Phase, Trajectory
from dymos.examples.cannonball.cannonball_ode import CannonballODE

from dymos.examples.cannonball.size_comp import CannonballSizeComp


def two_phase_cannonball_problem(transcription='radau-ps', optimizer='SLSQP',
                                 transcription_order=3, compressed=True, show_plots=False):

    p = Problem(model=Group())

    external_params = p.model.add_subsystem('external_params', IndepVarComp())

    external_params.add_output('radius', val=0.10, units='m')
    external_params.add_output('dens', val=7.87, units='g/cm**3')

    p.model.add_subsystem('size_comp', CannonballSizeComp())

    traj = p.model.add_subsystem('traj', Trajectory())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SLSQP'
    # p.driver.opt_settings['iSumm'] = 6
    p.driver.options['dynamic_simul_derivs'] = True

    # First Phase (ascent)
    ascent = Phase(transcription,
                   ode_class=CannonballODE,
                   num_segments=5,
                   transcription_order=transcription_order,
                   compressed=compressed)

    ascent = traj.add_phase('ascent', ascent)

    # All initial states except flight path angle are fixed
    # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
    ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100))
    ascent.set_state_options('r', fix_initial=True, fix_final=False)
    ascent.set_state_options('h', fix_initial=True, fix_final=False)
    ascent.set_state_options('gam', fix_initial=True, fix_final=True)
    ascent.set_state_options('v', fix_initial=True, fix_final=False)

    # Second Phase (descent)
    descent = Phase(transcription,
                    ode_class=CannonballODE,
                    num_segments=5,
                    transcription_order=transcription_order,
                    compressed=compressed)

    traj.add_phase('descent', descent)

    # All initial states and time are free (they will be linked to the final states of ascent.
    # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
    descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100), duration_ref=10)
    descent.set_state_options('r', fix_initial=False, fix_final=False)
    descent.set_state_options('h', fix_initial=False, fix_final=True)
    descent.set_state_options('gam', fix_initial=False, fix_final=False)
    descent.set_state_options('v', fix_initial=False, fix_final=False)

    descent.add_objective('r', loc='final', scaler=-0.01)

    # Add internally-managed design parameters to the trajectory.
    traj.add_design_parameter('CD', val=0.5, units=None, opt=False)
    traj.add_design_parameter('CL', val=0.0, units=None, opt=False)
    traj.add_design_parameter('T', val=0.0, units='N', opt=False)
    traj.add_design_parameter('alpha', val=0.0, units='deg', opt=False)

    # Add externally-provided design parameters to the trajectory.
    traj.add_design_parameter('mass', targets={'ascent': 'm', 'descent': 'm'},
                              val=1.0, units='kg', input_value=True)
    traj.add_design_parameter('S', val=0.005, units='m**2', input_value=True)

    # Link Phases (link time and all state variables)
    traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

    # Issue Connnections
    p.model.connect('external_params.radius', 'size_comp.radius')
    p.model.connect('external_params.dens', 'size_comp.dens')

    p.model.connect('size_comp.mass', 'traj.design_parameters:mass')
    p.model.connect('size_comp.S', 'traj.design_parameters:S')

    # Finish Problem Setup
    p.model.options['assembled_jac_type'] = 'csc'
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.driver.add_recorder(SqliteRecorder('ex_two_phase_cannonball.db'))

    p.setup(check=True)

    # Set Initial Guesses
    p.set_val('external_params.radius', 0.005, units='m')
    p.set_val('external_params.dens', 7.87, units='g/cm**3')

    p.set_val('traj.design_parameters:CD', 0.5)
    p.set_val('traj.design_parameters:CL', 0.0)
    p.set_val('traj.design_parameters:T', 0.0)

    p.set_val('traj.ascent.t_initial', 0.0)
    p.set_val('traj.ascent.t_duration', 10.0)

    p.set_val('traj.ascent.states:r', ascent.interpolate(ys=[0, 100], nodes='state_input'))
    p.set_val('traj.ascent.states:h', ascent.interpolate(ys=[0, 100], nodes='state_input'))
    p.set_val('traj.ascent.states:v', ascent.interpolate(ys=[200, 150], nodes='state_input'))
    p.set_val('traj.ascent.states:gam', ascent.interpolate(ys=[45, 0], nodes='state_input'), units='deg')

    p.set_val('traj.descent.t_initial', 10.0)
    p.set_val('traj.descent.t_duration', 10.0)

    p.set_val('traj.descent.states:r', descent.interpolate(ys=[100, 200], nodes='state_input'))
    p.set_val('traj.descent.states:h', descent.interpolate(ys=[100, 0], nodes='state_input'))
    p.set_val('traj.descent.states:v', descent.interpolate(ys=[150, 200], nodes='state_input'))
    p.set_val('traj.descent.states:gam', descent.interpolate(ys=[0, -45], nodes='state_input'), units='deg')

    p.run_model()

    p.run_driver()

    exp_out = traj.simulate(times=100)

    plt.plot(traj.get_values('r')['ascent'], traj.get_values('h')['ascent'], 'bo')
    plt.plot(traj.get_values('r')['descent'], traj.get_values('h')['descent'], 'ro')

    plt.plot(exp_out.get_values('r')['ascent'], exp_out.get_values('h')['ascent'], 'b--')
    plt.plot(exp_out.get_values('r')['descent'], exp_out.get_values('h')['descent'], 'r--')

    plt.figure()

    plt.plot(traj.get_values('time')['ascent'], traj.get_values('mass')['ascent'], 'bo')
    plt.plot(traj.get_values('time')['descent'], traj.get_values('mass')['descent'], 'ro')

    plt.plot(exp_out.get_values('time')['ascent'], exp_out.get_values('mass')['ascent'], 'b--')
    plt.plot(exp_out.get_values('time')['descent'], exp_out.get_values('mass')['descent'], 'r--')

    plt.show()


#         self.p['phase0.states:h'] = phase.interpolate(ys=[0, 0], nodes='state_input')
#         self.p['phase0.states:v'] = phase.interpolate(ys=[v0, v0], nodes='state_input')
#         self.p['phase0.states:gam'] = phase.interpolate(ys=[gam0, -gam0], nodes='state_input')


#
#     def test_cannonball_max_range(self):
#         self.p.setup()
#
#         v0 = 100.0
#         gam0 = np.radians(30.0)
#         g = 9.80665
#         t_duration = 10.0
#
#         phase = self.p.model.phase0
#
#         self.p['phase0.t_initial'] = 0.0
#         self.p['phase0.t_duration'] = t_duration
#
#         self.p['phase0.states:r'] = phase.interpolate(ys=[0, v0 * np.cos(gam0) * t_duration],
#                                                       nodes='state_disc')
#         self.p['phase0.states:h'] = phase.interpolate(ys=[0, 0], nodes='state_input')
#         self.p['phase0.states:v'] = phase.interpolate(ys=[v0, v0], nodes='state_input')
#         self.p['phase0.states:gam'] = phase.interpolate(ys=[gam0, -gam0], nodes='state_input')
#
#         self.p.run_driver()
#
#         exp_out = phase.simulate(times='all')
#
#         assert_rel_error(self, exp_out.get_values('h')[-1], 0.0, tolerance=0.001)
#         assert_rel_error(self, exp_out.get_values('r')[-1], v0**2 / g, tolerance=0.001)
#         assert_rel_error(self, exp_out.get_values('gam')[-1], -np.radians(45), tolerance=0.001)
#         assert_rel_error(self, exp_out.get_values('v')[-1], v0, tolerance=0.001)
#
#     p.set_val('ascent.t_initial', value=0.0)
#     p.set_val('ascent.t_duration', value=10)
#
#     p.set_val('ascent.states:r', value=ascent.interpolate(ys=[1, 1.5], nodes='state_input'))
#     p.set_val('ascent.states:theta', value=ascent.interpolate(ys=[0, 1.7], nodes='state_input'))
#     p.set_val('ascent.states:vr', value=ascent.interpolate(ys=[0, 0], nodes='state_input'))
#     p.set_val('ascent.states:vt', value=ascent.interpolate(ys=[1, 1], nodes='state_input'))
#     p.set_val('ascent.states:accel', value=ascent.interpolate(ys=[0.1, 0], nodes='state_input'))
#     p.set_val('ascent.states:deltav', value=ascent.interpolate(ys=[0, 0.1], nodes='state_input'),)
#     p.set_val('ascent.controls:u1',
#               value=ascent.interpolate(ys=[-3.5, 13.0], nodes='control_input'))
#     p.set_val('ascent.design_parameters:c', value=1.5)
#
#
#
#     p.set_val('coast.t_initial', value=2.25)
#     p.set_val('coast.t_duration', value=3.0)
#
#     p.set_val('coast.states:r', value=coast.interpolate(ys=[1.3, 1.5], nodes='state_input'))
#     p.set_val('coast.states:theta',
#               value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))
#     p.set_val('coast.states:vr', value=coast.interpolate(ys=[0.3285, 0], nodes='state_input'))
#     p.set_val('coast.states:vt', value=coast.interpolate(ys=[0.97, 1], nodes='state_input'))
#     p.set_val('coast.states:accel', value=coast.interpolate(ys=[0, 0], nodes='state_input'))
#     p.set_val('coast.controls:u1', value=coast.interpolate(ys=[0, 0], nodes='control_input'))
#     p.set_val('coast.design_parameters:c', value=1.5)
#
#     p.set_val('burn2.t_initial', value=5.25)
#     p.set_val('burn2.t_duration', value=1.75)
#
#     p.run_model()
#
#     exit(0)
#
#     # Plot results
#     exp_out = traj.simulate(times=50, num_procs=2,
#                             record_file='example_two_phase_cannonball_sim.db')
#
#     fig = plt.figure(figsize=(8, 4))
#     fig.suptitle('Two Burn Orbit Raise Solution')
#     ax_u1 = plt.subplot2grid((2, 2), (0, 0))
#     ax_deltav = plt.subplot2grid((2, 2), (1, 0))
#     ax_xy = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
#
#     span = np.linspace(0, 2 * np.pi, 100)
#     ax_xy.plot(np.cos(span), np.sin(span), 'k--', lw=1)
#     ax_xy.plot(3 * np.cos(span), 3 * np.sin(span), 'k--', lw=1)
#     ax_xy.set_xlim(-4.5, 4.5)
#     ax_xy.set_ylim(-4.5, 4.5)
#
#     ax_xy.set_xlabel('x ($R_e$)')
#     ax_xy.set_ylabel('y ($R_e$)')
#
#     ax_u1.set_xlabel('time ($TU$)')
#     ax_u1.set_ylabel('$u_1$ ($deg$)')
#     ax_u1.grid(True)
#
#     ax_deltav.set_xlabel('time ($TU$)')
#     ax_deltav.set_ylabel('${\Delta}v$ ($DU/TU$)')
#     ax_deltav.grid(True)
#
#     t_sol = traj.get_values('time', flat=True)
#     x_sol = traj.get_values('pos_x', flat=True)
#     y_sol = traj.get_values('pos_y', flat=True)
#     dv_sol = traj.get_values('deltav', flat=True)
#     u1_sol = traj.get_values('u1', units='deg', flat=True)
#
#     t_exp = exp_out.get_values('time', flat=True)
#     x_exp = exp_out.get_values('pos_x', flat=True)
#     y_exp = exp_out.get_values('pos_y', flat=True)
#     dv_exp = exp_out.get_values('deltav', flat=True)
#     u1_exp = exp_out.get_values('u1', units='deg', flat=True)
#
#     ax_u1.plot(t_sol, u1_sol, 'ro', ms=3)
#     ax_u1.plot(t_exp, u1_exp, 'b-')
#
#     ax_deltav.plot(t_sol, dv_sol, 'ro', ms=3)
#     ax_deltav.plot(t_exp, dv_exp, 'b-')
#
#     ax_xy.plot(x_sol, y_sol, 'ro', ms=3, label='implicit')
#     ax_xy.plot(x_exp, y_exp, 'b-', label='explicit')
#
#     if show_plots:
#         plt.show()
#
#     return p
#
#


if __name__ == '__main__':
    two_phase_cannonball_problem(optimizer='SLSQP', show_plots=True)
