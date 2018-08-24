from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, SqliteRecorder,\
    pyOptSparseDriver

from dymos import Phase, Trajectory, load_simulation_results
from dymos.examples.cannonball.cannonball_ode import CannonballODE

from dymos.examples.cannonball.size_comp import CannonballSizeComp


def two_phase_cannonball_problem(transcription='radau-ps', optimizer='SLSQP',
                                 transcription_order=3, compressed=True, show_plots=False):

    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.options['dynamic_simul_derivs'] = True

    external_params = p.model.add_subsystem('external_params', IndepVarComp())

    external_params.add_output('radius', val=0.10, units='m')
    external_params.add_output('dens', val=7.87, units='g/cm**3')

    external_params.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10)

    p.model.add_subsystem('size_comp', CannonballSizeComp())

    traj = p.model.add_subsystem('traj', Trajectory())

    # First Phase (ascent)
    ascent = Phase(transcription,
                   ode_class=CannonballODE,
                   num_segments=5,
                   transcription_order=transcription_order,
                   compressed=compressed)

    ascent = traj.add_phase('ascent', ascent)

    # All initial states except flight path angle are fixed
    # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
    ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100)
    ascent.set_state_options('r', fix_initial=True, fix_final=False)
    ascent.set_state_options('h', fix_initial=True, fix_final=False)
    ascent.set_state_options('gam', fix_initial=False, fix_final=True)
    ascent.set_state_options('v', fix_initial=False, fix_final=False)

    # Limit the muzzle energy
    ascent.add_boundary_constraint('kinetic_energy.ke', loc='initial', units='kg*m**2/s**2',
                                   upper=400000, lower=0, ref=100000)

    # Second Phase (descent)
    descent = Phase(transcription,
                    ode_class=CannonballODE,
                    num_segments=5,
                    transcription_order=transcription_order,
                    compressed=compressed)

    traj.add_phase('descent', descent)

    # All initial states and time are free (they will be linked to the final states of ascent.
    # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
    descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100), duration_ref=100)
    descent.set_state_options('r', fix_initial=False, fix_final=False)
    descent.set_state_options('h', fix_initial=False, fix_final=True)
    descent.set_state_options('gam', fix_initial=False, fix_final=False)
    descent.set_state_options('v', fix_initial=False, fix_final=False)

    descent.add_objective('r', loc='final', scaler=-1.0)

    # Add internally-managed design parameters to the trajectory.
    traj.add_design_parameter('CD', val=0.5, units=None, opt=False)
    traj.add_design_parameter('CL', val=0.0, units=None, opt=False)
    traj.add_design_parameter('T', val=0.0, units='N', opt=False)
    traj.add_design_parameter('alpha', val=0.0, units='deg', opt=False)

    # Add externally-provided design parameters to the trajectory.
    traj.add_input_parameter('mass', targets={'ascent': 'm', 'descent': 'm'}, val=1.0, units='kg')
    traj.add_input_parameter('S', val=0.005, units='m**2')

    # Link Phases (link time and all state variables)
    traj.link_phases(phases=['ascent', 'descent'], vars=['*'])

    # Issue Connnections
    p.model.connect('external_params.radius', 'size_comp.radius')
    p.model.connect('external_params.dens', 'size_comp.dens')

    p.model.connect('size_comp.mass', 'traj.input_parameters:mass')
    p.model.connect('size_comp.S', 'traj.input_parameters:S')

    # Finish Problem Setup
    p.model.options['assembled_jac_type'] = 'csc'
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.driver.add_recorder(SqliteRecorder('ex_two_phase_cannonball.db'))

    p.setup(check=True)

    # Set Initial Guesses
    p.set_val('external_params.radius', 0.05, units='m')
    p.set_val('external_params.dens', 7.87, units='g/cm**3')

    p.set_val('traj.design_parameters:CD', 0.1)
    p.set_val('traj.design_parameters:CL', 0.0)
    p.set_val('traj.design_parameters:T', 0.0)

    p.set_val('traj.ascent.t_initial', 0.0)
    p.set_val('traj.ascent.t_duration', 10.0)

    p.set_val('traj.ascent.states:r', ascent.interpolate(ys=[0, 100], nodes='state_input'))
    p.set_val('traj.ascent.states:h', ascent.interpolate(ys=[0, 100], nodes='state_input'))
    p.set_val('traj.ascent.states:v', ascent.interpolate(ys=[200, 150], nodes='state_input'))
    p.set_val('traj.ascent.states:gam', ascent.interpolate(ys=[25, 0], nodes='state_input'),
              units='deg')

    p.set_val('traj.descent.t_initial', 10.0)
    p.set_val('traj.descent.t_duration', 10.0)

    p.set_val('traj.descent.states:r', descent.interpolate(ys=[100, 200], nodes='state_input'))
    p.set_val('traj.descent.states:h', descent.interpolate(ys=[100, 0], nodes='state_input'))
    p.set_val('traj.descent.states:v', descent.interpolate(ys=[150, 200], nodes='state_input'))
    p.set_val('traj.descent.states:gam', descent.interpolate(ys=[0, -45], nodes='state_input'),
              units='deg')

    p.run_model()

    p.run_driver()

    exp_out = traj.simulate(times=100, record_file='ex_two_phase_cannonball_sim.db')

    exp_out_loaded = load_simulation_results('ex_two_phase_cannonball_sim.db')

    plt.plot(traj.get_values('r')['ascent'],
             traj.get_values('h')['ascent'],
             'bo')

    plt.plot(traj.get_values('r')['descent'],
             traj.get_values('h')['descent'],
             'ro')

    plt.plot(exp_out.get_values('r')['ascent'],
             exp_out.get_values('h')['ascent'],
             'b--')

    plt.plot(exp_out.get_values('r')['descent'],
             exp_out.get_values('h')['descent'],
             'r--')

    plt.plot(exp_out_loaded.get_values('r')['ascent'],
             exp_out_loaded.get_values('h')['ascent'],
             'm--')

    plt.plot(exp_out_loaded.get_values('r')['descent'],
             exp_out_loaded.get_values('h')['descent'],
             'c--')

    plt.figure()

    plt.plot(traj.get_values('time')['ascent'],
             traj.get_values('mass')['ascent'],
             'bo')

    plt.plot(traj.get_values('time')['descent'],
             traj.get_values('mass')['descent'],
             'ro')

    plt.plot(exp_out.get_values('time')['ascent'],
             exp_out.get_values('mass')['ascent'],
             'b--')

    plt.plot(exp_out.get_values('time')['descent'],
             exp_out.get_values('mass')['descent'],
             'r--')

    plt.plot(exp_out_loaded.get_values('time')['ascent'],
             exp_out_loaded.get_values('mass')['ascent'],
             'b--')

    plt.plot(exp_out_loaded.get_values('time')['descent'],
             exp_out_loaded.get_values('mass')['descent'],
             'r--')

    plt.figure()

    plt.suptitle('Kinetic Energy vs Time')

    plt.plot(traj.get_values('time')['ascent'],
             traj.get_values('kinetic_energy.ke')['ascent'],
             'bo')

    plt.plot(traj.get_values('time')['descent'],
             traj.get_values('kinetic_energy.ke')['descent'],
             'ro')

    plt.plot(exp_out.get_values('time')['ascent'],
             exp_out.get_values('kinetic_energy.ke')['ascent'],
             'b--')

    plt.plot(exp_out.get_values('time')['descent'],
             exp_out.get_values('kinetic_energy.ke')['descent'],
             'r--')

    plt.plot(exp_out_loaded.get_values('time')['ascent'],
             exp_out_loaded.get_values('kinetic_energy.ke')['ascent'],
             'b--')

    plt.plot(exp_out_loaded.get_values('time')['descent'],
             exp_out_loaded.get_values('kinetic_energy.ke')['descent'],
             'r--')

    plt.figure()

    plt.plot(traj.get_values('time')['ascent'],
             traj.get_values('gam', units='deg')['ascent'],
             'bo')
    plt.plot(traj.get_values('time')['descent'],
             traj.get_values('gam', units='deg')['descent'],
             'ro')

    plt.plot(exp_out.get_values('time')['ascent'],
             exp_out.get_values('gam', units='deg')['ascent'],
             'b--')

    plt.plot(exp_out.get_values('time')['descent'],
             exp_out.get_values('gam', units='deg')['descent'],
             'r--')

    plt.show()


if __name__ == '__main__':
    two_phase_cannonball_problem(optimizer='SLSQP', show_plots=True)
