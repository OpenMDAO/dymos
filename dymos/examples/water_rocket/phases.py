import numpy as np

import dymos as dm
from dymos.examples.water_rocket.water_propulsion_ode import WaterPropulsionODE


def new_propelled_ascent_phase(transcription):
    propelled_ascent = dm.Phase(ode_class=WaterPropulsionODE,
                                transcription=transcription)

    # All initial states except flight path angle and water volume are fixed
    # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
    # Final water volume is fixed (we will set it to zero so that phase ends when bottle empties)
    propelled_ascent.set_time_options(
        fix_initial=True, duration_bounds=(0.001, 0.5), duration_ref=0.1, units='s')
    propelled_ascent.add_state('r', units='m', rate_source='eom.r_dot',
                               fix_initial=True, fix_final=False, ref=100.0, defect_ref=100.0)
    propelled_ascent.add_state('h', units='m', rate_source='eom.h_dot', targets=['atmos.h'],
                               fix_initial=True, fix_final=False, ref=100.0, defect_ref=100.0)
    propelled_ascent.add_state('gam', units='deg', rate_source='eom.gam_dot', targets=['eom.gam'],
                               fix_initial=False, fix_final=False, lower=0, upper=85.0, ref=90)
    propelled_ascent.add_state('v', units='m/s', rate_source='eom.v_dot', targets=['dynamic_pressure.v', 'eom.v'],
                               fix_initial=True, fix_final=False, ref=100, defect_ref=100, lower=0.01)

    propelled_ascent.add_state('p', units='bar', rate_source='water_engine.pdot',
                               targets=['water_engine.p'], fix_initial=True, fix_final=False,
                               lower=1.02)
    propelled_ascent.add_state('V_w', units='L', rate_source='water_engine.Vdot',
                               targets=['water_engine.V_w', 'mass_adder.V_w'],
                               fix_initial=False, fix_final=True, ref=10, defect_ref=10,
                               lower=0)

    propelled_ascent.add_parameter(
        'S', targets=['aero.S'], units='m**2')
    propelled_ascent.add_parameter(
        'm_empty', targets=['mass_adder.m_empty'], units='kg')
    propelled_ascent.add_parameter(
        'V_b', targets=['water_engine.V_b'], units='m**3')

    propelled_ascent.add_timeseries_output('water_engine.F', 'T', units='N')

    return propelled_ascent


def new_ballistic_ascent_phase(transcription):
    ballistic_ascent = dm.Phase(ode_class=WaterPropulsionODE, transcription=transcription,
                                ode_init_kwargs={'ballistic': True})

    # All initial states are free (they will be  linked to the final stages of propelled_ascent).
    # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
    ballistic_ascent.set_time_options(
        fix_initial=False, initial_bounds=(0.001, 1), duration_bounds=(0.001, 10),
        duration_ref=1, units='s')

    ballistic_ascent.add_state('r', units='m', rate_source='eom.r_dot', fix_initial=False,
                               fix_final=False, ref=100.0, defect_ref=100.0)
    ballistic_ascent.add_state('h', units='m', rate_source='eom.h_dot', targets=['atmos.h'],
                               fix_initial=False, fix_final=False, ref=100.0, defect_ref=100.0)
    ballistic_ascent.add_state('gam', units='deg', rate_source='eom.gam_dot', targets=['eom.gam'],
                               fix_initial=False, fix_final=True, upper=89, ref=90, defect_ref=90)
    ballistic_ascent.add_state('v', units='m/s', rate_source='eom.v_dot',
                               targets=['dynamic_pressure.v', 'eom.v'], fix_initial=False,
                               fix_final=False, ref=100, defect_ref=100)

    ballistic_ascent.add_parameter('S', targets=['aero.S'], units='m**2')
    ballistic_ascent.add_parameter('m_empty', targets=['eom.m'], units='kg')

    return ballistic_ascent


def new_descent_phase(transcription):
    descent = dm.Phase(ode_class=WaterPropulsionODE, transcription=transcription,
                       ode_init_kwargs={'ballistic': True})

    # All initial states and time are free (they will be linked to the final states of ballistic_ascent).
    # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
    descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                             duration_ref=10, units='s')

    descent.add_state('r', units='m', rate_source='eom.r_dot', fix_initial=False, fix_final=False)
    descent.add_state('h', units='m', rate_source='eom.h_dot', targets=['atmos.h'],
                      fix_initial=False, fix_final=True)
    descent.add_state('gam', units='deg', rate_source='eom.gam_dot', targets=['eom.gam'],
                      fix_initial=False, fix_final=False)
    descent.add_state('v', units='m/s', rate_source='eom.v_dot',
                      targets=['dynamic_pressure.v', 'eom.v'],
                      fix_initial=False, fix_final=False,
                      ref=100, defect_ref=100)

    descent.add_parameter('S', targets=['aero.S'], units='m**2')
    descent.add_parameter('mass', targets=['eom.m'], units='kg')

    return descent


def new_water_rocket_trajectory(objective):
    tx_prop = dm.Radau(num_segments=50, order=3, compressed=True)
    tx_bal = dm.Radau(num_segments=10, order=3, compressed=True)
    tx_desc = dm.Radau(num_segments=10, order=3, compressed=True)
    traj = dm.Trajectory()

    # Add phases to trajectory
    propelled_ascent = traj.add_phase('propelled_ascent', new_propelled_ascent_phase(tx_prop))
    ballistic_ascent = traj.add_phase('ballistic_ascent', new_ballistic_ascent_phase(tx_bal))
    descent = traj.add_phase('descent', new_descent_phase(tx_desc))

    # Link phases
    traj.link_phases(phases=['propelled_ascent', 'ballistic_ascent'], vars=['*'])
    traj.link_phases(phases=['ballistic_ascent', 'descent'], vars=['*'])

    # Set objective function
    if objective == 'height':
        ballistic_ascent.add_objective('h', loc='final', ref=-1.0)
    elif objective == 'range':
        descent.add_objective('r', loc='final', ref=-100.0)
    else:
        raise ValueError(f"objective='{objective}' is not defined. Try using 'height' or 'range'")

    # Add design parameters to the trajectory.
    traj.add_parameter('CD',
                       targets={'propelled_ascent': ['aero.CD'],
                                'ballistic_ascent': ['aero.CD'],
                                'descent': ['aero.CD']},
                       val=0.3450, units=None, opt=False)
    traj.add_parameter('CL',
                       targets={'propelled_ascent': ['aero.CL'],
                                'ballistic_ascent': ['aero.CL'],
                                'descent': ['aero.CL']},
                       val=0.0, units=None, opt=False)
    traj.add_parameter('T',
                       targets={'ballistic_ascent': ['eom.T'],
                                'descent': ['eom.T']},
                       val=0.0, units='N', opt=False)
    traj.add_parameter('alpha',
                       targets={'propelled_ascent': ['eom.alpha'],
                                'ballistic_ascent': ['eom.alpha'],
                                'descent': ['eom.alpha']},
                       val=0.0, units='deg', opt=False)

    traj.add_parameter('m_empty', units='kg', val=0.15,
                       targets={'propelled_ascent': 'm_empty',
                                'ballistic_ascent': 'm_empty',
                                'descent': 'mass'},
                       lower=0.001, upper=1, ref=0.1,
                       opt=False)
    traj.add_parameter('V_b', units='m**3', val=2e-3,
                       targets={'propelled_ascent': 'V_b'},
                       opt=False)

    traj.add_parameter('S', units='m**2', val=np.pi*106e-3**2/4, opt=False)
    traj.add_parameter('A_out', units='m**2', val=np.pi*22e-3**2/4.,
                       targets={'propelled_ascent': ['water_engine.A_out']},
                       opt=False)
    traj.add_parameter('k', units=None, val=1.2, opt=False,
                       targets={'propelled_ascent': ['water_engine.k']})

    # Make parameter m_empty a design variable
    traj.set_parameter_options('m_empty', opt=True)

    return traj, {'propelled_ascent': propelled_ascent,
                  'ballistic_ascent': ballistic_ascent,
                  'descent': descent}


def set_sane_initial_guesses(problem, phases):
    p = problem
    # Set Initial Guesses
    p.set_val('traj.propelled_ascent.t_initial', 0.0)
    p.set_val('traj.propelled_ascent.t_duration', 0.3)

    p.set_val('traj.propelled_ascent.states:r',
              phases['propelled_ascent'].interp('r', [0, 3]))
    p.set_val('traj.propelled_ascent.states:h',
              phases['propelled_ascent'].interp('h', [0, 10]))
    # Set initial value for velocity as non-zero to avoid undefined EOM
    p.set_val('traj.propelled_ascent.states:v',
              phases['propelled_ascent'].interp('v', [0.1, 100]))
    p.set_val('traj.propelled_ascent.states:gam',
              phases['propelled_ascent'].interp('gam', [80, 80]),
              units='deg')
    p.set_val('traj.propelled_ascent.states:V_w',
              phases['propelled_ascent'].interp('V_w', [9, 0]),
              units='L')
    p.set_val('traj.propelled_ascent.states:p',
              phases['propelled_ascent'].interp('p', [6.5, 3.5]),
              units='bar')

    p.set_val('traj.ballistic_ascent.t_initial', 0.3)
    p.set_val('traj.ballistic_ascent.t_duration', 5)

    p.set_val('traj.ballistic_ascent.states:r',
              phases['ballistic_ascent'].interp('r', [0, 10]))
    p.set_val('traj.ballistic_ascent.states:h',
              phases['ballistic_ascent'].interp('h', [10, 100]))
    p.set_val('traj.ballistic_ascent.states:v',
              phases['ballistic_ascent'].interp('v', [60, 20]))
    p.set_val('traj.ballistic_ascent.states:gam',
              phases['ballistic_ascent'].interp('gam', [80, 0]),
              units='deg')

    p.set_val('traj.descent.t_initial', 10.0)
    p.set_val('traj.descent.t_duration', 10.0)

    p.set_val('traj.descent.states:r',
              phases['descent'].interp('r', [10, 20]))
    p.set_val('traj.descent.states:h',
              phases['descent'].interp('h', [10, 0]))
    p.set_val('traj.descent.states:v',
              phases['descent'].interp('v', [20, 60]))
    p.set_val('traj.descent.states:gam',
              phases['descent'].interp('gam', [0, -45]),
              units='deg')
