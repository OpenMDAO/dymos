import matplotlib
import matplotlib.pyplot as plt

import openmdao.api as om

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

SHOW_PLOTS = True


def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=True, optimizer='SLSQP'):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)
    elif transcription == 'runge-kutta':
        t = dm.RungeKutta(num_segments=num_segments,
                          order=transcription_order,
                          compressed=compressed)
    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    traj.add_phase('phase0', phase)

    p.model.add_subsystem('traj0', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                    units=BrachistochroneODE.states['x']['units'],
                    fix_initial=True, fix_final=False, solve_segments=False)
    phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                    units=BrachistochroneODE.states['y']['units'],
                    fix_initial=True, fix_final=False, solve_segments=False)
    phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                    units=BrachistochroneODE.states['v']['units'],
                    fix_initial=True, fix_final=False, solve_segments=False)

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', units='m/s**2', val=9.80665)

    phase.add_timeseries('timeseries2',
                         transcription=dm.Radau(num_segments=num_segments*5,
                                                order=transcription_order,
                                                compressed=compressed),
                         subset='control_input')

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=['unconnected_inputs'])

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 2.0

    p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
    p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
    p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
    p['traj0.phase0.parameters:g'] = 9.80665

    p.final_setup()
    return p

if __name__ == '__main__':
    p = brachistochrone_min_time(transcription='radau-ps', num_segments=8,
                                 transcription_order=3, compressed=True,
                                 optimizer='SLSQP')
