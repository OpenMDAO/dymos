import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import require_pyoptsparse

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@require_pyoptsparse(optimizer='SLSQP')
def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=True, optimizer='SLSQP', run_driver=True, force_alloc_complex=False,
                             solve_segments=False):
    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring(tol=1.0E-12)

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)
    elif transcription == 'shooting-gauss-lobatto':
        grid = dm.GaussLobattoGrid(num_segments=num_segments,
                                   nodes_per_seg=transcription_order,
                                   compressed=compressed)
        t = dm.ExplicitShooting(grid=grid)
    elif transcription == 'shooting-radau':
        grid = dm.RadauGrid(num_segments=num_segments,
                            nodes_per_seg=transcription_order + 1,
                            compressed=compressed)
        t = dm.ExplicitShooting(grid=grid)
    elif transcription == 'birkhoff':
        grid = dm.BirkhoffGrid(num_nodes=transcription_order + 1,
                               grid_type='cgl')
        t = dm.Birkhoff(grid=grid)
        # phase = dm.ImplicitPhase(ode_class=BrachistochroneODE, num_nodes=11)

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    p.model.add_subsystem('traj0', traj)
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10))

    phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=solve_segments)
    phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=solve_segments)

    # Note that by omitting the targets here Dymos will automatically attempt to connect
    # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=solve_segments)

    phase.add_control('theta',
                      continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', targets=['g'], units='m/s**2')

    phase.add_timeseries('timeseries2',
                         transcription=dm.Radau(num_segments=num_segments*5,
                                                order=transcription_order,
                                                compressed=compressed),
                         subset='control_input')

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)
    phase.add_path_constraint('y', lower=0, upper=20)
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    phase.add_timeseries_output('check')

    p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

    phase.set_simulate_options(method='RK23')

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 2.0

    if transcription.startswith('shooting') or transcription == 'birkhoff':
        p['traj0.phase0.initial_states:x'] = 0
        p['traj0.phase0.initial_states:y'] = 10
        p['traj0.phase0.initial_states:v'] = 0
    else:
        p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])

    p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
    p['traj0.phase0.parameters:g'] = 9.80665

    dm.run_problem(p, run_driver=run_driver, simulate=True, make_plots=True,
                   simulate_kwargs={'times_per_seg': 100})

    return p


if __name__ == '__main__':

    with dm.options.temporary(include_check_partials=True):
        p = brachistochrone_min_time(transcription='birkhoff', num_segments=1, run_driver=True,
                                     transcription_order=19, compressed=False, optimizer='SLSQP',
                                     solve_segments=False, force_alloc_complex=True)
