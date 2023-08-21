import matplotlib

import openmdao.api as om
from openmdao.utils.testing_utils import require_pyoptsparse

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


SHOW_PLOTS = True
# matplotlib.use('Agg')


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
        from dymos.transcriptions.pseudospectral.birkhoff import Birkhoff
        from dymos.transcriptions.pseudospectral.birkhoff_gl import BirkhoffGL
        from dymos.transcriptions.grid_data import BirkhoffGaussLobattoGrid
        # grid = dm.RadauGrid(num_segments=num_segments,
        #                     nodes_per_seg=transcription_order + 1,
        #                     compressed=compressed)
        # t = Birkhoff(grid=grid)

        grid = BirkhoffGaussLobattoGrid(num_segments=num_segments,
                                        nodes_per_seg=transcription_order + 1,
                                        compressed=compressed)
        t = BirkhoffGL(grid=grid)

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    p.model.add_subsystem('traj0', traj)
    traj.add_phase('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(1.8, 10))

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
    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.setup(check=['unconnected_inputs'], force_alloc_complex=force_alloc_complex)

    phase.set_simulate_options(method='RK23')

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 2.0

    if transcription.startswith('shooting'):
        p['traj0.phase0.initial_states:x'] = 0
        p['traj0.phase0.initial_states:y'] = 10
        p['traj0.phase0.initial_states:v'] = 0
    else:
        p['traj0.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj0.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])

    p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
    p['traj0.phase0.parameters:g'] = 9.80665

    dm.run_problem(p, run_driver=run_driver, simulate=False, make_plots=True)

    print(p.get_val('traj0.phase0.timeseries.time')[-1])
    print(p.get_val('traj0.phase0.timeseries.x')[-1])
    print(p.get_val('traj0.phase0.timeseries.y')[-1])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(p.get_val('traj0.phase0.timeseries.x'), p.get_val('traj0.phase0.timeseries.y'))
    plt.show()

    return p


if __name__ == '__main__':
    p = brachistochrone_min_time(transcription='birkhoff', num_segments=1, run_driver=True,
                                 transcription_order=14, compressed=False, optimizer='IPOPT',
                                 solve_segments=False, force_alloc_complex=True)
