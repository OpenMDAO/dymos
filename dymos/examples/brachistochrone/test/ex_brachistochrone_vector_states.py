import openmdao.api as om
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_vector_states_ode \
    import BrachistochroneVectorStatesODE


def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             compressed=True, optimizer='SLSQP',
                             dynamic_simul_derivs=True, force_alloc_complex=False,
                             solve_segments=False, run_driver=True):
    p = om.Problem(model=om.Group())

    if optimizer == 'SNOPT':
        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['iSumm'] = 6
    else:
        p.driver = om.ScipyOptimizeDriver()

    if dynamic_simul_derivs:
        p.driver.declare_coloring()

    if transcription == 'gauss-lobatto':
        transcription = dm.GaussLobatto(num_segments=num_segments,
                                        order=transcription_order,
                                        compressed=compressed)
        fix_final = not solve_segments
    elif transcription == 'radau-ps':
        transcription = dm.Radau(num_segments=num_segments,
                                 order=transcription_order,
                                 compressed=compressed)
        fix_final = not solve_segments

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                     transcription=transcription)
    traj.add_phase('phase0', phase)

    p.model.add_subsystem('traj0', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

    # can't fix final position if you're solving the segments

    phase.add_state('pos', fix_initial=True, fix_final=fix_final, solve_segments=solve_segments, ref=[1, 1])
    #
    phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=solve_segments)
    #
    phase.add_control('theta',
                      continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_parameter('g', opt=False, units='m/s**2', val=9.80665)

    if not fix_final:
        phase.add_boundary_constraint('pos', loc='final', equals=[10, 5])

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 1.8016

    pos0 = [0, 10]
    posf = [10, 5]

    p['traj0.phase0.states:pos'] = phase.interp('pos', [pos0, posf])
    p['traj0.phase0.states:v'] = phase.interp('v', [0, 9.9])
    p['traj0.phase0.controls:theta'] = phase.interp('theta', [5, 100])
    p['traj0.phase0.parameters:g'] = 9.80665

    p.run_model()
    if run_driver:
        p.run_driver()

    return p


if __name__ == '__main__':
    p = brachistochrone_min_time(transcription='radau-ps', num_segments=5, run_driver=True,
                                 transcription_order=5, compressed=False, optimizer='SLSQP',
                                 solve_segments=False, force_alloc_complex=True, dynamic_simul_derivs=True)
