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

    if transcription == 'runge-kutta':
        transcription = dm.RungeKutta(num_segments=num_segments, compressed=compressed)
    elif transcription == 'gauss-lobatto':
        transcription = dm.GaussLobatto(num_segments=num_segments,
                                        order=transcription_order,
                                        compressed=compressed)
    elif transcription == 'radau-ps':
        transcription = dm.Radau(num_segments=num_segments,
                                 order=transcription_order,
                                 compressed=compressed)

    phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                     transcription=transcription)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

    fix_final = not solve_segments  # can't fix final position if you're solving the segments

    phase.add_state('pos',
                    rate_source=BrachistochroneVectorStatesODE.states['pos']['rate_source'],
                    units=BrachistochroneVectorStatesODE.states['pos']['units'],
                    shape=BrachistochroneVectorStatesODE.states['pos']['shape'],
                    fix_initial=True, fix_final=fix_final, solve_segments=solve_segments)
    #
    phase.add_state('v',
                    rate_source=BrachistochroneVectorStatesODE.states['v']['rate_source'],
                    targets=BrachistochroneVectorStatesODE.states['v']['targets'],
                    units=BrachistochroneVectorStatesODE.states['v']['units'],
                    fix_initial=True, fix_final=False, solve_segments=solve_segments)
    #
    phase.add_control('theta',
                      targets=BrachistochroneVectorStatesODE.parameters['theta']['targets'],
                      continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_design_parameter('g',
                               targets=BrachistochroneVectorStatesODE.parameters['g']['targets'],
                               opt=False, units='m/s**2', val=9.80665)

    if not fix_final:
        phase.add_boundary_constraint('pos', loc='final', units='m', shape=(2,), equals=[10, 5])

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', scaler=10)

    p.model.linear_solver = om.DirectSolver()
    p.setup(check=True, force_alloc_complex=force_alloc_complex)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 1.8016

    pos0 = [0, 10]
    posf = [10, 5]

    p['phase0.states:pos'] = phase.interpolate(ys=[pos0, posf], nodes='state_input')
    p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
    p['phase0.design_parameters:g'] = 9.80665

    p.run_model()
    if run_driver:
        p.run_driver()

    return p


if __name__ == '__main__':
    p = brachistochrone_min_time(transcription='radau-ps', num_segments=5, run_driver=True,
                                 transcription_order=5, compressed=False, optimizer='SNOPT',
                                 solve_segments=True, force_alloc_complex=True, dynamic_simul_derivs=True)

    # p.model.list_outputs(print_arrays=True)

    # p.list_problem_vars(print_arrays=True, desvar_opts=['indices'])

    # for key, options in p.model.get_design_vars(get_sizes=True).items():
    #     print(key)
    #     print(options)

    # import numpy as np
    # with np.printoptions(linewidth=1024, edgeitems=500):
    #     p.check_totals(wrt='phase0.states:pos', method='cs', compact_print=True)

    import matplotlib.pyplot as plt
    plt.plot(p.get_val('phase0.timeseries.time')[:, 0], p.get_val('phase0.timeseries.states:pos')[:, 0], 'ro')
    plt.show()
