import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import openmdao.api as om

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

SHOW_PLOTS = True


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
    elif transcription == 'runge-kutta':
        t = dm.RungeKutta(num_segments=num_segments,
                          order=transcription_order,
                          compressed=compressed)
    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)
    p.model.add_subsystem('traj0', traj)
    traj.add_phase('phase0', phase)

    # p.model.add_subsystem('traj0', traj)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.add_state('x', rate_source=BrachistochroneODE.states['x']['rate_source'],
                    fix_initial=True, fix_final=False, solve_segments=solve_segments)
    phase.add_state('y', rate_source=BrachistochroneODE.states['y']['rate_source'],
                    fix_initial=True, fix_final=False, solve_segments=solve_segments)

    # Note that by omitting the targets here Dymos will automatically attempt to connect
    # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
    phase.add_state('v', rate_source=BrachistochroneODE.states['v']['rate_source'],
                    fix_initial=True, fix_final=False, solve_segments=solve_segments)

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

    p['traj0.phase0.t_initial'] = 0.0
    p['traj0.phase0.t_duration'] = 2.0

    p['traj0.phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
    p['traj0.phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
    p['traj0.phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    p['traj0.phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
    p['traj0.phase0.parameters:g'] = 9.80665

    dm.run_problem(p, run_driver=run_driver)

    # Plot results
    if SHOW_PLOTS:
        exp_out = traj.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('traj0.phase0.timeseries.states:x')
        y_imp = p.get_val('traj0.phase0.timeseries.states:y')

        x_exp = exp_out.get_val('traj0.phase0.timeseries.states:x')
        y_exp = exp_out.get_val('traj0.phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('traj0.phase0.timeseries.time_phase')
        y_imp = p.get_val('traj0.phase0.timeseries.controls:theta')

        x_exp = exp_out.get_val('traj0.phase0.timeseries.time_phase')
        y_exp = exp_out.get_val('traj0.phase0.timeseries.controls:theta')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('theta (rad)')
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.show()

    return p


if __name__ == '__main__':
    p = brachistochrone_min_time(transcription='radau-ps', num_segments=3, run_driver=True,
                                 transcription_order=5, compressed=False, optimizer='SNOPT',
                                 solve_segments=False, force_alloc_complex=True)
