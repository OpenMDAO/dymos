from __future__ import print_function, division, absolute_import

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import DeprecatedPhaseFactory
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE
from dymos.phase import Phase
from dymos.transcriptions import GaussLobatto, Radau

SHOW_PLOTS = True


def brachistochrone_min_time(num_segments=8, transcription_order=3, compressed=True):
    p = Problem(model=Group())

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Verify level'] = 3
    p.driver.options['dynamic_simul_derivs'] = True

    phase = Phase(ode_class=BrachistochroneODE,
                  transcription=Radau(num_segments=num_segments,
                                             order=transcription_order,
                                             compressed=True))

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    SOLVE_SEGS = True
    phase.set_state_options('x', fix_initial=True, fix_final=False, solve_segments=SOLVE_SEGS)
    phase.set_state_options('y', fix_initial=True, fix_final=False, solve_segments=SOLVE_SEGS)
    phase.set_state_options('v', fix_initial=True, fix_final=False, solve_segments=SOLVE_SEGS)

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_input_parameter('g', units='m/s**2', val=9.80665)

    phase.add_boundary_constraint('x', loc='final', equals=10)
    phase.add_boundary_constraint('y', loc='final', equals=5)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', scaler=10)

    p.model.linear_solver = DirectSolver()
    p.setup(check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 1.806

    p.set_val('phase0.states:x', phase.interpolate(ys=[0, 10], nodes='state_input'))
    p.set_val('phase0.states:y', phase.interpolate(ys=[10, 5], nodes='state_input'))
    p.set_val('phase0.states:v',  phase.interpolate(ys=[0, 9.9], nodes='state_input'))
    p.set_val('phase0.controls:theta',  phase.interpolate(ys=[0.01, 100.508], nodes='control_input'))
    p.set_val('phase0.input_parameters:g',  9.80665)

    p.run_driver()

    from openmdao.api import view_model
    view_model(p.model)

    # print(p.get_val('phase0.timeseries.controls:theta'))
    # print(p.get_val('phase0.timeseries.control_rates:theta_rate'))
    print(p.get_val('phase0.timeseries.states:x')[-1])
    print(p.get_val('phase0.timeseries.states:y')[-1])
    print(p.get_val('phase0.timeseries.states:v')[-1])

    print(p.get_val('phase0.final_boundary_constraints.final_value:x'))
    print(p.get_val('phase0.final_boundary_constraints.final_value:y'))

    # Plot results
    if SHOW_PLOTS:
        # exp_out = phase.simulate()

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        # x_exp = exp_out.get_val('phase0.timeseries.states:x')
        # y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        # ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.time_phase')
        y_imp = p.get_val('phase0.timeseries.controls:theta')

        # x_exp = exp_out.get_val('phase0.timeseries.time_phase')
        # y_exp = exp_out.get_val('phase0.timeseries.controls:theta')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        # ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('theta (rad)')
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.show()

    return p


if __name__ == '__main__':
    brachistochrone_min_time(num_segments=10, transcription_order=3, compressed=True)
