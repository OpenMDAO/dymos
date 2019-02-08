from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

SHOW_PLOTS = True


def brachistochrone_min_time(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
                             run_driver=True, top_level_jacobian='csc', compressed=True,
                             sim_record='brach_min_time_sim.db', optimizer='SLSQP',
                             dynamic_simul_derivs=True):
    p = Problem(model=Group())

    # if optimizer == 'SNOPT':
    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    #     p.driver.opt_settings['Major iterations limit'] = 100
    #     p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
    #     p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
    #     p.driver.opt_settings['iSumm'] = 6
    # else:
    #     p.driver = ScipyOptimizeDriver()

    p.driver.options['dynamic_simul_derivs'] = dynamic_simul_derivs

    phase = Phase(transcription,
                  ode_class=BrachistochroneODE,
                  num_segments=num_segments,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

    phase.set_state_options('x', fix_initial=True, fix_final=True)
    phase.set_state_options('y', fix_initial=True, fix_final=True)
    phase.set_state_options('v', fix_initial=True, fix_final=False)

    phase.add_control('theta', continuity=True, rate_continuity=True,
                      units='deg', lower=0.01, upper=179.9)

    phase.add_input_parameter('g', units='m/s**2', val=9.80665)

    # Minimize time at the end of the phase
    phase.add_objective('time_phase', loc='final', scaler=10)

    p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
    p.model.linear_solver = DirectSolver(assemble_jac=True)
    p.setup(check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
    p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
    p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
    p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100], nodes='control_input')
    p['phase0.input_parameters:g'] = 9.80665

    p.run_model()
    if run_driver:
        p.run_driver()

    # Plot results
    if SHOW_PLOTS:
        exp_out = phase.simulate(times=50, record_file=sim_record)

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')

        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = p.get_val('phase0.timeseries.time_phase')
        y_imp = p.get_val('phase0.timeseries.controls:theta')

        x_exp = exp_out.get_val('phase0.timeseries.time_phase')
        y_exp = exp_out.get_val('phase0.timeseries.controls:theta')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('theta (rad)')
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.show()

    return p


if __name__ == '__main__':
    brachistochrone_min_time(transcription='gauss-lobatto', num_segments=10, run_driver=True,
                             top_level_jacobian='csc', transcription_order=3, compressed=True,
                             optimizer='SNOPT')
    # brachistochrone_min_time(transcription='radau-ps', num_segments=10, run_driver=True,
    #                          top_level_jacobian='csc', transcription_order=3, compressed=True,
    #                          optimizer='SNOPT')
