from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

OPTIMIZER = 'SLSQP'
SHOW_PLOTS = False


def brachistochrone_min_time(
        transcription='gauss-lobatto', num_segments=8, run_driver=True, test_fix_initial=True,
        top_level_jacobian='csc', compressed=True,
        glm_formulation='solver-based', glm_integrator='GaussLegendre4', force_alloc_complex=False):
    p = Problem(model=Group())

    if OPTIMIZER == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    else:
        p.driver = ScipyOptimizeDriver()

    kwargs = {}
    if transcription != 'glm':
        kwargs['transcription_order'] = 3
    else:
        kwargs['formulation'] = glm_formulation
        kwargs['method_name'] = glm_integrator

    phase = Phase(transcription,
                  ode_class=BrachistochroneODE,
                  num_segments=num_segments,
                  compressed=compressed,
                  **kwargs)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

    if test_fix_initial:
        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)
        phase.add_boundary_constraint('x', loc='final', equals=10.)
        phase.add_boundary_constraint('y', loc='final', equals=5.)
    else:
        phase.add_boundary_constraint('x', loc='initial', equals=0.)
        phase.add_boundary_constraint('x', loc='final', equals=10.)
        phase.add_boundary_constraint('y', loc='initial', equals=10.)
        phase.add_boundary_constraint('y', loc='final', equals=5.)
        phase.add_boundary_constraint('v', loc='initial', equals=0.)

    if transcription != 'glm':
        rate_continuity = True
    else:
        rate_continuity = None

    phase.add_control('theta', units='deg', dynamic=True,
                      rate_continuity=rate_continuity, lower=0.01, upper=179.9)

    phase.add_control('g', units='m/s**2', dynamic=True, opt=False, val=9.80665)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', scaler=10)

    if transcription != 'glm':
        p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(mode='rev', check=True)
    else:
        p.setup(force_alloc_complex=force_alloc_complex)
        p.final_setup()
        p.set_solver_print(level=-1)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    if transcription != 'glm':
        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_disc')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_disc')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_disc')
    else:
        phase.set_values('x', [0, 10])
        phase.set_values('y', [10, 5])
        phase.set_values('v', [0, 9.9])
        phase.set_values('theta', [5, 100.5])

    p.run_model()
    if run_driver:
        p.run_driver()

    # Plot results
    if SHOW_PLOTS:
        exp_out = phase.simulate(times=np.linspace(
            p['phase0.t_initial'], p['phase0.t_initial'] + p['phase0.t_duration'], 50))

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = phase.get_values('x', nodes='all')
        y_imp = phase.get_values('y', nodes='all')

        x_exp = exp_out.get_values('x')
        y_exp = exp_out.get_values('y')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = phase.get_values('time', nodes='all')
        y_imp = phase.get_values('theta_rate2', nodes='all')

        x_exp = exp_out.get_values('time')
        y_exp = exp_out.get_values('theta_rate2')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('theta rate2 (rad/s**2)')
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.show()

    return p
