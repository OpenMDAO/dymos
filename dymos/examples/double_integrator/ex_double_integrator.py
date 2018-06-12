from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE


def double_integrator_direct_collocation(transcription='gauss-lobatto', top_level_jacobian='csc',
                                         optimizer='SLSQP', compressed=True):
    p = Problem(model=Group())

    if optimizer == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = optimizer
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    else:
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

    phase = Phase(transcription,
                  ode_class=DoubleIntegratorODE,
                  num_segments=20,
                  transcription_order=3,
                  compressed=compressed)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(1.0, 1.0))

    phase.set_state_options('x', fix_initial=True)
    phase.set_state_options('v', fix_initial=True, fix_final=True)

    phase.add_control('u', units='m/s**2', scaler=0.01, continuity=True, rate_continuity=False,
                      rate2_continuity=False, lower=-1.0, upper=1.0)

    # Maximize distance travelled in one second.
    phase.add_objective('x', loc='final', scaler=-1)

    p.model.options['assembled_jac_type'] = top_level_jacobian.lower()
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(mode='fwd', check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 1.0

    p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_disc')
    p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_disc')
    p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_disc')

    p.run_driver()

    return p


if __name__ == '__main__':
    prob = double_integrator_direct_collocation(transcription='gauss-lobatto', optimizer='SLSQP',
                                                show_plots=True, compressed=False)
