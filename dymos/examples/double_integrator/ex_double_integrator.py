from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE


def double_integrator_direct_collocation(transcription='gauss-lobatto', top_level_jacobian='csc',
                                         compressed=True):
    p = Problem(model=Group())
    p.driver = pyOptSparseDriver()
    p.driver.options['dynamic_simul_derivs'] = True
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Major iterations limit'] = 100

    phase = Phase(transcription,
                  ode_class=DoubleIntegratorODE,
                  num_segments=30,
                  transcription_order=3,
                  compressed=compressed)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(fix_initial=True, fix_duration=True)

    phase.set_state_options('x', fix_initial=True)
    phase.set_state_options('v', fix_initial=True, fix_final=True)

    phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                      rate2_continuity=False, lower=-1.0, upper=1.0)

    # Maximize distance travelled in one second.
    phase.add_objective('x', loc='final', scaler=-1)

    p.model.linear_solver = DirectSolver(assemble_jac=True)
    p.model.options['assembled_jac_type'] = top_level_jacobian.lower()

    p.setup(check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 1.0

    p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
    p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
    p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

    p.run_driver()

    return p


if __name__ == '__main__':  # pragma: no cover
    prob = double_integrator_direct_collocation(transcription='radau-ps', compressed=True)

    import matplotlib.pyplot as plt
    plt.plot(prob.model.phase0.get_values('time'), prob.model.phase0.get_values('x'), 'ro')
    plt.plot(prob.model.phase0.get_values('time'), prob.model.phase0.get_values('v'), 'bo')
    plt.plot(prob.model.phase0.get_values('time'), prob.model.phase0.get_values('u'), 'go')

    expout = prob.model.phase0.simulate(times=100)
    plt.plot(expout.get_values('time'), expout.get_values('x'), 'r-')
    plt.plot(expout.get_values('time'), expout.get_values('v'), 'b-')
    plt.plot(expout.get_values('time'), expout.get_values('u'), 'g-')

    plt.show()
