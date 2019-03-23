from __future__ import print_function, division, absolute_import

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import DeprecatedPhaseFactory
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE


def double_integrator_direct_collocation(transcription='gauss-lobatto', compressed=True):
    p = Problem(model=Group())
    p.driver = pyOptSparseDriver()
    p.driver.options['dynamic_simul_derivs'] = True

    phase = DeprecatedPhaseFactory(transcription,
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

    p.model.linear_solver = DirectSolver()

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

    time = prob.get_val('phase0.timeseries.time')
    x = prob.get_val('phase0.timeseries.states:x')
    v = prob.get_val('phase0.timeseries.states:v')
    u = prob.get_val('phase0.timeseries.controls:u')

    plt.plot(time, x, 'ro')
    plt.plot(time, v, 'bo')
    plt.plot(time, u, 'go')

    expout = prob.model.phase0.simulate()

    time = expout.get_val('phase0.timeseries.time')
    x = expout.get_val('phase0.timeseries.states:x')
    v = expout.get_val('phase0.timeseries.states:v')
    u = expout.get_val('phase0.timeseries.controls:u')

    plt.plot(time, x, 'r-')
    plt.plot(time, v, 'b-')
    plt.plot(time, u, 'g-')

    plt.show()
