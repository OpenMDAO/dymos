from __future__ import print_function, division, absolute_import

import unittest

import matplotlib
matplotlib.use('Agg')


class TestDoubleIntegratorForDocs(unittest.TestCase):

    def test_double_integrator_for_docs(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from dymos import Phase
        from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        phase = Phase('gauss-lobatto',
                      ode_class=DoubleIntegratorODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=True)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(1.0, 1.0))

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('v', fix_initial=True, fix_final=True)

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=True, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.setup(mode='fwd', check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_disc')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_disc')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_disc')

        p.run_driver()

        exp_out = phase.simulate(times=np.linspace(p['phase0.t_initial'],
                                                   p['phase0.t_duration'],
                                                   100),
                                 record=False)

        # Plot results
        fig, axes = plt.subplots(3, 1)
        fig.suptitle('Double Integrator Direct Collocation Solution')

        t_imp = phase.get_values('time', nodes='all')
        x_imp = phase.get_values('x', nodes='all')
        v_imp = phase.get_values('v', nodes='all')
        u_imp = phase.get_values('u', nodes='all')

        t_exp = exp_out.get_values('time')
        x_exp = exp_out.get_values('x')
        v_exp = exp_out.get_values('v')
        u_exp = exp_out.get_values('u')

        axes[0].plot(t_imp, x_imp, 'ro', label='implicit')
        axes[0].plot(t_exp, x_exp, 'b-', label='explicit')

        axes[0].set_xlabel('time (s)')
        axes[0].set_ylabel('x (m)')
        axes[0].grid(True)
        axes[0].legend(loc='best')

        axes[1].plot(t_imp, v_imp, 'ro', label='implicit')
        axes[1].plot(t_exp, v_exp, 'b-', label='explicit')

        axes[1].set_xlabel('time (s)')
        axes[1].set_ylabel('v (m/s)')
        axes[1].grid(True)
        axes[1].legend(loc='best')

        axes[2].plot(t_imp, u_imp, 'ro', label='implicit')
        axes[2].plot(t_exp, u_exp, 'b-', label='explicit')

        axes[2].set_xlabel('time (s)')
        axes[2].set_ylabel('u (m/s**2)')
        axes[2].grid(True)
        axes[2].legend(loc='best')

        plt.show()
