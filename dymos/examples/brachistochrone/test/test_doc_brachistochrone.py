from __future__ import print_function, absolute_import, division

import os
import unittest


class TestBrachistochroneExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_for_docs_gauss_lobatto(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(mode='rev')

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_disc')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_disc')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_disc')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, phase.get_values('time')[-1], 1.8016, tolerance=1.0E-3)

        # Generate the explicitly simulated trajectory
        t0 = p['phase0.t_initial']
        tf = t0 + p['phase0.t_duration']
        exp_out = phase.simulate(times=np.linspace(t0, tf, 50))

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = phase.get_values('x', nodes='all')
        y_imp = phase.get_values('y', nodes='all')

        x_exp = exp_out.get_values('x')
        y_exp = exp_out.get_values('y')

        ax.plot(x_imp, y_imp, 'ro', label='solution')
        ax.plot(x_exp, y_exp, 'b-', label='simulated')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()
