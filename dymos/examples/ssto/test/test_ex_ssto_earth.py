from __future__ import print_function, absolute_import, division

import os

import itertools
import unittest

import matplotlib
matplotlib.use('Agg')

from numpy.testing import assert_almost_equal

from parameterized import parameterized

import dymos.examples.ssto.ex_ssto_earth as ex_ssto_earth


class TestExampleSSTOEarth(unittest.TestCase):

    def tearDown(self):
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, 'phase0_sim.db')):
            os.remove('phase0_sim.db')

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['csc'],  # jacobian
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1]])
    )
    def test_results(self, transcription='gauss-lobatto', jacobian='csc', compressed=True):

        p = ex_ssto_earth.ssto_earth(transcription, num_seg=10, transcription_order=5,
                                     top_level_jacobian=jacobian, compressed=compressed)

        p.setup(check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 150.0
        p['phase0.states:x'] = p.model.phase0.interpolate(ys=[0, 1.15E5], nodes='state_input')
        p['phase0.states:y'] = p.model.phase0.interpolate(ys=[0, 1.85E5], nodes='state_input')
        p['phase0.states:vx'] = p.model.phase0.interpolate(ys=[0, 7796.6961], nodes='state_input')
        p['phase0.states:vy'] = p.model.phase0.interpolate(ys=[1.0E-6, 0], nodes='state_input')
        p['phase0.states:m'] = p.model.phase0.interpolate(ys=[117000, 1163], nodes='state_input')
        p['phase0.controls:theta'] = p.model.phase0.interpolate(ys=[1.5, -0.76],
                                                                nodes='control_input')

        # p.run_model()

        p.run_driver()

        # Ensure defects are zero
        for state in ['x', 'y', 'vx', 'vy', 'm']:
            assert_almost_equal(p['phase0.collocation_constraint.defects:{0}'.format(state)],
                                0.0, decimal=5)

            if not compressed:
                assert_almost_equal(p['phase0.continuity_comp.'
                                      'defect_states:{0}'.format(state)],
                                    0.0, decimal=5,
                                    err_msg='error in state continuity for state {0}'.format(state))

        # Ensure time found is the known solution
        assert_almost_equal(p['phase0.t_duration'], 143, decimal=0)

    def test_simulate_plot(self):
        from matplotlib import pyplot as plt
        import numpy as np
        from openmdao.utils.assert_utils import assert_rel_error

        import dymos.examples.ssto.ex_ssto_earth as ex_ssto_earth

        p = ex_ssto_earth.ssto_earth('gauss-lobatto', num_seg=20, transcription_order=5,
                                     top_level_jacobian='csc')

        p.setup()

        phase = p.model.phase0
        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 150.0
        p['phase0.states:x'] = phase.interpolate(ys=[0, 1.15E5], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[0, 1.85E5], nodes='state_input')
        p['phase0.states:vx'] = phase.interpolate(ys=[0, 7796.6961], nodes='state_input')
        p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='state_input')
        p['phase0.states:m'] = phase.interpolate(ys=[117000, 1163], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[1.5, -0.76], nodes='control_input')

        p.run_driver()

        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 100))

        ##############################
        # quick check of the results
        ##############################
        assert_rel_error(self, p.get_val('phase0.timeseries.states:y')[-1], 1.85E5, 1e-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:vx')[-1], 7796.6961, 1e-4)
        assert_rel_error(self, p.get_val('phase0.timeseries.states:vy')[-1], 0, 1e-4)

        # check if the boundary value constraints were satisfied in the simulation
        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:y')[-1], 1.85E5, .02)
        assert_rel_error(self, exp_out.get_val('phase0.timeseries.states:vx')[-1], 7796.6961, .02)

        ########################
        # plot the results
        ########################
        t_imp = p.get_val('phase0.timeseries.time')
        t_exp = exp_out.get_val('phase0.timeseries.time')
        x_imp = p.get_val('phase0.timeseries.states:x')
        x_exp = exp_out.get_val('phase0.timeseries.states:x')
        y_imp = p.get_val('phase0.timeseries.states:y')
        y_exp = exp_out.get_val('phase0.timeseries.states:y')
        vx_imp = p.get_val('phase0.timeseries.states:vx')
        vx_exp = exp_out.get_val('phase0.timeseries.states:vx')
        vy_imp = p.get_val('phase0.timeseries.states:vy')
        vy_exp = exp_out.get_val('phase0.timeseries.states:vy')
        theta_imp = p.get_val('phase0.timeseries.controls:theta', units='deg')
        theta_exp = exp_out.get_val('phase0.timeseries.controls:theta', units='deg')

        plt.figure(facecolor='white')
        plt.plot(x_imp, y_imp, 'bo', label='solution')
        plt.plot(x_exp, y_exp, 'r-', label='simulated')
        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.legend(loc='best', ncol=2)
        plt.grid()

        fig = plt.figure(facecolor='white')
        fig.suptitle('results for flat_earth_without_aero')

        axarr = fig.add_subplot(3, 1, 1)
        axarr.plot(t_imp, theta_imp, 'bo', label='solution')
        axarr.plot(t_exp, theta_exp, 'b-', label='simulated')

        axarr.set_xlabel('time, s')
        axarr.set_ylabel(r'$\theta$, deg')

        axarr = fig.add_subplot(3, 1, 3)

        axarr.plot(t_imp, vx_imp, 'bo', label='$v_x$ solution')
        axarr.plot(t_exp, vx_exp, 'b-', label='$v_x$ simulated')

        axarr.plot(t_imp, vy_imp, 'ro', label='$v_y$ solution')
        axarr.plot(t_exp, vy_exp, 'r-', label='$v_y$ simulated')

        axarr.set_xlabel('time, s')
        axarr.set_ylabel('velocity, m/s')
        axarr.legend(loc='best', ncol=2)

        plt.show()

if __name__ == "__main__":
    unittest.main()
