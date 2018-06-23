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
                          ['fwd'],  # derivative_mode
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1],
                                                                          p.args[2]])
    )
    def test_results(self, transcription='gauss-lobatto', jacobian='csc', derivative_mode='fwd',
                     compressed=True):

        p = ex_ssto_earth.ssto_earth(transcription, num_seg=10, transcription_order=5,
                                     top_level_jacobian=jacobian, compressed=compressed)

        p.setup(mode=derivative_mode, check=True)

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 150.0
        p['phase0.states:x'] = p.model.phase0.interpolate(ys=[0, 1.15E5], nodes='state_disc')
        p['phase0.states:y'] = p.model.phase0.interpolate(ys=[0, 1.85E5], nodes='state_disc')
        p['phase0.states:vx'] = p.model.phase0.interpolate(ys=[0, 7796.6961], nodes='state_disc')
        p['phase0.states:vy'] = p.model.phase0.interpolate(ys=[1.0E-6, 0], nodes='state_disc')
        p['phase0.states:m'] = p.model.phase0.interpolate(ys=[117000, 1163], nodes='state_disc')
        p['phase0.controls:theta'] = p.model.phase0.interpolate(ys=[1.5, -0.76],
                                                                nodes='control_disc')

        # p.run_model()

        p.run_driver()

        # Ensure defects are zero
        for state in ['x', 'y', 'vx', 'vy', 'm']:
            assert_almost_equal(p['phase0.collocation_constraint.defects:{0}'.format(state)],
                                0.0, decimal=5)

            if not compressed:
                assert_almost_equal(p['phase0.continuity_constraint.'
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

        p.setup(mode='fwd')

        phase = p.model.phase0
        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 150.0
        p['phase0.states:x'] = phase.interpolate(ys=[0, 1.15E5], nodes='state_disc')
        p['phase0.states:y'] = phase.interpolate(ys=[0, 1.85E5], nodes='state_disc')
        p['phase0.states:vx'] = phase.interpolate(ys=[0, 7796.6961], nodes='state_disc')
        p['phase0.states:vy'] = phase.interpolate(ys=[1.0E-6, 0], nodes='state_disc')
        p['phase0.states:m'] = phase.interpolate(ys=[117000, 1163], nodes='state_disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[1.5, -0.76], nodes='control_disc')

        p.run_driver()

        exp_out = phase.simulate(times=np.linspace(0, p['phase0.t_duration'], 100))

        ##############################
        # quick check of the results
        ##############################
        assert_rel_error(self, phase.get_values('y')[-1], 1.85E5, 1e-4)
        assert_rel_error(self, phase.get_values('vx')[-1], 7796.6961, 1e-4)
        assert_rel_error(self, phase.get_values('vy')[-1], 0, 1e-4)

        # check if the boundary value constraints were satisfied in the simulation
        assert_rel_error(self, exp_out.get_values('y')[-1], 1.85E5, 1e-2)
        assert_rel_error(self, exp_out.get_values('vx')[-1], 7796.6961, 1e-2)
        # there is a small amount of discretization error here
        assert_rel_error(self, exp_out.get_values('vy')[-1], 8.55707245, 1e-2)

        ########################
        # plot the results
        ########################
        plt.figure(facecolor='white')
        plt.plot(phase.get_values('x'), phase.get_values('y'), 'bo', label='solution')
        plt.plot(exp_out.get_values('x'), exp_out.get_values('y'), 'r-', label='simulated')
        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.legend(loc='best', ncol=2)
        plt.grid()

        fig = plt.figure(facecolor='white')
        fig.suptitle('results for flat_earth_without_aero')

        axarr = fig.add_subplot(3, 1, 1)
        axarr.plot(phase.get_values('time'),
                   phase.get_values('theta', units='deg'), 'bo', label='solution')
        axarr.plot(exp_out.get_values('time'),
                   exp_out.get_values('theta', units='deg'), 'b-', label='simulated')

        axarr.set_xlabel('time, s')
        axarr.set_ylabel(r'$\theta$, deg')

        axarr = fig.add_subplot(3, 1, 2)

        axarr.plot(phase.get_values('x'),
                   phase.get_values('y'), 'bo', label='$v_x$ solution')
        axarr.plot(exp_out.get_values('x'),
                   exp_out.get_values('y'), 'b-', label='$v_x$ simulated')

        axarr.plot(phase.get_values('x'),
                   phase.get_values('y'), 'ro', label='$v_y$ solution')
        axarr.plot(exp_out.get_values('x'),
                   exp_out.get_values('y'), 'r-', label='$v_y$ simulated')

        axarr.set_xlabel('downrange, m')
        axarr.set_ylabel('altitude, m')
        axarr.legend(loc='best', ncol=2)

        axarr = fig.add_subplot(3, 1, 3)

        axarr.plot(phase.get_values('time'),
                   phase.get_values('vx'), 'bo', label='$v_x$ solution')
        axarr.plot(exp_out.get_values('time'),
                   exp_out.get_values('vx'), 'b-', label='$v_x$ simulated')

        axarr.plot(phase.get_values('time'),
                   phase.get_values('vy'), 'ro', label='$v_y$ solution')
        axarr.plot(exp_out.get_values('time'),
                   exp_out.get_values('vy'), 'r-', label='$v_y$ simulated')

        axarr.set_xlabel('time, s')
        axarr.set_ylabel('velocity, m/s')
        axarr.legend(loc='best', ncol=2)

        plt.show()

if __name__ == "__main__":
    unittest.main()
