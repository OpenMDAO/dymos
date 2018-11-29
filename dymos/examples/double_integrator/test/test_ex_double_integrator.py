from __future__ import print_function, absolute_import, division

import itertools
import unittest

# from numpy.testing import assert_almost_equal

from parameterized import parameterized

from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE
import dymos.examples.double_integrator.ex_double_integrator as ex_double_integrator


class TestDoubleIntegratorExample(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['dense', 'csc'],  # jacobian
                          ['compressed', 'uncompressed'],  # compressed transcription
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1],
                                                                          p.args[2]])
    )
    def test_ex_double_integrator(self, transcription='radau-ps', jacobian='csc',
                                  compressed='compressed'):
        ex_double_integrator.SHOW_PLOTS = False
        p = ex_double_integrator.double_integrator_direct_collocation(
            transcription, top_level_jacobian=jacobian, compressed=compressed == 'compressed')

        x0 = p.model.phase0.get_values('x')[0]
        xf = p.model.phase0.get_values('x')[-1]

        v0 = p.model.phase0.get_values('v')[0]
        vf = p.model.phase0.get_values('v')[-1]

        assert_rel_error(self, x0, 0.0, tolerance=1.0E-4)
        assert_rel_error(self, xf, 0.25, tolerance=1.0E-4)

        assert_rel_error(self, v0, 0.0, tolerance=1.0E-4)
        assert_rel_error(self, vf, 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_input_times(self, transcription='radau-ps',
                                              compressed=True):
        """
        Tests that externally connected t_initial and t_duration function as expected.
        """

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

        times_ivc = p.model.add_subsystem('times_ivc', IndepVarComp(),
                                          promotes_outputs=['t0', 'tp'])
        times_ivc.add_output(name='t0', val=0.0, units='s')
        times_ivc.add_output(name='tp', val=1.0, units='s')

        phase = Phase(transcription,
                      ode_class=DoubleIntegratorODE,
                      num_segments=20,
                      transcription_order=3,
                      compressed=compressed)

        p.model.add_subsystem('phase0', phase)

        p.model.connect('t0', 'phase0.t_initial')
        p.model.connect('tp', 'phase0.t_duration')

        phase.set_time_options(input_initial=True, input_duration=True)

        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('v', fix_initial=True, fix_final=True)

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup(check=True)

        p['t0'] = 0.0
        p['tp'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()
