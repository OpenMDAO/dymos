from __future__ import print_function, absolute_import, division

import itertools
import os
import unittest

from parameterized import parameterized

from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

import dymos as dm
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE


def double_integrator_direct_collocation(transcription='gauss-lobatto', compressed=True):

    p = Problem(model=Group())
    p.driver = pyOptSparseDriver()
    p.driver.declare_coloring()

    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=30, order=3, compressed=compressed)
    elif transcription == "radau-ps":
        t = dm.Radau(num_segments=30, order=3, compressed=compressed)
    else:
        raise ValueError('invalid transcription')

    traj = p.model.add_subsystem('traj', dm.Trajectory())

    phase = traj.add_phase('phase0', dm.Phase(ode_class=DoubleIntegratorODE, transcription=t))

    phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

    phase.set_state_options('x', fix_initial=True, rate_source='v', units='m')
    phase.set_state_options('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')

    phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                      rate2_continuity=False, lower=-1.0, upper=1.0)

    # Maximize distance travelled in one second.
    phase.add_objective('x', loc='final', scaler=-1)

    p.model.linear_solver = DirectSolver()

    p.setup(check=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 1.0

    p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
    p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
    p['traj.phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

    p.run_driver()

    return p


class TestDoubleIntegratorExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    @parameterized.expand(
        itertools.product(['gauss-lobatto', 'radau-ps'],  # transcription
                          ['compressed', 'uncompressed'],  # compressed transcription
                          ), testcase_func_name=lambda f, n, p: '_'.join(['test_results',
                                                                          p.args[0],
                                                                          p.args[1]])
    )
    def test_ex_double_integrator(self, transcription='radau-ps', compressed='compressed'):
        p = double_integrator_direct_collocation(transcription,
                                                 compressed=compressed == 'compressed')

        x = p.get_val('traj.phase0.timeseries.states:x')
        v = p.get_val('traj.phase0.timeseries.states:v')

        assert_rel_error(self, x[0], 0.0, tolerance=1.0E-4)
        assert_rel_error(self, x[-1], 0.25, tolerance=1.0E-4)

        assert_rel_error(self, v[0], 0.0, tolerance=1.0E-4)
        assert_rel_error(self, v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_input_times(self, compressed=True):
        """
        Tests that externally connected t_initial and t_duration function as expected.
        """

        p = Problem(model=Group())
        p.driver = pyOptSparseDriver()
        p.driver.declare_coloring()

        times_ivc = p.model.add_subsystem('times_ivc', IndepVarComp(),
                                          promotes_outputs=['t0', 'tp'])
        times_ivc.add_output(name='t0', val=0.0, units='s')
        times_ivc.add_output(name='tp', val=1.0, units='s')

        transcription = dm.Radau(num_segments=20, order=3, compressed=compressed)
        phase = dm.Phase(ode_class=DoubleIntegratorODE, transcription=transcription)
        p.model.add_subsystem('phase0', phase)

        p.model.connect('t0', 'phase0.t_initial')
        p.model.connect('tp', 'phase0.t_duration')

        phase.set_time_options(input_initial=True, input_duration=True, units='s')

        phase.set_state_options('x', fix_initial=True, rate_source='v', units='m')
        phase.set_state_options('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = DirectSolver()

        p.setup(check=True)

        p['t0'] = 0.0
        p['tp'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()


if __name__ == "__main__":

    unittest.main()
