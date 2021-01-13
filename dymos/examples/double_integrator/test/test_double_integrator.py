import itertools
import os
import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.double_integrator.double_integrator_ode import DoubleIntegratorODE


def double_integrator_direct_collocation(transcription='gauss-lobatto', compressed=True):

    p = om.Problem(model=om.Group())
    p.driver = om.pyOptSparseDriver()
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

    phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')
    phase.add_state('x', fix_initial=True, rate_source='v', units='m', shape=(1, ))

    phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                      rate2_continuity=False, shape=(1, ), lower=-1.0, upper=1.0)

    # Maximize distance travelled in one second.
    phase.add_objective('x', loc='final', scaler=-1)

    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = 1.0

    p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
    p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
    p['traj.phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

    p.run_driver()

    return p


@use_tempdirs
class TestDoubleIntegratorExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def _assert_results(self, p, traj=True, tol=1.0E-4):
        if traj:
            x = p.get_val('traj.phase0.timeseries.states:x')
            v = p.get_val('traj.phase0.timeseries.states:v')
        else:
            x = p.get_val('phase0.timeseries.states:x')
            v = p.get_val('phase0.timeseries.states:v')

        assert_near_equal(x[0], 0.0, tolerance=tol)
        assert_near_equal(x[-1], 0.25, tolerance=tol)

        assert_near_equal(v[0], 0.0, tolerance=tol)
        assert_near_equal(v[-1], 0.0, tolerance=tol)

    def test_ex_double_integrator_gl_compressed(self):
        p = double_integrator_direct_collocation('gauss-lobatto',
                                                 compressed=True)

        x = p.get_val('traj.phase0.timeseries.states:x')
        v = p.get_val('traj.phase0.timeseries.states:v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_gl_uncompressed(self):
        p = double_integrator_direct_collocation('gauss-lobatto',
                                                 compressed=False)

        x = p.get_val('traj.phase0.timeseries.states:x')
        v = p.get_val('traj.phase0.timeseries.states:v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_radau_compressed(self):
        p = double_integrator_direct_collocation('radau-ps',
                                                 compressed=True)

        x = p.get_val('traj.phase0.timeseries.states:x')
        v = p.get_val('traj.phase0.timeseries.states:v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_radau_uncompressed(self):
        p = double_integrator_direct_collocation('radau-ps',
                                                 compressed=False)

        x = p.get_val('traj.phase0.timeseries.states:x')
        v = p.get_val('traj.phase0.timeseries.states:v')

        assert_near_equal(x[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(x[-1], 0.25, tolerance=1.0E-4)

        assert_near_equal(v[0], 0.0, tolerance=1.0E-4)
        assert_near_equal(v[-1], 0.0, tolerance=1.0E-4)

    def test_ex_double_integrator_input_times_uncompressed(self):
        """
        Tests that externally connected t_initial and t_duration function as expected.
        """
        compressed = False
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        times_ivc = p.model.add_subsystem('times_ivc', om.IndepVarComp(),
                                          promotes_outputs=['t0', 'tp'])
        times_ivc.add_output(name='t0', val=0.0, units='s')
        times_ivc.add_output(name='tp', val=1.0, units='s')

        transcription = dm.Radau(num_segments=20, order=3, compressed=compressed)
        phase = dm.Phase(ode_class=DoubleIntegratorODE, transcription=transcription)
        p.model.add_subsystem('phase0', phase)

        p.model.connect('t0', 'phase0.t_initial')
        p.model.connect('tp', 'phase0.t_duration')

        phase.set_time_options(input_initial=True, input_duration=True, units='s')

        phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')
        phase.add_state('x', fix_initial=True, rate_source='v', units='m')

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, shape=(1,), lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['t0'] = 0.0
        p['tp'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()

        self._assert_results(p, traj=False)
        exp_out = phase.simulate()
        self._assert_results(exp_out, traj=False, tol=1.0E-2)

    def test_ex_double_integrator_input_times_compressed(self):
        """
        Tests that externally connected t_initial and t_duration function as expected.
        """
        compressed = True
        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        times_ivc = p.model.add_subsystem('times_ivc', om.IndepVarComp(),
                                          promotes_outputs=['t0', 'tp'])
        times_ivc.add_output(name='t0', val=0.0, units='s')
        times_ivc.add_output(name='tp', val=1.0, units='s')

        transcription = dm.Radau(num_segments=20, order=3, compressed=compressed)
        phase = dm.Phase(ode_class=DoubleIntegratorODE, transcription=transcription)
        p.model.add_subsystem('phase0', phase)

        p.model.connect('t0', 'phase0.t_initial')
        p.model.connect('tp', 'phase0.t_duration')

        phase.set_time_options(input_initial=True, input_duration=True, units='s')

        phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')
        phase.add_state('x', fix_initial=True, rate_source='v', units='m')

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, shape=(1, ), lower=-1.0, upper=1.0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['t0'] = 0.0
        p['tp'] = 1.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()

    def test_double_integrator_rk4(self, compressed=True):

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver()
        p.driver.declare_coloring()

        t = dm.RungeKutta(num_segments=30, order=3, compressed=compressed)

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0', dm.Phase(ode_class=DoubleIntegratorODE, transcription=t))

        phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

        phase.add_state('v', fix_initial=True, fix_final=False, rate_source='u', shape=(1, ), units='m/s')
        phase.add_state('x', fix_initial=True, rate_source='v', units='m')

        phase.add_control('u', units='m/s**2', scaler=0.01, continuity=False, rate_continuity=False,
                          rate2_continuity=False, shape=(1, ), lower=-1.0, upper=1.0)

        phase.add_boundary_constraint(name='v', loc='final', equals=0)

        # Maximize distance travelled in one second.
        phase.add_objective('x', loc='final', scaler=-1)

        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 1.0

        p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 0.25], nodes='state_input')
        p['traj.phase0.states:v'] = phase.interpolate(ys=[0, 0], nodes='state_input')
        p['traj.phase0.controls:u'] = phase.interpolate(ys=[1, -1], nodes='control_input')

        p.run_driver()

        return p


if __name__ == "__main__":

    unittest.main()
