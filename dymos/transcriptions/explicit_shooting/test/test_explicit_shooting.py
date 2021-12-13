import unittest
import warnings

import numpy as np

import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class Simple2StateODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s**2')
        self.add_input('y', shape=(nn,), units='s')
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('p', shape=(nn,), units='s**2')

        self.add_output('x_dot', shape=(nn,), units='s')
        self.add_output('y_dot', shape=(nn,), units=None)

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='x_dot', wrt='p', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='y_dot', wrt='y', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='y_dot', wrt='t', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        t = inputs['t']
        p = inputs['p']
        outputs['x_dot'] = x - t**2 + p
        outputs['y_dot'] = -2 * y + t**3 * np.exp(-2*t)

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['x_dot', 't'] = -2*t
        partials['y_dot', 't'] = -np.exp(-2*t) * t**2 * (2 * t - 3)


class Simple1StateODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('y', shape=(nn,), units=None)
        self.add_input('t', shape=(nn,), units='s')

        self.add_output('y_dot', shape=(nn,), units='1/s')

        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='y_dot', wrt='y', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='y_dot', wrt='t', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        y = inputs['y']
        t = inputs['t']
        outputs['y_dot'] = -2 * y + t**3 * np.exp(-2*t)

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['y_dot', 't'] = -np.exp(-2*t) * t**2 * (2 * t - 3)


@use_tempdirs
class TestExplicitShooting(unittest.TestCase):

    # @require_pyoptsparse(optimizer='SNOPT')
    def test_1_state_run_model(self):
        prob = om.Problem()

        tx = dm.transcriptions.ExplicitShooting(num_segments=1, grid='gauss-lobatto', method='rk4',
                                                order=3, num_steps_per_segment=10, compressed=True)

        phase = dm.Phase(ode_class=Simple1StateODE, transcription=tx)

        phase.set_time_options(targets=['t'], units='s')

        # automatically discover states
        phase.set_state_options('y', targets=['y'], rate_source='y_dot')

        prob.model.add_subsystem('phase0', phase)

        prob.setup(force_alloc_complex=False)

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 1.0)
        prob.set_val('phase0.states:y', 1.0)

        prob.run_model()

        with np.printoptions(linewidth=1024):
            cpd = prob.check_partials(compact_print=True)

        assert_check_partials(cpd, rtol=1.0E-5)

    # @require_pyoptsparse(optimizer='SNOPT')
    def test_2_states_run_model(self):

        for method in ['rk4', 'euler', '3/8', 'ralston', 'rkf', 'rkck', 'dopri']:
            with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):

                prob = om.Problem()

                tx = dm.transcriptions.ExplicitShooting(num_segments=2, grid='gauss-lobatto',
                                                        method=method, order=3,
                                                        num_steps_per_segment=50, compressed=True)

                phase = dm.Phase(ode_class=Simple2StateODE, transcription=tx)

                phase.set_time_options(targets=['t'], units='s')

                # automatically discover states
                phase.set_state_options('x', targets=['x'], rate_source='x_dot')
                phase.set_state_options('y', targets=['y'], rate_source='y_dot')

                phase.add_parameter('p', targets=['p'])

                prob.model.add_subsystem('phase0', phase)

                prob.setup(force_alloc_complex=True)

                prob.set_val('phase0.t_initial', 0.0)
                prob.set_val('phase0.t_duration', 1.0)
                prob.set_val('phase0.states:x', 0.5)
                prob.set_val('phase0.states:y', 1.0)
                prob.set_val('phase0.parameters:p', 1)

                prob.run_model()

                t_f = prob.get_val('phase0.integrator.t_final')
                x_f = prob.get_val('phase0.integrator.states_out:x')
                y_f = prob.get_val('phase0.integrator.states_out:y')

                if method == 'euler':
                    tol = 5.0E-2
                else:
                    tol = 1.0E-3

                assert_near_equal(t_f, 1.0)
                assert_near_equal(x_f[-1, ...], 2.64085909, tolerance=tol)
                assert_near_equal(y_f[-1, ...], 0.1691691, tolerance=tol)

                with np.printoptions(linewidth=1024):
                    cpd = prob.check_partials(compact_print=True, method='cs')
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_brachistochrone_explicit_shooting(self):

        for method in ['rk4', 'ralston']:
            for compressed in [True, False]:
                with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):
                    prob = om.Problem()

                    prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

                    tx = dm.ExplicitShooting(num_segments=3, grid='gauss-lobatto',
                                             method=method, order=5,
                                             num_steps_per_segment=5,
                                             compressed=compressed)

                    phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                    phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                    # automatically discover states
                    phase.set_state_options('x', fix_initial=True)
                    phase.set_state_options('y', fix_initial=True)
                    phase.set_state_options('v', fix_initial=True)

                    phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
                    phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9,
                                      ref=90., rate2_continuity=True)

                    phase.add_boundary_constraint('x', loc='final', equals=10.0)
                    phase.add_boundary_constraint('y', loc='final', equals=5.0)

                    prob.model.add_subsystem('phase0', phase)

                    phase.add_objective('time', loc='final')

                    prob.setup(force_alloc_complex=True)

                    prob.set_val('phase0.t_initial', 0.0)
                    prob.set_val('phase0.t_duration', 2)
                    prob.set_val('phase0.states:x', 0.0)
                    prob.set_val('phase0.states:y', 10.0)
                    prob.set_val('phase0.states:v', 1.0E-6)
                    prob.set_val('phase0.parameters:g', 1.0, units='m/s**2')
                    prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

                    prob.run_driver()

                    x = prob.get_val('phase0.timeseries.states:x')
                    y = prob.get_val('phase0.timeseries.states:y')
                    t = prob.get_val('phase0.timeseries.time')
                    theta = prob.get_val('phase0.timeseries.controls:theta')
                    theta_rate = prob.get_val('phase0.timeseries.control_rates:theta_rate')
                    theta_rate2 = prob.get_val('phase0.timeseries.control_rates:theta_rate2')

                    assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
                    assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)
                    assert_near_equal(t[-1, ...], 1.8016, tolerance=1.0E-2)

                    # Test the continuity constraints
                    tol = 1.0E-6
                    assert_near_equal(theta[1:-2:2, ...], theta[2::2, ...], tolerance=tol)
                    assert_near_equal(theta_rate[1:-2:2, ...], theta_rate[2::2, ...], tolerance=tol)
                    assert_near_equal(theta_rate2[1:-2:2, ...], theta_rate2[2::2, ...], tolerance=tol)

                    with np.printoptions(linewidth=1024):
                        cpd = prob.check_partials(compact_print=True, method='cs', out_stream=None)
                        assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_brachistochrone_explicit_shooting_path_constraint(self):

        for method in ['rk4', 'ralston']:
            with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):
                prob = om.Problem()

                prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

                tx = dm.transcriptions.ExplicitShooting(num_segments=10, grid='gauss-lobatto', method=method,
                                                        order=3, num_steps_per_segment=5, compressed=True)

                phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=9, units='m/s**2', opt=True, lower=9, upper=9.80665)
                phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)
                phase.add_path_constraint('ydot', lower=-100, upper=0)

                prob.model.add_subsystem('phase0', phase)

                phase.add_objective('time', loc='final')

                prob.setup(force_alloc_complex=True)

                prob.set_val('phase0.t_initial', 0.0)
                prob.set_val('phase0.t_duration', 2)
                prob.set_val('phase0.states:x', 0.0)
                prob.set_val('phase0.states:y', 10.0)
                prob.set_val('phase0.states:v', 1.0E-6)
                prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

                prob.run_driver()

                x = prob.get_val('phase0.timeseries.states:x')
                y = prob.get_val('phase0.timeseries.states:y')
                ydot = prob.get_val('phase0.timeseries.ydot')

                assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
                self.assertTrue(np.all(ydot < 1.0E-6), msg='Not all elements of path constraint satisfied')
                assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)

                with np.printoptions(linewidth=1024):
                    cpd = prob.check_partials(compact_print=False, method='cs', out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_brachistochrone_explicit_shooting_path_constraint_polynomial_control(self):
        for method in ['euler', '3/8']:
            with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):
                prob = om.Problem()

                prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

                tx = dm.transcriptions.ExplicitShooting(num_segments=1, grid='gauss-lobatto', method=method,
                                                        order=3, num_steps_per_segment=50, compressed=True)

                phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=9, units='m/s**2', opt=True, lower=9, upper=9.80665)
                phase.add_polynomial_control('theta', order=5, val=45.0, units='deg', lower=1.0E-6, upper=179.9)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)
                phase.add_path_constraint('ydot', lower=-100, upper=0)

                prob.model.add_subsystem('phase0', phase)

                phase.add_objective('time', loc='final')

                prob.setup(force_alloc_complex=True)

                prob.set_val('phase0.t_initial', 0.0)
                prob.set_val('phase0.t_duration', 2)
                prob.set_val('phase0.states:x', 0.0)
                prob.set_val('phase0.states:y', 10.0)
                prob.set_val('phase0.states:v', 1.0E-6)
                prob.set_val('phase0.polynomial_controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

                prob.run_driver()

                x = prob.get_val('phase0.timeseries.states:x')
                y = prob.get_val('phase0.timeseries.states:y')
                ydot = prob.get_val('phase0.timeseries.ydot')

                assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
                self.assertTrue(np.all(ydot < 1.0E-6), msg='Not all elements of path constraint satisfied')
                assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)

                with np.printoptions(linewidth=1024):
                    cpd = prob.check_partials(compact_print=False, method='cs', out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_brachistochrone_explicit_shooting_path_constraint_renamed(self):

        for method in ['rk4', 'ralston']:
            with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):
                prob = om.Problem()

                prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

                tx = dm.transcriptions.ExplicitShooting(num_segments=10, grid='gauss-lobatto', method=method,
                                                        order=3, num_steps_per_segment=5, compressed=True)

                phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)
                phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)
                phase.add_path_constraint('ydot', constraint_name='foo', lower=-100, upper=0)

                prob.model.add_subsystem('phase0', phase)

                phase.add_objective('time', loc='final')

                prob.setup(force_alloc_complex=True)

                prob.set_val('phase0.t_initial', 0.0)
                prob.set_val('phase0.t_duration', 2)
                prob.set_val('phase0.states:x', 0.0)
                prob.set_val('phase0.states:y', 10.0)
                prob.set_val('phase0.states:v', 1.0E-6)
                prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

                prob.run_driver()

                x = prob.get_val('phase0.timeseries.states:x')
                y = prob.get_val('phase0.timeseries.states:y')
                ydot = prob.get_val('phase0.timeseries.foo')

                assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
                self.assertTrue(np.all(ydot <= 1.0E-6), msg='Not all elements of path constraint satisfied')
                assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)

                with np.printoptions(linewidth=1024):
                    cpd = prob.check_partials(compact_print=True, method='cs', out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_brachistochrone_explicit_shooting_path_constraint_invalid_renamed(self):
        for method in ['rk4', 'ralston']:
            with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):
                prob = om.Problem()

                prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

                tx = dm.transcriptions.ExplicitShooting(num_segments=10, grid='gauss-lobatto', method=method,
                                                        order=3, num_steps_per_segment=5, compressed=True)

                phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)
                phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)
                phase.add_path_constraint('y', constraint_name='foo', lower=5, upper=100)

                prob.model.add_subsystem('phase0', phase)

                phase.add_objective('time', loc='final')

                with warnings.catch_warnings(record=True) as ctx:
                    warnings.simplefilter('always')
                    prob.setup(check=True)

                self.assertIn("Option 'constraint_name' on path constraint y is only valid for "
                              "ODE outputs. The option is being ignored.", [str(w.message) for w in ctx])

    @require_pyoptsparse(optimizer='SLSQP')
    def test_brachistochrone_explicit_shooting_polynomial_control(self):
        prob = om.Problem()

        prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

        tx = dm.transcriptions.ExplicitShooting(num_segments=3, grid='radau-ps', method='rk4',
                                                order=3, num_steps_per_segment=10, compressed=True)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

        # automatically discover states
        phase.set_state_options('x', fix_initial=True)
        phase.set_state_options('y', fix_initial=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
        phase.add_polynomial_control('theta', order=9, val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

        phase.add_boundary_constraint('x', loc='final', equals=10.0)
        phase.add_boundary_constraint('y', loc='final', equals=5.0)

        prob.model.add_subsystem('phase0', phase)

        phase.add_objective('time', loc='final')

        prob.setup(force_alloc_complex=True)

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 2)
        prob.set_val('phase0.states:x', 0.0)
        prob.set_val('phase0.states:y', 10.0)
        prob.set_val('phase0.states:v', 1.0E-6)
        prob.set_val('phase0.parameters:g', 1.0, units='m/s**2')
        prob.set_val('phase0.polynomial_controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

        prob.run_driver()

        x = prob.get_val('phase0.timeseries.states:x')
        y = prob.get_val('phase0.timeseries.states:y')
        t = prob.get_val('phase0.timeseries.time')
        tp = prob.get_val('phase0.timeseries.time_phase')

        assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-5)
        assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-5)
        assert_near_equal(t[-1, ...], 1.8016, tolerance=5.0E-3)
        assert_near_equal(tp[-1, ...], 1.8016, tolerance=5.0E-3)

        with np.printoptions(linewidth=1024):
            cpd = prob.check_partials(compact_print=False, method='cs', out_stream=None)
            assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_explicit_shooting_timeseries_ode_output(self):

        for method in ['rk4', 'ralston']:
            with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):
                prob = om.Problem()

                prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

                tx = dm.transcriptions.ExplicitShooting(num_segments=5, grid='gauss-lobatto', method=method,
                                                        order=3, num_steps_per_segment=10, compressed=True)

                phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
                phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)

                phase.add_timeseries_output('*')

                prob.model.add_subsystem('phase0', phase)

                phase.add_objective('time', loc='final')

                prob.setup(force_alloc_complex=True)

                prob.set_val('phase0.t_initial', 0.0)
                prob.set_val('phase0.t_duration', 2)
                prob.set_val('phase0.states:x', 0.0)
                prob.set_val('phase0.states:y', 10.0)
                prob.set_val('phase0.states:v', 1.0E-6)
                prob.set_val('phase0.parameters:g', 1.0, units='m/s**2')
                prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

                dm.run_problem(prob, run_driver=True)

                x = prob.get_val('phase0.timeseries.states:x')
                y = prob.get_val('phase0.timeseries.states:y')
                v = prob.get_val('phase0.timeseries.states:v')
                theta = prob.get_val('phase0.timeseries.controls:theta', units='rad')
                check = prob.get_val('phase0.timeseries.check')
                t = prob.get_val('phase0.timeseries.time')

                assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
                assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)
                assert_near_equal(t[-1, ...], 1.8016, tolerance=1.0E-2)
                assert_near_equal(check, v / np.sin(theta))

                with np.printoptions(linewidth=1024):
                    cpd = prob.check_partials(method='cs', out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_explicit_shooting_unknown_timeseries(self):

        for method in ['euler']:
            with self.subTest(f"test brachistochrone explicit shooting with method '{method}'"):
                prob = om.Problem()

                prob.driver = om.pyOptSparseDriver(optimizer='SLSQP')

                tx = dm.transcriptions.ExplicitShooting(num_segments=5, grid='gauss-lobatto', method=method,
                                                        order=3, num_steps_per_segment=10, compressed=True)

                phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
                phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)

                phase.add_timeseries_output('*')
                phase.add_timeseries_output('foo')

                prob.model.add_subsystem('phase0', phase)

                phase.add_objective('time', loc='final')

                msg = "The following timeseries outputs were requested but not found in the " \
                      "ODE: foo"

                with warnings.catch_warnings(record=True) as ctx:
                    warnings.simplefilter('always')
                    prob.setup(force_alloc_complex=True)

                self.assertIn(msg, [str(w.message) for w in ctx])

    # @require_pyoptsparse(optimizer='SNOPT')
    def test_brachistochrone_static_gravity_explicit_shooting(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        #
        # Initialize the Problem and the optimization driver
        #
        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        #
        # Create a trajectory and add a phase to it
        #
        traj = p.model.add_subsystem('traj', dm.Trajectory())

        phase = traj.add_phase('phase0',
                               dm.Phase(ode_class=BrachistochroneODE,
                                        ode_init_kwargs={'static_gravity': True},
                                        transcription=dm.ExplicitShooting(num_segments=10, num_steps_per_segment=10)))

        #
        # Set the variables
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        targets=None,
                        units='m',
                        fix_initial=True)

        phase.add_state('y', rate_source='ydot',
                        targets=None,
                        units='m',
                        fix_initial=True)

        phase.add_state('v', rate_source='vdot',
                        targets=['v'],
                        units='m/s',
                        fix_initial=True)

        phase.add_control('theta', targets=['theta'],
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], static_target=True, opt=False)

        #
        # Constrain the final values of x and y
        #
        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        #
        # Minimize time at the end of the phase
        #
        phase.add_objective('time', loc='final', scaler=10)

        #
        # Setup the Problem
        #
        p.setup()

        #
        # Set the initial values
        # The initial time is fixed, and we set that fixed value here.
        # The optimizer is allowed to modify t_duration, but an initial guess is provided here.
        #
        p.set_val('traj.phase0.t_initial', 0.0)
        p.set_val('traj.phase0.t_duration', 2.0)

        # Guesses for states are provided at all state_input nodes.
        # We use the phase.interpolate method to linearly interpolate values onto the state input nodes.
        # Since fix_initial=True for all states and fix_final=True for x and y, the initial or final
        # values of the interpolation provided here will not be changed by the optimizer.
        p.set_val('traj.phase0.states:x', phase.interp('x', [0, 10]))
        p.set_val('traj.phase0.states:y', phase.interp('y', [10, 5]))
        p.set_val('traj.phase0.states:v', phase.interp('v', [0, 9.9]))

        # Guesses for controls are provided at all control_input node.
        # Here phase.interpolate is used to linearly interpolate values onto the control input nodes.
        p.set_val('traj.phase0.controls:theta', phase.interp('theta', [5, 100.5]))

        # Set the value for gravitational acceleration.
        p.set_val('traj.phase0.parameters:g', 9.80665)

        #
        # Solve for the optimal trajectory
        #
        dm.run_problem(p, simulate=False)

        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)
