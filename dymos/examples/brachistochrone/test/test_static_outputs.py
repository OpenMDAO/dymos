import warnings
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm


class BrachODEStaticOutput(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('static_gravity', types=(bool,), default=False,
                             desc='If True, treat gravity as a static (scalar) input, rather than '
                                  'having different values at each node.')

    def setup(self):
        nn = self.options['num_nodes']
        g_default_val = 9.80665 if self.options['static_gravity'] else 9.80665 * np.ones(nn)

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=g_default_val, desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.ones(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s',
                        tags=['dymos.state_rate_source:x', 'dymos.state_units:m'])

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s',
                        tags=['dymos.state_rate_source:y', 'dymos.state_units:m'])

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2',
                        tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s'])

        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        self.add_output('foo', val=np.eye(2), desc='a static matrix to be output', units='m/s**2')

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs', show_summary=True, show_sparsity=True)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

        np.fill_diagonal(outputs['foo'], g)


@use_tempdirs
class TestStaticODEOutput(unittest.TestCase):

    def _test_static_ode_output(self, transcription):

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=BrachODEStaticOutput,
                         transcription=transcription(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that the it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(1.0, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0.01, upper=np.pi-0.01, shape=(1,))

        phase.add_timeseries_output('*')

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            p.setup(check=True)
            expected_warning = "Cannot add ODE output foo to the timeseries output. It is sized " \
                               "such that its first dimension != num_nodes."
            self.assertIn(expected_warning, [str(w.message) for w in ctx])

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        p.set_val('traj.phase0.t_initial', 0.0, units='s')
        p.set_val('traj.phase0.t_duration', 5.0, units='s')

        p.set_val('traj.phase0.states:x',
                  phase.interp('x', [0, 10]),
                  units='m')

        p.set_val('traj.phase0.states:y',
                  phase.interp('y', [10, 5]),
                  units='m')

        p.set_val('traj.phase0.states:v',
                  phase.interp('v', [0, 5]),
                  units='m/s')

        p.set_val('traj.phase0.controls:theta',
                  phase.interp('theta', [0.01, 90]),
                  units='deg')

        # Run the driver to solve the problem
        with warnings.catch_warnings(record=True) as ctx:
            warnings.simplefilter('always')
            dm.run_problem(p, simulate=True, make_plots=False)
            expected_warning = "Cannot add ODE output foo to the timeseries output. It is sized " \
                               "such that its first dimension != num_nodes."
            self.assertIn(expected_warning, [str(w.message) for w in ctx])

        sol = om.CaseReader('dymos_solution.db').get_case('final')
        sim = om.CaseReader('dymos_simulation.db').get_case('final')

        with self.assertRaises(expected_exception=KeyError) as e:
            sol.get_val('traj.phase0.timeseries.foo')
        self.assertEqual(str(e.exception), "'Variable name \"traj.phase0.timeseries.foo\" not found.'")

        with self.assertRaises(expected_exception=KeyError) as e:
            sim.get_val('traj.phase0.timeseries.foo')
        self.assertEqual(str(e.exception), "'Variable name \"traj.phase0.timeseries.foo\" not found.'")

    def test_static_ode_output_gl(self):
        self._test_static_ode_output(dm.GaussLobatto)

    def test_static_ode_output_radau(self):
        self._test_static_ode_output(dm.Radau)
