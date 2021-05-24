import os
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from openmdao.utils.testing_utils import use_tempdirs

import openmdao.api as om
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestBrachistochroneStaticGravity(unittest.TestCase):

    def _make_problem(self, tx):
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
                                        transcription=tx))

        #
        # Set the variables
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', rate_source='xdot',
                        targets=None,
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('y', rate_source='ydot',
                        targets=None,
                        units='m',
                        fix_initial=True, fix_final=True, solve_segments=False)

        phase.add_state('v', rate_source='vdot',
                        targets=['v'],
                        units='m/s',
                        fix_initial=True, fix_final=False, solve_segments=False)

        phase.add_control('theta', targets=['theta'],
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_objective('time', loc='final', scaler=10)
        return p, phase

    def tearDown(self):
        for filename in ['total_coloring.pkl', 'SLSQP.out', 'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_conflicting_static_target(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.GaussLobatto(num_segments=10))

        phase.add_parameter('g', targets=['g'], static_target=False, opt=False)
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "User has specified 'static_target = False' for parameter g,but one or more " \
                       "targets is tagged with 'dymos.static_target': g"

        self.assertEqual(str(e.exception), expected_msg)

    def test_control_to_static_target_fails_gl(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.GaussLobatto(num_segments=10))

        phase.add_control('g', opt=False)
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Control 'g' cannot be connected to its targets because one or more " \
                       "targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_control_to_static_target_fails_radau(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.Radau(num_segments=10))

        phase.add_control('g', opt=False)
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Control 'g' cannot be connected to its targets because one or more " \
                       "targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_control_rate_to_static_target_fails_gl(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.GaussLobatto(num_segments=10))

        phase.add_control('foo', rate_targets=['g'], opt=False, shape=(1,), units='m/s')
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Control rate of 'foo' cannot be connected to its targets because one or " \
                       "more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_control_rate_to_static_target_fails_radau(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.Radau(num_segments=10))

        phase.add_control('foo', rate_targets=['g'], opt=False, shape=(1,), units='m/s')
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Control rate of 'foo' cannot be connected to its targets because one or " \
                       "more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_control_rate2_to_static_target_fails_gl(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.GaussLobatto(num_segments=10))

        phase.add_control('foo', rate2_targets=['g'], opt=False, shape=(1,), units='m/s')
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Control rate2 of 'foo' cannot be connected to its targets because one or " \
                       "more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_control_rate2_to_static_target_fails_radau(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.Radau(num_segments=10))

        phase.add_control('foo', rate2_targets=['g'], opt=False, shape=(1,), units='m/s')
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Control rate2 of 'foo' cannot be connected to its targets because one or " \
                       "more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_polynomial_control_to_static_target_fails_gl(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.GaussLobatto(num_segments=10))

        phase.add_polynomial_control('g', opt=False, order=5)
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Polynomial control 'g' cannot be connected to its targets because one or " \
                       "more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_polynomial_control_to_static_target_fails_radau(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.Radau(num_segments=10))

        phase.add_polynomial_control('g', opt=False, order=5)
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Polynomial control 'g' cannot be connected to its targets because one or " \
                       "more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_polynomial_control_rate_to_static_target_fails_gl(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.GaussLobatto(num_segments=10))

        phase.add_polynomial_control('foo', opt=False, order=5, shape=(1,), units='m/s',
                                     rate_targets=['g'])
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Rate of polynomial control 'foo' cannot be connected to its targets " \
                       "because one or more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_polynomial_control_rate_to_static_target_fails_radau(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.Radau(num_segments=10))

        phase.add_polynomial_control('foo', opt=False, order=5, shape=(1,), units='m/s',
                                     rate_targets=['g'])
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Rate of polynomial control 'foo' cannot be connected to its targets " \
                       "because one or more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_polynomial_control_rate2_to_static_target_fails_gl(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.GaussLobatto(num_segments=10))

        phase.add_polynomial_control('foo', opt=False, order=5, shape=(1,), units='m/s',
                                     rate2_targets=['g'])
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Rate2 of polynomial control 'foo' cannot be connected to its targets " \
                       "because one or more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))

    def test_polynomial_control_rate2_to_static_target_fails_radau(self):
        """ Tests that control cannot be connected to target tagged as 'dymos.static_target'. """
        p, phase = self._make_problem(dm.Radau(num_segments=10))

        phase.add_polynomial_control('foo', opt=False, order=5, shape=(1,), units='m/s',
                                     rate2_targets=['g'])
        with self.assertRaises(ValueError) as e:
            p.setup()

        expected_msg = "Rate2 of polynomial control 'foo' cannot be connected to its targets " \
                       "because one or more targets are tagged with 'dymos.static_target'."

        self.assertEqual(expected_msg, str(e.exception))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
