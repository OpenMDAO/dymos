from __future__ import print_function, division, absolute_import

import numpy as np
import unittest

from dymos import declare_time, declare_state, declare_parameter

from openmdao.api import ExplicitComponent


class _BrachistochroneODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)


class TestODEOptions(unittest.TestCase):

    def test_declare_time(self):

        @declare_time(units='s', targets=['foo', 'bar'])
        class B(_BrachistochroneODE):
            pass

        self.assertIn('ode_options', B.__dict__, msg='System does not have ode_options metadata')
        self.assertEqual(B.ode_options._time_options['units'], 's')
        self.assertEqual(B.ode_options._time_options['targets'], ['foo', 'bar'])

    def test_declare_state(self):

        @declare_state('x', rate_source='xdot', units='m')
        @declare_state('y', rate_source='ydot', units='m')
        @declare_state('v', rate_source='vdot', targets='v', units='m/s')
        class B(_BrachistochroneODE):
            pass

        self.assertIn('ode_options', B.__dict__, msg='System does not have ode_options metadata')
        self.assertEqual(B.ode_options._states['x']['units'], 'm')
        self.assertEqual(B.ode_options._states['x']['rate_source'], 'xdot')
        self.assertEqual(B.ode_options._states['y']['units'], 'm')
        self.assertEqual(B.ode_options._states['y']['rate_source'], 'ydot')
        self.assertEqual(B.ode_options._states['v']['units'], 'm/s')
        self.assertEqual(B.ode_options._states['v']['rate_source'], 'vdot')
        self.assertEqual(B.ode_options._states['v']['targets'], ['v'])

    def test_invalid_state_name(self):
        with self.assertRaises(NameError) as e:
            @declare_state('x', rate_source='xdot', units='m')
            @declare_state('foo.x', rate_source='foox_dot', units='m')
            @declare_state('v', rate_source='vdot', targets='v', units='m/s')
            class B(_BrachistochroneODE):
                pass
        self.assertEqual(str(e.exception), "'foo.x' is not a valid OpenMDAO variable name.")

    def test_declare_parameters(self):

        @declare_parameter('theta', targets='theta', units='rad')
        @declare_parameter('g', units='m/s**2', targets=['g'], dynamic=False)
        class B(_BrachistochroneODE):
            pass

        self.assertIn('ode_options', B.__dict__, msg='System does not have ode_options metadata')
        self.assertEqual(B.ode_options._parameters['theta']['targets'], ['theta'])
        self.assertEqual(B.ode_options._parameters['theta']['units'], 'rad')

    def test_invalid_parameter_name(self):
        with self.assertRaises(NameError) as e:
            @declare_parameter('theta', targets='theta', units='rad')
            @declare_parameter('g?', units='m/s**2', targets=['g'], dynamic=False)
            class B(_BrachistochroneODE):
                pass
        self.assertEqual(str(e.exception), "'g?' is not a valid OpenMDAO variable name.")

    def test_all(self):

        @declare_time(units='s', targets=['foo', 'bar'])
        @declare_state('x', rate_source='xdot', units='m')
        @declare_state('y', rate_source='ydot', units='m')
        @declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
        @declare_parameter('theta', targets=['theta'], units='rad')
        @declare_parameter('g', units='m/s**2', targets=['g'], dynamic=False)
        class B(_BrachistochroneODE):
            pass

        self.assertIn('ode_options', B.__dict__, msg='System does not have ode_options metadata')
        self.assertEqual(B.ode_options._time_options['units'], 's')
        self.assertEqual(B.ode_options._time_options['targets'], ['foo', 'bar'])
        self.assertEqual(B.ode_options._states['x']['units'], 'm')
        self.assertEqual(B.ode_options._states['x']['rate_source'], 'xdot')
        self.assertEqual(B.ode_options._states['y']['units'], 'm')
        self.assertEqual(B.ode_options._states['y']['rate_source'], 'ydot')
        self.assertEqual(B.ode_options._states['v']['units'], 'm/s')
        self.assertEqual(B.ode_options._states['v']['rate_source'], 'vdot')
        self.assertEqual(B.ode_options._states['v']['targets'], ['v'])
        self.assertEqual(B.ode_options._parameters['theta']['targets'], ['theta'])
        self.assertEqual(B.ode_options._parameters['theta']['units'], 'rad')

    def test_str(self):

        @declare_time(units='s', targets=['foo', 'bar'])
        @declare_state('x', rate_source='xdot', units='m')
        @declare_state('y', rate_source='ydot', units='m')
        @declare_state('v', rate_source='vdot', targets=['v'], units='m/s')
        @declare_parameter('theta', targets=['theta'], units='rad')
        @declare_parameter('g', units='m/s**2', targets=['g'], dynamic=False)
        class B(_BrachistochroneODE):
            pass

        s = str(B.ode_options)

        expected = """Time Options:
    targets: ['foo', 'bar']
    units: s
State Options:
    v
        rate_source: vdot
        targets: ['v']
        shape: (1,)
        units: m/s
    x
        rate_source: xdot
        targets: []
        shape: (1,)
        units: m
    y
        rate_source: ydot
        targets: []
        shape: (1,)
        units: m
Parameter Options:
    g
        targets: ['g']
        shape: (1,)
        dynamic: True
        units: m/s**2
    theta
        targets: ['theta']
        shape: (1,)
        dynamic: True
        units: rad"""

        self.assertEqual(s, expected)


if __name__ == "__main__":
    unittest.main()
