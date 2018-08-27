from __future__ import print_function, division, absolute_import

import os
import os.path
import unittest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal

from openmdao.api import ExplicitComponent, Group

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


OPTIMIZER = 'SLSQP'
SHOW_PLOTS = False


class _A(object):
    pass


class _B(Group):
    pass


class _C(ExplicitComponent):
    pass


class _D(ExplicitComponent):
    ode_options = None


class TestPhaseBase(unittest.TestCase):

    def test_invalid_ode_class_wrong_class(self):

        with self.assertRaises(ValueError) as e:
            phase = Phase('gauss-lobatto',
                          ode_class=_A,
                          num_segments=8,
                          transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class must be derived from openmdao.core.System.')

    def test_invalid_ode_class_no_metadata(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=_B,
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_no_metadata2(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=_C,
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_invalid_metadata(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=_D,
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_instance(self):

        with self.assertRaises(ValueError) as e:
            Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE(),
                  num_segments=8,
                  transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class must be a class, not an instance.')

    def test_invalid_design_parameter_name(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with self.assertRaises(ValueError) as e:

            p.add_design_parameter('foo')

        expected = 'foo is not a controllable parameter in the ODE system.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_design_parameter_as_design_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_design_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as a design parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_control_as_design_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_control('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as a control.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_input_parameter_as_design_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_input_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_design_parameter('theta')

        expected = 'theta has already been added as an input parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_options_nonoptimal_design_param(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.add_design_parameter('g', opt=False, lower=5, upper=10, ref0=5, ref=10,
                                   scaler=1, adder=0)

        expected = 'Invalid options for non-optimal design parameter "g":' \
                   'lower, upper, scaler, adder, ref, ref0'

        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message), expected)

    def test_invalid_input_parameter_name(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with self.assertRaises(ValueError) as e:

            p.add_input_parameter('foo')

        expected = 'foo is not a controllable parameter in the ODE system.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_design_parameter_as_input_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_design_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as a design parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_control_as_input_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_control('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as a control.'
        self.assertEqual(str(e.exception), expected)

    def test_add_existing_input_parameter_as_input_parameter(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        p.add_input_parameter('theta')

        with self.assertRaises(ValueError) as e:
            p.add_input_parameter('theta')

        expected = 'theta has already been added as an input parameter.'
        self.assertEqual(str(e.exception), expected)

    def test_invalid_options_nonoptimal_control(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            p.add_control('theta', opt=False, lower=5, upper=10, ref0=5, ref=10,
                          scaler=1, adder=0)

        expected = 'Invalid options for non-optimal control "theta":' \
                   'lower, upper, scaler, adder, ref, ref0'

        self.assertEqual(len(w), 1)
        self.assertEqual(str(w[0].message), expected)

    def test_invalid_boundary_loc(self):

        p = Phase('gauss-lobatto',
                  ode_class=BrachistochroneODE,
                  num_segments=8,
                  transcription_order=3)

        with self.assertRaises(ValueError) as e:
            p.add_boundary_constraint('x', loc='foo')

        expected = 'Invalid boundary constraint location "foo". Must be "initial" or "final".'
        self.assertEqual(str(e.exception), expected)


if __name__ == '__main__':
    unittest.main()
