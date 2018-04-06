from __future__ import print_function, division, absolute_import

import os
import os.path
import unittest

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
            phase = Phase('gauss-lobatto',
                          ode_class=_B,
                          num_segments=8,
                          transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_no_metadata2(self):

        with self.assertRaises(ValueError) as e:
            phase = Phase('gauss-lobatto',
                          ode_class=_C,
                          num_segments=8,
                          transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_invalid_metadata(self):

        with self.assertRaises(ValueError) as e:
            phase = Phase('gauss-lobatto',
                          ode_class=_D,
                          num_segments=8,
                          transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class has no ODE metadata.  '
                                           'Use @declare_time, @declare_stateand @declare_control '
                                           'to assign ODE metadata.')

    def test_invalid_ode_class_instance(self):

        with self.assertRaises(ValueError) as e:
            phase = Phase('gauss-lobatto',
                          ode_class=BrachistochroneODE(),
                          num_segments=8,
                          transcription_order=3)
        self.assertEqual(str(e.exception), 'ode_class must be a class, not an instance.')


if __name__ == '__main__':
    unittest.main()
