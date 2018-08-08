from __future__ import print_function, division, absolute_import

import os
import sys
import unittest
import warnings

from openmdao.api import Problem, Group

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestPhaseTimeOptions(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_invalid_options(self, transcription='gauss-lobatto'):
        p = Problem(model=Group())

        phase = Phase(transcription,
                      ode_class=BrachistochroneODE,
                      num_segments=8,
                      transcription_order=3)

        p.model.add_subsystem('phase0', phase)

        expected_msg0 = 'Phase time options have no effect because fix_initial=True for ' \
                        'phase "phase0": initial_bounds, initial_scaler, initial_adder, ' \
                        'initial_ref, initial_ref0'
        expected_msg1 = 'Phase time options have no effect because fix_duration=True for' \
                        ' phase "phase0": duration_bounds, duration_scaler, ' \
                        'duration_adder, duration_ref, duration_ref0'

        if sys.version_info >= (3, 3):
            with self.assertWarns(RuntimeWarning) as ctx:
                phase.set_time_options(fix_initial=True, fix_duration=True,
                                       initial_bounds=(1.0, 5.0), initial_adder=0.0,
                                       initial_scaler=1.0, initial_ref0=0.0,
                                       initial_ref=1.0, duration_bounds=(1.0, 5.0),
                                       duration_adder=0.0, duration_scaler=1.0, duration_ref0=0.0,
                                       duration_ref=1.0)
                self.assertEqual(len(ctx.warnings), 2,
                                 msg='set_time_options failed to raise two warnings')
                self.assertEqual(str(ctx.warnings[0].message), expected_msg0)
                self.assertEqual(str(ctx.warnings[1].message), expected_msg1)
        else:
            with warnings.catch_warnings(record=True) as ctx:
                warnings.simplefilter('always')
                phase.set_time_options(fix_initial=True, fix_duration=True,
                                       initial_bounds=(1.0, 5.0), initial_adder=0.0,
                                       initial_scaler=1.0, initial_ref0=0.0,
                                       initial_ref=1.0, duration_bounds=(1.0, 5.0),
                                       duration_adder=0.0, duration_scaler=1.0,
                                       duration_ref0=0.0, duration_ref=1.0)
                self.assertEqual(len(ctx), 2,
                                 msg='set_time_options failed to raise two warnings')
                self.assertEqual(str(ctx[0].message), expected_msg0)
                self.assertEqual(str(ctx[1].message), expected_msg1)


if __name__ == '__main__':
    unittest.main()
