from __future__ import print_function, absolute_import, division

import os
import unittest


class TestBrachistochroneRecordingExample(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['brachistochrone_solution.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_brachistochrone_recording_for_docs(self):
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, CSCJacobian, DirectSolver, \
            SqliteRecorder, CaseReader
        from openmdao.utils.assert_utils import assert_rel_error
        from dymos import Phase
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = Problem(model=Group())
        p.driver = ScipyOptimizeDriver()

        phase = Phase('gauss-lobatto',
                      ode_class=BrachistochroneODE,
                      num_segments=10)

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)

        phase.add_control('theta', units='deg', dynamic=True,
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_control('g', units='m/s**2', dynamic=False, opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.jacobian = CSCJacobian()
        p.model.linear_solver = DirectSolver()

        # Recording
        rec = SqliteRecorder('brachistochrone_solution.db')

        p.driver.recording_options['record_desvars'] = True
        p.driver.recording_options['record_responses'] = True
        p.driver.recording_options['record_objectives'] = True
        p.driver.recording_options['record_constraints'] = True

        p.model.recording_options['record_metadata'] = True

        p.driver.add_recorder(rec)
        p.model.add_recorder(rec)
        phase.add_recorder(rec)

        p.setup(mode='rev')

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='disc')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='disc')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='disc')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='all')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, phase.get_values('time')[-1], 1.8016, tolerance=1.0E-3)

        cr = CaseReader('brachistochrone_solution.db')

        outputs = dict([(o[0], o[1]) for o in cr.list_outputs(units=True, shape=True,
                                                              out_stream=None)])

        assert_rel_error(self, p['phase0.controls:theta'],
                         outputs['phase0.indep_controls.controls:theta']['value'])

        phase0_metadata = cr.system_metadata['phase0']['component_metadata']

        num_segments = phase0_metadata['num_segments']
        transcription_order = phase0_metadata['transcription_order']
        segment_ends = phase0_metadata['segment_ends']
        ode_class = phase0_metadata['ode_class']

        print(num_segments)
        print(transcription_order)
        print(ode_class)
        print(segment_ends)
