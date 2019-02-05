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
        from openmdao.api import Problem, Group, ScipyOptimizeDriver, DirectSolver, \
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

        phase.add_control('theta', units='deg', rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_design_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = DirectSolver(assemble_jac=True)
        p.model.options['assembled_jac_type'] = 'csc'

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

        p.setup()

        p['phase0.t_initial'] = 0.0
        p['phase0.t_duration'] = 2.0

        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='state_input')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='state_input')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='state_input')
        p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='control_input')

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_rel_error(self, p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

        cr = CaseReader('brachistochrone_solution.db')
        system_cases = cr.list_cases('root')
        case = cr.get_case(system_cases[-1])

        outputs = dict([(o[0], o[1]) for o in case.list_outputs(units=True, shape=True,
                                                                out_stream=None)])

        assert_rel_error(self, p['phase0.controls:theta'],
                         outputs['phase0.indep_controls.controls:theta']['value'])

        phase0_options = cr.system_metadata['phase0']['component_options']

        num_segments = phase0_options['num_segments']
        transcription_order = phase0_options['transcription_order']
        segment_ends = phase0_options['segment_ends']
        ode_class = phase0_options['ode_class']

        print(num_segments)
        print(transcription_order)
        print(ode_class)
        print(segment_ends)
