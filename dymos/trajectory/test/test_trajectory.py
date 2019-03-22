from __future__ import print_function, division, absolute_import

import os
import unittest

import numpy as np

from openmdao.api import Problem, DirectSolver, SqliteRecorder
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase, Trajectory
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE


class TestTrajectory(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['test_trajectory_rec.db', 'coloring.json']:
            if os.path.exists(filename):
                os.remove(filename)

    @classmethod
    def setUpClass(cls):

        cls.traj = Trajectory()
        p = cls.p = Problem(model=cls.traj)

        # Since we're only testing features like get_values that don't rely on a converged
        # solution, no driver is attached.  We'll just invoke run_model.

        # First Phase (burn)
        burn1 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=4,
                      transcription_order=3,
                      compressed=True)

        cls.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10))
        burn1.set_state_options('r', fix_initial=True, fix_final=False)
        burn1.set_state_options('theta', fix_initial=True, fix_final=False)
        burn1.set_state_options('vr', fix_initial=True, fix_final=False, defect_scaler=0.1)
        burn1.set_state_options('vt', fix_initial=True, fix_final=False, defect_scaler=0.1)
        burn1.set_state_options('accel', fix_initial=True, fix_final=False)
        burn1.set_state_options('deltav', fix_initial=True, fix_final=False)
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg')
        burn1.add_design_parameter('c', opt=False, val=1.5)

        # Second Phase (Coast)

        coast = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=10,
                      transcription_order=3,
                      compressed=True)

        cls.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
        coast.set_state_options('r', fix_initial=False, fix_final=False)
        coast.set_state_options('theta', fix_initial=False, fix_final=False)
        coast.set_state_options('vr', fix_initial=False, fix_final=False)
        coast.set_state_options('vt', fix_initial=False, fix_final=False)
        coast.set_state_options('accel', fix_initial=True, fix_final=True)
        coast.set_state_options('deltav', fix_initial=False, fix_final=False)
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_design_parameter('c', opt=False, val=1.5)

        # Third Phase (burn)

        burn2 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=3,
                      transcription_order=3,
                      compressed=True)

        cls.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
        burn2.set_state_options('r', fix_initial=False, fix_final=True, defect_scaler=1.0)
        burn2.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.set_state_options('vr', fix_initial=False, fix_final=True, defect_scaler=0.1)
        burn2.set_state_options('vt', fix_initial=False, fix_final=True, defect_scaler=0.1)
        burn2.set_state_options('accel', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.set_state_options('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          ref0=0, ref=10)
        burn2.add_design_parameter('c', opt=False, val=1.5)

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        cls.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                             vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        cls.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = DirectSolver()

        p.model.add_recorder(SqliteRecorder('test_trajectory_rec.db'))

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interpolate(ys=[1, 1.5], nodes='state_input'))
        p.set_val('burn1.states:theta', value=burn1.interpolate(ys=[0, 1.7], nodes='state_input'))
        p.set_val('burn1.states:vr', value=burn1.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('burn1.states:vt', value=burn1.interpolate(ys=[1, 1], nodes='state_input'))
        p.set_val('burn1.states:accel', value=burn1.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1], nodes='state_input'))
        p.set_val('burn1.controls:u1',
                  value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))
        p.set_val('burn1.design_parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interpolate(ys=[1.3, 1.5], nodes='state_input'))
        p.set_val('coast.states:theta',
                  value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))
        p.set_val('coast.states:vr', value=coast.interpolate(ys=[0.3285, 0], nodes='state_input'))
        p.set_val('coast.states:vt', value=coast.interpolate(ys=[0.97, 1], nodes='state_input'))
        p.set_val('coast.states:accel', value=coast.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('coast.controls:u1', value=coast.interpolate(ys=[0, 0], nodes='control_input'))
        p.set_val('coast.design_parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interpolate(ys=[1, 3], nodes='state_input'))
        p.set_val('burn2.states:theta', value=burn2.interpolate(ys=[0, 4.0], nodes='state_input'))
        p.set_val('burn2.states:vr', value=burn2.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('burn2.states:vt',
                  value=burn2.interpolate(ys=[1, np.sqrt(1 / 3)], nodes='state_input'))
        p.set_val('burn2.states:accel', value=burn2.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('burn2.states:deltav',
                  value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
        p.set_val('burn2.controls:u1', value=burn2.interpolate(ys=[1, 1], nodes='control_input'))
        p.set_val('burn2.design_parameters:c', value=1.5)

        p.run_model()

    def test_linked_phases(self):
        burn1_accel = self.p.get_val('burn1.states:accel')
        burn2_accel = self.p.get_val('burn2.states:accel')
        accel_link_error = self.p.get_val('linkages.burn1|burn2_accel')
        assert_rel_error(self, accel_link_error, burn2_accel[0]-burn1_accel[-1])


class TestInvalidLinkages(unittest.TestCase):

    def test_invalid_linkage_variable(self):
        traj = Trajectory()
        p = p = Problem(model=traj)

        # Since we're only testing features like get_values that don't rely on a converged
        # solution, no driver is attached.  We'll just invoke run_model.

        # First Phase (burn)
        burn1 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=4,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10))
        burn1.set_state_options('r', fix_initial=True, fix_final=False)
        burn1.set_state_options('theta', fix_initial=True, fix_final=False)
        burn1.set_state_options('vr', fix_initial=True, fix_final=False, defect_scaler=0.1)
        burn1.set_state_options('vt', fix_initial=True, fix_final=False, defect_scaler=0.1)
        burn1.set_state_options('accel', fix_initial=True, fix_final=False)
        burn1.set_state_options('deltav', fix_initial=True, fix_final=False)
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg')
        burn1.add_design_parameter('c', opt=False, val=1.5)

        # Second Phase (Coast)

        coast = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=10,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
        coast.set_state_options('r', fix_initial=False, fix_final=False)
        coast.set_state_options('theta', fix_initial=False, fix_final=False)
        coast.set_state_options('vr', fix_initial=False, fix_final=False)
        coast.set_state_options('vt', fix_initial=False, fix_final=False)
        coast.set_state_options('accel', fix_initial=True, fix_final=True)
        coast.set_state_options('deltav', fix_initial=False, fix_final=False)
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_design_parameter('c', opt=False, val=1.5)

        # Third Phase (burn)

        burn2 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=3,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
        burn2.set_state_options('r', fix_initial=False, fix_final=True, defect_scaler=1.0)
        burn2.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.set_state_options('vr', fix_initial=False, fix_final=True, defect_scaler=0.1)
        burn2.set_state_options('vt', fix_initial=False, fix_final=True, defect_scaler=0.1)
        burn2.set_state_options('accel', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.set_state_options('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          ref0=0, ref=10)
        burn2.add_design_parameter('c', opt=False, val=1.5)

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'burn2'], vars=['foo', 'bar'])

        # Finish Problem Setup
        p.model.linear_solver = DirectSolver()

        p.model.add_recorder(SqliteRecorder('test_trajectory_rec.db'))

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        self.assertEqual(str(e.exception), 'Cannot find linkage variable \'bar\' in '
                                           'phase \'burn1\'.  Only states, time, controls, '
                                           'or parameters may be linked via link_phases.')
