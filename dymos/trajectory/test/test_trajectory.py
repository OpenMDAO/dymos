import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE


class TestTrajectory(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['test_trajectory_rec.db', 'total_coloring.pkl']:
            if os.path.exists(filename):
                os.remove(filename)

    @classmethod
    def setUpClass(cls):

        cls.traj = dm.Trajectory()
        p = cls.p = om.Problem(model=cls.traj)

        # Since we're only testing features like get_values that don't rely on a converged
        # solution, no driver is attached.  We'll just invoke run_model.

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        cls.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        cls.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', targets=['u1'], opt=False, val=0.0, units='deg')
        coast.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        cls.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        cls.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                             vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        cls.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.model.add_recorder(om.SqliteRecorder('test_trajectory_rec.db'))

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
        assert_near_equal(accel_link_error, burn2_accel[0]-burn1_accel[-1])


class TestInvalidLinkages(unittest.TestCase):

    def test_invalid_linkage_variable(self):
        traj = dm.Trajectory()
        p = om.Problem(model=traj)

        # Since we're only testing features like get_values that don't rely on a converged
        # solution, no driver is attached.  We'll just invoke run_model.

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', targets=['u1'], opt=False, val=0.0, units='deg')
        coast.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'burn2'], vars=['u1', 'bar'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.model.add_recorder(om.SqliteRecorder('test_trajectory_rec.db'))

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        self.assertEqual(str(e.exception), 'Cannot find linkage variable \'bar\' in '
                                           'phase \'burn1\'.  Only states, time, controls, '
                                           'or parameters may be linked via link_phases.')

    def test_invalid_linkage_phase(self):
        p = om.Problem(model=om.Group())

        traj = dm.Trajectory()
        p.model.add_subsystem('traj', subsys=traj)

        # Since we're only testing features like get_values that don't rely on a converged
        # solution, no driver is attached.  We'll just invoke run_model.

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', targets=['u1'], opt=False, val=0.0, units='deg')
        coast.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_design_parameter('c', opt=False, val=1.5, targets=['c'], units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'foo'], vars=['u1', 'u1'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.model.add_recorder(om.SqliteRecorder('test_trajectory_rec.db'))

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        self.assertEqual(str(e.exception), 'Invalid linkage.  Phase \'foo\' does not exist in '
                                           'trajectory \'traj\'.')
