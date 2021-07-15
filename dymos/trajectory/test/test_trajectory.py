import os
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE


@use_tempdirs
class TestTrajectory(unittest.TestCase):

    def tearDown(self):
        for filename in ['test_trajectory_rec.db', 'total_coloring.pkl']:
            if os.path.exists(filename):
                os.remove(filename)

    def setUp(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

    def test_linked_phases(self):
        burn1_accel = self.p.get_val('burn1.states:accel')
        burn2_accel = self.p.get_val('burn2.states:accel')
        accel_link_error = self.p.get_val('linkages.burn1:accel_final|burn2:accel_initial')
        assert_near_equal(accel_link_error, burn1_accel[-1]-burn2_accel[0])


@use_tempdirs
class TestLinkages(unittest.TestCase):

    def tearDown(self):
        for filename in ['test_trajectory_rec.db', 'total_coloring.pkl']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_linked_controls(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['u1'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

        burn1_u1_final = p.get_val('burn1.timeseries.controls:u1')[-1, ...]
        burn2_u1_initial = p.get_val('burn2.timeseries.controls:u1')[0, ...]

        u1_linkage_error = p.get_val('linkages.burn1:u1_final|burn2:u1_initial')
        assert_near_equal(u1_linkage_error, burn1_u1_final - burn2_u1_initial)

    def test_linked_parameters(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['c'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

        burn1_c_final = p.get_val('burn1.timeseries.parameters:c')[-1, ...]
        burn2_c_initial = p.get_val('burn2.timeseries.parameters:c')[0, ...]

        c_linkage_error = p.get_val('linkages.burn1:c_final|burn2:c_initial')
        assert_near_equal(c_linkage_error, burn1_c_final - burn2_c_initial)

    def test_linked_control_to_polynomial_control(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_polynomial_control('u1', order=2, units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['u1'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses
        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.polynomial_controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

        burn1_u1_final = p.get_val('burn1.timeseries.controls:u1')[-1, ...]
        burn2_u1_initial = p.get_val('burn2.timeseries.polynomial_controls:u1')[0, ...]

        u1_linkage_error = p.get_val('linkages.burn1:u1_final|burn2:u1_initial')
        assert_near_equal(u1_linkage_error, burn1_u1_final - burn2_u1_initial)

    def test_linked_control_rate(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['u1_rate'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

        burn1_u1_final = p.get_val('burn1.timeseries.control_rates:u1_rate')[-1, ...]
        burn2_u1_initial = p.get_val('burn2.timeseries.control_rates:u1_rate')[0, ...]

        u1_linkage_error = p.get_val('linkages.burn1:u1_rate_final|burn2:u1_rate_initial')
        assert_near_equal(u1_linkage_error, burn1_u1_final - burn2_u1_initial)

    def test_linked_control_rate2(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=5))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=5))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['u1_rate2'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

        burn1_u1_final = p.get_val('burn1.timeseries.control_rates:u1_rate2')[-1, ...]
        burn2_u1_initial = p.get_val('burn2.timeseries.control_rates:u1_rate2')[0, ...]

        u1_linkage_error = p.get_val('linkages.burn1:u1_rate2_final|burn2:u1_rate2_initial')
        assert_near_equal(u1_linkage_error, burn1_u1_final - burn2_u1_initial)

    def test_linked_polynomial_control_rate(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=5))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=5))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_polynomial_control('u1', units='deg', order=2, scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['u1_rate'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses

        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.polynomial_controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

        burn1_u1_final = p.get_val('burn1.timeseries.control_rates:u1_rate')[-1, ...]
        burn2_u1_initial = p.get_val('burn2.timeseries.polynomial_control_rates:u1_rate')[0, ...]

        u1_linkage_error = p.get_val('linkages.burn1:u1_rate_final|burn2:u1_rate_initial')
        assert_near_equal(u1_linkage_error, burn1_u1_final - burn2_u1_initial)

    def test_linked_polynomial_control_rate2(self):

        self.traj = dm.Trajectory()
        p = self.p = om.Problem(model=self.traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=5))

        self.traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        self.traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=5))

        self.traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_polynomial_control('u1', units='deg', order=2, scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        self.traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                              vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])
        self.traj.link_phases(phases=['burn1', 'burn2'], vars=['u1_rate2'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True)

        # Set Initial Guesses
        p.set_val('burn1.t_initial', value=0.0)
        p.set_val('burn1.t_duration', value=2.25)

        p.set_val('burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))
        p.set_val('burn1.parameters:c', value=1.5)

        p.set_val('coast.t_initial', value=2.25)
        p.set_val('coast.t_duration', value=3.0)

        p.set_val('coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('coast.states:accel', value=coast.interp('accel', [0, 0]))
        p.set_val('coast.controls:u1', value=coast.interp('u1', [0, 0]))
        p.set_val('coast.parameters:c', value=1.5)

        p.set_val('burn2.t_initial', value=5.25)
        p.set_val('burn2.t_duration', value=1.75)

        p.set_val('burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('burn2.polynomial_controls:u1', value=burn2.interp('u1', [1, 1]))
        p.set_val('burn2.parameters:c', value=1.5)

        p.run_model()

        burn1_u1_final = p.get_val('burn1.timeseries.control_rates:u1_rate2')[-1, ...]
        burn2_u1_initial = p.get_val('burn2.timeseries.polynomial_control_rates:u1_rate2')[0, ...]

        u1_linkage_error = p.get_val('linkages.burn1:u1_rate2_final|burn2:u1_rate2_initial')
        assert_near_equal(u1_linkage_error, burn1_u1_final - burn2_u1_initial)


@use_tempdirs
class TestInvalidLinkages(unittest.TestCase):

    def test_invalid_linkage_variable(self):
        traj = dm.Trajectory()
        p = om.Problem(model=traj)

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'burn2'], vars=['u1', 'bar'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        self.assertEqual(str(e.exception), 'Error in linking bar from burn1 to bar in burn2: '
                                           'Unable to find variable \'bar\' in phase '
                                           '\'phases.burn1\' or its ODE.')

    def test_invalid_linkage_phase(self):
        p = om.Problem(model=om.Group())

        traj = dm.Trajectory()
        p.model.add_subsystem('traj', subsys=traj)

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'foo'], vars=['u1', 'u1'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        self.assertEqual(str(e.exception), 'Invalid linkage.  Phase \'foo\' does not exist in '
                                           'trajectory \'traj\'.')

    def test_invalid_linkage_args(self):
        p = om.Problem(model=om.Group())

        traj = dm.Trajectory()
        p.model.add_subsystem('traj', subsys=traj)

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        with self.assertWarns(UserWarning) as w:
            traj.add_linkage_constraint(phase_a='burn1', phase_b='burn2', var_a='accel', var_b='accel',
                                        lower=-5, upper=5, ref0=-5, ref=5, linear=True, connected=True)

        expected_warning = 'Invalid option in linkage between burn1:accel and burn2:accel in trajectory traj. ' \
                           'The following options for add_linkage_constraint were specified but ' \
                           'not valid when option \'connected\' is True: lower upper ref0 ref linear'

        self.assertEqual(expected_warning, str(w.warning))


    def test_linkage_units(self):
        """
        Test that a linkage variable being linked to mutliple phases is done using the correct
        units each time when the linkage units are different.
        """
        import numpy as np
        from scipy.interpolate import interp1d
        import openmdao.api as om
        import dymos as dm
        from dymos.models.atmosphere.atmos_1976 import USatm1976Data

        '''
        The is a test version of the multi-phase cannonball problem. It is used 
        to demonstrate the unit linkage problem found by the GASPy team. For 
        ease of reading, each line of code in the ODE that has been changed in 
        to demonstrate the test case includes a comment that looks like this:

        ###test modification here###

        This should allow for easy searching through the code for the relevant
        changes.
        '''

        ## note: the values of h in phases 'ascent and 'descent are supposed to be linked, however, when we print them out with the same units
        ## we see that the final magnitude of h in ascent is equal to the initial magnitude of h in descent only if we assume
        ## that h in ascent is in ft and h in descent is in m. However, they are supposed to be the same units! This can be
        ## further explored by changing the units on the linkage constraint between ascent and descent. (line 240)

        ## Also note: this issue does not exist if there are only two phases. This is why the 'start' phase was introduced.

        ## Additional note: this model does not converge to a 0-1, it has an iteration limit. For this testing purpose
        ## convergence is irrelevant.

        ## Final note: By commenting out line 240 and commenting in line 241 I would have expected dymos to internally convert
        ## the units, but instead it throws an error.


        class CannonballSizeComp(om.ExplicitComponent):

            def setup(self):
                self.add_input(name='radius', val=1.0, desc='cannonball radius', units='m')
                self.add_input(name='dens', val=7870., desc='cannonball density', units='kg/m**3')

                self.add_output(name='mass', shape=(1,), desc='cannonball mass', units='kg')
                self.add_output(name='S', shape=(1,), desc='aerodynamic reference area', units='m**2')

                self.declare_partials(of='mass', wrt='dens')
                self.declare_partials(of='mass', wrt='radius')

                self.declare_partials(of='S', wrt='radius')

            def compute(self, inputs, outputs):
                radius = inputs['radius']
                dens = inputs['dens']

                outputs['mass'] = (4/3.) * dens * np.pi * radius ** 3
                outputs['S'] = np.pi * radius ** 2

            def compute_partials(self, inputs, partials):
                radius = inputs['radius']
                dens = inputs['dens']

                partials['mass', 'dens'] = (4/3.) * np.pi * radius ** 3
                partials['mass', 'radius'] = 4. * dens * np.pi * radius ** 2

                partials['S', 'radius'] = 2 * np.pi * radius


        class CannonballODE(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('num_nodes', types=int)
                self.options.declare('alt_unit', values=('ft', 'm', 'NM')) ###test modification here###

            def setup(self):
                nn = self.options['num_nodes']

                # static parameters
                self.add_input('m', units='kg')
                self.add_input('S', units='m**2')
                # 0.5 good assumption for a sphere
                self.add_input('CD', 0.5)

                # time varying inputs
                self.add_input('h', units=self.options['alt_unit'], shape=nn) ###test modification here###
                self.add_input('v', units='m/s', shape=nn)
                self.add_input('gam', units='rad', shape=nn)

                # state rates
                self.add_output('v_dot', shape=nn, units='m/s**2', tags=['dymos.state_rate_source:v'])
                self.add_output('gam_dot', shape=nn, units='rad/s', tags=['dymos.state_rate_source:gam'])
                self.add_output('h_dot', shape=nn, units=self.options['alt_unit']+'/s', tags=['dymos.state_rate_source:h']) ###test modification here###
                self.add_output('r_dot', shape=nn, units='m/s', tags=['dymos.state_rate_source:r'])
                self.add_output('ke', shape=nn, units='J')

                self.declare_coloring(wrt='*', method='cs')

                alt_data = USatm1976Data.alt * om.unit_conversion('ft', 'm')[0]
                rho_data = USatm1976Data.rho * om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
                self.rho_interp = interp1d(np.array(alt_data, dtype=complex),
                                            np.array(rho_data, dtype=complex),
                                            kind='linear')

            def compute(self, inputs, outputs):

                gam = inputs['gam']
                v = inputs['v']
                h = inputs['h']
                m = inputs['m']
                S = inputs['S']
                CD = inputs['CD']

                if self.options['alt_unit'] == 'ft': ###test modification here###
                    h = h*0.3048 ###test modification here### (convert to units necessary for component)
                elif self.options['alt_unit'] == 'NM': ###test modification here###
                    h = h*1852 ###test modification here### (convert to units necessary for component)

                GRAVITY = 9.80665  # m/s**2

                # handle complex-step gracefully from the interpolant
                if np.iscomplexobj(h):
                    rho = self.rho_interp(inputs['h'])
                else:
                    rho = self.rho_interp(inputs['h']).real

                q = 0.5*rho*inputs['v']**2
                qS = q * S
                D = qS * CD
                cgam = np.cos(gam)
                sgam = np.sin(gam)
                outputs['v_dot'] = - D/m-GRAVITY*sgam
                outputs['gam_dot'] = -(GRAVITY/v)*cgam

                if self.options['alt_unit'] == 'ft': ###test modification here###
                    outputs['h_dot'] = v*sgam/0.3048 ###test modification here### (convert to units necessary for component)
                elif self.options['alt_unit'] == 'NM': ###test modification here###
                    outputs['h_dot'] = v*sgam/1852 ###test modification here### (convert to units necessary for component)
                outputs['h_dot'] = v*sgam

                outputs['r_dot'] = v*cgam
                outputs['ke'] = 0.5*m*v**2

        #############################################
        # Setup the Dymos problem
        #############################################

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.declare_coloring()

        p.driver.opt_settings['Major iterations limit'] = 5

        p.model.add_subsystem('size_comp', CannonballSizeComp(),
                                promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10,
                                ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)

        ################ PHASE 1 ######################
        start = dm.Phase(ode_class=CannonballODE, transcription=transcription,
                          ode_init_kwargs={'alt_unit': 'm'}) # added a third phase because problem does not occur with only two phases

        start = traj.add_phase('start', start)

        start.set_time_options(fix_initial=True, duration_bounds=(1, 100),
                                duration_ref=100, units='s')
        start.set_state_options('r', fix_initial=True, fix_final=False)
        start.set_state_options('h', fix_initial=True, fix_final=False)
        start.set_state_options('gam', fix_initial=False, fix_final=False)
        start.set_state_options('v', fix_initial=False, fix_final=False)

        start.add_parameter('S', units='m**2', static_target=True)
        start.add_parameter('m', units='kg', static_target=True)

        # Limit the muzzle energy
        start.add_boundary_constraint('ke', loc='initial',
                                        upper=400000, lower=0, ref=100000) ###test modification here###

        start.add_boundary_constraint('h', loc='final', equals=2, units='m') ###test modification here###

        ################ PHASE 2 ######################

        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription,
                          ode_init_kwargs={'alt_unit': 'm'}) #modified this phase to take in unit option

        ascent = traj.add_phase('ascent', ascent)

        ascent.set_time_options(fix_initial=False, duration_bounds=(1, 100),
                                duration_ref=100, units='s')
        ascent.set_state_options('r', fix_initial=False, fix_final=False)
        ascent.set_state_options('h', fix_initial=False, fix_final=False)
        ascent.set_state_options('gam', fix_initial=False, fix_final=True)
        ascent.set_state_options('v', fix_initial=False, fix_final=False)

        ascent.add_parameter('S', units='m**2', static_target=True)
        ascent.add_parameter('m', units='kg', static_target=True)

        ################ PHASE 3 ######################

        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription,
                           ode_init_kwargs={'alt_unit': 'ft'}) #modified this phase to take in unit option

        traj.add_phase('descent', descent)

        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                    duration_ref=100, units='s')
        descent.add_state('r')
        descent.add_state('h', fix_initial=False, fix_final=True)
        descent.add_state('gam', fix_initial=False, fix_final=False)
        descent.add_state('v', fix_initial=False, fix_final=False)

        descent.add_parameter('S', units='m**2', static_target=True)
        descent.add_parameter('m', units='kg', static_target=True)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                            targets={'ascent': ['CD'], 'descent': ['CD']},
                            val=0.5, units=None, opt=False, static_target=True)

        traj.add_parameter('m', units='kg', val=1.0,
                            targets={'ascent': 'mass', 'descent': 'mass'}, static_target=True)

        # In this case, by omitting targets, we're connecting these
        # parameters to parameters with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005, static_target=True)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['start', 'ascent'], vars=['*'])

        # Now add another h linkage with different units.
        traj.add_linkage_constraint('ascent', 'descent', 'h', 'h', 'final', 'initial', units='ft')
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'v', 'gam'])

        # Issue Connections
        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # A linear solver at the top level can improve performance.
        p.model.linear_solver = om.DirectSolver()

        # Finish Problem Setup
        p.setup()

        #############################################
        # Set constants and initial guesses
        #############################################
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        p.set_val('traj.parameters:CD', 0.5)

        p.set_val('traj.start.t_initial', 0.0)
        p.set_val('traj.start.t_duration', 10.0)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.start.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.start.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.start.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.start.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')

        #####################################################
        # Run the optimization and final explicit simulation
        #####################################################
        dm.run_problem(p)

        h0_final = p.get_val('traj.start.timeseries.states:h', units='m')[-1, ...]
        h1_initial = p.get_val('traj.ascent.timeseries.states:h', units='m')[0, ...]
        h1_final = p.get_val('traj.ascent.timeseries.states:h', units='m')[-1, ...]
        h2_initial = p.get_val('traj.descent.timeseries.states:h', units='m')[0, ...]

        assert_near_equal(h0_final, h1_initial, tolerance=1.0E-6)
        assert_near_equal(h1_final, h2_initial, tolerance=1.0E-6)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
