from __future__ import print_function, division, absolute_import

import os
import unittest

import numpy as np

from openmdao.api import Problem, DirectSolver, SqliteRecorder
from openmdao.utils.assert_utils import assert_rel_error

from dymos import Phase, Trajectory, load_simulation_results
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE


class TestTrajectorySimulationResults(unittest.TestCase):

    def instantiate_problem(self, idx):

        traj = Trajectory()
        p = Problem(model=traj)

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
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        rec_file = 'two_burn_orbit_raise_example_{0}.db'.format(idx)
        p.driver.add_recorder(SqliteRecorder(rec_file))

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

        # Plot results
        sim_rec_file = 'traj_sim_{0}.db'.format(idx)
        exp_out = traj.simulate(times=50, record_file=sim_rec_file)

        loaded_exp_out = load_simulation_results(sim_rec_file)

        return exp_out, loaded_exp_out, rec_file, sim_rec_file

    def cleanup(self, filenames):
        for f in filenames:
            if os.path.exists(f):
                os.remove(f)

    def test_returned_and_loaded_equivalent(self):

        exp_out, loaded_exp_out, rec_file, sim_rec_file = self.instantiate_problem(0)

        for phase in ('burn1', 'coast', 'burn2'):
            t_returned = exp_out.get_values('time')[phase]
            r_returned = exp_out.get_values('r')[phase]

            t_loaded = loaded_exp_out.get_values('time')[phase]
            r_loaded = loaded_exp_out.get_values('r')[phase]

            assert_rel_error(self, t_returned, t_loaded)
            assert_rel_error(self, r_returned, r_loaded)

        self.cleanup([rec_file, sim_rec_file])

    def test_returned_and_loaded_flattened_equivalent(self):

        exp_out, loaded_exp_out, rec_file, sim_rec_file = self.instantiate_problem(1)

        for var in ('time', 'r', 'theta', 'u1', 'c'):

            returned = exp_out.get_values(var, flat=True)
            loaded = loaded_exp_out.get_values(var, flat=True)

            assert_rel_error(self, returned, loaded)

        self.cleanup([rec_file, sim_rec_file])

    def test_return_flattened(self):

        exp_out, loaded_exp_out, rec_file, sim_rec_file = self.instantiate_problem(2)

        t_flat_returned = exp_out.get_values('time', flat=True)
        r_flat_returned = exp_out.get_values('r', flat=True)

        t_flat_loaded = loaded_exp_out.get_values('time', flat=True)
        r_flat_loaded = loaded_exp_out.get_values('r', flat=True)

        start_idx = 0

        for phase in ('burn1', 'coast', 'burn2'):
            t_returned = exp_out.get_values('time')[phase]
            r_returned = exp_out.get_values('r')[phase]

            t_loaded = loaded_exp_out.get_values('time')[phase]
            r_loaded = loaded_exp_out.get_values('r')[phase]

            num_returned = len(t_returned)
            num_loaded = len(t_loaded)

            self.assertEqual(num_returned, num_loaded)

            assert_rel_error(self,
                             t_flat_returned[start_idx: start_idx + num_returned, ...],
                             t_returned)
            assert_rel_error(self,
                             t_flat_loaded[start_idx: start_idx + num_loaded, ...],
                             t_loaded)

            assert_rel_error(self,
                             r_flat_returned[start_idx: start_idx + num_returned, ...],
                             r_returned)

            assert_rel_error(self,
                             r_flat_loaded[start_idx: start_idx + num_loaded, ...],
                             r_loaded)

            start_idx += num_returned

        self.cleanup([rec_file, sim_rec_file])

    def test_str_phases_arg(self):

        exp_out, loaded_exp_out, rec_file, sim_rec_file = self.instantiate_problem(2)

        t_flat_returned_str = exp_out.get_values('time', phases="burn1", flat=True)
        t_flat_returned_list = exp_out.get_values('time', phases=["burn1"], flat=True)

        self.assertTrue(np.all(t_flat_returned_str==t_flat_returned_list))

        self.cleanup([rec_file, sim_rec_file])

    def test_nonexistent_var(self):

        exp_out, loaded_exp_out, rec_file, sim_rec_file = self.instantiate_problem(3)

        with self.assertRaises(KeyError) as e:
            loaded_exp_out.get_values('foo')
            self.assertEqual(str(e.exception), 'Variable "foo" not found in trajectory '
                                               'simulation results.')

        self.cleanup([rec_file, sim_rec_file])


if __name__ == "__main__": 
    unittest.main()
