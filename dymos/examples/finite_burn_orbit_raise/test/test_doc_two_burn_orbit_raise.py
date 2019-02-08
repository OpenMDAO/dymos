from __future__ import print_function, division, absolute_import

import os
import unittest

import matplotlib
# matplotlib.use('Agg')


class TestTwoBurnOrbitRaiseForDocs(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['coloring.json', 'two_burn_orbit_raise_example_for_docs.db', 'SLSQP.out',
                         'SNOPT_print.out', 'SNOPT_summary.out']:
            if os.path.exists(filename):
                os.remove(filename)

    def test_two_burn_orbit_raise_for_docs(self):
        import numpy as np

        import matplotlib.pyplot as plt

        from openmdao.api import Problem, Group, pyOptSparseDriver, DirectSolver, SqliteRecorder
        from openmdao.utils.assert_utils import assert_rel_error
        from openmdao.utils.general_utils import set_pyoptsparse_opt

        from dymos import Phase, Trajectory
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = Trajectory()
        p = Problem(model=Group())
        p.model.add_subsystem('traj', traj)

        p.driver = pyOptSparseDriver()
        _, optimizer = set_pyoptsparse_opt('SNOPT', fallback=False)
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.options['dynamic_simul_derivs'] = True

        traj.add_design_parameter('c', opt=False, val=1.5, units='DU/TU')

        # First Phase (burn)

        burn1 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=10,
                      transcription_order=3,
                      compressed=True)

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10))
        burn1.set_state_options('r', fix_initial=True, fix_final=False, defect_scaler=100.0)
        burn1.set_state_options('theta', fix_initial=True, fix_final=False, defect_scaler=100.0)
        burn1.set_state_options('vr', fix_initial=True, fix_final=False, defect_scaler=100.0)
        burn1.set_state_options('vt', fix_initial=True, fix_final=False, defect_scaler=100.0)
        burn1.set_state_options('accel', fix_initial=True, fix_final=False)
        burn1.set_state_options('deltav', fix_initial=True, fix_final=False)
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01, rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30)

        # Second Phase (Coast)

        coast = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=10,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10)
        coast.set_state_options('r', fix_initial=False, fix_final=False, defect_scaler=100.0)
        coast.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=100.0)
        coast.set_state_options('vr', fix_initial=False, fix_final=False, defect_scaler=100.0)
        coast.set_state_options('vt', fix_initial=False, fix_final=False, defect_scaler=100.0)
        coast.set_state_options('accel', fix_initial=True, fix_final=True)
        coast.set_state_options('deltav', fix_initial=False, fix_final=False)
        coast.add_control('u1', opt=False, val=0.0, units='deg')

        # Third Phase (burn)

        burn2 = Phase('gauss-lobatto',
                      ode_class=FiniteBurnODE,
                      num_segments=10,
                      transcription_order=3,
                      compressed=True)

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10)
        burn2.set_state_options('r', fix_initial=False, fix_final=True, defect_scaler=100.0)
        burn2.set_state_options('theta', fix_initial=False, fix_final=False, defect_scaler=100.0)
        burn2.set_state_options('vr', fix_initial=False, fix_final=True, defect_scaler=100.0)
        burn2.set_state_options('vt', fix_initial=False, fix_final=True, defect_scaler=100.0)
        burn2.set_state_options('accel', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.set_state_options('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0)
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01, rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30)

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        burn1.add_timeseries_output('pos_x', units='DU')
        coast.add_timeseries_output('pos_x', units='DU')
        burn2.add_timeseries_output('pos_x', units='DU')

        burn1.add_timeseries_output('pos_y', units='DU')
        coast.add_timeseries_output('pos_y', units='DU')
        burn2.add_timeseries_output('pos_y', units='DU')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup

        p.model.options['assembled_jac_type'] = 'csc'
        p.model.linear_solver = DirectSolver(assemble_jac=True)

        p.driver.add_recorder(SqliteRecorder('two_burn_orbit_raise_example_for_docs.db'))

        p.setup(check=True)

        # Set Initial Guesses
        p.set_val('traj.design_parameters:c', value=1.5)

        p.set_val('traj.burn1.t_initial', value=0.0)
        p.set_val('traj.burn1.t_duration', value=2.25)

        p.set_val('traj.burn1.states:r', value=burn1.interpolate(ys=[1, 1.5], nodes='state_input'))
        p.set_val('traj.burn1.states:theta', value=burn1.interpolate(ys=[0, 1.7], nodes='state_input'))
        p.set_val('traj.burn1.states:vr', value=burn1.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('traj.burn1.states:vt', value=burn1.interpolate(ys=[1, 1], nodes='state_input'))
        p.set_val('traj.burn1.states:accel', value=burn1.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('traj.burn1.states:deltav',
                  value=burn1.interpolate(ys=[0, 0.1], nodes='state_input'), )
        p.set_val('traj.burn1.controls:u1',
                  value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))

        p.set_val('traj.coast.t_initial', value=2.25)
        p.set_val('traj.coast.t_duration', value=3.0)

        p.set_val('traj.coast.states:r', value=coast.interpolate(ys=[1.3, 1.5], nodes='state_input'))
        p.set_val('traj.coast.states:theta',
                  value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))
        p.set_val('traj.coast.states:vr', value=coast.interpolate(ys=[0.3285, 0], nodes='state_input'))
        p.set_val('traj.coast.states:vt', value=coast.interpolate(ys=[0.97, 1], nodes='state_input'))
        p.set_val('traj.coast.states:accel', value=coast.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('traj.coast.controls:u1', value=coast.interpolate(ys=[0, 0], nodes='control_input'))

        p.set_val('traj.burn2.t_initial', value=5.25)
        p.set_val('traj.burn2.t_duration', value=1.75)

        p.set_val('traj.burn2.states:r', value=burn2.interpolate(ys=[1, 3], nodes='state_input'))
        p.set_val('traj.burn2.states:theta', value=burn2.interpolate(ys=[0, 4.0], nodes='state_input'))
        p.set_val('traj.burn2.states:vr', value=burn2.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('traj.burn2.states:vt',
                  value=burn2.interpolate(ys=[1, np.sqrt(1 / 3)], nodes='state_input'))
        p.set_val('traj.burn2.states:accel', value=burn2.interpolate(ys=[0.1, 0], nodes='state_input'))
        p.set_val('traj.burn2.states:deltav',
                  value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
        p.set_val('traj.burn2.controls:u1', value=burn2.interpolate(ys=[1, 1], nodes='control_input'))

        p.run_driver()

        assert_rel_error(self,
                         p.get_val('traj.burn2.timeseries.states:deltav')[-1],
                         0.3995,
                         tolerance=2.0E-3)

        # Plot results
        exp_out = traj.simulate(times=50)

        fig = plt.figure(figsize=(8, 4))
        fig.suptitle('Two Burn Orbit Raise Solution')
        ax_u1 = plt.subplot2grid((2, 2), (0, 0))
        ax_deltav = plt.subplot2grid((2, 2), (1, 0))
        ax_xy = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

        span = np.linspace(0, 2 * np.pi, 100)
        ax_xy.plot(np.cos(span), np.sin(span), 'k--', lw=1)
        ax_xy.plot(3 * np.cos(span), 3 * np.sin(span), 'k--', lw=1)
        ax_xy.set_xlim(-4.5, 4.5)
        ax_xy.set_ylim(-4.5, 4.5)

        ax_xy.set_xlabel('x ($R_e$)')
        ax_xy.set_ylabel('y ($R_e$)')

        ax_u1.set_xlabel('time ($TU$)')
        ax_u1.set_ylabel('$u_1$ ($deg$)')
        ax_u1.grid(True)

        ax_deltav.set_xlabel('time ($TU$)')
        ax_deltav.set_ylabel('${\Delta}v$ ($DU/TU$)')
        ax_deltav.grid(True)

        t_sol = dict((phs, p.get_val('traj.{0}.timeseries.time'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        x_sol = dict((phs, p.get_val('traj.{0}.timeseries.pos_x'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        y_sol = dict((phs, p.get_val('traj.{0}.timeseries.pos_y'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        dv_sol = dict((phs, p.get_val('traj.{0}.timeseries.states:deltav'.format(phs)))
                      for phs in ['burn1', 'coast', 'burn2'])
        u1_sol = dict((phs, p.get_val('traj.{0}.timeseries.controls:u1'.format(phs), units='deg'))
                      for phs in ['burn1', 'burn2'])

        t_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.time'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        x_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.pos_x'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        y_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.pos_y'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        dv_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.states:deltav'.format(phs)))
                      for phs in ['burn1', 'coast', 'burn2'])
        u1_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.controls:u1'.format(phs),
                                            units='deg'))
                      for phs in ['burn1', 'burn2'])

        for phs in ['burn1', 'coast', 'burn2']:
            try:
                ax_u1.plot(t_sol[phs], u1_sol[phs], 'ro', ms=3)
                ax_u1.plot(t_exp[phs], u1_exp[phs], 'b-')
            except KeyError:
                pass

            ax_deltav.plot(t_sol[phs], dv_sol[phs], 'ro', ms=3)
            ax_deltav.plot(t_exp[phs], dv_exp[phs], 'b-')

            ax_xy.plot(x_sol[phs], y_sol[phs], 'ro', ms=3, label='implicit')
            ax_xy.plot(x_exp[phs], y_exp[phs], 'b-', label='explicit')

        plt.show()


if __name__ == '__main__':
    unittest.main()
