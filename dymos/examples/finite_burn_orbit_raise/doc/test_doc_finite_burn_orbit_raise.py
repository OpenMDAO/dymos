import unittest

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.style.use('ggplot')

from openmdao.utils.general_utils import set_pyoptsparse_opt
from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs


_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)


@use_tempdirs
class TestFiniteBurnOrbitRaise(unittest.TestCase):

    @unittest.skipIf(optimizer != 'IPOPT', 'IPOPT not available')
    @save_for_docs
    def test_finite_burn_orbit_raise(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.declare_coloring()

        traj = dm.Trajectory()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=5, order=3, compressed=False))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01, rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30)
        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=5, order=3, compressed=False))

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 50), duration_ref=50,
                               units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')

        coast.add_parameter('u1', opt=False, val=0.0, units='deg', targets=['u1'])

        # Third Phase (burn)
        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=5, order=3, compressed=False))

        traj.add_phase('coast', coast)
        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 50), duration_bounds=(.5, 10), initial_ref=10,
                               units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True, defect_scaler=1000.0,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True, defect_scaler=1000.0,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_scaler=1.0,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0,
                        rate_source='deltav_dot', units='DU/TU')

        burn2.add_objective('deltav', loc='final', scaler=100.0)

        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01, lower=-90, upper=90)

        burn1.add_timeseries_output('pos_x')
        coast.add_timeseries_output('pos_x')
        burn2.add_timeseries_output('pos_x')

        burn1.add_timeseries_output('pos_y')
        coast.add_timeseries_output('pos_y')
        burn2.add_timeseries_output('pos_y')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        p.model.add_subsystem('traj', subsys=traj)

        # Finish Problem Setup

        # Needed to move the direct solver down into the phases for use with MPI.
        #  - After moving down, used fewer iterations (about 30 less)

        p.driver.add_recorder(om.SqliteRecorder('two_burn_orbit_raise_example.db'))

        p.setup(check=True, mode='fwd')

        # Set Initial Guesses
        p.set_val('traj.parameters:c', value=1.5, units='DU/TU')

        burn1 = p.model.traj.phases.burn1
        burn2 = p.model.traj.phases.burn2
        coast = p.model.traj.phases.coast

        p.set_val('traj.burn1.t_initial', value=0.0)
        p.set_val('traj.burn1.t_duration', value=2.25)
        p.set_val('traj.burn1.states:r', value=burn1.interpolate(ys=[1, 1.5],
                                                                 nodes='state_input'))
        p.set_val('traj.burn1.states:theta', value=burn1.interpolate(ys=[0, 1.7],
                                                                     nodes='state_input'))
        p.set_val('traj.burn1.states:vr', value=burn1.interpolate(ys=[0, 0],
                                                                  nodes='state_input'))
        p.set_val('traj.burn1.states:vt', value=burn1.interpolate(ys=[1, 1],
                                                                  nodes='state_input'))
        p.set_val('traj.burn1.states:accel', value=burn1.interpolate(ys=[0.1, 0],
                                                                     nodes='state_input'))
        p.set_val('traj.burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1],
                                                                      nodes='state_input'))
        p.set_val('traj.burn1.controls:u1',
                  value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))

        p.set_val('traj.coast.t_initial', value=2.25)
        p.set_val('traj.coast.t_duration', value=3.0)

        p.set_val('traj.coast.states:r', value=coast.interpolate(ys=[1.3, 1.5],
                                                                 nodes='state_input'))
        p.set_val('traj.coast.states:theta',
                  value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))

        p.set_val('traj.coast.states:vr', value=coast.interpolate(ys=[0.3285, 0],
                                                                  nodes='state_input'))
        p.set_val('traj.coast.states:vt', value=coast.interpolate(ys=[0.97, 1],
                                                                  nodes='state_input'))
        p.set_val('traj.coast.states:accel', value=coast.interpolate(ys=[0, 0],
                                                                     nodes='state_input'))

        p.set_val('traj.burn2.t_initial', value=5.25)
        p.set_val('traj.burn2.t_duration', value=1.75)

        p.set_val('traj.burn2.states:r', value=burn2.interpolate(ys=[1, 3.],
                                                                 nodes='state_input'))
        p.set_val('traj.burn2.states:theta', value=burn2.interpolate(ys=[0, 4.0],
                                                                     nodes='state_input'))
        p.set_val('traj.burn2.states:vr', value=burn2.interpolate(ys=[0, 0],
                                                                  nodes='state_input'))
        p.set_val('traj.burn2.states:vt',
                  value=burn2.interpolate(ys=[1, np.sqrt(1 / 3.)],
                                          nodes='state_input'))
        p.set_val('traj.burn2.states:deltav',
                  value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
        p.set_val('traj.burn2.states:accel', value=burn2.interpolate(ys=[0.1, 0],
                                                                     nodes='state_input'))

        p.set_val('traj.burn2.controls:u1', value=burn2.interpolate(ys=[0, 0],
                                                                    nodes='control_input'))

        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                          tolerance=2.0E-3)

        #
        # Plot results
        #
        traj = p.model.traj
        exp_out = traj.simulate()

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
                ax_u1.plot(t_exp[phs], u1_exp[phs], '-', marker=None, color='C0')
                ax_u1.plot(t_sol[phs], u1_sol[phs], 'o', mfc='C1', mec='C1', ms=3)
            except KeyError:
                pass

            ax_deltav.plot(t_exp[phs], dv_exp[phs], '-', marker=None, color='C0')
            ax_deltav.plot(t_sol[phs], dv_sol[phs], 'o', mfc='C1', mec='C1', ms=3)

            ax_xy.plot(x_exp[phs], y_exp[phs], '-', marker=None, color='C0', label='explicit')
            ax_xy.plot(x_sol[phs], y_sol[phs], 'o', mfc='C1', mec='C1', ms=3, label='implicit')

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
