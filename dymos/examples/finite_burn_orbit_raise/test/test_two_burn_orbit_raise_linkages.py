import unittest
import warnings

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from dymos.utils.introspection import get_promoted_vars

try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
except ImportError:
    plt = None

@require_pyoptsparse(optimizer='IPOPT')
@use_tempdirs
class TestTwoBurnOrbitRaiseLinkages(unittest.TestCase):

    def test_two_burn_orbit_raise_gl_radau_gl_changing_units_error(self):
        import openmdao.api as om

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01,
                          rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30, targets=['u1'])

        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')

        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        units='rad', rate_source='theta_dot', targets=['theta'])
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=False, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

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
                        rate_source='at_dot', targets=['accel'], units='km/s**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', targets=['u1'], rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        burn1.add_timeseries_output('pos_x')
        coast.add_timeseries_output('pos_x')
        burn2.add_timeseries_output('pos_x')

        burn1.add_timeseries_output('pos_y')
        coast.add_timeseries_output('pos_y')
        burn2.add_timeseries_output('pos_y')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'vr', 'vt', 'deltav'])
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        with self.assertRaises(ValueError) as e:
            p.setup(check=True, force_alloc_complex=True)

        expected_exception = 'traj: Linkage units were not specified but the units of burn1.accel (DU/TU**2) and ' \
                             'burn2.accel (km/s**2) are not equivalent. Units for this linkage constraint must ' \
                             'be specified explicitly.'

        self.assertEqual(expected_exception, str(e.exception))

    def test_two_burn_orbit_raise_gl_radau_gl_equivalent_units_error(self):
        import openmdao.api as om

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='nmi')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01,
                          rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30, targets=['u1'])

        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')

        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='nmi')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        units='rad', rate_source='theta_dot', targets=['theta'])
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=False, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', targets=['r'], units='NM')
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

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        burn1.add_timeseries_output('pos_x')
        coast.add_timeseries_output('pos_x')
        burn2.add_timeseries_output('pos_x')

        burn1.add_timeseries_output('pos_y')
        coast.add_timeseries_output('pos_y')
        burn2.add_timeseries_output('pos_y')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'vr', 'vt', 'deltav'])
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

    @unittest.skipIf(plt is None, "This test requires matplotlib")
    def test_two_burn_orbit_raise_gl_radau_gl_constrained(self):
        import numpy as np

        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=False, rate2_continuity=False, units='deg',
                          scaler=0.01,
                          rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30, targets=['u1'])

        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')

        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        units='rad', rate_source='theta_dot', targets=['theta'])
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=False, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

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
        burn2.add_control('u1', targets=['u1'], rate_continuity=False, rate2_continuity=False,
                          units='deg', scaler=0.01, lower=-30, upper=30)

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        burn1.add_timeseries_output('pos_x')
        coast.add_timeseries_output('pos_x')
        burn2.add_timeseries_output('pos_x')

        burn1.add_timeseries_output('pos_y')
        coast.add_timeseries_output('pos_y')
        burn2.add_timeseries_output('pos_y')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'vr', 'vt', 'deltav'], scaler=1.0E-6, linear=True)
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'],
                         locs=('final', 'initial'), scaler=1.0E-6, linear=True)

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        # Set Initial Guesses
        p.set_val('traj.parameters:c', val=1.5)

        p.set_val('traj.burn1.t_initial', val=0.0)
        p.set_val('traj.burn1.t_duration', val=2.25)

        p.set_val('traj.burn1.states:r', val=burn1.interp('r', [1, 1.5]))
        p.set_val('traj.burn1.states:theta', val=burn1.interp('theta', [0, 1.7]))
        p.set_val('traj.burn1.states:vr', val=burn1.interp('vr', [0, 0]))
        p.set_val('traj.burn1.states:vt', val=burn1.interp('vt', [1, 1]))
        p.set_val('traj.burn1.states:accel', val=burn1.interp('accel', [0.1, 0]))
        p.set_val('traj.burn1.states:deltav', val=burn1.interp('deltav', [0, 0.1]), )
        p.set_val('traj.burn1.controls:u1', val=burn1.interp('u1', [-3.5, 13.0]))

        p.set_val('traj.coast.t_initial', val=2.25)
        p.set_val('traj.coast.t_duration', val=3.0)

        p.set_val('traj.coast.states:r', val=coast.interp('r', [1.3, 1.5]))
        p.set_val('traj.coast.states:theta', val=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('traj.coast.states:vr', val=coast.interp('vr', [0.3285, 0]))
        p.set_val('traj.coast.states:vt', val=coast.interp('vt', [0.97, 1]))
        p.set_val('traj.coast.states:accel', val=coast.interp('accel', [0, 0]))

        p.set_val('traj.burn2.t_initial', val=5.25)
        p.set_val('traj.burn2.t_duration', val=1.75)

        p.set_val('traj.burn2.states:r', val=burn2.interp('r', [1.8, 3]))
        p.set_val('traj.burn2.states:theta', val=burn2.interp('theta', [3.2, 4.0]))
        p.set_val('traj.burn2.states:vr', val=burn2.interp('vr', [.5, 0]))
        p.set_val('traj.burn2.states:vt', val=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('traj.burn2.states:accel', val=burn2.interp('accel', [0.1, 0]))
        p.set_val('traj.burn2.states:deltav', val=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('traj.burn2.controls:u1', val=burn2.interp('u1', [1, 1]))

        p.run_driver()

        assert_near_equal(p.get_val('traj.burn2.timeseries.deltav')[-1],
                          0.3995,
                          tolerance=2.0E-3)

        # Plot results
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
        ax_deltav.set_ylabel('${\Delta}v$ ($DU/TU$)')  # nopep8: W605
        ax_deltav.grid(True)

        t_sol = dict((phs, p.get_val('traj.{0}.timeseries.time'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        x_sol = dict((phs, p.get_val('traj.{0}.timeseries.pos_x'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        y_sol = dict((phs, p.get_val('traj.{0}.timeseries.pos_y'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        dv_sol = dict((phs, p.get_val('traj.{0}.timeseries.deltav'.format(phs)))
                      for phs in ['burn1', 'coast', 'burn2'])
        u1_sol = dict((phs, p.get_val('traj.{0}.timeseries.u1'.format(phs), units='deg'))
                      for phs in ['burn1', 'burn2'])

        t_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.time'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        x_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.pos_x'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        y_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.pos_y'.format(phs)))
                     for phs in ['burn1', 'coast', 'burn2'])
        dv_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.deltav'.format(phs)))
                      for phs in ['burn1', 'coast', 'burn2'])
        u1_exp = dict((phs, exp_out.get_val('traj.{0}.timeseries.u1'.format(phs),
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

    @unittest.skipIf(plt is None, "This test requires matplotlib")
    def test_two_burn_orbit_raise_link_control_to_param(self):
        import numpy as np

        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['mu_init'] = 1e-3
        p.driver.opt_settings['max_iter'] = 500
        p.driver.opt_settings['acceptable_tol'] = 1e-3
        p.driver.opt_settings['constr_viol_tol'] = 1e-3
        p.driver.opt_settings['compl_inf_tol'] = 1e-3
        p.driver.opt_settings['acceptable_iter'] = 0
        p.driver.opt_settings['tol'] = 1e-3
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'  # for faster convergence
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['mu_strategy'] = 'monotone'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)

        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01,
                          rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30, targets=['u1'])

        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3, solve_segments='forward'))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')

        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        units='rad', rate_source='theta_dot', targets=['theta'])
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=False, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', opt=True, units='deg', lower=-30, upper=30, scaler=0.01)

        coast.nonlinear_solver = om.NewtonSolver(iprint=0, solve_subsystems=True)

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

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

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        burn1.add_timeseries_output('pos_x')
        coast.add_timeseries_output('pos_x')
        burn2.add_timeseries_output('pos_x')

        burn1.add_timeseries_output('pos_y')
        coast.add_timeseries_output('pos_y')
        burn2.add_timeseries_output('pos_y')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'vr', 'vt', 'deltav'])

        # Here u1 is a control in the burns and a parameter in the coast
        traj.link_phases(phases=['burn1', 'coast', 'burn2'], vars=['u1'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        with warnings.catch_warnings(record=True) as w:
            p.setup(check=True, force_alloc_complex=True)

        # Set Initial Guesses
        p.set_val('traj.parameters:c', val=1.5)

        p.set_val('traj.burn1.t_initial', val=0.0)
        p.set_val('traj.burn1.t_duration', val=2.25)

        p.set_val('traj.burn1.states:r', val=burn1.interp('r', [1, 1.5]))
        p.set_val('traj.burn1.states:theta', val=burn1.interp('theta', [0, 1.7]))
        p.set_val('traj.burn1.states:vr', val=burn1.interp('vr', [0, 0]))
        p.set_val('traj.burn1.states:vt', val=burn1.interp('vt', [1, 1]))
        p.set_val('traj.burn1.states:accel', val=burn1.interp('accel', [0.1, 0]))
        p.set_val('traj.burn1.states:deltav', val=burn1.interp('deltav', [0, 0.1]), )
        p.set_val('traj.burn1.controls:u1', val=burn1.interp('u1', [-3.5, 13.0]))

        p.set_val('traj.coast.t_initial', val=2.25)
        p.set_val('traj.coast.t_duration', val=3.0)

        p.set_val('traj.coast.states:r', val=coast.interp('r', [1.3, 1.5]))
        p.set_val('traj.coast.states:theta', val=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('traj.coast.states:vr', val=coast.interp('vr', [0.3285, 0]))
        p.set_val('traj.coast.states:vt', val=coast.interp('vt', [0.97, 1]))
        p.set_val('traj.coast.states:accel', val=coast.interp('accel', [0, 0]))

        p.set_val('traj.burn2.t_initial', val=5.25)
        p.set_val('traj.burn2.t_duration', val=1.75)

        p.set_val('traj.burn2.states:r', val=burn2.interp('r', [1.8, 3]))
        p.set_val('traj.burn2.states:theta', val=burn2.interp('theta', [3.2, 4.0]))
        p.set_val('traj.burn2.states:vr', val=burn2.interp('vr', [.5, 0]))
        p.set_val('traj.burn2.states:vt', val=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('traj.burn2.states:accel', val=burn2.interp('accel', [0.1, 0]))
        p.set_val('traj.burn2.states:deltav', val=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('traj.burn2.controls:u1', val=burn2.interp('u1', [1, 1]))

        p.run_driver()

        # This tolerance is loosened because we're testing that the control
        # stays continuous across the trajectory phases, which isn't necessarily optimal.
        assert_near_equal(p.get_val('traj.burn2.timeseries.deltav')[-1],
                          0.3995,
                          tolerance=0.05)

        burn1_u1_final = p.get_val('traj.burn1.timeseries.u1')[-1, ...]
        coast_u1_initial = p.get_val('traj.coast.parameter_vals:u1')[0, ...]
        coast_u1_final = p.get_val('traj.coast.parameter_vals:u1')[-1, ...]
        burn2_u1_initial = p.get_val('traj.burn2.timeseries.u1')[0, ...]

        assert_near_equal(burn1_u1_final - coast_u1_initial, 0.0, 1e-12)
        assert_near_equal(coast_u1_final - burn2_u1_initial, 0.0, 1e-12)

    @use_tempdirs
    @unittest.skipIf(plt is None, "This test requires matplotlib")
    def test_two_burn_orbit_raise_gl_list_add_timeseries_output(self):
        import numpy as np

        import matplotlib.pyplot as plt

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01,
                          rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30, targets=['u1'])

        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')

        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        units='rad', rate_source='theta_dot', targets=['theta'])
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=False, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg')

        # Third Phase (burn)
        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

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

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        # demonstrate adding multiple variables at once using list in add_timeseries_output
        # test dict for units, ignore mismatched unit
        burn1.add_timeseries_output(['pos_x', 'pos_y'], units={'pos_x': 'm', 'junk': 'lbm'})
        coast.add_timeseries_output(['pos_x', 'pos_y'])
        # test list for units
        burn2.add_timeseries_output(['pos_x', 'pos_y'], units=['m', 'm'])

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'vr', 'vt', 'deltav'])
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        # Set Initial Guesses
        p.set_val('traj.parameters:c', val=1.5)

        p.set_val('traj.burn1.t_initial', val=0.0)
        p.set_val('traj.burn1.t_duration', val=2.25)

        p.set_val('traj.burn1.states:r', val=burn1.interp('r', [1, 1.5]))
        p.set_val('traj.burn1.states:theta', val=burn1.interp('theta', [0, 1.7]))
        p.set_val('traj.burn1.states:vr', val=burn1.interp('vr', [0, 0]))
        p.set_val('traj.burn1.states:vt', val=burn1.interp('vt', [1, 1]))
        p.set_val('traj.burn1.states:accel', val=burn1.interp('accel', [0.1, 0]))
        p.set_val('traj.burn1.states:deltav', val=burn1.interp('deltav', [0, 0.1]), )
        p.set_val('traj.burn1.controls:u1', val=burn1.interp('u1', [-3.5, 13.0]))

        p.set_val('traj.coast.t_initial', val=2.25)
        p.set_val('traj.coast.t_duration', val=3.0)

        p.set_val('traj.coast.states:r', val=coast.interp('r', [1.3, 1.5]))
        p.set_val('traj.coast.states:theta', val=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('traj.coast.states:vr', val=coast.interp('vr', [0.3285, 0]))
        p.set_val('traj.coast.states:vt', val=coast.interp('vt', [0.97, 1]))
        p.set_val('traj.coast.states:accel', val=coast.interp('accel', [0, 0]))

        p.set_val('traj.burn2.t_initial', val=5.25)
        p.set_val('traj.burn2.t_duration', val=1.75)

        p.set_val('traj.burn2.states:r', val=burn2.interp('r', [1.8, 3]))
        p.set_val('traj.burn2.states:theta', val=burn2.interp('theta', [3.2, 4.0]))
        p.set_val('traj.burn2.states:vr', val=burn2.interp('vr', [.5, 0]))
        p.set_val('traj.burn2.states:vt', val=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('traj.burn2.states:accel', val=burn2.interp('accel', [0.1, 0]))
        p.set_val('traj.burn2.states:deltav', val=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('traj.burn2.controls:u1', val=burn2.interp('u1', [1, 1]))

        p.run_driver()

        assert_near_equal(p.get_val('traj.burn2.timeseries.deltav')[-1],
                          0.3995,
                          tolerance=2.0E-3)

        # check units
        un = get_promoted_vars(p.model, iotypes=('input', 'output'))
        assert un['traj.burn1.timeseries.pos_x']['units'] == 'm'
        assert un['traj.burn1.timeseries.pos_y']['units'] == 'DU'
        assert un['traj.burn2.timeseries.pos_x']['units'] == 'm'
        assert un['traj.burn2.timeseries.pos_y']['units'] == 'm'

    @use_tempdirs
    def test_two_burn_orbit_raise_gl_wildcard_add_timeseries_output(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01,
                          rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30, targets=['u1'])

        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')

        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        units='rad', rate_source='theta_dot', targets=['theta'])
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=False, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg')

        # Third Phase (burn)
        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

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

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        # demonstrate adding multiple variables at once using wildcard in add_timeseries_output
        burn1.add_timeseries_output('pos_*')  # match partial variable in ODE outputs (Gauss Lobatto)
        coast.add_timeseries_output('pos_*')  # match partial variable in ODE outputs (Radau)
        burn2.add_timeseries_output('*')      # add all ODE outputs

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'vr', 'vt', 'deltav'])
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        # Set Initial Guesses
        p.set_val('traj.parameters:c', val=1.5)

        p.set_val('traj.burn1.t_initial', val=0.0)
        p.set_val('traj.burn1.t_duration', val=2.25)

        p.set_val('traj.burn1.states:r', val=burn1.interp('r', [1, 1.5]))
        p.set_val('traj.burn1.states:theta', val=burn1.interp('theta', [0, 1.7]))
        p.set_val('traj.burn1.states:vr', val=burn1.interp('vr', [0, 0]))
        p.set_val('traj.burn1.states:vt', val=burn1.interp('vt', [1, 1]))
        p.set_val('traj.burn1.states:accel', val=burn1.interp('accel', [0.1, 0]))
        p.set_val('traj.burn1.states:deltav', val=burn1.interp('deltav', [0, 0.1]), )
        p.set_val('traj.burn1.controls:u1', val=burn1.interp('u1', [-3.5, 13.0]))

        p.set_val('traj.coast.t_initial', val=2.25)
        p.set_val('traj.coast.t_duration', val=3.0)

        p.set_val('traj.coast.states:r', val=coast.interp('r', [1.3, 1.5]))
        p.set_val('traj.coast.states:theta', val=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('traj.coast.states:vr', val=coast.interp('vr', [0.3285, 0]))
        p.set_val('traj.coast.states:vt', val=coast.interp('vt', [0.97, 1]))
        p.set_val('traj.coast.states:accel', val=coast.interp('accel', [0, 0]))

        p.set_val('traj.burn2.t_initial', val=5.25)
        p.set_val('traj.burn2.t_duration', val=1.75)

        p.set_val('traj.burn2.states:r', val=burn2.interp('r', [1.8, 3]))
        p.set_val('traj.burn2.states:theta', val=burn2.interp('theta', [3.2, 4.0]))
        p.set_val('traj.burn2.states:vr', val=burn2.interp('vr', [.5, 0]))
        p.set_val('traj.burn2.states:vt', val=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('traj.burn2.states:accel', val=burn2.interp('accel', [0.1, 0]))
        p.set_val('traj.burn2.states:deltav', val=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('traj.burn2.controls:u1', val=burn2.interp('u1', [1, 1]))

        p.run_driver()

        assert_near_equal(p.get_val('traj.burn2.timeseries.deltav')[-1],
                          0.3995,
                          tolerance=2.0E-3)

        # test ODE output wildcard matching in solve_ivp
        exp_out = traj.simulate()

        for prob in (p, exp_out):
            # check timeseries: pos_x = r * cos(theta) and pos_y = r * sin(theta)
            for phs in ['burn1', 'coast', 'burn2']:
                x = np.array(prob.get_val('traj.{0}.timeseries.pos_x'.format(phs))).flatten()
                y = np.array(prob.get_val('traj.{0}.timeseries.pos_y'.format(phs))).flatten()
                t = np.array(prob.get_val('traj.{0}.timeseries.theta'.format(phs))).flatten()
                r = np.array(prob.get_val('traj.{0}.timeseries.r'.format(phs))).flatten()

                xerr = x - r * np.cos(t)
                yerr = y - r * np.sin(t)

                assert_near_equal(np.sqrt(np.mean(xerr * xerr)), 0.0)
                assert_near_equal(np.sqrt(np.mean(yerr * yerr)), 0.0)

    @use_tempdirs
    def test_two_burn_orbit_raise_radau_wildcard_add_timeseries_output(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3, compressed=True))

        burn1 = traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', targets=['theta'], units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01,
                          rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                          lower=-30, upper=30, targets=['u1'])

        # Second Phase (Coast)
        coast = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')

        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', targets=['r'], units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        units='rad', rate_source='theta_dot', targets=['theta'])
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', targets=['vr'], units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', targets=['vt'], units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=False, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', targets=['accel'], units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_parameter('u1', targets=['u1'], opt=False, val=0.0, units='deg')

        # Third Phase (burn)
        burn2 = dm.Phase(ode_class=FiniteBurnODE,
                         transcription=dm.Radau(num_segments=10, order=3, compressed=True))

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

        burn2.add_objective('deltav', loc='final', scaler=1.0)

        # demonstrate adding multiple variables at once using wildcard in add_timeseries_output
        burn1.add_timeseries_output('pos_*')  # match partial variable in ODE outputs (Radau)
        coast.add_timeseries_output('pos_*')  # match partial variable in ODE outputs (Radau)
        burn2.add_timeseries_output('*')      # add all ODE outputs

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'vr', 'vt', 'deltav'])
        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.setup(check=True, force_alloc_complex=True)

        # Set Initial Guesses
        p.set_val('traj.parameters:c', val=1.5)

        p.set_val('traj.burn1.t_initial', val=0.0)
        p.set_val('traj.burn1.t_duration', val=2.25)

        p.set_val('traj.burn1.states:r', val=burn1.interp('r', [1, 1.5]))
        p.set_val('traj.burn1.states:theta', val=burn1.interp('theta', [0, 1.7]))
        p.set_val('traj.burn1.states:vr', val=burn1.interp('vr', [0, 0]))
        p.set_val('traj.burn1.states:vt', val=burn1.interp('vt', [1, 1]))
        p.set_val('traj.burn1.states:accel', val=burn1.interp('accel', [0.1, 0]))
        p.set_val('traj.burn1.states:deltav', val=burn1.interp('deltav', [0, 0.1]), )
        p.set_val('traj.burn1.controls:u1', val=burn1.interp('u1', [-3.5, 13.0]))

        p.set_val('traj.coast.t_initial', val=2.25)
        p.set_val('traj.coast.t_duration', val=3.0)

        p.set_val('traj.coast.states:r', val=coast.interp('r', [1.3, 1.5]))
        p.set_val('traj.coast.states:theta', val=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('traj.coast.states:vr', val=coast.interp('vr', [0.3285, 0]))
        p.set_val('traj.coast.states:vt', val=coast.interp('vt', [0.97, 1]))
        p.set_val('traj.coast.states:accel', val=coast.interp('accel', [0, 0]))

        p.set_val('traj.burn2.t_initial', val=5.25)
        p.set_val('traj.burn2.t_duration', val=1.75)

        p.set_val('traj.burn2.states:r', val=burn2.interp('r', [1.8, 3]))
        p.set_val('traj.burn2.states:theta', val=burn2.interp('theta', [3.2, 4.0]))
        p.set_val('traj.burn2.states:vr', val=burn2.interp('vr', [.5, 0]))
        p.set_val('traj.burn2.states:vt', val=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('traj.burn2.states:accel', val=burn2.interp('accel', [0.1, 0]))
        p.set_val('traj.burn2.states:deltav', val=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('traj.burn2.controls:u1', val=burn2.interp('u1', [1, 1]))

        p.run_driver()

        assert_near_equal(p.get_val('traj.burn2.timeseries.deltav')[-1],
                          0.3995,
                          tolerance=2.0E-3)

        # test ODE output wildcard matching in solve_ivp
        exp_out = traj.simulate()

        for prob in (p, exp_out):
            # check timeseries: pos_x = r * cos(theta) and pos_y = r * sin(theta)
            for phs in ['burn1', 'coast', 'burn2']:
                x = np.array(prob.get_val('traj.{0}.timeseries.pos_x'.format(phs))).flatten()
                y = np.array(prob.get_val('traj.{0}.timeseries.pos_y'.format(phs))).flatten()
                t = np.array(prob.get_val('traj.{0}.timeseries.theta'.format(phs))).flatten()
                r = np.array(prob.get_val('traj.{0}.timeseries.r'.format(phs))).flatten()

                xerr = x - r * np.cos(t)
                yerr = y - r * np.sin(t)

                assert_near_equal(np.sqrt(np.mean(xerr*xerr)), 0.0)
                assert_near_equal(np.sqrt(np.mean(yerr*yerr)), 0.0)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
