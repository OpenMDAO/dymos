import unittest
import numpy as np
import openmdao.api as om
import dymos as dm

from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal


@require_pyoptsparse(optimizer='SLSQP')
@use_tempdirs
class TestTwoPhaseOrbitDoubleLinked(unittest.TestCase):

    def test_two_phase_orbit_double_linked(self):
        traj = dm.Trajectory()
        p = om.Problem(model=om.Group())
        p.model.add_subsystem('traj', traj)

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'

        p.driver.declare_coloring()

        traj.add_parameter('c', opt=False, val=0.0, units='DU/TU',
                           targets={'phase0': ['c']})

        traj.add_parameter('accel', opt=False, val=0.0, units='DU/TU**2',
                           targets={'phase0': ['accel']})

        traj.add_parameter('u1', opt=False, val=0.0, units='deg',
                           targets={'phase0': ['u1']})

        # First half of the orbit
        phase0 = dm.Phase(ode_class=FiniteBurnODE,
                          transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        phase0 = traj.add_phase('phase0', phase0)

        phase0.set_time_options(fix_initial=True, fix_duration=True, units='TU')
        phase0.add_state('r', fix_initial=False, fix_final=False,
                         rate_source='r_dot', targets=['r'], units='DU')
        phase0.add_state('theta', fix_initial=True, fix_final=False,
                         rate_source='theta_dot', targets=['theta'], units='rad')
        phase0.add_state('vr', fix_initial=False, fix_final=False,
                         rate_source='vr_dot', targets=['vr'], units='DU/TU')
        phase0.add_state('vt', fix_initial=False, fix_final=False,
                         rate_source='vt_dot', targets=['vt'], units='DU/TU')

        phase0.add_objective('time', loc='final')

        # Second half of the orbit
        phase1 = dm.Phase(ode_class=FiniteBurnODE,
                          transcription=dm.GaussLobatto(num_segments=10, order=3, compressed=True))

        phase1 = traj.add_phase('phase1', phase1)

        phase1.set_time_options(fix_initial=True, fix_duration=True, units='TU')
        phase1.add_state('r', fix_initial=False, fix_final=False,
                         rate_source='r_dot', targets=['r'], units='DU')
        phase1.add_state('theta', fix_initial=False, fix_final=False,
                         rate_source='theta_dot', targets=['theta'], units='rad')
        phase1.add_state('vr', fix_initial=False, fix_final=False,
                         rate_source='vr_dot', targets=['vr'], units='DU/TU')
        phase1.add_state('vt', fix_initial=False, fix_final=False,
                         rate_source='vt_dot', targets=['vt'], units='DU/TU')

        traj.link_phases(phases=['phase0', 'phase1'], vars=['r', 'theta', 'vr', 'vt'], connected=False)
        traj.link_phases(phases=['phase1', 'phase0'], vars=['r', 'vr', 'vt'], connected=False)

        p.setup()
        phase0.set_time_val(initial=0.0, duration=np.pi)
        phase0.set_state_val('r', vals=[2, 1])
        phase0.set_state_val('theta', vals=[0, np.pi])
        phase0.set_state_val('vr', vals=[0, 0])
        phase0.set_state_val('vt', vals=[2, 1])

        phase1.set_time_val(initial=np.pi, duration=np.pi)
        phase0.set_state_val('r', vals=[1, 1])
        phase0.set_state_val('theta', vals=[0, np.pi])
        phase0.set_state_val('vr', vals=[0, 0])
        phase0.set_state_val('vt', vals=[1, 1])

        dm.run_problem(p, make_plots=True)

        r_1 = p.get_val('traj.phase0.timeseries.r')
        theta_2 = p.get_val('traj.phase1.timeseries.theta')

        assert_near_equal(r_1[0], 1.0, tolerance=1e-9)
        assert_near_equal(theta_2[-1], 2*np.pi, tolerance=1e-9)
