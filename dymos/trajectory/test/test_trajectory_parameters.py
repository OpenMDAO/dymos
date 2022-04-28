import unittest

import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestPhaseParameterPromotion(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_promotes_parameter(self):
        transcription = 'radau-ps'
        optimizer = 'SLSQP'
        num_segments = 10
        transcription_order = 3
        compressed = False

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        OPT, OPTIMIZER = set_pyoptsparse_opt(optimizer, fallback=True)
        p.driver.options['optimizer'] = OPTIMIZER
        p.driver.declare_coloring()

        if transcription == 'gauss-lobatto':
            t = dm.GaussLobatto(num_segments=num_segments,
                                order=transcription_order,
                                compressed=compressed)
        elif transcription == 'radau-ps':
            t = dm.Radau(num_segments=num_segments,
                         order=transcription_order,
                         compressed=compressed)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=t)

        traj.add_phase('phase0', phase, promotes_inputs=['t_initial', 't_duration', 'parameters:g'])

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='s')

        phase.add_state('x', fix_initial=True, fix_final=False, solve_segments=False,
                        units='m', rate_source='xdot')
        phase.add_state('y', fix_initial=True, fix_final=False, solve_segments=False,
                        units='m', rate_source='ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, solve_segments=False,
                        units='m/s', rate_source='vdot', targets=['v'])

        phase.add_control('theta', continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9, targets=['theta'])

        phase.add_parameter('g', units='m/s**2', val=9.80665, targets=['g'])

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True)

        p.set_val('traj.t_initial', 0.0)
        p.set_val('traj.t_duration', 2.0)

        p.set_val('traj.phase0.states:x', phase.interp('x', ys=[0, 10]))
        p.set_val('traj.phase0.states:y', phase.interp('y', ys=[10, 5]))
        p.set_val('traj.phase0.states:v', phase.interp('v', ys=[0, 9.9]))
        p.set_val('traj.phase0.controls:theta', phase.interp('theta', ys=[5, 100]))
        p.set_val('traj.parameters:g', 9.80665)

        p.run_driver()

        assert_near_equal(p['traj.t_duration'], 1.8016, tolerance=1.0E-4)


@use_tempdirs
class TestParameterIntrospection(unittest.TestCase):

    def test_parameter_introspection_targets_none_no_valid_parameter_targets(self):
        import openmdao.api as om

        import dymos as dm

        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        p = om.Problem()

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                         transcription=dm.GaussLobatto(num_segments=5, compressed=False))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                        ref=100.0, defect_ref=1.0E2,
                        rate_source='flight_dynamics.h_dot')

        phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.v_dot')

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                        ref=1.0, defect_ref=1.0,
                        rate_source='flight_dynamics.gam_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='prop.m_dot')

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False)

        # Unit introspection for phase params. This works.
        phase.add_parameter('S', val=49.2386)

        # Unit introspection for traj params. This doesn't work.
        traj.add_parameter('Isp', val=1600.0, targets=None)

        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        expected = 'No target was found for trajectory parameter `Isp` in any phase.\n' \
                   'Option `targets=None` but no phase in the trajectory has a parameter named `Isp`.'

        self.assertEqual(str(e.exception), expected)

    def test_parameter_introspection_targets_dict_no_valid_parameter_targets(self):
        import openmdao.api as om

        import dymos as dm

        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        p = om.Problem()

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                         transcription=dm.GaussLobatto(num_segments=5, compressed=False))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                        ref=100.0, defect_ref=1.0E2,
                        rate_source='flight_dynamics.h_dot')

        phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.v_dot')

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                        ref=1.0, defect_ref=1.0,
                        rate_source='flight_dynamics.gam_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='prop.m_dot')

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False)

        # Unit introspection for phase params. This works.
        phase.add_parameter('S', val=49.2386)

        # Unit introspection for traj params. This doesn't work.
        traj.add_parameter('Isp', val=1600.0, targets={'phase0': None})

        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        expected = 'No target was found for trajectory parameter `Isp` in any phase.\n' \
                   'Option `targets` is a dictionary keyed by phase name but target for each phase is None.'

        self.assertEqual(str(e.exception), expected)

    def test_parameter_introspection_targets_dict_no_valid_parameter_targets(self):
        import openmdao.api as om

        import dymos as dm

        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        p = om.Problem()

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                         transcription=dm.GaussLobatto(num_segments=5, compressed=False))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                        ref=100.0, defect_ref=1.0E2,
                        rate_source='flight_dynamics.h_dot')

        phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.v_dot')

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                        ref=1.0, defect_ref=1.0,
                        rate_source='flight_dynamics.gam_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='prop.m_dot')

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False)

        # Unit introspection for phase params. This works.
        phase.add_parameter('S', val=49.2386)

        # Unit introspection for traj params. This doesn't work.
        traj.add_parameter('Isp', val=1600.0, targets={'phase0': 'Isp'})

        phase.add_parameter('throttle', val=1.0, opt=False, targets=['throttle'])

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        expected = 'Invalid target for trajectory `traj` parameter `Isp` in phase `phase0`.\n' \
                   "Target for phase `phase0` is 'Isp' but the phase has no such parameter."

        self.assertEqual(str(e.exception), expected)

    def test_parameter_introspection_targets_dict_no_valid_ode_targets(self):
        import openmdao.api as om

        import dymos as dm

        from dymos.examples.min_time_climb.min_time_climb_ode import MinTimeClimbODE

        p = om.Problem()

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=MinTimeClimbODE,
                         transcription=dm.GaussLobatto(num_segments=5, compressed=False))

        traj.add_phase('phase0', phase)

        p.model.add_subsystem('traj', traj)

        phase.set_time_options(fix_initial=True, duration_bounds=(50, 400),
                               duration_ref=100.0)

        phase.add_state('r', fix_initial=True, lower=0, upper=1.0E6, units='m',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='flight_dynamics.r_dot')

        phase.add_state('h', fix_initial=True, lower=0, upper=20000.0, units='m',
                        ref=100.0, defect_ref=1.0E2,
                        rate_source='flight_dynamics.h_dot')

        phase.add_state('v', fix_initial=True, lower=10.0, units='m/s',
                        ref=1.0E2, defect_ref=1.0E2,
                        rate_source='flight_dynamics.v_dot')

        phase.add_state('gam', fix_initial=True, lower=-1.5, upper=1.5, units='rad',
                        ref=1.0, defect_ref=1.0,
                        rate_source='flight_dynamics.gam_dot')

        phase.add_state('m', fix_initial=True, lower=10.0, upper=1.0E5, units='kg',
                        ref=1.0E3, defect_ref=1.0E3,
                        rate_source='prop.m_dot')

        phase.add_control('alpha', units='deg', lower=-8.0, upper=8.0, scaler=1.0,
                          rate_continuity=True, rate_continuity_scaler=100.0,
                          rate2_continuity=False)

        # Unit introspection for phase params. This works.
        phase.add_parameter('S', val=49.2386)

        # Error, no such ODE target in the phase.
        traj.add_parameter('Isp', val=1600.0, targets={'phase0': ['foo']})

        with self.assertRaises(ValueError) as e:
            p.setup(check=True)

        expected = "Invalid parameter in phase `traj.phases.phase0`.\n" \
                   "Parameter `Isp` has invalid target(s).\n" \
                   "No such ODE input: 'foo'."

        self.assertEqual(str(e.exception), expected)


if __name__ == '__main__':
    unittest.main()
