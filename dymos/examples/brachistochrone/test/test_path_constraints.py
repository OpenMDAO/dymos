import unittest

import numpy as np
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


@use_tempdirs
class TestBrachistochronePathConstraints(unittest.TestCase):

    def test_control_rate_path_constraint_gl(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g',
                            units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate', lower=0, upper=100)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_control_val('theta', [5, 101.5])

        # Solve for the optimal trajectory
        p.run_driver()

        # Test the results
        assert_near_equal(p.get_val('phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)

    def test_control_rate2_path_constraint_gl(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10, order=5))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate2', lower=-200, upper=200)

        p.model.linear_solver = om.DirectSolver()
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_control_val('theta', [5, 101.5])

        # Solve for the optimal trajectory
        failed = p.run_driver()
        self.assertFalse(failed, msg='optimization failed')

        # Test the results
        rate_path = 'control_rates:theta_rate2' \
            if phase.timeseries_options['use_prefix'] else 'theta_rate2'
        self.assertGreaterEqual(np.min(p.get_val(f'phase0.timeseries.{rate_path}')), -200.000001)
        self.assertLessEqual(np.max(p.get_val(f'phase0.timeseries.{rate_path}')), 200.000001)

    def test_control_rate_path_constraint_radau(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10,
                                                compressed=False))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate', lower=0, upper=100)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_control_val('theta', [5, 101.5])

        # Solve for the optimal trajectory
        failed = p.run_driver()
        self.assertFalse(failed, msg='optimization failed')

        # Test the results
        rate_path = 'control_rates:theta_rate' \
            if phase.timeseries_options['use_prefix'] else 'theta_rate'
        self.assertGreaterEqual(np.min(p.get_val(f'phase0.timeseries.{rate_path}')), -0.000001)
        self.assertLessEqual(np.max(p.get_val(f'phase0.timeseries.{rate_path}')), 100.000001)

    def test_control_rate2_path_constraint_radau(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.ScipyOptimizeDriver()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10,
                                                compressed=False))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g',
                            units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_path_constraint('theta_rate2', lower=-200, upper=200)

        p.model.linear_solver = om.DirectSolver()
        p.model.options['assembled_jac_type'] = 'csc'

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_control_val('theta', [5, 101.5])

        # Solve for the optimal trajectory
        failed = p.run_driver()
        self.assertFalse(failed, msg='optimization failed')

        # Test the results
        rate_path = 'control_rates:theta_rate2' \
            if phase.timeseries_options['use_prefix'] else 'theta_rate2'
        self.assertGreaterEqual(np.min(p.get_val(f'phase0.timeseries.{rate_path}')), -200.000001)
        self.assertLessEqual(np.max(p.get_val(f'phase0.timeseries.{rate_path}')), 200.000001)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_state_path_constraint_radau(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver(optimizer='IPOPT')

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.Radau(num_segments=10,
                                                compressed=False))

        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('y', lower=5, upper=100)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_control_val('theta', [5, 101.5])

        # Solve for the optimal trajectory
        failed = p.run_driver()
        self.assertFalse(failed, msg='optimization failed')

        # Test the results
        state_path = 'states:y' if phase.timeseries_options['use_prefix'] else 'y'
        self.assertGreaterEqual(np.min(p.get_val(f'traj0.phase0.timeseries.{state_path}')), 4.999999)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_state_path_constraint_gl(self):
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        p = om.Problem(model=om.Group())
        p.driver = om.pyOptSparseDriver(optimizer='IPOPT')

        traj = dm.Trajectory()

        phase = dm.Phase(ode_class=BrachistochroneODE,
                         transcription=dm.GaussLobatto(num_segments=10,
                                                       compressed=False))

        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_path_constraint('y', lower=5, upper=100)

        p.model.linear_solver = om.DirectSolver()

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0, 9.9])

        phase.set_control_val('theta', [5, 101.5])

        # Solve for the optimal trajectory
        failed = p.run_driver()
        self.assertFalse(failed, msg='optimization failed')

        # Test the results
        state_path = 'states:y' if phase.timeseries_options['use_prefix'] else 'y'
        self.assertGreaterEqual(np.min(p.get_val(f'traj0.phase0.timeseries.{state_path}')), 4.999999)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
