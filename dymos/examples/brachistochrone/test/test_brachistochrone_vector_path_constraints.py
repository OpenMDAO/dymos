import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_vector_states_ode \
    import BrachistochroneVectorStatesODE


@use_tempdirs
class TestBrachistochroneVectorPathConstraints(unittest.TestCase):

    def test_brachistochrone_vector_state_path_constraints_radau_partial_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta_rate', loc='final', equals=0.0)
        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0.0)
        phase.add_path_constraint('pos', indices=[1], lower=5)

        phase.add_timeseries_output('pos_dot')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        phase.set_time_val(initial=0.0, duration=1.8016)

        pos0 = [0, 10]
        posf = [10, 5]

        phase.set_state_val('pos', [pos0, posf])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        assert_near_equal(np.min(p.get_val('phase0.timeseries.pos')[:, 1]), 5.0,
                          tolerance=1.0E-3)

    def test_brachistochrone_vector_ode_path_constraints_radau_partial_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('pos_dot', indices=[1],
                                  lower=-4, upper=4)

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        phase.set_time_val(initial=0.0, duration=1.8016)

        pos0 = [0, 10]
        posf = [10, 5]

        phase.set_state_val('pos', [pos0, posf])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        assert_near_equal(np.min(p.get_val('phase0.timeseries.pos_dot')[:, -1]),
                          -4,
                          tolerance=1.0E-2)

    def test_brachistochrone_vector_ode_path_constraints_radau_no_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.Radau(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('pos_dot', lower=-4, upper=12)

        phase.add_timeseries_output('pos_dot')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        phase.set_time_val(initial=0.0, duration=1.8016)

        pos0 = [0, 10]
        posf = [10, 5]

        phase.set_state_val('pos', [pos0, posf])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        assert_near_equal(np.min(p.get_val('phase0.timeseries.pos_dot')[:, -1]),
                          -4,
                          tolerance=1.0E-2)

    def test_brachistochrone_vector_state_path_constraints_gl_partial_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.GaussLobatto(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg', rate_continuity=True, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('pos', indices=[1], lower=5)

        phase.add_timeseries_output('pos_dot')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        phase.set_time_val(initial=0.0, duration=1.8016)

        pos0 = [0, 10]
        posf = [10, 5]

        phase.set_state_val('pos', [pos0, posf])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        assert_near_equal(np.min(p.get_val('phase0.timeseries.pos')[:, 1]),
                          5,
                          tolerance=1.0E-2)

    def test_brachistochrone_vector_ode_path_constraints_gl_partial_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.GaussLobatto(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg',
                          rate_continuity=True, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_boundary_constraint('theta_rate', loc='final', equals=0.0)
        phase.add_boundary_constraint('theta_rate2', loc='final', equals=0.0)
        phase.add_path_constraint('pos_dot', indices=[1],
                                  lower=-4, upper=4)

        phase.add_timeseries_output('pos_dot')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        phase.set_time_val(initial=0.0, duration=1.8016)

        pos0 = [0, 10]
        posf = [10, 5]

        phase.set_state_val('pos', [pos0, posf])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        assert_near_equal(np.min(p.get_val('phase0.timeseries.pos_dot')[:, 1]), -4.0,
                          tolerance=1.0E-3)

    def test_brachistochrone_vector_ode_path_constraints_gl_no_indices(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring()

        phase = dm.Phase(ode_class=BrachistochroneVectorStatesODE,
                         transcription=dm.GaussLobatto(num_segments=20, order=3))

        p.model.add_subsystem('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('pos', fix_initial=True, fix_final=True)
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='deg',
                          rate_continuity=False, lower=0.01, upper=179.9)

        phase.add_parameter('g', units='m/s**2', opt=False, val=9.80665)

        phase.add_path_constraint('pos_dot', lower=-4, upper=12)

        phase.add_timeseries_output('pos_dot')

        # Minimize time at the end of the phase
        phase.add_objective('time', loc='final', scaler=10)

        p.model.linear_solver = om.DirectSolver()
        p.setup(check=True, force_alloc_complex=True)

        phase.set_time_val(initial=0.0, duration=1.8016)

        pos0 = [0, 10]
        posf = [10, 5]

        phase.set_state_val('pos', [pos0, posf])
        phase.set_state_val('v', [0, 9.9])
        phase.set_control_val('theta', [5, 100])
        phase.set_parameter_val('g', 9.80665)

        p.run_driver()

        assert_near_equal(np.min(p.get_val('phase0.timeseries.pos_dot')[:, -1]),
                          -4,
                          tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
