import unittest

import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


@use_tempdirs
class TestBrachistochroneBirkhoffConstraints(unittest.TestCase):

    def test_brachistochrone_control_prefix(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring(tol=1.0E-12)

        grid = dm.BirkhoffGrid(num_nodes=25, grid_type='lgl')
        tx = dm.Birkhoff(grid=grid)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.timeseries_options['use_prefix'] = True
        p.model.add_subsystem('traj', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg')

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_path_constraint('theta', lower=0.01, upper=179.9)
        phase.add_boundary_constraint('theta', loc='final', lower=0.01, upper=179.9)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.set_solver_print(0)

        p.setup()

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 1.5

        p['traj.phase0.initial_states:x'] = 0.0
        p['traj.phase0.initial_states:y'] = 10.0
        p['traj.phase0.initial_states:v'] = 0.0

        p['traj.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj.phase0.parameters:g'] = 9.80665

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-4)

    def test_brachistochrone_control_no_prefix(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring(tol=1.0E-12)

        grid = dm.BirkhoffGrid(num_nodes=25, grid_type='lgl')
        tx = dm.Birkhoff(grid=grid)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.timeseries_options['use_prefix'] = False
        p.model.add_subsystem('traj', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg')

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_path_constraint('theta', lower=0.01, upper=179.9)
        phase.add_boundary_constraint('theta', loc='final', lower=0.01, upper=179.9)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.set_solver_print(0)

        p.setup()

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 1.5

        p['traj.phase0.initial_states:x'] = 0.0
        p['traj.phase0.initial_states:y'] = 10.0
        p['traj.phase0.initial_states:v'] = 0.0

        p['traj.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj.phase0.parameters:g'] = 9.80665

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-4)

    def test_brachistochrone_ode_prefix(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring(tol=1.0E-12)

        grid = dm.BirkhoffGrid(num_nodes=25, grid_type='lgl')
        tx = dm.Birkhoff(grid=grid)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.timeseries_options['use_prefix'] = True
        p.model.add_subsystem('traj', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_boundary_constraint('check', loc='final', lower=-50, upper=50)
        phase.add_path_constraint('check', upper=100, lower=-100)
        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.set_solver_print(0)

        p.setup()

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 1.5

        p['traj.phase0.initial_states:x'] = 0.0
        p['traj.phase0.initial_states:y'] = 10.0
        p['traj.phase0.initial_states:v'] = 0.0

        p['traj.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj.phase0.parameters:g'] = 9.80665

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-4)

    def test_brachistochrone_ode_no_prefix(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring(tol=1.0E-12)

        grid = dm.BirkhoffGrid(num_nodes=25, grid_type='lgl')
        tx = dm.Birkhoff(grid=grid)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        phase.timeseries_options['use_prefix'] = False
        p.model.add_subsystem('traj', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=False)
        phase.add_state('y', fix_initial=True, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)
        phase.add_boundary_constraint('check', loc='final', lower=-50, upper=50)
        phase.add_path_constraint('check', upper=100, lower=-100)

        # Minimize time at the end of the phase
        phase.add_objective('time_phase', loc='final', scaler=10)

        p.set_solver_print(0)

        p.setup()

        p['traj.phase0.t_initial'] = 0.0
        p['traj.phase0.t_duration'] = 1.5

        p['traj.phase0.initial_states:x'] = 0.0
        p['traj.phase0.initial_states:y'] = 10.0
        p['traj.phase0.initial_states:v'] = 0.0

        p['traj.phase0.states:x'] = phase.interp('x', [0, 10])
        p['traj.phase0.states:y'] = phase.interp('y', [10, 5])
        p['traj.phase0.states:v'] = phase.interp('v', [0, 9.9])
        p['traj.phase0.controls:theta'] = phase.interp('theta', [5, 100])
        p['traj.phase0.parameters:g'] = 9.80665

        p.run_driver()
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-4)
