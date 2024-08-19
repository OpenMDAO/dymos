import os
import unittest

import numpy as np

import openmdao.api as om
import dymos as dm

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=True)


@use_tempdirs
class TestDuplicateConstraints(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def run_asserts(self, p, tol=0.01):

        t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
        tf = p.get_val('traj0.phase0.timeseries.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.x')[0]
        xf = p.get_val('traj0.phase0.timeseries.x')[-1]

        y0 = p.get_val('traj0.phase0.timeseries.y')[0]
        yf = p.get_val('traj0.phase0.timeseries.y')[-1]

        v0 = p.get_val('traj0.phase0.timeseries.v')[0]
        vf = p.get_val('traj0.phase0.timeseries.v')[-1]

        g = p.get_val('traj0.phase0.timeseries.g')[0]

        thetaf = p.get_val('traj0.phase0.timeseries.theta', units='deg')[-1]

        assert_near_equal(t_initial, 0.0, tolerance=0.01)
        assert_near_equal(x0, 0.0, tolerance=0.01)
        assert_near_equal(y0, 10.0, tolerance=0.01)
        assert_near_equal(v0, 0.0, tolerance=0.01)

        assert_near_equal(tf, 1.8016, tolerance=0.01)
        assert_near_equal(xf, 10.0, tolerance=0.01)
        assert_near_equal(yf, 5.0, tolerance=0.01)
        assert_near_equal(vf, 9.902, tolerance=0.01)
        assert_near_equal(g, 9.80665, tolerance=0.01)

        assert_near_equal(thetaf, 100.12, tolerance=0.01)

    def test_duplicate_initial_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
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

        phase.add_boundary_constraint('x', loc='initial', equals=0)

        # Now mistakenly add a boundary constraint at the same loc, variable, and indices.
        with self.assertRaises(ValueError) as e:
            phase.add_boundary_constraint('x', loc='initial', equals=10)

        expected = 'Cannot add new initial boundary constraint for variable `x` and indices None. One already exists.'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_final_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
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

        phase.add_boundary_constraint('x', loc='final', equals=0)

        # Now mistakenly add a boundary constraint at the same loc, variable, and indices.
        with self.assertRaises(ValueError) as e:
            phase.add_boundary_constraint('x', loc='final', equals=10)

        expected = 'Cannot add new final boundary constraint for variable `x` and indices None. One already exists.'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_path_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
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

        phase.add_path_constraint('theta', lower=0)

        # Now mistakenly add a boundary constraint at the same loc, variable, and indices.
        with self.assertRaises(ValueError) as e:
            phase.add_path_constraint('theta', upper=100)

        expected = 'Cannot add new path constraint for variable `theta` and indices None. One already exists.'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_initial_constraint_equivalent_indices(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='initial', equals=0)
        phase.add_boundary_constraint('y', loc='initial', equals=10)
        phase.add_boundary_constraint('v', loc='initial', equals=0)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent indices.
        phase.add_boundary_constraint('x', loc='initial', equals=10, indices=[0])

        phase.add_objective('time', loc='final')

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `x` are ' \
                   'used in multiple initial boundary constraints:\n{0}'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_final_constraint_equivalent_indices(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='initial', equals=0)
        phase.add_boundary_constraint('y', loc='initial', equals=10)
        phase.add_boundary_constraint('v', loc='initial', equals=0)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent indices.
        phase.add_boundary_constraint('x', loc='final', equals=10, indices=[0])

        phase.add_objective('time', loc='final')

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `x` are ' \
                   'used in multiple final boundary constraints:\n{0}'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_path_constraint_equivalent_indices(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='initial', equals=0)
        phase.add_boundary_constraint('y', loc='initial', equals=10)
        phase.add_boundary_constraint('v', loc='initial', equals=0)

        phase.add_boundary_constraint('x', loc='final', equals=10)
        phase.add_boundary_constraint('y', loc='final', equals=5)

        # Now mistakenly add a path constraint at the same loc, variable, and different but equivalent indices
        phase.add_path_constraint('theta', upper=100)
        phase.add_path_constraint('theta', lower=0, indices=[0])

        phase.add_objective('time', loc='final')

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `theta` are ' \
                   'used in multiple path constraints:\n{0}'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_initial_constraint_equivalent_negative_indices(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='initial', equals=0)

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent negative indices.
        phase.add_boundary_constraint('x', loc='initial', equals=10, indices=[-1])

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `x` are ' \
                   'used in multiple initial boundary constraints:\n{0}'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_final_constraint_equivalent_negative_indices(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='final', equals=10)

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent negative indices.
        phase.add_boundary_constraint('x', loc='final', equals=10, indices=[-1])

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `x` are ' \
                   'used in multiple final boundary constraints:\n{0}'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_path_constraint_equivalent_negative_indices(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        # Now mistakenly add a path constraint at the same loc, variable, and different but equivalent negative indices
        phase.add_path_constraint('theta', upper=100)
        phase.add_path_constraint('theta', lower=0, indices=[-1])

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `theta` are ' \
                   'used in multiple path constraints:\n{0}'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_initial_constraint_indies_as_ndarray(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2')

        phase.add_boundary_constraint('x', loc='initial', equals=0, indices=np.array([0], dtype=int))

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent negative indices.
        phase.add_boundary_constraint('x', loc='initial', equals=10, indices=np.array([-1], dtype=int))

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `x` are ' \
                   'used in multiple initial boundary constraints:\n{0}'

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_parameter_constraint(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=False, fix_final=False)
        phase.add_state('y', fix_initial=False, fix_final=False)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=False, fix_final=False)

        phase.add_control('theta',
                          continuity=True, rate_continuity=True,
                          units='deg', lower=0.01, upper=179.9)

        phase.add_parameter('g', targets=['g'], units='m/s**2', opt=True)

        # Add a parameter as both a boundary and path constraint. This is illegal since the parameter value cannot
        # change during the phase.  A user must choose either a path constraint or a boundary constraint.
        phase.add_boundary_constraint('g', loc='initial', upper=9.80665)
        phase.add_path_constraint('g', lower=0)

        with self.assertRaises(RuntimeError) as e:
            p.setup()

        expected = ("In phase traj0.phases.phase0, parameter `g` is subject to multiple boundary or path constraints.\n"
                    "Parameters are single values that do not change in time, and may only be used in a single "
                    "boundary or path constraint.")

        self.assertEqual(str(e.exception), expected)

    def test_duplicate_path_constraint_different_constraint_name(self):
        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver(tol=1.0E-6)
        p.driver.declare_coloring(tol=1.0E-12)

        tx = dm.Radau(num_segments=10, order=3)

        traj = dm.Trajectory()
        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)
        p.model.add_subsystem('traj0', traj)
        traj.add_phase('phase0', phase)

        phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

        phase.add_state('x', fix_initial=True, fix_final=True)
        phase.add_state('y', fix_initial=True, fix_final=True)

        # Note that by omitting the targets here Dymos will automatically attempt to connect
        # to a top-level input named 'v' in the ODE, and connect to nothing if it's not found.
        phase.add_state('v', fix_initial=True, fix_final=False)

        phase.add_control('theta', units='rad', lower=1E-6, upper=np.pi)

        phase.add_parameter('g', units='m/s**2')

        # Constrain the ODE output "check", but with different names in different places.
        phase.add_boundary_constraint('check', loc='final', lower=-50, upper=50)
        phase.add_path_constraint('check', constraint_name='bounded_check', upper=100, lower=-100)
        phase.add_objective('time', loc='final')

        p.setup()

        phase.set_time_val(initial=0.0, duration=2.0)

        phase.set_state_val('x', [0, 10])
        phase.set_state_val('y', [10, 5])
        phase.set_state_val('v', [0.001, 9.9])
        phase.set_control_val('theta', [5, 90], units='deg')
        phase.set_parameter_val('g', 9.80665)

        p.run_model()

        v = p.get_val('traj0.phase0.timeseries.v')
        theta = p.get_val('traj0.phase0.timeseries.theta', units='rad')

        check_calc = v / np.sin(theta)
        check_1 = p.get_val('traj0.phase0.timeseries.check')
        check_2 = p.get_val('traj0.phase0.timeseries.bounded_check')

        assert_near_equal(check_calc, check_1)
        assert_near_equal(check_calc, check_2)
