import os
import unittest
from numpy.testing import assert_almost_equal

import openmdao.api as om
import dymos as dm

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.testing_utils import use_tempdirs
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT', fallback=True)


@use_tempdirs
class TestDuplicateConstraints(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        for filename in ['phase0_sim.db', 'brachistochrone_sim.db']:
            if os.path.exists(filename):
                os.remove(filename)

    def run_asserts(self, p):

        t_initial = p.get_val('traj0.phase0.timeseries.time')[0]
        tf = p.get_val('traj0.phase0.timeseries.time')[-1]

        x0 = p.get_val('traj0.phase0.timeseries.states:x')[0]
        xf = p.get_val('traj0.phase0.timeseries.states:x')[-1]

        y0 = p.get_val('traj0.phase0.timeseries.states:y')[0]
        yf = p.get_val('traj0.phase0.timeseries.states:y')[-1]

        v0 = p.get_val('traj0.phase0.timeseries.states:v')[0]
        vf = p.get_val('traj0.phase0.timeseries.states:v')[-1]

        g = p.get_val('traj0.phase0.timeseries.parameters:g')[0]

        thetaf = p.get_val('traj0.phase0.timeseries.controls:theta')[-1]

        assert_almost_equal(t_initial, 0.0)
        assert_almost_equal(x0, 0.0)
        assert_almost_equal(y0, 10.0)
        assert_almost_equal(v0, 0.0)

        assert_almost_equal(tf, 1.8016, decimal=4)
        assert_almost_equal(xf, 10.0, decimal=3)
        assert_almost_equal(yf, 5.0, decimal=3)
        assert_almost_equal(vf, 9.902, decimal=3)
        assert_almost_equal(g, 9.80665, decimal=3)

        assert_almost_equal(thetaf, 100.12, decimal=0)

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

        expected = 'Cannot add new initial boundary constraint for variable `x` and indices None. ' \
                   'One already exists.'

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

        expected = 'Cannot add new final boundary constraint for variable `x` and indices None. ' \
                   'One already exists.'

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

        expected = 'Cannot add new path constraint for variable `theta` and indices None. ' \
                   'One already exists.'

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

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent indices
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

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent_negative indices.
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

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent_negative indices.
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

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent_negative indices
        phase.add_path_constraint('theta', upper=100)
        phase.add_path_constraint('theta', lower=0, indices=[-1])

        with self.assertRaises(ValueError) as e:
            p.setup()

        expected = 'Duplicate constraint in phase traj0.phases.phase0. The following indices of `theta` are ' \
                   'used in multiple path constraints:\n{0}'

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

        # Now mistakenly add a boundary constraint at the same loc, variable, and different but equivalent_negative indices
        phase.add_boundary_constraint('g', loc='initial', upper=9.80665)
        phase.add_path_constraint('g', lower=0)

        with self.assertRaises(RuntimeError) as e:
            p.setup()

        expected = ("In phase traj0.phases.phase0, parameter `g` is subject to multiple boundary or path constraints.\n"
                    "Parameters are single values that do not change in time, and may only be used in a single "
                    "boundary or path constraint.")

        self.assertEqual(str(e.exception), expected)
