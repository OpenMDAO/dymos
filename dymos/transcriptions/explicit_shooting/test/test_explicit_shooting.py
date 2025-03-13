import unittest
import warnings

import numpy as np

import openmdao.api as om
import dymos as dm
from dymos import options as dymos_options

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class Simple2StateODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s**2')
        self.add_input('y', shape=(nn,), units='s')
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('p', shape=(nn,), units='s**2')

        self.add_output('x_dot', shape=(nn,), units='s')
        self.add_output('y_dot', shape=(nn,), units=None)

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='x_dot', wrt='p', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='y_dot', wrt='y', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='y_dot', wrt='t', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        t = inputs['t']
        p = inputs['p']
        outputs['x_dot'] = x - t**2 + p
        outputs['y_dot'] = -2 * y + t**3 * np.exp(-2*t)

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['x_dot', 't'] = -2*t
        partials['y_dot', 't'] = -np.exp(-2*t) * t**2 * (2 * t - 3)


class Simple1StateODE(om.ExplicitComponent):
    """
    A simple ODE from https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf
    """
    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('y', shape=(nn,), units=None)
        self.add_input('t', shape=(nn,), units='s')

        self.add_output('y_dot', shape=(nn,), units='1/s')

        ar = np.arange(nn, dtype=int)

        self.declare_partials(of='y_dot', wrt='y', rows=ar, cols=ar, val=-2.0)
        self.declare_partials(of='y_dot', wrt='t', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        y = inputs['y']
        t = inputs['t']
        outputs['y_dot'] = -2 * y + t**3 * np.exp(-2*t)

    def compute_partials(self, inputs, partials):
        t = inputs['t']
        partials['y_dot', 't'] = -np.exp(-2*t) * t**2 * (2 * t - 3)


@use_tempdirs
class TestExplicitShooting(unittest.TestCase):

    def test_1_state_run_model(self):

        dymos_options['include_check_partials'] = True

        prob = om.Problem()

        phase = dm.Phase(ode_class=Simple1StateODE,
                         transcription=dm.ExplicitShooting(grid=dm.GaussLobattoGrid(num_segments=1,
                                                                                    nodes_per_seg=3,
                                                                                    compressed=True)))

        phase.set_time_options(targets=['t'], units='s')

        # automatically discover states
        phase.set_state_options('y', targets=['y'], rate_source='y_dot')

        prob.model.add_subsystem('phase0', phase)

        prob.setup(force_alloc_complex=False)

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 1.0)
        prob.set_val('phase0.initial_states:y', 1.0)

        prob.run_model()

        with np.printoptions(linewidth=1024):
            cpd = prob.check_partials(compact_print=True, out_stream=None)

        assert_check_partials(cpd, rtol=1.0E-5)

        dymos_options['include_check_partials'] = False

    def test_2_states_run_model(self):

        dymos_options['include_check_partials'] = True

        prob = om.Problem()

        phase = dm.Phase(ode_class=Simple2StateODE,
                         transcription=dm.ExplicitShooting(grid=dm.GaussLobattoGrid(num_segments=2,
                                                                                    nodes_per_seg=3,
                                                                                    compressed=True)))

        phase.set_time_options(targets=['t'], units='s')

        # automatically discover states
        phase.set_state_options('x', targets=['x'], rate_source='x_dot')
        phase.set_state_options('y', targets=['y'], rate_source='y_dot')

        phase.add_parameter('p', targets=['p'])

        prob.model.add_subsystem('phase0', phase)

        prob.setup(force_alloc_complex=True)

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 1.0)
        prob.set_val('phase0.initial_states:x', 0.5)
        prob.set_val('phase0.initial_states:y', 1.0)
        prob.set_val('phase0.parameters:p', 1)

        prob.run_model()

        t_f = prob.get_val('phase0.integrator.t_final')
        x_f = prob.get_val('phase0.integrator.states_out:x')
        y_f = prob.get_val('phase0.integrator.states_out:y')

        assert_near_equal(t_f, 1.0)
        assert_near_equal(x_f[-1, ...], 2.64085909, tolerance=1.0E-5)
        assert_near_equal(y_f[-1, ...], 0.1691691, tolerance=1.0E-5)

        with np.printoptions(linewidth=1024):
            cpd = prob.check_partials(compact_print=True, method='fd', out_stream=None)
            assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

        dymos_options['include_check_partials'] = False

    def test_brachistochrone_explicit_shooting(self):

        dymos_options['include_check_partials'] = True

        input_grid = dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=3, compressed=False)

        output_grids = {'same': input_grid,
                        'more_dense': dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=7, compressed=True),
                        'radau': dm.RadauGrid(num_segments=3, nodes_per_seg=4, compressed=True),
                        'uniform': dm.UniformGrid(num_segments=3, nodes_per_seg=11)}

        for output_grid_type in ('same', 'more_dense', 'radau', 'uniform'):

            with self.subTest(f'output_grid = {output_grid_type}'):
                prob = om.Problem()

                phase = dm.Phase(ode_class=BrachistochroneODE,
                                 transcription=dm.ExplicitShooting(grid=input_grid,
                                                                   output_grid=output_grids[output_grid_type]))

                traj = prob.model.add_subsystem('traj0', dm.Trajectory())
                traj.add_phase('phase0', phase)

                prob.driver = om.ScipyOptimizeDriver()

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
                phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9,
                                  ref=90., rate_continuity=True, rate2_continuity=False)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)

                phase.add_objective('time', loc='final')

                prob.setup(force_alloc_complex=True)

                prob.set_val('traj0.phase0.t_initial', 0.0)
                prob.set_val('traj0.phase0.t_duration', 2)
                prob.set_val('traj0.phase0.initial_states:x', 0.0)
                prob.set_val('traj0.phase0.initial_states:y', 10.0)
                prob.set_val('traj0.phase0.initial_states:v', 0.1)
                prob.set_val('traj0.phase0.parameters:g', 9.80665, units='m/s**2')
                prob.set_val('traj0.phase0.controls:theta', phase.interp('theta', ys=[10, 50]), units='deg')

                # prob.run_model()

                dm.run_problem(prob)

                x = prob.get_val('traj0.phase0.timeseries.x')
                y = prob.get_val('traj0.phase0.timeseries.y')
                t = prob.get_val('traj0.phase0.timeseries.time')

                assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
                assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)
                assert_near_equal(t[-1, ...], 1.8016, tolerance=1.0E-2)

                with np.printoptions(linewidth=1024):
                    cpd = prob.check_partials(compact_print=True, method='cs',
                                              excludes=['traj0.phases.phase0.integrator'],
                                              out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

                    cpd = prob.check_partials(compact_print=True, method='fd',
                                              includes=['traj0.phases.phase0.integrator'],
                                              out_stream=None)
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

        dymos_options['include_check_partials'] = False

    def test_brachistochrone_explicit_shooting_path_constraint(self):

        dymos_options['include_check_partials'] = True

        input_grid = dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=3, compressed=True)

        output_grids = {'same': input_grid,
                        'more_dense': dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=7, compressed=True),
                        'radau': dm.RadauGrid(num_segments=3, nodes_per_seg=4, compressed=True),
                        'uniform': dm.UniformGrid(num_segments=3, nodes_per_seg=11)}

        for output_grid_type in ('same', 'more_dense', 'radau', 'uniform'):
            with self.subTest(f'output_grid = {output_grid_type}'):
                prob = om.Problem()

                phase = dm.Phase(ode_class=BrachistochroneODE,
                                 transcription=dm.ExplicitShooting(grid=input_grid,
                                                                   output_grid=output_grids[output_grid_type]))

                traj = prob.model.add_subsystem('traj0', dm.Trajectory())
                traj.add_phase('phase0', phase)

                prob.driver = om.ScipyOptimizeDriver()

                phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                # automatically discover states
                phase.set_state_options('x', fix_initial=True)
                phase.set_state_options('y', fix_initial=True)
                phase.set_state_options('v', fix_initial=True)

                phase.add_parameter('g', val=9, units='m/s**2', opt=True, lower=9, upper=9.80665)
                phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

                phase.add_boundary_constraint('x', loc='final', equals=10.0)
                phase.add_boundary_constraint('y', loc='final', equals=5.0)
                phase.add_path_constraint('ydot', lower=-100, upper=0)

                phase.add_objective('time', loc='final')

                prob.setup(force_alloc_complex=True)

                prob.set_val('traj0.phase0.t_initial', 0.0)
                prob.set_val('traj0.phase0.t_duration', 2)
                prob.set_val('traj0.phase0.initial_states:x', 0.0)
                prob.set_val('traj0.phase0.initial_states:y', 10.0)
                prob.set_val('traj0.phase0.initial_states:v', 1.0E-6)
                prob.set_val('traj0.phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

                prob.run_driver()

                x = prob.get_val('traj0.phase0.timeseries.x')
                y = prob.get_val('traj0.phase0.timeseries.y')
                ydot = prob.get_val('traj0.phase0.timeseries.ydot')

                assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
                self.assertTrue(np.all(ydot < 1.0E-6), msg='Not all elements of path constraint satisfied')
                assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)

                with np.printoptions(linewidth=1024):
                    cpd = prob.check_partials(compact_print=True, method='cs', out_stream=None,
                                              excludes=['traj0.phases.phase0.integrator'])
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

                    cpd = prob.check_partials(compact_print=True, method='fd', out_stream=None,
                                              includes=['traj0.phases.phase0.integrator'])
                    assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

        dymos_options['include_check_partials'] = False

    def test_brachistochrone_explicit_shooting_path_constraint_polynomial_control(self):

        dymos_options['include_check_partials'] = True

        for output_grid_type in ('same', 'more_dense', 'radau', 'uniform'):
            for compressed in (True, False):
                for path_rename in (True, False):
                    with self.subTest(f'output_grid = {output_grid_type}  compressed = {compressed}  '
                                      f'rename_path_const = {path_rename}'):
                        print(f'output_grid = {output_grid_type}  compressed = {compressed}  ')
                        prob = om.Problem()

                        input_grid = dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=3, compressed=compressed)

                        output_grids = {'same': input_grid,
                                        'more_dense': dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=7,
                                                                          compressed=True),
                                        'radau': dm.RadauGrid(num_segments=3, nodes_per_seg=4, compressed=True),
                                        'uniform': dm.UniformGrid(num_segments=3, nodes_per_seg=11)}

                        phase = dm.Phase(ode_class=BrachistochroneODE,
                                         transcription=dm.ExplicitShooting(grid=input_grid,
                                                                           output_grid=output_grids[output_grid_type]))

                        traj = prob.model.add_subsystem('traj0', dm.Trajectory())
                        traj.add_phase('phase0', phase)

                        prob.driver = om.ScipyOptimizeDriver()

                        phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

                        # automatically discover states
                        phase.set_state_options('x', fix_initial=True)
                        phase.set_state_options('y', fix_initial=True)
                        phase.set_state_options('v', fix_initial=True)

                        phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
                        phase.add_control('theta', order=2, val=45.0, units='deg', opt=True,
                                          lower=1.0E-6, upper=179.9, ref=90., control_type='polynomial')

                        phase.add_boundary_constraint('x', loc='final', equals=10.0)
                        phase.add_boundary_constraint('y', loc='final', equals=5.0)
                        phase.add_path_constraint('ydot', constraint_name='foo' if path_rename else None,
                                                  lower=-100, upper=0)

                        phase.add_objective('time', loc='final')

                        prob.setup(force_alloc_complex=True)

                        prob.set_val('traj0.phase0.t_initial', 0.0)
                        prob.set_val('traj0.phase0.t_duration', 2)
                        prob.set_val('traj0.phase0.initial_states:x', 0.0)
                        prob.set_val('traj0.phase0.initial_states:y', 10.0)
                        prob.set_val('traj0.phase0.initial_states:v', 1.0E-6)
                        prob.set_val('traj0.phase0.parameters:g', 9.80665, units='m/s**2')
                        prob.set_val('traj0.phase0.controls:theta',
                                     phase.interp('theta', ys=[0.01, 50]), units='deg')

                        dm.run_problem(prob)

                        x = prob.get_val('traj0.phase0.timeseries.x')
                        y = prob.get_val('traj0.phase0.timeseries.y')
                        t = prob.get_val('traj0.phase0.timeseries.time')

                        assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-5)
                        assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-5)
                        assert_near_equal(t[-1, ...], 1.807379, tolerance=1.0E-5)

                        with np.printoptions(linewidth=1024):
                            cpd = prob.check_partials(compact_print=True, method='cs', out_stream=None,
                                                      excludes=['traj0.phases.phase0.integrator'])
                            assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

                            cpd = prob.check_partials(compact_print=True, method='fd', out_stream=None,
                                                      includes=['traj0.phases.phase0.integrator'])
                            assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

        dymos_options['include_check_partials'] = False

    def test_brachistochrone_explicit_shooting_path_constraint_invalid_renamed(self):

        dymos_options['include_check_partials'] = True

        for output_grid_type in ('same', 'more_dense', 'radau', 'uniform'):
            prob = om.Problem()

            input_grid = dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=3, compressed=True)

            output_grids = {'same': input_grid,
                            'more_dense': dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=7,
                                                              compressed=True),
                            'radau': dm.RadauGrid(num_segments=3, nodes_per_seg=4, compressed=True),
                            'uniform': dm.UniformGrid(num_segments=3, nodes_per_seg=11)}

            phase = dm.Phase(ode_class=BrachistochroneODE,
                             transcription=dm.ExplicitShooting(grid=input_grid,
                                                               output_grid=output_grids[output_grid_type]))

            phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

            # automatically discover states
            phase.set_state_options('x', fix_initial=True)
            phase.set_state_options('y', fix_initial=True)
            phase.set_state_options('v', fix_initial=True)

            phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)
            phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

            phase.add_boundary_constraint('x', loc='final', equals=10.0)
            phase.add_boundary_constraint('y', loc='final', equals=5.0)
            phase.add_path_constraint('y', constraint_name='foo', lower=5, upper=100)

            prob.model.add_subsystem('phase0', phase)

            phase.add_objective('time', loc='final')

            with warnings.catch_warnings(record=True) as ctx:
                warnings.simplefilter('always')
                prob.setup(check=True)

            self.assertIn("<class 'openmdao.utils.om_warnings.UnusedOptionWarning'>: "
                          "Option 'constraint_name' on path constraint y is only valid for "
                          "ODE outputs. The option is being ignored.", [str(w.message) for w in ctx])

        dymos_options['include_check_partials'] = False

    def test_explicit_shooting_timeseries_ode_output(self):

        dymos_options['include_check_partials'] = True

        for output_grid_type in ('same', 'more_dense', 'radau', 'uniform'):
            prob = om.Problem()

            prob.driver = om.ScipyOptimizeDriver()

            input_grid = dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=3, compressed=True)

            output_grids = {'same': input_grid,
                            'more_dense': dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=7,
                                                              compressed=True),
                            'radau': dm.RadauGrid(num_segments=3, nodes_per_seg=4, compressed=True),
                            'uniform': dm.UniformGrid(num_segments=3, nodes_per_seg=11)}

            phase = dm.Phase(ode_class=BrachistochroneODE,
                             transcription=dm.ExplicitShooting(grid=input_grid,
                                                               output_grid=output_grids[output_grid_type]))

            phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

            # automatically discover states
            phase.set_state_options('x', fix_initial=True)
            phase.set_state_options('y', fix_initial=True)
            phase.set_state_options('v', fix_initial=True)

            phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
            phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

            phase.add_boundary_constraint('x', loc='final', equals=10.0)
            phase.add_boundary_constraint('y', loc='final', equals=5.0)

            phase.add_timeseries_output('*')

            prob.model.add_subsystem('phase0', phase)

            phase.add_objective('time', loc='final')

            prob.setup(force_alloc_complex=True)

            prob.set_val('phase0.t_initial', 0.0)
            prob.set_val('phase0.t_duration', 2)
            prob.set_val('phase0.initial_states:x', 0.0)
            prob.set_val('phase0.initial_states:y', 10.0)
            prob.set_val('phase0.initial_states:v', 1.0E-6)
            prob.set_val('phase0.parameters:g', 1.0, units='m/s**2')
            prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 90]), units='deg')

            dm.run_problem(prob, run_driver=True)

            x = prob.get_val('phase0.timeseries.x')
            y = prob.get_val('phase0.timeseries.y')
            v = prob.get_val('phase0.timeseries.v')
            theta = prob.get_val('phase0.timeseries.theta', units='rad')
            check = prob.get_val('phase0.timeseries.check')
            t = prob.get_val('phase0.timeseries.time')

            assert_near_equal(x[-1, ...], 10.0, tolerance=1.0E-3)
            assert_near_equal(y[-1, ...], 5.0, tolerance=1.0E-3)
            assert_near_equal(t[-1, ...], 1.8016, tolerance=1.0E-2)
            assert_near_equal(check, v / np.sin(theta))

            with np.printoptions(linewidth=1024):
                cpd = prob.check_partials(method='cs', out_stream=None, excludes=['phase0.integrator'])
                assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

                cpd = prob.check_partials(method='fd', out_stream=None, includes=['phase0.integrator'])
                assert_check_partials(cpd, atol=1.0E-5, rtol=1.0E-5)

            dymos_options['include_check_partials'] = False

    def test_explicit_shooting_unknown_timeseries(self):

        for output_grid_type in ('same', 'more_dense', 'radau', 'uniform'):
            prob = om.Problem()

            input_grid = dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=3, compressed=True)

            output_grids = {'same': input_grid,
                            'more_dense': dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=7,
                                                              compressed=True),
                            'radau': dm.RadauGrid(num_segments=3, nodes_per_seg=4, compressed=True),
                            'uniform': dm.UniformGrid(num_segments=3, nodes_per_seg=11)}

            phase = dm.Phase(ode_class=BrachistochroneODE,
                             transcription=dm.ExplicitShooting(grid=input_grid,
                                                               output_grid=output_grids[output_grid_type]))

            phase.set_time_options(units='s', fix_initial=True, duration_bounds=(1.0, 10.0))

            # automatically discover states
            phase.set_state_options('x', fix_initial=True)
            phase.set_state_options('y', fix_initial=True)
            phase.set_state_options('v', fix_initial=True)

            phase.add_parameter('g', val=1.0, units='m/s**2', opt=True, lower=1, upper=9.80665)
            phase.add_control('theta', val=45.0, units='deg', opt=True, lower=1.0E-6, upper=179.9)

            phase.add_boundary_constraint('x', loc='final', equals=10.0)
            phase.add_boundary_constraint('y', loc='final', equals=5.0)

            phase.add_timeseries_output('*')
            phase.add_timeseries_output('foo')

            prob.model.add_subsystem('phase0', phase)

            phase.add_objective('time', loc='final')

            msg = "Unable to find the source 'foo' in the ODE."

            with self.assertRaises(ValueError) as e:
                prob.setup()

            self.assertIn(msg, str(e.exception))

    def test_brachistochrone_static_gravity_explicit_shooting(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal
        import dymos as dm
        from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

        for output_grid_type in ('same', 'more_dense', 'radau', 'uniform'):
            p = om.Problem()

            input_grid = dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=3, compressed=True)

            output_grids = {'same': input_grid,
                            'more_dense': dm.GaussLobattoGrid(num_segments=3, nodes_per_seg=7,
                                                              compressed=True),
                            'radau': dm.RadauGrid(num_segments=3, nodes_per_seg=4, compressed=True),
                            'uniform': dm.UniformGrid(num_segments=3, nodes_per_seg=11)}

            phase = dm.Phase(ode_class=BrachistochroneODE,
                             ode_init_kwargs={'static_gravity': True},
                             transcription=dm.ExplicitShooting(grid=input_grid,
                                                               output_grid=output_grids[output_grid_type]))

            traj = p.model.add_subsystem('traj', dm.Trajectory())

            traj.add_phase('phase0', phase)

            #
            # Set the variables
            #
            phase.set_time_options(fix_initial=True, duration_bounds=(.5, 10))

            phase.add_state('x', rate_source='xdot',
                            targets=None,
                            units='m',
                            fix_initial=True)

            phase.add_state('y', rate_source='ydot',
                            targets=None,
                            units='m',
                            fix_initial=True)

            phase.add_state('v', rate_source='vdot',
                            targets=['v'],
                            units='m/s',
                            fix_initial=True)

            phase.add_control('theta', targets=['theta'],
                              continuity=True, rate_continuity=True,
                              units='deg', lower=0.01, upper=179.9)

            phase.add_parameter('g', targets=['g'], static_target=True, opt=False)

            #
            # Constrain the final values of x and y
            #
            phase.add_boundary_constraint('x', loc='final', equals=10)
            phase.add_boundary_constraint('y', loc='final', equals=5)

            #
            # Minimize time at the end of the phase
            #
            phase.add_objective('time', loc='final', scaler=10)

            #
            # Set the optimization driver
            #
            p.driver = om.ScipyOptimizeDriver()

            #
            # Setup the Problem
            #
            p.setup()

            #
            # Set the initial values
            # The initial time is fixed, and we set that fixed value here.
            # The optimizer is allowed to modify t_duration, but an initial guess is provided here.
            #
            p.set_val('traj.phase0.t_initial', 0.0)
            p.set_val('traj.phase0.t_duration', 2.0)

            # Guesses for states are provided at all state_input nodes.
            # We use the phase.interpolate method to linearly interpolate values onto the state input nodes.
            # Since fix_initial=True for all states and fix_final=True for x and y, the initial or final
            # values of the interpolation provided here will not be changed by the optimizer.
            p.set_val('traj.phase0.initial_states:x', phase.interp('x', [0, 10]))
            p.set_val('traj.phase0.initial_states:y', phase.interp('y', [10, 5]))
            p.set_val('traj.phase0.initial_states:v', phase.interp('v', [0, 9.9]))

            # Guesses for controls are provided at all control_input node.
            # Here phase.interpolate is used to linearly interpolate values onto the control input nodes.
            p.set_val('traj.phase0.controls:theta', phase.interp('theta', [5, 100.5]))

            # Set the value for gravitational acceleration.
            p.set_val('traj.phase0.parameters:g', 9.80665)

            #
            # Solve for the optimal trajectory
            #
            dm.run_problem(p, simulate=False)

            assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-3)
