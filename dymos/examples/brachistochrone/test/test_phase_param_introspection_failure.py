import unittest
import numpy as np
import openmdao.api as om
import dymos as dm

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.units import convert_units


class _TestEOM(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))
        self.options.declare('foo_units', allow_none=True, default=None)
        self.options.declare('foo_shapes', allow_none=True, default=None)
        self.options.declare('foo_static', default=[])

    def setup(self):
        num_nodes = self.options['num_nodes']

        foo_shape = (num_nodes,) if self.options['foo_shapes'] is None \
            else (num_nodes,) + self.options['foo_shapes']['vdot_comp']

        foo_unit = 'kg' if self.options['foo_units'] is None else self.options['foo_units']['vdot_comp']

        foo_tags = ['dymos.static_target']\
            if 'vdot_comp' in self.options['foo_static'] and self.options['foo_static']['vdot_comp'] else []

        foo_shape = (1,) if 'dymos.static_target' in foo_tags else foo_shape

        vdot_comp = om.ExecComp(['vdot = g * cos(theta)',
                                 'bar = foo'],
                                vdot={'shape': (num_nodes,), 'units': 'm/s**2'},
                                g={'val': 9.80665, 'units': 'm/s**2'},
                                theta={'shape': (num_nodes,), 'units': 'rad'},
                                foo={'shape': foo_shape, 'units': foo_unit, 'tags': foo_tags},
                                bar={'shape': foo_shape, 'units': foo_unit})

        foo_shape = (num_nodes,) if self.options['foo_shapes'] is None \
            else (num_nodes,) + self.options['foo_shapes']['xdot_comp']

        foo_unit = 'kg' if self.options['foo_units'] is None else self.options['foo_units']['xdot_comp']

        foo_tags = ['dymos.static_target']\
            if 'xdot_comp' in self.options['foo_static'] and self.options['foo_static']['xdot_comp'] else []

        foo_shape = (1,) if 'dymos.static_target' in foo_tags else foo_shape

        xdot_comp = om.ExecComp(['xdot = v * sin(theta)',
                                 'bar = foo'],
                                xdot={'shape': (num_nodes,), 'units': 'm/s'},
                                v={'shape': (num_nodes,), 'units': 'm/s'},
                                theta={'shape': (num_nodes,), 'units': 'rad'},
                                foo={'shape': foo_shape, 'units': foo_unit, 'tags': foo_tags},
                                bar={'shape': foo_shape, 'units': foo_unit})

        foo_shape = (num_nodes,) if self.options['foo_shapes'] is None \
            else (num_nodes,) + self.options['foo_shapes']['ydot_comp']

        foo_unit = 'kg' if self.options['foo_units'] is None else self.options['foo_units']['ydot_comp']

        foo_tags = ['dymos.static_target']\
            if 'ydot_comp' in self.options['foo_static'] and self.options['foo_static']['ydot_comp'] else []

        foo_shape = (1,) if 'dymos.static_target' in foo_tags else foo_shape

        ydot_comp = om.ExecComp(['ydot = -v * cos(theta)',
                                 'bar = foo'],
                                ydot={'shape': (num_nodes,), 'units': 'm/s'},
                                v={'shape': (num_nodes,), 'units': 'm/s'},
                                theta={'shape': (num_nodes,), 'units': 'rad'},
                                foo={'shape': foo_shape, 'units': foo_unit, 'tags': foo_tags},
                                bar={'shape': foo_shape, 'units': foo_unit})

        self.add_subsystem('vdot_comp', vdot_comp)
        self.add_subsystem('xdot_comp', xdot_comp)
        self.add_subsystem('ydot_comp', ydot_comp)


@use_tempdirs
class TestParameterShapes(unittest.TestCase):

    def test_valid_parameters(self):
        import numpy as np
        import openmdao.api as om
        import dymos as dm

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=_TestEOM,
                         transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                        rate_source='xdot_comp.xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m',
                        rate_source='ydot_comp.ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot_comp.vdot', targets=['xdot_comp.v', 'ydot_comp.v'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi,
                          targets=['xdot_comp.theta', 'ydot_comp.theta', 'vdot_comp.theta'])

        phase.add_parameter('foo',
                            opt=False,
                            targets=['xdot_comp.foo', 'ydot_comp.foo', 'vdot_comp.foo'])

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.
        phase.set_time_val(initial=0.0, duration=2.0, units='s')
        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 9.9], units='m/s')
        phase.set_control_val('theta', [90, 90], units='deg')
        phase.set_parameter_val('foo', 5.0)

        # Run the driver to solve the problem
        p.run_driver()

        self.assertEqual((1,), phase.parameter_options['foo']['shape'])
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.parameter_vals:foo')[-1], 5.0, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.xdot_comp.foo'), 5.0*np.ones(10,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.ydot_comp.foo'), 5.0*np.ones(10,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.vdot_comp.foo'), 5.0*np.ones(10,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.xdot_comp.foo'), 5.0*np.ones(20,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.ydot_comp.foo'), 5.0*np.ones(20,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.vdot_comp.foo'), 5.0*np.ones(20,), tolerance=1.0E-5)

    def test_invalid_params_different_target_shapes(self):
        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=_TestEOM,
                         transcription=dm.GaussLobatto(num_segments=10, order=3),
                         ode_init_kwargs={'foo_shapes': {'xdot_comp': (2,),
                                                         'ydot_comp': (2,),
                                                         'vdot_comp': (2,)}})

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                        rate_source='xdot_comp.xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m',
                        rate_source='ydot_comp.ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot_comp.vdot', targets=['xdot_comp.v', 'ydot_comp.v'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi,
                          targets=['xdot_comp.theta', 'ydot_comp.theta', 'vdot_comp.theta'])

        phase.add_parameter('foo', shape=(1,),
                            opt=False,
                            targets=['xdot_comp.foo', 'ydot_comp.foo', 'vdot_comp.foo'])

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        with self.assertRaises(RuntimeError) as e:
            p.setup(check=True)

        expected = ("Shape provided to parameter `foo` differs from its targets.\n"
                    "Given shape: (1,)\n"
                    "Target shapes:\n"
                    "{'xdot_comp.foo': (2,), 'ydot_comp.foo': (2,), 'vdot_comp.foo': (2,)}")
        self.assertEqual(expected, str(e.exception))

    def test_invalid_params_different_target_shapes_introspection_failure(self):
        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=_TestEOM,
                         transcription=dm.GaussLobatto(num_segments=10, order=3),
                         ode_init_kwargs={'foo_shapes': {'xdot_comp': (1,),
                                                         'ydot_comp': (2,),
                                                         'vdot_comp': (3,)}})

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                        rate_source='xdot_comp.xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m',
                        rate_source='ydot_comp.ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot_comp.vdot', targets=['xdot_comp.v', 'ydot_comp.v'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi,
                          targets=['xdot_comp.theta', 'ydot_comp.theta', 'vdot_comp.theta'])

        phase.add_parameter('foo',
                            opt=False,
                            targets=['xdot_comp.foo', 'ydot_comp.foo', 'vdot_comp.foo'])

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        with self.assertRaises(RuntimeError) as e:
            p.setup(check=True)

        expected = ('Invalid targets for parameter `foo`.\n'
                    'Targets have multiple shapes.\n'
                    "{'xdot_comp.foo': (1,), 'ydot_comp.foo': (2,), 'vdot_comp.foo': (3,)}")
        self.assertEqual(expected, str(e.exception))


@use_tempdirs
class TestParameterUnits(unittest.TestCase):

    def test_valid_parameters(self):
        import numpy as np
        import openmdao.api as om
        import dymos as dm

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=_TestEOM,
                         transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                        rate_source='xdot_comp.xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m',
                        rate_source='ydot_comp.ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot_comp.vdot', targets=['xdot_comp.v', 'ydot_comp.v'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi,
                          targets=['xdot_comp.theta', 'ydot_comp.theta', 'vdot_comp.theta'])

        phase.add_parameter('foo',
                            units='lbm',
                            opt=False,
                            targets=['xdot_comp.foo', 'ydot_comp.foo', 'vdot_comp.foo'])

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.

        phase.set_time_val(initial=0.0, duration=2.0, units='s')
        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 9.9], units='m/s')
        phase.set_control_val('theta', [90, 90], units='deg')
        phase.set_parameter_val('foo', 5.0)

        # Run the driver to solve the problem
        p.run_driver()

        expected = convert_units(5.0, 'lbm', 'kg')

        self.assertEqual((1,), phase.parameter_options['foo']['shape'])
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.parameter_vals:foo')[-1], 5.0, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.xdot_comp.foo'), expected*np.ones(10,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.ydot_comp.foo'), expected*np.ones(10,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.vdot_comp.foo'), expected*np.ones(10,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.xdot_comp.foo'), expected*np.ones(20,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.ydot_comp.foo'), expected*np.ones(20,), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.vdot_comp.foo'), expected*np.ones(20,), tolerance=1.0E-5)

    def test_invalid_params_different_target_units_introspection_failure(self):
        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=_TestEOM,
                         transcription=dm.GaussLobatto(num_segments=10, order=3),
                         ode_init_kwargs={'foo_units': {'xdot_comp': 'kg',
                                                        'ydot_comp': 'lbm',
                                                        'vdot_comp': 'slug'}})

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                        rate_source='xdot_comp.xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m',
                        rate_source='ydot_comp.ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot_comp.vdot', targets=['xdot_comp.v', 'ydot_comp.v'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi,
                          targets=['xdot_comp.theta', 'ydot_comp.theta', 'vdot_comp.theta'])

        phase.add_parameter('foo',
                            opt=False,
                            targets=['xdot_comp.foo', 'ydot_comp.foo', 'vdot_comp.foo'])

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        with self.assertRaises(RuntimeError) as e:
            p.setup(check=True)

        expected = ("Unable to automatically assign units based on targets.\n"
                    "Targets have multiple units assigned:\n"
                    "{'xdot_comp.foo': 'kg', 'ydot_comp.foo': 'lbm', 'vdot_comp.foo': 'slug'}.\n"
                    "Either promote targets and use set_input_defaults to assign common\n"
                    "units, or explicitly provide units to the variable.")

        self.assertEqual(expected, str(e.exception))


@use_tempdirs
class TestMixedStaticDynamicParameterTargets(unittest.TestCase):

    def test_all_static(self):
        import numpy as np
        import openmdao.api as om
        import dymos as dm

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=_TestEOM,
                         transcription=dm.GaussLobatto(num_segments=10, order=3),
                         ode_init_kwargs={'foo_static': {'xdot_comp': True, 'ydot_comp': True, 'vdot_comp': True}})

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                        rate_source='xdot_comp.xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m',
                        rate_source='ydot_comp.ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot_comp.vdot', targets=['xdot_comp.v', 'ydot_comp.v'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi,
                          targets=['xdot_comp.theta', 'ydot_comp.theta', 'vdot_comp.theta'])

        phase.add_parameter('foo',
                            units='lbm',
                            opt=False,
                            targets=['xdot_comp.foo', 'ydot_comp.foo', 'vdot_comp.foo'])

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.
        phase.set_time_val(initial=0.0, duration=2.0, units='s')
        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 9.9], units='m/s')
        phase.set_control_val('theta', [90, 90], units='deg')
        phase.set_parameter_val('foo', 5.0)

        # Run the driver to solve the problem
        p.run_driver()

        expected = convert_units(5.0, 'lbm', 'kg')

        self.assertEqual((1,), phase.parameter_options['foo']['shape'])
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.parameter_vals:foo')[-1], 5.0, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.xdot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.ydot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.vdot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.xdot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.ydot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.vdot_comp.foo'), expected, tolerance=1.0E-5)

    def test_mixed_static(self):
        import numpy as np
        import openmdao.api as om
        import dymos as dm

        #
        # Define the OpenMDAO problem
        #
        p = om.Problem(model=om.Group())

        #
        # Define a Trajectory object
        #
        traj = dm.Trajectory()

        p.model.add_subsystem('traj', subsys=traj)

        #
        # Define a Dymos Phase object with GaussLobatto Transcription
        #
        phase = dm.Phase(ode_class=_TestEOM,
                         transcription=dm.GaussLobatto(num_segments=10, order=3),
                         ode_init_kwargs={'foo_static': {'xdot_comp': True, 'ydot_comp': True, 'vdot_comp': False}})

        traj.add_phase(name='phase0', phase=phase)

        #
        # Set the time options
        # Time has no targets in our ODE.
        # We fix the initial time so that it is not a design variable in the optimization.
        # The duration of the phase is allowed to be optimized, but is bounded on [0.5, 10].
        #
        phase.set_time_options(fix_initial=True, duration_bounds=(0.5, 10.0), units='s')

        #
        # Set the time options
        # Initial values of positions and velocity are all fixed.
        # The final value of position are fixed, but the final velocity is a free variable.
        # The equations of motion are not functions of position, so 'x' and 'y' have no targets.
        # The rate source points to the output in the ODE which provides the time derivative of the
        # given state.
        phase.add_state('x', fix_initial=True, fix_final=True, units='m',
                        rate_source='xdot_comp.xdot')
        phase.add_state('y', fix_initial=True, fix_final=True, units='m',
                        rate_source='ydot_comp.ydot')
        phase.add_state('v', fix_initial=True, fix_final=False, units='m/s',
                        rate_source='vdot_comp.vdot', targets=['xdot_comp.v', 'ydot_comp.v'])

        # Define theta as a control.
        phase.add_control(name='theta', units='rad', lower=0, upper=np.pi,
                          targets=['xdot_comp.theta', 'ydot_comp.theta', 'vdot_comp.theta'])

        phase.add_parameter('foo',
                            units='lbm',
                            opt=False,
                            targets=['xdot_comp.foo', 'ydot_comp.foo', 'vdot_comp.foo'])

        # Minimize final time.
        phase.add_objective('time', loc='final')

        # Set the driver.
        p.driver = om.ScipyOptimizeDriver()

        # Allow OpenMDAO to automatically determine our sparsity pattern.
        # Doing so can significant speed up the execution of Dymos.
        p.driver.declare_coloring()

        # Setup the problem
        p.setup(check=True)

        # Now that the OpenMDAO problem is setup, we can set the values of the states.
        phase.set_time_val(initial=0.0, duration=2.0, units='s')
        phase.set_state_val('x', [0, 10], units='m')
        phase.set_state_val('y', [10, 5], units='m')
        phase.set_state_val('v', [0, 9.9], units='m/s')
        phase.set_control_val('theta', [90, 90], units='deg')
        phase.set_parameter_val('foo', 5.0)

        # Run the driver to solve the problem
        p.run_driver()

        expected = convert_units(5.0, 'lbm', 'kg')

        self.assertEqual((1,), phase.parameter_options['foo']['shape'])
        self.assertEqual({'xdot_comp.foo', 'ydot_comp.foo'}, phase.parameter_options['foo']['static_targets'])
        assert_near_equal(p.get_val('traj.phase0.timeseries.time')[-1], 1.8016, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.parameter_vals:foo')[-1], 5.0, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.xdot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.ydot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_col.vdot_comp.foo'), expected*np.ones(10), tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.xdot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.ydot_comp.foo'), expected, tolerance=1.0E-5)
        assert_near_equal(p.get_val('traj.phase0.rhs_disc.vdot_comp.foo'), expected*np.ones(20), tolerance=1.0E-5)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
