import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse

import dymos as dm

from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class ParamComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('falling_object:mass', np.array([0.3, 0.5, 0.9]), units='kg')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        pass


class ODEGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('p1', ParamComp(), promotes=['*'])
        self.add_subsystem('brach', BrachistochroneODE(num_nodes=nn), promotes=['*'])


def make_traj(transcription='gauss-lobatto', transcription_order=3, compressed=False,
              connected=False, param_mode='param_sequence'):

    t = {'gauss-lobatto': dm.GaussLobatto(num_segments=5, order=transcription_order, compressed=compressed),
         'radau': dm.Radau(num_segments=20, order=transcription_order, compressed=compressed)}

    traj = dm.Trajectory()

    if param_mode == 'param_sequence':
        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'coast': ['c'], 'burn2': ['c']})
    elif param_mode == 'param_sequence_missing_phase':
        traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                           targets={'burn1': ['c'], 'burn2': ['c']})
    elif param_mode == 'param_no_targets':
        traj.add_parameter('c', val=1.5, units='DU/TU')

    # First Phase (burn)

    burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

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
                    rate_source='deltav_dot', targets=None, units='DU/TU')
    burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg', scaler=0.01,
                      rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                      lower=-30, upper=30, targets=['u1'])
    # Second Phase (Coast)
    coast = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 50), duration_ref=50, units='TU')
    coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='r_dot', targets=['r'], units='DU')
    coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='theta_dot', targets=['theta'], units='rad')
    coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='vr_dot', targets=['vr'], units='DU/TU')
    coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='vt_dot', targets=['vt'], units='DU/TU')
    coast.add_state('accel', fix_initial=True, fix_final=True,
                    rate_source='at_dot', targets=['accel'], units='DU/TU**2')
    coast.add_state('deltav', fix_initial=False, fix_final=False,
                    rate_source='deltav_dot', units='DU/TU', targets=None)

    coast.add_parameter('u1', opt=False, val=0.0, units='deg', targets=['u1'])

    # Third Phase (burn)
    burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    traj.add_phase('coast', coast)
    traj.add_phase('burn2', burn2)

    burn2.set_time_options(initial_bounds=(0.5, 50), duration_bounds=(.5, 10), initial_ref=10, units='TU')
    burn2.add_state('r', fix_initial=False, fix_final=True, defect_scaler=100.0,
                    rate_source='r_dot', targets=['r'], units='DU')
    burn2.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='theta_dot', targets=['theta'], units='rad')
    burn2.add_state('vr', fix_initial=False, fix_final=True, defect_scaler=1000.0,
                    rate_source='vr_dot', targets=['vr'], units='DU/TU')
    burn2.add_state('vt', fix_initial=False, fix_final=True, defect_scaler=1000.0,
                    rate_source='vt_dot', targets=['vt'], units='DU/TU')
    burn2.add_state('accel', fix_initial=False, fix_final=False, defect_scaler=1.0,
                    rate_source='at_dot', targets=['accel'], units='DU/TU**2')
    burn2.add_state('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0,
                    rate_source='deltav_dot', units='DU/TU', targets=None)

    burn2.add_objective('deltav', loc='final', scaler=100.0)

    burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                      scaler=0.01, lower=-180, upper=180, targets=['u1'])

    if 'sequence_missing_phase' in param_mode:
        coast.add_parameter('c', val=0.0, units='DU/TU', targets=['c'])
    elif 'no_targets' in param_mode:
        burn1.add_parameter('c', val=0.0, units='DU/TU', targets=['c'])
        coast.add_parameter('c', val=0.0, units='DU/TU', targets=['c'])
        burn2.add_parameter('c', val=0.0, units='DU/TU', targets=['c'])

    burn1.add_timeseries_output('pos_x')
    coast.add_timeseries_output('pos_x')
    burn2.add_timeseries_output('pos_x')

    burn1.add_timeseries_output('pos_y')
    coast.add_timeseries_output('pos_y')
    burn2.add_timeseries_output('pos_y')

    # Link Phases
    if connected:
        traj.link_phases(phases=['burn1', 'coast'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'],
                         connected=True)

        # No direct connections to the end of a phase.
        traj.link_phases(phases=['burn2', 'coast'],
                         vars=['r', 'theta', 'vr', 'vt', 'deltav'],
                         locs=('final', 'final'))
        traj.link_phases(phases=['burn2', 'coast'],
                         vars=['time'], locs=('final', 'final'))

        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'],
                         locs=('final', 'final'))

    else:
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

        traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

    return traj


def two_burn_orbit_raise_problem(transcription='gauss-lobatto', optimizer='SLSQP', r_target=3.0,
                                 transcription_order=3, compressed=False, run_driver=True, simulate=True,
                                 show_output=True, connected=False, param_mode='param_sequence'):

    p = om.Problem(model=om.Group())

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    p.driver.declare_coloring()
    if optimizer == 'SNOPT':
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        if show_output:
            p.driver.opt_settings['iSumm'] = 6
    elif optimizer == 'IPOPT':
        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        p.driver.opt_settings['print_level'] = 0
        p.driver.opt_settings['linear_solver'] = 'mumps'
        p.driver.opt_settings['mu_strategy'] = 'adaptive'
        # p.driver.opt_settings['derivative_test'] = 'first-order'

    traj = make_traj(transcription=transcription, transcription_order=transcription_order,
                     compressed=compressed, connected=connected, param_mode=param_mode)
    p.model.add_subsystem('orbit_transfer', subsys=traj)

    # Finish Problem Setup

    # Needed to move the direct solver down into the phases for use with MPI.
    #  - After moving down, used fewer iterations (about 30 less)

    p.driver.add_recorder(om.SqliteRecorder('two_burn_orbit_raise_example.db'))

    p.setup(check=True)

    # Set Initial Guesses
    p.set_val('orbit_transfer.parameters:c', val=1.5, units='DU/TU')

    burn1 = p.model.orbit_transfer.phases.burn1
    burn2 = p.model.orbit_transfer.phases.burn2
    coast = p.model.orbit_transfer.phases.coast

    if burn1 in p.model.orbit_transfer.phases._subsystems_myproc:
        p.set_val('orbit_transfer.burn1.t_initial', val=0.0)
        p.set_val('orbit_transfer.burn1.t_duration', val=2.25)
        p.set_val('orbit_transfer.burn1.states:r', val=burn1.interp('r', [1, 1.5]))
        p.set_val('orbit_transfer.burn1.states:theta', val=burn1.interp('theta', [0, 1.7]))
        p.set_val('orbit_transfer.burn1.states:vr', val=burn1.interp('vr', [0, 0]))
        p.set_val('orbit_transfer.burn1.states:vt', val=burn1.interp('vt', [1, 1]))
        p.set_val('orbit_transfer.burn1.states:accel', val=burn1.interp('accel', [0.1, 0]))
        p.set_val('orbit_transfer.burn1.states:deltav', val=burn1.interp('deltav', [0, 0.1]))
        p.set_val('orbit_transfer.burn1.controls:u1', val=burn1.interp('u1', [-3.5, 13.0]))

    if coast in p.model.orbit_transfer.phases._subsystems_myproc:
        p.set_val('orbit_transfer.coast.t_initial', val=2.25)
        p.set_val('orbit_transfer.coast.t_duration', val=3.0)

        p.set_val('orbit_transfer.coast.states:r', val=coast.interp('r', [1.3, 1.5]))
        p.set_val('orbit_transfer.coast.states:theta', val=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('orbit_transfer.coast.states:vr', val=coast.interp('vr', [0.3285, 0]))
        p.set_val('orbit_transfer.coast.states:vt', val=coast.interp('vt', [0.97, 1]))
        p.set_val('orbit_transfer.coast.states:accel', val=coast.interp('accel', [0, 0]))

    if burn2 in p.model.orbit_transfer.phases._subsystems_myproc:
        p.set_val('orbit_transfer.burn2.t_initial', val=5.25)
        p.set_val('orbit_transfer.burn2.t_duration', val=1.75)

        p.set_val('orbit_transfer.burn2.states:r', val=burn2.interp('r', [1, r_target]))
        p.set_val('orbit_transfer.burn2.states:theta', val=burn2.interp('theta', [0, 4.0]))
        p.set_val('orbit_transfer.burn2.states:vr', val=burn2.interp('vr', [0, 0]))
        p.set_val('orbit_transfer.burn2.states:vt', val=burn2.interp('vt', [1, np.sqrt(1 / r_target)]))
        p.set_val('orbit_transfer.burn2.states:deltav', val=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('orbit_transfer.burn2.states:accel', val=burn2.interp('accel', [0.1, 0]))

        p.set_val('orbit_transfer.burn2.controls:u1', val=burn2.interp('u1', [0, 0]))

    dm.run_problem(p, run_driver=run_driver, simulate=simulate)

    return p


@use_tempdirs
class TestTrajectoryParameters(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    def test_param_explicit_connections_to_sequence(self):
        """
        Test that, when setting up a trajectory parameter, we can explicitly provide a sequence
        in each phase as targets and a corresponding parameter for the phase will
        automatically be added.
        """
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer='IPOPT', run_driver=False,
                                         simulate=False, show_output=False, param_mode='param_sequence')

        traj_c = p.get_val('orbit_transfer.parameter_vals:c')
        burn1_c = p.get_val('orbit_transfer.burn1.parameter_vals:c')
        coast_c = p.get_val('orbit_transfer.coast.parameter_vals:c')
        burn2_c = p.get_val('orbit_transfer.burn2.parameter_vals:c')

        assert_near_equal(burn1_c, traj_c)
        assert_near_equal(coast_c, traj_c)
        assert_near_equal(burn2_c, traj_c)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_param_explicit_connections_to_sequence_missing_phase(self):
        """
        Test that, when setting up a trajectory parameter with a phase omitted from input,
        that we attempt to connect to an existing input variable in that phase of the same name.
        """
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer='IPOPT', run_driver=False, simulate=False,
                                         show_output=False,
                                         param_mode='param_sequence_missing_phase')

        traj_c = p.get_val('orbit_transfer.parameter_vals:c')
        burn1_c = p.get_val('orbit_transfer.burn1.parameter_vals:c')
        coast_c = p.get_val('orbit_transfer.coast.parameter_vals:c')
        burn2_c = p.get_val('orbit_transfer.burn2.parameter_vals:c')

        assert_near_equal(burn1_c, traj_c)
        assert_near_equal(coast_c, traj_c)
        assert_near_equal(burn2_c, traj_c)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_param_no_targets(self):
        """
        Test that, when setting up a trajectory parameter with a phase omitted from input,
        that we attempt to connect to an existing input variable in that phase of the same name.
        """
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer='IPOPT', run_driver=False,
                                         show_output=False, simulate=False,
                                         param_mode='param_no_targets')

        traj_c = p.get_val('orbit_transfer.parameter_vals:c')
        burn1_c = p.get_val('orbit_transfer.burn1.parameter_vals:c')
        coast_c = p.get_val('orbit_transfer.coast.parameter_vals:c')
        burn2_c = p.get_val('orbit_transfer.burn2.parameter_vals:c')

        assert_near_equal(burn1_c, traj_c)
        assert_near_equal(coast_c, traj_c)
        assert_near_equal(burn2_c, traj_c)


@use_tempdirs
class TestPhaseParameterPromotion(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_promotes_parameter(self):

        for transcription in ['radau-ps', 'gauss-lobatto']:
            with self.subTest(msg=transcription):
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
class TestTrajectoryParametersDefinition(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_parameter_shape_definition(self):

        for transcription in ['radau-ps', 'gauss-lobatto']:
            with self.subTest(msg=transcription):
                optimizer = 'SLSQP'
                num_segments = 10
                transcription_order = 3
                compressed = False

                p = om.Problem(model=om.Group())

                p.driver = om.pyOptSparseDriver(print_results=False)
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

                traj = p.model.add_subsystem('traj', dm.Trajectory())
                phase = traj.add_phase('phase0',
                                       dm.Phase(ode_class=ODEGroup,
                                                transcription=t),
                                       promotes_inputs=['t_initial', 't_duration', 'parameters:g'])

                # p.model.add_subsystem('p1', ParamComp(), promotes=['*'])

                # Add a vector parameter.
                traj.add_parameter('falling_object:mass',
                                   opt=False,
                                   units='kg',
                                   shape=(3,),
                                   static_target=True,
                                   targets={'phase0': ['falling_object:mass']})

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
                phase.set_control_val('theta', [5, 100])
                phase.set_parameter_val('g', 9.80665)

                # Solve for the optimal trajectory
                p.run_driver()

                # Test the results
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
