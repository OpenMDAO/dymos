import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.general_utils import set_pyoptsparse_opt
_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE
from openmdao.utils.testing_utils import use_tempdirs


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
                                 transcription_order=3, compressed=False,
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
        p.driver.opt_settings['print_level'] = 4
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
    p.set_val('orbit_transfer.parameters:c', value=1.5, units='DU/TU')

    burn1 = p.model.orbit_transfer.phases.burn1
    burn2 = p.model.orbit_transfer.phases.burn2
    coast = p.model.orbit_transfer.phases.coast

    if burn1 in p.model.orbit_transfer.phases._subsystems_myproc:
        p.set_val('orbit_transfer.burn1.t_initial', value=0.0)
        p.set_val('orbit_transfer.burn1.t_duration', value=2.25)
        p.set_val('orbit_transfer.burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('orbit_transfer.burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('orbit_transfer.burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('orbit_transfer.burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('orbit_transfer.burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('orbit_transfer.burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('orbit_transfer.burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))

    if coast in p.model.orbit_transfer.phases._subsystems_myproc:
        p.set_val('orbit_transfer.coast.t_initial', value=2.25)
        p.set_val('orbit_transfer.coast.t_duration', value=3.0)

        p.set_val('orbit_transfer.coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('orbit_transfer.coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('orbit_transfer.coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('orbit_transfer.coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('orbit_transfer.coast.states:accel', value=coast.interp('accel', [0, 0]))

    if burn2 in p.model.orbit_transfer.phases._subsystems_myproc:
        p.set_val('orbit_transfer.burn2.t_initial', value=5.25)
        p.set_val('orbit_transfer.burn2.t_duration', value=1.75)

        p.set_val('orbit_transfer.burn2.states:r', value=burn2.interp('r', [1, r_target]))
        p.set_val('orbit_transfer.burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('orbit_transfer.burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('orbit_transfer.burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / r_target)]))
        p.set_val('orbit_transfer.burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('orbit_transfer.burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))

        p.set_val('orbit_transfer.burn2.controls:u1', value=burn2.interp('u1', [0, 0]))

    dm.run_problem(p, simulate=True)

    return p


@use_tempdirs
class TestTrajectoryParameters(unittest.TestCase):

    def test_param_explicit_connections_to_sequence(self):
        """
        Test that, when setting up a trajectory parameter, we can explicitly provide a sequence
        in each phase as targets and a corresponding parameter for the phase will
        automatically be added.
        """
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, param_mode='param_sequence')

        if p.model.orbit_transfer.phases.burn2 in p.model.orbit_transfer.phases._subsystems_myproc:
            assert_near_equal(p.get_val('orbit_transfer.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)

    def test_param_explicit_connections_to_sequence_missing_phase(self):
        """
        Test that, when setting up a trajectory parameter with a phase omitted from input,
        that we attempt to connect to an existing input variable in that phase of the same name.
        """
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False,
                                         param_mode='param_sequence_missing_phase')

        if p.model.orbit_transfer.phases.burn2 in p.model.orbit_transfer.phases._subsystems_myproc:
            assert_near_equal(p.get_val('orbit_transfer.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)

    def test_param_no_targets(self):
        """
        Test that, when setting up a trajectory parameter with a phase omitted from input,
        that we attempt to connect to an existing input variable in that phase of the same name.
        """
        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False,
                                         param_mode='param_no_targets')

        if p.model.orbit_transfer.phases.burn2 in p.model.orbit_transfer.phases._subsystems_myproc:
            assert_near_equal(p.get_val('orbit_transfer.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
