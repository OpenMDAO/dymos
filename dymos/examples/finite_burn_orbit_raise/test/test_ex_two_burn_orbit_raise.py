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
              connected=False):

    t = {'gauss-lobatto': dm.GaussLobatto(num_segments=5, order=transcription_order, compressed=compressed),
         'radau': dm.Radau(num_segments=20, order=transcription_order, compressed=compressed),
         'runge-kutta': dm.RungeKutta(num_segments=5, compressed=compressed)}

    traj = dm.Trajectory()

    traj.add_parameter('c', opt=False, val=1.5, units='DU/TU',
                       targets={'burn1': ['c'], 'burn2': ['c']})

    # First Phase (burn)

    burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    burn1 = traj.add_phase('burn1', burn1)

    burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
    burn1.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                    rate_source='r_dot', units='DU')
    burn1.add_state('theta', fix_initial=True, fix_final=False, defect_scaler=100.0,
                    rate_source='theta_dot', units='rad')
    burn1.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=100.0,
                    rate_source='vr_dot', units='DU/TU')
    burn1.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=100.0,
                    rate_source='vt_dot', units='DU/TU')
    burn1.add_state('accel', fix_initial=True, fix_final=False,
                    rate_source='at_dot', units='DU/TU**2')
    burn1.add_state('deltav', fix_initial=True, fix_final=False,
                    rate_source='deltav_dot', units='DU/TU')
    burn1.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg', scaler=0.01,
                      rate_continuity_scaler=0.001, rate2_continuity_scaler=0.001,
                      lower=-30, upper=30)
    # Second Phase (Coast)
    coast = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 50), duration_ref=50, units='TU')
    coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='r_dot', units='DU')
    coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='theta_dot', units='rad')
    coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='vr_dot', units='DU/TU')
    coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                    rate_source='vt_dot', units='DU/TU')
    coast.add_state('accel', fix_initial=True, fix_final=True,
                    rate_source='at_dot', units='DU/TU**2')
    coast.add_state('deltav', fix_initial=False, fix_final=False,
                    rate_source='deltav_dot', units='DU/TU')

    coast.add_parameter('u1', opt=False, val=0.0, units='deg')
    coast.add_parameter('c', val=0.0, units='DU/TU')

    # Third Phase (burn)
    burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=t[transcription])

    if connected:
        traj.add_phase('burn2', burn2)
        traj.add_phase('coast', coast)

        burn2.set_time_options(initial_bounds=(1.0, 60), duration_bounds=(-10.0, -0.5),
                               initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=True, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=True, fix_final=False, defect_scaler=1000.0,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=True, fix_final=False, defect_scaler=1000.0,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_scaler=1.0,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False, defect_scaler=1.0,
                        rate_source='deltav_dot', units='DU/TU')

        burn2.add_objective('deltav', loc='initial', scaler=100.0)

        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01, lower=-180, upper=180)
    else:
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
                        rate_source='deltav_dot', units='DU/TU')

        burn2.add_objective('deltav', loc='final', scaler=100.0)

        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True, units='deg',
                          scaler=0.01, lower=-180, upper=180, targets=['u1'])

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
                                 show_output=True, connected=False):

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
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['linear_solver'] = 'mumps'
        p.driver.opt_settings['mu_strategy'] = 'adaptive'
        # p.driver.opt_settings['derivative_test'] = 'first-order'

    traj = make_traj(transcription=transcription, transcription_order=transcription_order,
                     compressed=compressed, connected=connected)
    p.model.add_subsystem('traj', subsys=traj)

    # Needed to move the direct solver down into the phases for use with MPI.
    #  - After moving down, used fewer iterations (about 30 less)

    p.setup(check=True)

    # Set Initial Guesses
    p.set_val('traj.parameters:c', value=1.5, units='DU/TU')

    burn1 = p.model.traj.phases.burn1
    burn2 = p.model.traj.phases.burn2
    coast = p.model.traj.phases.coast

    if burn1 in p.model.traj.phases._subsystems_myproc:
        p.set_val('traj.burn1.t_initial', value=0.0)
        p.set_val('traj.burn1.t_duration', value=2.25)
        p.set_val('traj.burn1.states:r', value=burn1.interpolate(ys=[1, 1.5],
                                                                 nodes='state_input'))
        p.set_val('traj.burn1.states:theta', value=burn1.interpolate(ys=[0, 1.7],
                  nodes='state_input'))
        p.set_val('traj.burn1.states:vr', value=burn1.interpolate(ys=[0, 0],
                                                                  nodes='state_input'))
        p.set_val('traj.burn1.states:vt', value=burn1.interpolate(ys=[1, 1],
                                                                  nodes='state_input'))
        p.set_val('traj.burn1.states:accel', value=burn1.interpolate(ys=[0.1, 0],
                  nodes='state_input'))
        p.set_val('traj.burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1],
                  nodes='state_input'))
        p.set_val('traj.burn1.controls:u1',
                  value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))

    if coast in p.model.traj.phases._subsystems_myproc:
        p.set_val('traj.coast.t_initial', value=2.25)
        p.set_val('traj.coast.t_duration', value=3.0)

        p.set_val('traj.coast.states:r', value=coast.interpolate(ys=[1.3, 1.5],
                  nodes='state_input'))
        p.set_val('traj.coast.states:theta',
                  value=coast.interpolate(ys=[2.1767, 1.7], nodes='state_input'))

        p.set_val('traj.coast.states:vr', value=coast.interpolate(ys=[0.3285, 0],
                  nodes='state_input'))
        p.set_val('traj.coast.states:vt', value=coast.interpolate(ys=[0.97, 1],
                  nodes='state_input'))
        p.set_val('traj.coast.states:accel', value=coast.interpolate(ys=[0, 0],
                  nodes='state_input'))

    if burn2 in p.model.traj.phases._subsystems_myproc:
        if connected:
            p.set_val('traj.burn2.t_initial', value=7.0)
            p.set_val('traj.burn2.t_duration', value=-1.75)

            p.set_val('traj.burn2.states:r', value=burn2.interpolate(ys=[r_target, 1],
                      nodes='state_input'))
            p.set_val('traj.burn2.states:theta', value=burn2.interpolate(ys=[4.0, 0.0],
                      nodes='state_input'))
            p.set_val('traj.burn2.states:vr', value=burn2.interpolate(ys=[0, 0],
                                                                      nodes='state_input'))
            p.set_val('traj.burn2.states:vt',
                      value=burn2.interpolate(ys=[np.sqrt(1 / r_target), 1],
                                              nodes='state_input'))
            p.set_val('traj.burn2.states:deltav',
                      value=burn2.interpolate(ys=[0.2, 0.1], nodes='state_input'))
            p.set_val('traj.burn2.states:accel', value=burn2.interpolate(ys=[0., 0.1],
                      nodes='state_input'))

        else:
            p.set_val('traj.burn2.t_initial', value=5.25)
            p.set_val('traj.burn2.t_duration', value=1.75)

            p.set_val('traj.burn2.states:r', value=burn2.interpolate(ys=[1, r_target],
                      nodes='state_input'))
            p.set_val('traj.burn2.states:theta', value=burn2.interpolate(ys=[0, 4.0],
                      nodes='state_input'))
            p.set_val('traj.burn2.states:vr', value=burn2.interpolate(ys=[0, 0],
                                                                      nodes='state_input'))
            p.set_val('traj.burn2.states:vt',
                      value=burn2.interpolate(ys=[1, np.sqrt(1 / r_target)],
                                              nodes='state_input'))
            p.set_val('traj.burn2.states:deltav',
                      value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
            p.set_val('traj.burn2.states:accel', value=burn2.interpolate(ys=[0.1, 0],
                      nodes='state_input'))

        p.set_val('traj.burn2.controls:u1', value=burn2.interpolate(ys=[0, 0],
                  nodes='control_input'))

    p.run_driver()

    return p


@use_tempdirs
class TestExampleTwoBurnOrbitRaise(unittest.TestCase):

    @unittest.skipIf(optimizer is not 'IPOPT', 'IPOPT not available')
    def test_ex_two_burn_orbit_raise(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[-1], 0.3995,
                              tolerance=2.0E-3)


# This test is separate because connected phases aren't directly parallelizable.
@use_tempdirs
class TestExampleTwoBurnOrbitRaiseConnected(unittest.TestCase):

    @unittest.skipIf(optimizer is not 'IPOPT', 'IPOPT not available')
    def test_ex_two_burn_orbit_raise_connected(self):
        optimizer = 'IPOPT'

        p = two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                         compressed=False, optimizer=optimizer,
                                         show_output=False, connected=True)

        if p.model.traj.phases.burn2 in p.model.traj.phases._subsystems_myproc:
            assert_near_equal(p.get_val('traj.burn2.states:deltav')[0], 0.3995,
                              tolerance=4.0E-3)


class TestExampleTwoBurnOrbitRaiseMPI(TestExampleTwoBurnOrbitRaise):
    N_PROCS = 3


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
