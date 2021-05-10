import importlib
import os
import shutil
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.general_utils import set_pyoptsparse_opt
_, optimizer = set_pyoptsparse_opt('IPOPT', fallback=True)

import dymos as dm
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE
from openmdao.utils.testing_utils import use_tempdirs

bokeh_available = importlib.util.find_spec('bokeh') is not None


def make_traj(transcription='gauss-lobatto', transcription_order=3, compressed=False):

    t = {'gauss-lobatto': dm.GaussLobatto(num_segments=5, order=transcription_order, compressed=compressed),
         'radau': dm.Radau(num_segments=20, order=transcription_order, compressed=compressed)}

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

    traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                     vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav'])

    traj.link_phases(phases=['burn1', 'burn2'], vars=['accel'])

    return traj


def two_burn_orbit_raise_problem(transcription='gauss-lobatto', optimizer='SLSQP', r_target=3.0,
                                 transcription_order=3, compressed=False,
                                 show_output=True):

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
                     compressed=compressed)
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

        p.set_val('traj.burn1.states:r', value=burn1.interp('r', [1, 1.5]))
        p.set_val('traj.burn1.states:theta', value=burn1.interp('theta', [0, 1.7]))
        p.set_val('traj.burn1.states:vr', value=burn1.interp('vr', [0, 0]))
        p.set_val('traj.burn1.states:vt', value=burn1.interp('vt', [1, 1]))
        p.set_val('traj.burn1.states:accel', value=burn1.interp('accel', [0.1, 0]))
        p.set_val('traj.burn1.states:deltav', value=burn1.interp('deltav', [0, 0.1]))
        p.set_val('traj.burn1.controls:u1', value=burn1.interp('u1', [-3.5, 13.0]))

    if coast in p.model.traj.phases._subsystems_myproc:
        p.set_val('traj.coast.t_initial', value=2.25)
        p.set_val('traj.coast.t_duration', value=3.0)

        p.set_val('traj.coast.states:r', value=coast.interp('r', [1.3, 1.5]))
        p.set_val('traj.coast.states:theta', value=coast.interp('theta', [2.1767, 1.7]))
        p.set_val('traj.coast.states:vr', value=coast.interp('vr', [0.3285, 0]))
        p.set_val('traj.coast.states:vt', value=coast.interp('vt', [0.97, 1]))
        p.set_val('traj.coast.states:accel', value=coast.interp('accel', [0, 0]))

    if burn2 in p.model.traj.phases._subsystems_myproc:
        p.set_val('traj.burn2.t_initial', value=5.25)
        p.set_val('traj.burn2.t_duration', value=1.75)

        p.set_val('traj.burn2.states:r', value=burn2.interp('r', [1, 3]))
        p.set_val('traj.burn2.states:theta', value=burn2.interp('theta', [0, 4.0]))
        p.set_val('traj.burn2.states:vr', value=burn2.interp('vr', [0, 0]))
        p.set_val('traj.burn2.states:vt', value=burn2.interp('vt', [1, np.sqrt(1 / 3)]))
        p.set_val('traj.burn2.states:accel', value=burn2.interp('accel', [0.1, 0]))
        p.set_val('traj.burn2.states:deltav', value=burn2.interp('deltav', [0.1, 0.2]))
        p.set_val('traj.burn2.controls:u1', value=burn2.interp('u1', [1, 1]))

    dm.run_problem(p, run_driver=False, simulate=True, make_plots=True)

    return p


@use_tempdirs
class TestExampleTwoBurnOrbitRaise(unittest.TestCase):

    def tearDown(self):
        if os.path.isdir('plots'):
            shutil.rmtree('plots')

    @unittest.skipIf(not bokeh_available, 'bokeh unavailable')
    def test_bokeh_plots(self):
        dm.options['plots'] = 'bokeh'

        two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                     compressed=False, optimizer='SLSQP', show_output=False)

        self.assertSetEqual({'plots.html'}, set(os.listdir('plots')))

    def test_mpl_plots(self):
        dm.options['plots'] = 'matplotlib'

        two_burn_orbit_raise_problem(transcription='gauss-lobatto', transcription_order=3,
                                     compressed=False, optimizer='SLSQP', show_output=False)

        expected_files = {'control_rates_u1_rate.png', 'state_rates_r.png', 'states_deltav.png',
                          'states_r.png', 'state_rates_accel.png', 'state_rates_deltav.png',
                          'states_accel.png', 'controls_u1.png', 'states_vr.png', 'pos_x.png',
                          'states_vt.png', 'pos_y.png', 'parameters_u1.png', 'states_theta.png',
                          'control_rates_u1_rate2.png', 'state_rates_vt.png', 'time_phase.png',
                          'parameters_c.png', 'state_rates_theta.png', 'state_rates_vr.png'}

        self.assertSetEqual(expected_files, set(os.listdir('plots')))


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
