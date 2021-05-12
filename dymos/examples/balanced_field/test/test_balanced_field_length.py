import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.balanced_field.balanced_field_ode import BalancedFieldODEComp
from dymos.utils.testing_utils import require_pyoptsparse


@use_tempdirs
class TestBalancedFieldLengthRestart(unittest.TestCase):

    def _make_problem(self):
        p = om.Problem()

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['derivative_test'] = 'first-order'

        p.driver.declare_coloring()

        # First Phase: Brake release to V1 - both engines operable
        br_to_v1 = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                            ode_init_kwargs={'mode': 'runway'})
        br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0)
        br_to_v1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0)
        br_to_v1.add_state('v', fix_initial=True, lower=0.0001, ref=100.0, defect_ref=100.0)
        br_to_v1.add_parameter('alpha', val=0.0, opt=False, units='deg')
        br_to_v1.add_timeseries_output('*')

        # Second Phase: Rejected takeoff at V1 - no engines operable
        rto = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                       ode_init_kwargs={'mode': 'runway'})
        rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        rto.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rto.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        rto.add_parameter('alpha', val=0.0, opt=False, units='deg')
        rto.add_timeseries_output('*')

        # Third Phase: V1 to Vr - single engine operable
        v1_to_vr = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                            ode_init_kwargs={'mode': 'runway'})
        v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        v1_to_vr.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        v1_to_vr.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        v1_to_vr.add_parameter('alpha', val=0.0, opt=False, units='deg')
        v1_to_vr.add_timeseries_output('*')

        # Fourth Phase: Rotate - single engine operable
        rotate = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                          ode_init_kwargs={'mode': 'runway'})
        rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0)
        rotate.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rotate.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        rotate.add_polynomial_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10, val=[0, 10])
        rotate.add_timeseries_output('*')

        # Fifth Phase: Climb to target speed and altitude at end of runway.
        climb = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=5),
                         ode_init_kwargs={'mode': 'climb'})
        climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0)
        climb.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        climb.add_state('h', fix_initial=True, lower=0.0, ref=1.0, defect_ref=1.0)
        climb.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0)
        climb.add_state('gam', fix_initial=True, lower=0.0, ref=0.05, defect_ref=0.05)
        climb.add_control('alpha', opt=True, units='deg', lower=-10, upper=15, ref=10)
        climb.add_timeseries_output('*')

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_v1', br_to_v1)
        traj.add_phase('rto', rto)
        traj.add_phase('v1_to_vr', v1_to_vr)
        traj.add_phase('rotate', rotate)
        traj.add_phase('climb', climb)

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={'br_to_v1': ['m'], 'v1_to_vr': ['m'], 'rto': ['m'],
                                    'rotate': ['m'], 'climb': ['m']})

        traj.add_parameter('T_nominal', val=27000 * 2, opt=False, units='lbf', static_target=True,
                           desc='nominal aircraft thrust',
                           targets={'br_to_v1': ['T']})

        traj.add_parameter('T_engine_out', val=27000, opt=False, units='lbf', static_target=True,
                           desc='thrust under a single engine',
                           targets={'v1_to_vr': ['T'], 'rotate': ['T'], 'climb': ['T']})

        traj.add_parameter('T_shutdown', val=0.0, opt=False, units='lbf', static_target=True,
                           desc='thrust when engines are shut down for rejected takeoff',
                           targets={'rto': ['T']})

        traj.add_parameter('mu_r_nominal', val=0.03, opt=False, units=None, static_target=True,
                           desc='nominal runway friction coeffcient',
                           targets={'br_to_v1': ['mu_r'], 'v1_to_vr': ['mu_r'],  'rotate': ['mu_r']})

        traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, static_target=True,
                           desc='runway friction coefficient under braking',
                           targets={'rto': ['mu_r']})

        traj.add_parameter('h_runway', val=0., opt=False, units='ft', static_target=False,
                           desc='runway altitude',
                           targets={'br_to_v1': ['h'], 'v1_to_vr': ['h'], 'rto': ['h'],
                                    'rotate': ['h']})

        traj.add_parameter('rho', val=1.225, opt=False, units='kg/m**3', static_target=True,
                           desc='atmospheric density',
                           targets={'br_to_v1': ['rho'], 'v1_to_vr': ['rho'], 'rto': ['rho'],
                                    'rotate': ['rho']})

        traj.add_parameter('S', val=124.7, opt=False, units='m**2', static_target=True,
                           desc='aerodynamic reference area',
                           targets={'br_to_v1': ['S'], 'v1_to_vr': ['S'], 'rto': ['S'],
                                    'rotate': ['S'], 'climb': ['S']})

        traj.add_parameter('CD0', val=0.03, opt=False, units=None, static_target=True,
                           desc='zero-lift drag coefficient',
                           targets={f'{phase}': ['CD0'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate' 'climb']})

        traj.add_parameter('AR', val=9.45, opt=False, units=None, static_target=True,
                           desc='wing aspect ratio',
                           targets={f'{phase}': ['AR'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                     'rto', 'rotate' 'climb']})

        traj.add_parameter('e', val=801, opt=False, units=None, static_target=True,
                           desc='Oswald span efficiency factor',
                           targets={f'{phase}': ['e'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                    'rto', 'rotate' 'climb']})

        traj.add_parameter('span', val=35.7, opt=False, units='m', static_target=True,
                           desc='wingspan',
                           targets={f'{phase}': ['span'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                       'rto', 'rotate' 'climb']})

        traj.add_parameter('h_w', val=1.0, opt=False, units='m', static_target=True,
                           desc='height of wing above CG',
                           targets={f'{phase}': ['h_w'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate' 'climb']})

        traj.add_parameter('CL0', val=0.5, opt=False, units=None, static_target=True,
                           desc='zero-alpha lift coefficient',
                           targets={f'{phase}': ['CL0'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate' 'climb']})

        traj.add_parameter('CL_max', val=2.0, opt=False, units=None, static_target=True,
                           desc='maximum lift coefficient for linear fit',
                           targets={f'{phase}': ['CL_max'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                         'rto', 'rotate' 'climb']})

        # Standard "end of first phase to beginning of second phase" linkages
        traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])

        # Less common "final value of r must be the match at ends of two phases".
        traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
                                    phase_b='climb', var_b='r', loc_b='final',
                                    ref=1000)

        # Define the constraints and objective for the optimal control problem
        rto.add_boundary_constraint('v', loc='final', upper=0.001, ref=100, linear=True)

        rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

        climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
        climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
        climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
        climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=1.2)

        rto.add_objective('r', loc='final', ref=1000.0)

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p.set_val('traj.br_to_v1.t_initial', 0)
        p.set_val('traj.br_to_v1.t_duration', 35)
        p.set_val('traj.br_to_v1.states:r', br_to_v1.interp('r', [0, 2500.0]))
        p.set_val('traj.br_to_v1.states:v', br_to_v1.interp('v', [0.0001, 100.0]))
        p.set_val('traj.br_to_v1.parameters:alpha', 0, units='deg')

        p.set_val('traj.v1_to_vr.t_initial', 35)
        p.set_val('traj.v1_to_vr.t_duration', 35)
        p.set_val('traj.v1_to_vr.states:r', v1_to_vr.interp('r', [2500, 300.0]))
        p.set_val('traj.v1_to_vr.states:v', v1_to_vr.interp('v', [100, 110.0]))
        p.set_val('traj.v1_to_vr.parameters:alpha', 0.0, units='deg')

        p.set_val('traj.rto.t_initial', 35)
        p.set_val('traj.rto.t_duration', 1)
        p.set_val('traj.rto.states:r', rto.interp('r', [2500, 5000.0]))
        p.set_val('traj.rto.states:v', rto.interp('v', [110, 0.0001]))
        p.set_val('traj.rto.parameters:alpha', 0.0, units='deg')

        p.set_val('traj.rotate.t_initial', 35)
        p.set_val('traj.rotate.t_duration', 5)
        p.set_val('traj.rotate.states:r', rotate.interp('r', [1750, 1800.0]))
        p.set_val('traj.rotate.states:v', rotate.interp('v', [80, 85.0]))
        p.set_val('traj.rotate.polynomial_controls:alpha', 0.0, units='deg')

        p.set_val('traj.climb.t_initial', 30)
        p.set_val('traj.climb.t_duration', 20)
        p.set_val('traj.climb.states:r', climb.interp('r', [5000, 5500.0]), units='ft')
        p.set_val('traj.climb.states:v', climb.interp('v', [160, 170.0]), units='kn')
        p.set_val('traj.climb.states:h', climb.interp('h', [0, 35.0]), units='ft')
        p.set_val('traj.climb.states:gam', climb.interp('gam', [0, 5.0]), units='deg')
        p.set_val('traj.climb.controls:alpha', 5.0, units='deg')

        return p

    @require_pyoptsparse(optimizer='IPOPT')
    def test_restart_from_sol(self):
        p = self._make_problem()
        dm.run_problem(p, run_driver=True, simulate=False)

        sol_results = om.CaseReader('dymos_solution.db').get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.states:r')[-1], 2016, tolerance=0.01)

        dm.run_problem(p, run_driver=True, simulate=True, restart='dymos_solution.db')

        sol_results = om.CaseReader('dymos_solution.db').get_case('final')
        sim_results = om.CaseReader('dymos_simulation.db').get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.states:r')[-1], 2016, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.climb.timeseries.states:r')[-1], 2016, tolerance=0.01)

        assert_near_equal(sol_results.get_val('traj.rto.timeseries.states:r')[-1], 2016, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.rto.timeseries.states:r')[-1], 2016, tolerance=0.01)

    @require_pyoptsparse(optimizer='IPOPT')
    def test_restart_from_sim(self):
        p = self._make_problem()
        dm.run_problem(p, run_driver=True, simulate=True)

        sol_results = om.CaseReader('dymos_solution.db').get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.states:r')[-1], 2016,
                          tolerance=0.01)

        dm.run_problem(p, run_driver=True, simulate=True, restart='dymos_simulation.db')

        sol_results = om.CaseReader('dymos_solution.db').get_case('final')
        sim_results = om.CaseReader('dymos_simulation.db').get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.states:r')[-1], 2016,
                          tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.climb.timeseries.states:r')[-1], 2016,
                          tolerance=0.01)

        assert_near_equal(sol_results.get_val('traj.rto.timeseries.states:r')[-1], 2016,
                          tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.rto.timeseries.states:r')[-1], 2016,
                          tolerance=0.01)


@use_tempdirs
class TestBalancedFieldLengthDefaultValues(unittest.TestCase):

    def test_default_vals_stick(self):
        """
        Make the balanced field problem without any set_val calls after setup.
        """
        p = om.Problem()

        # First Phase: Brake release to V1 - both engines operable
        br_to_v1 = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                            ode_init_kwargs={'mode': 'runway'})
        br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0,
                                  initial_val=0.0, duration_val=35.0)
        br_to_v1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0,
                           val=br_to_v1.interp(ys=[0, 2500.0], nodes='state_input'))
        br_to_v1.add_state('v', fix_initial=True, lower=0.0001, ref=100.0, defect_ref=100.0,
                           val=br_to_v1.interp(ys=[0.0001, 100.0], nodes='state_input'))
        br_to_v1.add_parameter('alpha', val=0.0, opt=False, units='deg')
        br_to_v1.add_timeseries_output('*')

        # Second Phase: Rejected takeoff at V1 - no engines operable
        rto = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                       ode_init_kwargs={'mode': 'runway'})
        rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0,
                             initial_val=35.0, duration_val=35.0)
        rto.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
                      val=rto.interp(ys=[2500, 5000.0], nodes='state_input'))
        rto.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0,
                      val=rto.interp(ys=[110, 0.0001], nodes='state_input'))
        rto.add_parameter('alpha', val=0.0, opt=False, units='deg')
        rto.add_timeseries_output('*')

        # Third Phase: V1 to Vr - single engine operable
        v1_to_vr = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                            ode_init_kwargs={'mode': 'runway'})
        v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0,
                                  initial_val=35.0, duration_val=35.0)
        v1_to_vr.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
                           val=v1_to_vr.interp(ys=[2500, 300.0], nodes='state_input'))
        v1_to_vr.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0,
                           val=v1_to_vr.interp(ys=[100, 110.0], nodes='state_input'))
        v1_to_vr.add_parameter('alpha', val=0.0, opt=False, units='deg')
        v1_to_vr.add_timeseries_output('*')

        # Fourth Phase: Rotate - single engine operable
        rotate = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=3),
                          ode_init_kwargs={'mode': 'runway'})
        rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0,
                                initial_val=70.0, duration_val=5.0)
        rotate.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
                         val=rotate.interp(ys=[1750, 1800.0], nodes='state_input'))
        rotate.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0,
                         val=rotate.interp(ys=[80, 85.0], nodes='state_input'))
        rotate.add_polynomial_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10, val=[0, 10])
        rotate.add_timeseries_output('*')

        # Fifth Phase: Climb to target speed and altitude at end of runway.
        climb = dm.Phase(ode_class=BalancedFieldODEComp, transcription=dm.Radau(num_segments=5),
                         ode_init_kwargs={'mode': 'climb'})
        climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0,
                               initial_val=75.0, duration_val=10.0)
        climb.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0,
                        val=climb.interp(ys=[1800, 2500.0], nodes='state_input'))
        climb.add_state('h', fix_initial=True, lower=0.0, ref=1.0, defect_ref=1.0, val=0.0)
        climb.add_state('v', fix_initial=False, lower=0.0001, ref=100.0, defect_ref=100.0,
                        val=climb.interp(ys=[85, 90], nodes='state_input'))
        climb.add_state('gam', fix_initial=True, lower=0.0, ref=0.05, defect_ref=0.05,
                        val=climb.interp(ys=[0, 0.05], nodes='state_input'))
        climb.add_control('alpha', opt=True, units='deg', lower=-10, upper=15, ref=10,
                          val=climb.interp(ys=[0.01, 0.01], nodes='control_input'))
        climb.add_timeseries_output('*')

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_v1', br_to_v1)
        traj.add_phase('rto', rto)
        traj.add_phase('v1_to_vr', v1_to_vr)
        traj.add_phase('rotate', rotate)
        traj.add_phase('climb', climb)

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={'br_to_v1': ['m'], 'v1_to_vr': ['m'], 'rto': ['m'],
                                    'rotate': ['m'], 'climb': ['m']})

        traj.add_parameter('T_nominal', val=27000 * 2, opt=False, units='lbf', static_target=True,
                           desc='nominal aircraft thrust', targets={'br_to_v1': ['T']})

        traj.add_parameter('T_engine_out', val=27000, opt=False, units='lbf', static_target=True,
                           desc='thrust under a single engine',
                           targets={'v1_to_vr': ['T'], 'rotate': ['T'], 'climb': ['T']})

        traj.add_parameter('T_shutdown', val=0.0, opt=False, units='lbf', static_target=True,
                           desc='thrust when engines are shut down for rejected takeoff',
                           targets={'rto': ['T']})

        traj.add_parameter('mu_r_nominal', val=0.03, opt=False, units=None, static_target=True,
                           desc='nominal runway friction coeffcient',
                           targets={'br_to_v1': ['mu_r'], 'v1_to_vr': ['mu_r'],  'rotate': ['mu_r']})

        traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, static_target=True,
                           desc='runway friction coefficient under braking',
                           targets={'rto': ['mu_r']})

        traj.add_parameter('h_runway', val=0., opt=False, units='ft', static_target=False,
                           desc='runway altitude',
                           targets={'br_to_v1': ['h'], 'v1_to_vr': ['h'], 'rto': ['h'],
                                    'rotate': ['h']})

        traj.add_parameter('rho', val=1.225, opt=False, units='kg/m**3', static_target=True,
                           desc='atmospheric density',
                           targets={'br_to_v1': ['rho'], 'v1_to_vr': ['rho'], 'rto': ['rho'],
                                    'rotate': ['rho']})

        traj.add_parameter('S', val=124.7, opt=False, units='m**2', static_target=True,
                           desc='aerodynamic reference area',
                           targets={'br_to_v1': ['S'], 'v1_to_vr': ['S'], 'rto': ['S'],
                                    'rotate': ['S'], 'climb': ['S']})

        traj.add_parameter('CD0', val=0.03, opt=False, units=None, static_target=True,
                           desc='zero-lift drag coefficient',
                           targets={f'{phase}': ['CD0'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate' 'climb']})

        traj.add_parameter('AR', val=9.45, opt=False, units=None, static_target=True,
                           desc='wing aspect ratio',
                           targets={f'{phase}': ['AR'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                     'rto', 'rotate' 'climb']})

        traj.add_parameter('e', val=801, opt=False, units=None, static_target=True,
                           desc='Oswald span efficiency factor',
                           targets={f'{phase}': ['e'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                    'rto', 'rotate' 'climb']})

        traj.add_parameter('span', val=35.7, opt=False, units='m', static_target=True,
                           desc='wingspan',
                           targets={f'{phase}': ['span'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                       'rto', 'rotate' 'climb']})

        traj.add_parameter('h_w', val=1.0, opt=False, units='m', static_target=True,
                           desc='height of wing above CG',
                           targets={f'{phase}': ['h_w'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate' 'climb']})

        traj.add_parameter('CL0', val=0.5, opt=False, units=None, static_target=True,
                           desc='zero-alpha lift coefficient',
                           targets={f'{phase}': ['CL0'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                      'rto', 'rotate' 'climb']})

        traj.add_parameter('CL_max', val=2.0, opt=False, units=None, static_target=True,
                           desc='maximum lift coefficient for linear fit',
                           targets={f'{phase}': ['CL_max'] for phase in ['br_to_v1', 'v1_to_vr',
                                                                         'rto', 'rotate' 'climb']})

        # Standard "end of first phase to beginning of second phase" linkages
        traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])

        # Less common "final value of r must be the match at ends of two phases".
        traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
                                    phase_b='climb', var_b='r', loc_b='final',
                                    ref=1000)

        # Define the constraints and objective for the optimal control problem
        rto.add_boundary_constraint('v', loc='final', upper=0.001, ref=100, linear=True)

        rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

        climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
        climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
        climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
        climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=1.2)

        rto.add_objective('r', loc='final', ref=1000.0)

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=True)

        p.run_model()

        assert_near_equal(p.get_val('traj.rotate.t_initial'), 70)
        assert_near_equal(p.get_val('traj.rotate.t_duration'), 5)
        assert_near_equal(p.get_val('traj.rotate.polynomial_controls:alpha'), np.array([[0, 10]]).T)
        assert_near_equal(p.get_val('traj.climb.controls:alpha'),
                          p.model.traj.phases.climb.interp('', [0.01, 0.01], nodes='control_input'))
        assert_near_equal(p.get_val('traj.climb.states:gam'),
                          p.model.traj.phases.climb.interp(ys=[0.0, 0.05], nodes='state_input'))
        assert_near_equal(p.get_val('traj.climb.states:h'),
                          p.model.traj.phases.climb.interp(ys=[0.0, 0.0], nodes='state_input'))
        assert_near_equal(p.get_val('traj.v1_to_vr.parameters:alpha'), 0.0)
