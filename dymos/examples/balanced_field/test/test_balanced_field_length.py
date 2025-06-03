from packaging.version import Version
import unittest

import numpy as np

import openmdao
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.balanced_field.balanced_field_ode import BalancedFieldODEComp
from dymos.examples.balanced_field.balanced_field_length import make_balanced_field_length_problem


@use_tempdirs
class TestBalancedFieldLengthRestart(unittest.TestCase):

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipUnless(Version(openmdao.__version__) > Version("3.23"),
                         reason='Test requires OpenMDAO 3.23.0 or later.')
    def test_make_plots(self):
        p = make_balanced_field_length_problem(ode_class=BalancedFieldODEComp, tx=dm.Radau(num_segments=3))
        dm.run_problem(p, run_driver=True, simulate=True, make_plots=True)

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipUnless(Version(openmdao.__version__) > Version("3.23"),
                         reason='Test requires OpenMDAO 3.23.0 or later.')
    def test_restart_from_sol(self):
        p = make_balanced_field_length_problem(ode_class=BalancedFieldODEComp, tx=dm.Radau(num_segments=3))
        dm.run_problem(p, run_driver=True, simulate=False)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'

        sol_results = om.CaseReader(sol_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        dm.run_problem(p, run_driver=True, simulate=True, restart=sol_db)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol_results = om.CaseReader(sol_db).get_case('final')
        sim_results = om.CaseReader(sim_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        assert_near_equal(sol_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)

    @require_pyoptsparse(optimizer='IPOPT')
    @unittest.skipUnless(Version(openmdao.__version__) > Version("3.23"),
                         reason='Test requires OpenMDAO 3.23.0 or later.')
    def test_restart_from_sim(self):
        p = make_balanced_field_length_problem(ode_class=BalancedFieldODEComp, tx=dm.Radau(num_segments=3))
        dm.run_problem(p, run_driver=True, simulate=True)

        sol_db = p.get_outputs_dir() / 'dymos_solution.db'
        sim_db = p.model.traj.sim_prob.get_outputs_dir() / 'dymos_simulation.db'

        sol_results = om.CaseReader(sol_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        dm.run_problem(p, run_driver=True, simulate=True, restart=sim_db)

        sol_results = om.CaseReader(sol_db).get_case('final')
        sim_results = om.CaseReader(sim_db).get_case('final')

        assert_near_equal(sol_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.climb.timeseries.r')[-1], 2197, tolerance=0.01)

        assert_near_equal(sol_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)
        assert_near_equal(sim_results.get_val('traj.rto.timeseries.r')[-1], 2197, tolerance=0.01)


@use_tempdirs
class TestBalancedFieldLengthDefaultValues(unittest.TestCase):

    def test_default_vals_stick(self):
        """
        Make the balanced field problem without any set_val calls after setup.
        """
        ode_class = BalancedFieldODEComp
        tx = dm.GaussLobatto(num_segments=5, order=3, compressed=True)

        p = om.Problem()

        # First Phase: Brake release to V1 - both engines operable
        br_to_v1 = dm.Phase(ode_class=ode_class, transcription=tx,
                            ode_init_kwargs={'mode': 'runway'})
        br_to_v1.set_time_options(fix_initial=True, duration_bounds=(1, 1000), duration_ref=10.0)
        br_to_v1.add_state('r', fix_initial=True, lower=0, ref=1000.0, defect_ref=1000.0)
        br_to_v1.add_state('v', fix_initial=True, lower=0, ref=100.0, defect_ref=100.0)
        br_to_v1.add_parameter('alpha', val=0.0, opt=False, units='deg')
        br_to_v1.add_timeseries_output('*')

        # Second Phase: Rejected takeoff at V1 - no engines operable
        rto = dm.Phase(ode_class=ode_class, transcription=tx,
                       ode_init_kwargs={'mode': 'runway'})
        rto.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        rto.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rto.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        rto.add_parameter('alpha', val=0.0, opt=False, units='deg')
        rto.add_timeseries_output('*')

        # Third Phase: V1 to Vr - single engine operable
        v1_to_vr = dm.Phase(ode_class=ode_class, transcription=tx,
                            ode_init_kwargs={'mode': 'runway'})
        v1_to_vr.set_time_options(fix_initial=False, duration_bounds=(1, 1000), duration_ref=1.0)
        v1_to_vr.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        v1_to_vr.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        v1_to_vr.add_parameter('alpha', val=0.0, opt=False, units='deg')
        v1_to_vr.add_timeseries_output('*')

        # Fourth Phase: Rotate - single engine operable
        rotate = dm.Phase(ode_class=ode_class, transcription=tx,
                          ode_init_kwargs={'mode': 'runway'})
        rotate.set_time_options(fix_initial=False, initial_val=35.0, duration_val=5.0,
                                duration_bounds=(1.0, 5), duration_ref=1.0)
        rotate.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        rotate.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        rotate.add_control('alpha', order=1, opt=True, units='deg', lower=0, upper=10, ref=10,
                           val=[0, 10], control_type='polynomial')
        rotate.add_timeseries_output('*')

        # Fifth Phase: Climb to target speed and altitude at end of runway.
        climb = dm.Phase(ode_class=ode_class, transcription=tx,
                         ode_init_kwargs={'mode': 'climb'})
        climb.set_time_options(fix_initial=False, duration_bounds=(1, 100), duration_ref=1.0)
        climb.add_state('r', fix_initial=False, lower=0, ref=1000.0, defect_ref=1000.0)
        climb.add_state('h', fix_initial=True, lower=0, ref=1.0, defect_ref=1.0,
                        val=np.linspace(0.0, 35.0, 6) * 0.3048)
        climb.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
        climb.add_state('gam', fix_initial=True, lower=0, ref=0.05, defect_ref=0.05,
                        val=np.radians(np.linspace(0.0, 5.0, 6)))
        climb.add_control('alpha', opt=True, val=5.0, units='deg', lower=-10, upper=15, ref=10)
        climb.add_timeseries_output('*')

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_v1', br_to_v1)
        traj.add_phase('rto', rto)
        traj.add_phase('v1_to_vr', v1_to_vr)
        traj.add_phase('rotate', rotate)
        traj.add_phase('climb', climb)

        all_phases = ['br_to_v1', 'v1_to_vr', 'rto', 'rotate', 'climb']
        groundroll_phases = ['br_to_v1', 'v1_to_vr', 'rto', 'rotate']

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={phase: ['m'] for phase in all_phases})

        # Handle parameters which change from phase to phase.
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
                           desc='nominal runway friction coefficient',
                           targets={'br_to_v1': ['mu_r'], 'v1_to_vr': ['mu_r'],  'rotate': ['mu_r']})

        traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, static_target=True,
                           desc='runway friction coefficient under braking',
                           targets={'rto': ['mu_r']})

        traj.add_parameter('h_runway', val=0., opt=False, units='ft',
                           desc='runway altitude',
                           targets={phase: ['h'] for phase in groundroll_phases})

        # Here we're omitting some constants that are common throughout all phases for the sake of brevity.
        # Their correct defaults are specified in add_input calls to `wrap_ode_func`.

        # Standard "end of first phase to beginning of second phase" linkages
        # Alpha changes from being a parameter in v1_to_vr to a polynomial control
        # in rotate, to a dynamic control in `climb`.
        traj.link_phases(['br_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'])
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['rotate', 'climb'], vars=['time', 'r', 'v', 'alpha'])
        traj.link_phases(['br_to_v1', 'rto'], vars=['time', 'r', 'v'])

        # Less common "final value of r must match at ends of two phases".
        traj.add_linkage_constraint(phase_a='rto', var_a='r', loc_a='final',
                                    phase_b='climb', var_b='r', loc_b='final',
                                    ref=1000)

        # Define the constraints and objective for the optimal control problem
        v1_to_vr.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.2, ref=100)

        rto.add_boundary_constraint('v', loc='final', equals=0., ref=100, linear=True)

        rotate.add_boundary_constraint('F_r', loc='final', equals=0, ref=100000)

        climb.add_boundary_constraint('h', loc='final', equals=35, ref=35, units='ft', linear=True)
        climb.add_boundary_constraint('gam', loc='final', equals=5, ref=5, units='deg', linear=True)
        climb.add_path_constraint('gam', lower=0, upper=5, ref=5, units='deg')
        climb.add_boundary_constraint('v_over_v_stall', loc='final', lower=1.25, ref=1.25)

        rto.add_objective('r', loc='final', ref=1000.0)

        #
        # Setup the problem and set the initial guess
        #
        p.setup(check=False)

        p.run_model()

        assert_near_equal(p.get_val('traj.rotate.t_initial'), 35)
        assert_near_equal(p.get_val('traj.rotate.t_duration'), 5)
        assert_near_equal(p.get_val('traj.rotate.controls:alpha'), np.array([[0, 10]]).T)
        assert_near_equal(p.get_val('traj.climb.controls:alpha', units='deg'),
                          p.model.traj.phases.climb.interp('', [5, 5], nodes='control_input'))
        assert_near_equal(p.get_val('traj.climb.states:gam', units='deg'),
                          p.model.traj.phases.climb.interp(ys=[0.0, 5.0], nodes='state_input'))
        assert_near_equal(p.get_val('traj.climb.states:h', units='ft'),
                          p.model.traj.phases.climb.interp(ys=[0.0, 35.0], nodes='state_input'))
        assert_near_equal(p.get_val('traj.v1_to_vr.parameters:alpha'), 0.0)
