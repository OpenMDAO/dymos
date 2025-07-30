import time
import unittest

from openmdao.utils.testing_utils import use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

try:
    import jax
except ImportError:
    jax = None

if jax is not None:
    import jax
    from dymos.examples.balanced_field.balanced_field_jax_ode import BalancedFieldJaxODEComp
else:
    jax = None


@use_tempdirs
class TestBalancedFieldLengthForDocs(unittest.TestCase):

    @unittest.skipIf(jax is None, 'Test requires jax')
    def test_balanced_field_length_solved(self):
        import openmdao.api as om
        import dymos as dm

        NODES_PER_SEG = 9

        p = om.Problem()

        # Brake release to v_ef - both engines operable
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=NODES_PER_SEG)
        br_to_vef = dm.Phase(ode_class=BalancedFieldJaxODEComp, transcription=tx,
                             ode_init_kwargs={'mode': 'runway',
                                              'attitude_input': 'pitch',
                                              'control': 'gam_rate'})
        br_to_vef.set_time_options(fix_initial=True, fix_duration=True,)
        br_to_vef.add_state('r', fix_initial=True, lower=0)
        br_to_vef.add_state('v', fix_initial=True, lower=0)
        br_to_vef.add_parameter('pitch', val=0.0, opt=False, units='deg')
        br_to_vef.add_parameter('v_ef', val=100.0, opt=False, units='kn')
        br_to_vef.add_calc_expr('v_to_go = v - v_ef',
                                v={'units': 'kn'},
                                v_ef={'units': 'kn'},
                                v_to_go={'units': 'kn'})
        br_to_vef.add_boundary_balance(param='t_duration', name='v_to_go', tgt_val=0.0, loc='final')
        br_to_vef.add_timeseries_output('*')

        # Engine failure to v1 - decision reaction time
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=NODES_PER_SEG)
        vef_to_v1 = dm.Phase(ode_class=BalancedFieldJaxODEComp, transcription=tx,
                             ode_init_kwargs={'mode': 'runway',
                                              'attitude_input': 'pitch',
                                              'control': 'gam_rate'})
        vef_to_v1.set_time_options(fix_initial=True, fix_duration=True, )
        vef_to_v1.add_state('r', fix_initial=True, lower=0)
        vef_to_v1.add_state('v', fix_initial=True, lower=0)
        vef_to_v1.add_parameter('pitch', val=0.0, opt=False, units='deg')
        vef_to_v1.add_parameter('v1', val=150.0, opt=False, units='kn')
        vef_to_v1.add_calc_expr('v_to_go = v - v1',
                                v={'units': 'kn'},
                                v1={'units': 'kn'},
                                v_to_go={'units': 'kn'})
        vef_to_v1.add_boundary_balance(param='t_duration', name='v_to_go', tgt_val=0.0, loc='final')
        vef_to_v1.add_timeseries_output('*')

        # Rejected takeoff at V1 - no engines operable - decelerate to stop
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=NODES_PER_SEG)
        rto = dm.Phase(ode_class=BalancedFieldJaxODEComp, transcription=tx,
                       ode_init_kwargs={'mode': 'runway',
                                        'attitude_input': 'pitch',
                                        'control': 'gam_rate'})
        rto.set_time_options(fix_initial=True, fix_duration=True)
        rto.add_state('r', fix_initial=False, lower=0)
        rto.add_state('v', fix_initial=False, lower=0)
        rto.add_parameter('pitch', val=0.0, opt=False, units='deg')
        rto.add_boundary_balance(param='t_duration', name='v', tgt_val=0.0, loc='final')
        rto.add_timeseries_output('*')

        # V1 to Vr - Single Engine Operable
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=NODES_PER_SEG)
        v1_to_vr = dm.Phase(ode_class=BalancedFieldJaxODEComp, transcription=tx,
                            ode_init_kwargs={'mode': 'runway',
                                             'attitude_input': 'pitch',
                                             'control': 'gam_rate'})
        v1_to_vr.set_time_options(fix_initial=True, fix_duration=True)
        v1_to_vr.add_state('r', fix_initial=False, lower=0)
        v1_to_vr.add_state('v', fix_initial=False, lower=0)
        v1_to_vr.add_parameter('pitch', val=0.0, opt=False, units='deg')
        v1_to_vr.add_boundary_balance(param='t_duration', name='v_over_v_stall', tgt_val=1.11)
        v1_to_vr.add_timeseries_output('*')

        # Rotation to liftoff - single engine operable
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=NODES_PER_SEG)
        rotate = dm.Phase(ode_class=BalancedFieldJaxODEComp, transcription=tx,
                          ode_init_kwargs={'mode': 'runway',
                                           'attitude_input': 'pitch',
                                           'control': 'gam_rate'})
        rotate.set_time_options(fix_initial=True, fix_duration=True)
        rotate.add_state('r', fix_initial=False, lower=0)
        rotate.add_state('v', fix_initial=False, lower=0)
        rotate.add_state('pitch', rate_source='pitch_rate', fix_initial=True, lower=0, upper=15, units='deg')
        rotate.add_parameter('pitch_rate', opt=False, units='deg/s')
        rotate.add_boundary_balance(param='t_duration', name='F_r', tgt_val=0.0, )
        rotate.add_timeseries_output('*')
        rotate.add_timeseries_output('alpha', units='deg')

        # Liftoff and rotate until climb gradient is achieved
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=NODES_PER_SEG)
        liftoff_to_climb_gradient = dm.Phase(ode_class=BalancedFieldJaxODEComp, transcription=tx,
                                             ode_init_kwargs={'mode': 'runway',
                                                              'attitude_input': 'pitch',
                                                              'control': 'attitude'})
        liftoff_to_climb_gradient.set_time_options(fix_initial=True, fix_duration=True)
        liftoff_to_climb_gradient.set_state_options('r', fix_initial=True, lower=0)
        liftoff_to_climb_gradient.set_state_options('h', fix_initial=True, rate_source='h_dot', lower=0)
        liftoff_to_climb_gradient.set_state_options('v', fix_initial=True, lower=0)
        liftoff_to_climb_gradient.set_state_options('gam', fix_initial=True, rate_source='gam_dot', lower=0)
        liftoff_to_climb_gradient.set_state_options('pitch', fix_initial=True, rate_source='pitch_rate',
                                                    opt=False, units='deg')
        liftoff_to_climb_gradient.add_parameter('pitch_rate', opt=False, units='deg/s')
        liftoff_to_climb_gradient.add_parameter('mu_r', opt=False, val=0.0, units=None)
        liftoff_to_climb_gradient.add_boundary_balance(param='t_duration', name='climb_gradient',
                                                       tgt_val=0.05, loc='final', lower=0, upper=10)
        liftoff_to_climb_gradient.add_timeseries_output('alpha', units='deg')
        liftoff_to_climb_gradient.add_timeseries_output('h', units='ft')
        liftoff_to_climb_gradient.add_timeseries_output('*')

        # Sixth Phase: Assume constant flight path angle until 35ft altitude is reached.
        tx = dm.PicardShooting(num_segments=1, nodes_per_seg=NODES_PER_SEG)
        climb_to_obstacle_clearance = dm.Phase(ode_class=BalancedFieldJaxODEComp, transcription=tx,
                                               ode_init_kwargs={'mode': 'runway',
                                                                'attitude_input': 'pitch',
                                                                'control': 'gam_rate'})
        climb_to_obstacle_clearance.set_time_options(fix_initial=True, fix_duration=True)
        climb_to_obstacle_clearance.set_state_options('r', fix_initial=True, lower=0)
        climb_to_obstacle_clearance.set_state_options('h', fix_initial=True, rate_source='h_dot', lower=0)
        climb_to_obstacle_clearance.set_state_options('v', fix_initial=True, lower=0)
        climb_to_obstacle_clearance.set_state_options('gam', fix_initial=True, rate_source='gam_dot', lower=0)
        climb_to_obstacle_clearance.set_state_options('pitch', fix_initial=True,
                                                      rate_source='pitch_rate', opt=False, units='deg')
        climb_to_obstacle_clearance.add_parameter('pitch_rate', opt=False, units='deg/s')
        climb_to_obstacle_clearance.add_parameter('gam_rate', opt=False, units='deg/s')
        climb_to_obstacle_clearance.add_parameter('mu_r', opt=False, val=0.0, units=None)
        climb_to_obstacle_clearance.add_boundary_balance(param='t_duration', name='h',
                                                         tgt_val=35, eq_units='ft', loc='final', lower=0.01, upper=100)
        climb_to_obstacle_clearance.add_timeseries_output('alpha', units='deg')
        climb_to_obstacle_clearance.add_timeseries_output('h', units='ft')
        climb_to_obstacle_clearance.add_timeseries_output('*')

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory(parallel_phases=False)
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_vef', br_to_vef)
        traj.add_phase('vef_to_v1', vef_to_v1)
        traj.add_phase('rto', rto)
        traj.add_phase('v1_to_vr', v1_to_vr)
        traj.add_phase('rotate', rotate)
        traj.add_phase('liftoff_to_climb_gradient', liftoff_to_climb_gradient)
        traj.add_phase('climb_to_obstacle_clearance', climb_to_obstacle_clearance)

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={phase: ['m'] for phase in
                                    ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr', 'rotate',
                                     'liftoff_to_climb_gradient', 'climb_to_obstacle_clearance']})

        traj.add_parameter('T_nominal', val=27000 * 2, opt=False, units='lbf', static_target=True,
                           desc='nominal aircraft thrust',
                           targets={'br_to_vef': ['T']})

        traj.add_parameter('T_engine_out', val=27000, opt=False, units='lbf', static_target=True,
                           desc='thrust under a single engine',
                           targets={phase: ['T'] for phase in ['vef_to_v1', 'v1_to_vr', 'rotate',
                                                               'liftoff_to_climb_gradient',
                                                               'climb_to_obstacle_clearance']})

        traj.add_parameter('T_shutdown', val=0.0, opt=False, units='lbf', static_target=True,
                           desc='thrust when engines are shut down for rejected takeoff',
                           targets={'rto': ['T']})

        traj.add_parameter('mu_r_nominal', val=0.03, opt=False, units=None, static_target=True,
                           desc='nominal runway friction coefficient',
                           targets={phase: ['mu_r'] for phase in ['br_to_vef', 'vef_to_v1', 'v1_to_vr', 'rotate']})

        traj.add_parameter('mu_r_braking', val=0.3, opt=False, units=None, static_target=True,
                           desc='runway friction coefficient under braking',
                           targets={'rto': ['mu_r']})

        traj.add_parameter('h_runway', val=0., opt=False, units='ft',
                           desc='runway altitude',
                           targets={phase: ['h'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr', 'rotate']})

        traj.add_parameter('rho', val=1.225, opt=False, units='kg/m**3', static_target=True,
                           desc='atmospheric density',
                           targets={phase: ['rho'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                 'rotate', 'liftoff_to_climb_gradient',
                                                                 'climb_to_obstacle_clearance']})

        traj.add_parameter('S', val=124.7, opt=False, units='m**2', static_target=True,
                           desc='aerodynamic reference area',
                           targets={f'{phase}': ['S'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                    'rotate', 'liftoff_to_climb_gradient',
                                                                    'climb_to_obstacle_clearance']})

        traj.add_parameter('CD0', val=0.03, opt=False, units=None, static_target=True,
                           desc='zero-lift drag coefficient',
                           targets={f'{phase}': ['CD0'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                      'rotate', 'liftoff_to_climb_gradient',
                                                                      'climb_to_obstacle_clearance']})

        traj.add_parameter('AR', val=9.45, opt=False, units=None, static_target=True,
                           desc='wing aspect ratio',
                           targets={f'{phase}': ['AR'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                     'rotate', 'liftoff_to_climb_gradient',
                                                                     'climb_to_obstacle_clearance']})

        traj.add_parameter('e', val=801, opt=False, units=None, static_target=True,
                           desc='Oswald span efficiency factor',
                           targets={f'{phase}': ['e'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                    'rotate', 'liftoff_to_climb_gradient',
                                                                    'climb_to_obstacle_clearance']})

        traj.add_parameter('span', val=35.7, opt=False, units='m', static_target=True,
                           desc='wingspan',
                           targets={f'{phase}': ['span'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                       'rotate', 'liftoff_to_climb_gradient',
                                                                       'climb_to_obstacle_clearance']})

        traj.add_parameter('h_w', val=1.0, opt=False, units='m', static_target=True,
                           desc='height of wing above CG',
                           targets={f'{phase}': ['h_w'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                      'rotate', 'liftoff_to_climb_gradient',
                                                                      'climb_to_obstacle_clearance']})

        traj.add_parameter('CL0', val=0.5, opt=False, units=None, static_target=True,
                           desc='zero-alpha lift coefficient',
                           targets={f'{phase}': ['CL0'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                      'rotate', 'liftoff_to_climb_gradient',
                                                                      'climb_to_obstacle_clearance']})

        traj.add_parameter('CL_max', val=2.0, opt=False, units=None, static_target=True,
                           desc='maximum lift coefficient for linear fit',
                           targets={f'{phase}': ['CL_max'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                         'rotate', 'liftoff_to_climb_gradient',
                                                                         'climb_to_obstacle_clearance']})

        traj.add_parameter('alpha_max', val=10.0, opt=False, units='deg', static_target=True,
                           desc='angle of attack at maximum lift',
                           targets={f'{phase}': ['alpha_max'] for phase in ['br_to_vef', 'vef_to_v1', 'rto', 'v1_to_vr',
                                                                            'rotate', 'liftoff_to_climb_gradient',
                                                                            'climb_to_obstacle_clearance']})

        # Linkages are done via connection to avoid the need for an optimizer to resolve the linkage constraints.
        traj.link_phases(['br_to_vef', 'vef_to_v1'], vars=['time', 'r', 'v'], connected=True)
        traj.link_phases(['vef_to_v1', 'rto'], vars=['time', 'r', 'v'], connected=True)
        traj.link_phases(['vef_to_v1', 'v1_to_vr'], vars=['time', 'r', 'v'], connected=True)
        traj.link_phases(['v1_to_vr', 'rotate'], vars=['time', 'r', 'v'], connected=True)
        traj.link_phases(['rotate', 'liftoff_to_climb_gradient'], vars=['time', 'r', 'v', 'pitch'],
                         connected=True)
        traj.link_phases(['liftoff_to_climb_gradient', 'climb_to_obstacle_clearance'],
                         vars=['time', 'h', 'gam', 'r', 'v', 'pitch'],
                         connected=True)

        # We need a balance comp to satisfy residuals that cannot be satisfied by connection.
        # - the final range of `rto` and `climb_to_obstacle_clearance` match
        # - the velocity at engine failure is such that the time from engine failure to v1 is the reaction time.
        #
        # Since the trajectory components are added during the setup/configure process, this
        # balance will show up as the first item in the trajectory.
        traj_balance_comp = om.BalanceComp()
        traj.add_subsystem('traj_balance_comp', traj_balance_comp)
        traj.options['auto_order'] = True

        # Vary v1 such that the final range after RTO is the same as the final range after achieving 35ft altitude.
        traj_balance_comp.add_balance('v1', units='kn', eq_units='ft', val=130,
                                      lower=50, upper=180,
                                      lhs_name='r_obstacle', rhs_name='r_rto')
        traj.connect('rto.final_states:r', 'traj_balance_comp.r_rto')
        traj.connect('climb_to_obstacle_clearance.final_states:r', 'traj_balance_comp.r_obstacle')
        traj.connect('v1', 'vef_to_v1.parameters:v1')
        traj.promotes('traj_balance_comp', outputs=['v1'])

        # Vary v_ef such that the time from engine failure to v1 is the reaction time.
        traj_balance_comp.add_balance('v_ef', units='kn', eq_units='s', val=80,
                                      lower=20, upper=180,
                                      lhs_name='ef_to_v1_duration', rhs_name='t_react')
        traj.connect('vef_to_v1.t_duration_val', 'traj_balance_comp.ef_to_v1_duration')
        traj.connect('v_ef', 'br_to_vef.parameters:v_ef')
        traj.promotes('traj_balance_comp', inputs=['t_react'], outputs=['v_ef'])

        traj.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=2, maxiter=100)
        traj.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        traj.linear_solver = om.DirectSolver()

        #
        # Setup the problem and set the initial guess
        #
        p.setup()

        traj.set_val('t_react', 1.0, units='s')
        traj.set_val('v_ef', 130.0, units='kn')

        br_to_vef.set_time_val(initial=0.0, duration=35.0)
        br_to_vef.set_state_val('r', [0, 2000.0])
        br_to_vef.set_state_val('v', [0.0, 80.0])
        br_to_vef.set_parameter_val('pitch', 0.0, units='deg')

        vef_to_v1.set_time_val(initial=0.0, duration=35.0)
        vef_to_v1.set_state_val('r', [2000, 2500.0])
        vef_to_v1.set_state_val('v', [80.0, 100.0])
        vef_to_v1.set_parameter_val('pitch', 0.0, units='deg')
        vef_to_v1.set_parameter_val('v1', 140.0, units='kn')

        rto.set_time_val(initial=30.0, duration=35.0)
        rto.set_state_val('r', [2500, 5000.0])
        rto.set_state_val('v', [100, 0.0], units='kn')
        rto.set_parameter_val('pitch', 0.0, units='deg')

        rotate.set_time_val(initial=30.0, duration=5.0)
        rotate.set_state_val('r', [1750, 1800.0])
        rotate.set_state_val('v', [80, 85.0])
        rotate.set_state_val('pitch', [0.0, 10], units='deg')
        rotate.set_parameter_val('pitch_rate', val=2.0, units='deg/s')

        liftoff_to_climb_gradient.set_time_val(initial=35.0, duration=4.1)
        liftoff_to_climb_gradient.set_state_val('r', [1800, 2000.0], units='ft')
        liftoff_to_climb_gradient.set_state_val('v', [160, 170.0], units='kn')
        liftoff_to_climb_gradient.set_state_val('h', [0.0, 35.0], units='ft')
        liftoff_to_climb_gradient.set_state_val('gam', [0.0, 5.0], units='deg')
        liftoff_to_climb_gradient.set_state_val('pitch', [5.0, 15.0], units='deg')
        liftoff_to_climb_gradient.set_parameter_val('pitch_rate', 2.0, units='deg/s')
        liftoff_to_climb_gradient.set_parameter_val('mu_r', 0.0, units=None)

        climb_to_obstacle_clearance.set_time_val(initial=40.0, duration=2)
        climb_to_obstacle_clearance.set_state_val('r', [2000, 5000.0], units='ft')
        climb_to_obstacle_clearance.set_state_val('v', [160, 170.0], units='kn')
        climb_to_obstacle_clearance.set_state_val('h', [25.0, 35.0], units='ft')
        climb_to_obstacle_clearance.set_state_val('gam', [0.0, 5.0], units='deg')
        climb_to_obstacle_clearance.set_state_val('pitch', [5.0, 15.0], units='deg')
        climb_to_obstacle_clearance.set_parameter_val('pitch_rate', 2.0, units='deg/s')
        climb_to_obstacle_clearance.set_parameter_val('gam_rate', 0.0, units='deg/s')
        climb_to_obstacle_clearance.set_parameter_val('mu_r', 0.0, units=None)

        start = time.perf_counter()
        dm.run_problem(p, run_driver=False, simulate=True, make_plots=True)
        end = time.perf_counter()

        elapsed_time = end - start
        print(f'Elapsed time: {elapsed_time:.6f} seconds')

        t_react = p.get_val('traj.t_react', units='s')[0]
        v_ef = p.get_val('traj.v_ef', units='kn')[0]
        v1 = p.get_val('traj.v1', units='kn')[0]
        vr = p.get_val('traj.rotate.initial_states:v', units='kn')[0]
        v2 = p.get_val('traj.climb_to_obstacle_clearance.final_states:v', units='kn')[0, 0]
        v_over_v_stall = p.get_val('traj.climb_to_obstacle_clearance.timeseries.v_over_v_stall')[-1, 0]
        runway_length = p.get_val('traj.rto.final_states:r', units='m')[0, 0]
        obstacle_dist = p.get_val('traj.climb_to_obstacle_clearance.final_states:r', units='m')[0, 0]

        print(f"{'Balanced Field Length':<21}: {runway_length:6.1f} ft")
        print(f"{'V_ef':<21}: {v_ef:6.1f} kts")
        print(f"{'V_1':<21}: {v1:6.1f} kts")
        print(f"{'V_R':<21}: {vr:6.1f} kts")
        print(f"{'V_2':<21}: {v2:6.1f} kts ({v_over_v_stall:4.3f} * v_stall)")
        print(f"{'Assumed Reaction Time':<21}: {t_react:6.1f} s")

        assert_near_equal(runway_length, obstacle_dist, tolerance=0.01)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
