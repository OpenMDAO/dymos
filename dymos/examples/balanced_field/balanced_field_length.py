import openmdao.api as om
import dymos as dm


def make_balanced_field_length_problem(ode_class, tx):
    """
    Create a balanced field length problem and optionally set default values into it.

    Parameters
    ----------
    ode_class : System class
        The Dymos ODE System class.
    tx_class : Transcription
        Transcription to use.

    Returns
    -------
    _type_
        _description_
    """
    p = om.Problem()

    p.driver = om.pyOptSparseDriver()
    p.driver.declare_coloring()

    # Use IPOPT if available, with fallback to SLSQP
    p.driver.options['optimizer'] = 'IPOPT'
    p.driver.options['print_results'] = True

    p.driver.opt_settings['print_level'] = 0
    p.driver.opt_settings['mu_strategy'] = 'adaptive'

    p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
    p.driver.opt_settings['mu_init'] = 0.01
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'

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
    rotate.set_time_options(fix_initial=False, duration_bounds=(1.0, 5), duration_ref=1.0)
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
    climb.add_state('h', fix_initial=True, lower=0, ref=1.0, defect_ref=1.0)
    climb.add_state('v', fix_initial=False, lower=0, ref=100.0, defect_ref=100.0)
    climb.add_state('gam', fix_initial=True, lower=0, ref=0.05, defect_ref=0.05)
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
    p.setup(check=True)

    br_to_v1.set_time_val(initial=0.0, duration=35.0)
    br_to_v1.set_state_val('r', [0, 2500.0])
    br_to_v1.set_state_val('v', [0.0001, 100.0])
    br_to_v1.set_parameter_val('alpha', 0.0, units='deg')

    v1_to_vr.set_time_val(initial=35.0, duration=35.0)
    v1_to_vr.set_state_val('r', [2500, 300.0])
    v1_to_vr.set_state_val('v', [100, 110.0])
    v1_to_vr.set_parameter_val('alpha', 0.0, units='deg')

    rto.set_time_val(initial=35.0, duration=1.0)
    rto.set_state_val('r', [2500, 5000.0])
    rto.set_state_val('v', [110, 0.0001])
    rto.set_parameter_val('alpha', 0.0, units='deg')

    rotate.set_time_val(initial=35.0, duration=5.0)
    rotate.set_state_val('r', [1750, 1800.0])
    rotate.set_state_val('v', [80, 85.0])
    rotate.set_control_val('alpha', 0.0, units='deg')

    climb.set_time_val(initial=30.0, duration=20.0)
    climb.set_state_val('r', [5000, 5500.0], units='ft')
    climb.set_state_val('v', [160, 170.0], units='kn')
    climb.set_state_val('h', [0.0, 35.0], units='ft')
    climb.set_state_val('gam', [0.0, 5.0], units='deg')
    climb.set_control_val('alpha', 5.0, units='deg')

    return p
