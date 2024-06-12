import unittest

try:
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
except ImportError:
    plt = None

from openmdao.utils.testing_utils import use_tempdirs, require_pyoptsparse


@use_tempdirs
class TestTwoPhaseCannonballODEOutputLinkage(unittest.TestCase):

    @require_pyoptsparse(optimizer='SLSQP')
    def test_two_phase_cannonball_ode_output_linkage(self):
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.cannonball.size_comp import CannonballSizeComp
        from dymos.examples.cannonball.cannonball_ode import CannonballODE

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        p.model.add_subsystem('size_comp', CannonballSizeComp(), promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, units='m', rate_source='r_dot')
        ascent.add_state('h', fix_initial=True, fix_final=False, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        ascent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial',
                                       upper=400000, lower=0, ref=100000)

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', fix_initial=False, fix_final=True, units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        descent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False, static_target=True)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'mass', 'descent': 'mass'})

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005)

        # Link Phases (link time and all state variables)
        # Note velocity is not included here.  Doing so is equivalent to linking kinetic energy,
        # and causes a duplicate row in the constraint jacobian.
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'h', 'gam'], connected=True)

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.driver.add_recorder(om.SqliteRecorder('ex_two_phase_cannonball.db'))

        p.setup()

        # Set Initial Guesses
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        traj.set_parameter_val('CD', 0.5)

        ascent.set_time_val(initial=0.0, duration=10.0)

        ascent.set_state_val('r', [0, 100])
        ascent.set_state_val('h', [0, 100])
        ascent.set_state_val('v', [200, 150])
        ascent.set_state_val('gam', [25, 0], units='deg')

        descent.set_time_val(initial=10.0, duration=10.0)

        descent.set_state_val('r', [100, 200])
        descent.set_state_val('h', [100, 0])
        descent.set_state_val('v', [150, 200])
        descent.set_state_val('gam', [0, -45], units='deg')

        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)

        print('optimal radius: {0:6.4f} m '.format(p.get_val('radius',
                                                             units='m')[0]))
        print('cannonball mass: {0:6.4f} kg '.format(p.get_val('size_comp.mass',
                                                               units='kg')[0]))
        print('launch angle: {0:6.4f} '
              'deg '.format(p.get_val('traj.ascent.timeseries.gam',  units='deg')[0, 0]))
        print('maximum range: {0:6.4f} '
              'm '.format(p.get_val('traj.descent.timeseries.r')[-1, 0]))

        assert_near_equal(p.get_val('traj.linkages.ascent:ke_final|descent:ke_initial'), 0.0)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_traj_param_target_none(self):
        # Tests a bug where you couldn't specify None as a target for a specific phase.
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.cannonball.size_comp import CannonballSizeComp
        from dymos.examples.cannonball.cannonball_ode import CannonballODE

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        p.model.add_subsystem('size_comp', CannonballSizeComp(), promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, units='m', rate_source='r_dot')
        ascent.add_state('h', fix_initial=True, fix_final=False, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        ascent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000, shape=(1,))

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', fix_initial=False, fix_final=True, units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        descent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False, static_target=True)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        traj.add_parameter('m', units='kg', val=1.0,
                           targets={'ascent': 'mass', 'descent': 'mass'}, static_target=True)

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_parameter('S', units='m**2', val=0.005, static_target=True)

        # Link Phases (link time and all state variables)
        # Note velocity is not included here.  Doing so is equivalent to linking kinetic energy,
        # and causes a duplicate row in the constraint jacobian.
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'h', 'gam'], connected=True)

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.driver.add_recorder(om.SqliteRecorder('ex_two_phase_cannonball.db'))

        p.setup()

        # Set Initial Guesses
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        traj.set_parameter_val('CD', 0.5)

        ascent.set_time_val(initial=0.0, duration=10.0)

        ascent.set_state_val('r', [0, 100])
        ascent.set_state_val('h', [0, 100])
        ascent.set_state_val('v', [200, 150])
        ascent.set_state_val('gam', [25, 0], units='deg')

        descent.set_time_val(initial=10.0, duration=10.0)

        descent.set_state_val('r', [100, 200])
        descent.set_state_val('h', [100, 0])
        descent.set_state_val('v', [150, 200])
        descent.set_state_val('gam', [0, -45], units='deg')

        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)

    @require_pyoptsparse(optimizer='SLSQP')
    def test_traj_param_target_unspecified_units(self):
        # Tests a bug where you couldn't specify None as a target for a specific phase.
        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.cannonball.size_comp import CannonballSizeComp
        from dymos.examples.cannonball.cannonball_ode import CannonballODE

        p = om.Problem(model=om.Group())

        p.driver = om.pyOptSparseDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        p.model.add_subsystem('size_comp', CannonballSizeComp(), promotes_inputs=['radius', 'dens'])
        p.model.set_input_defaults('dens', val=7.87, units='g/cm**3')
        p.model.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10, units='m')

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        ascent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        ascent = traj.add_phase('ascent', ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        ascent.set_time_options(fix_initial=True, duration_bounds=(1, 100), duration_ref=100, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, units='m', rate_source='r_dot')
        ascent.add_state('h', fix_initial=True, fix_final=False, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        ascent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000, shape=(1,))

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=100, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', fix_initial=False, fix_final=True, units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2', static_target=True)
        descent.add_parameter('mass', targets=['m'], units='kg', static_target=True)

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, opt=False, static_target=True)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        traj.add_parameter('m', val=1.0,
                           targets={'ascent': 'mass', 'descent': 'mass'}, static_target=True)

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_parameter('S', val=0.005, static_target=True)

        # Link Phases (link time and all state variables)
        # Note velocity is not included here.  Doing so is equivalent to linking kinetic energy,
        # and causes a duplicate row in the constraint jacobian.
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'h', 'gam'], connected=True)

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.driver.add_recorder(om.SqliteRecorder('ex_two_phase_cannonball.db'))

        p.setup()

        # Set Initial Guesses
        p.set_val('radius', 0.05, units='m')
        p.set_val('dens', 7.87, units='g/cm**3')

        traj.set_parameter_val('CD', 0.5)

        ascent.set_time_val(initial=0.0, duration=10.0)

        ascent.set_state_val('r', [0, 100])
        ascent.set_state_val('h', [0, 100])
        ascent.set_state_val('v', [200, 150])
        ascent.set_state_val('gam', [25, 0], units='deg')

        descent.set_time_val(initial=10.0, duration=10.0)

        descent.set_state_val('r', [100, 200])
        descent.set_state_val('h', [100, 0])
        descent.set_state_val('v', [150, 200])
        descent.set_state_val('gam', [0, -45], units='deg')

        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
