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
    def test_link_fixed_states_final_to_initial(self):
        """ Test that linking phases with states that are fixed at the linkage point raises an exception. """

        import openmdao.api as om

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
        ascent.add_state('h', fix_initial=True, fix_final=False,  units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2')
        ascent.add_parameter('mass', targets=['m'], units='kg')

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
        descent.add_state('h', fix_initial=False, fix_final=True,  units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=True, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2')
        descent.add_parameter('mass', targets=['m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False)

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
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'h', 'gam'], connected=False)

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        with self.assertRaises(ValueError) as e:
            p.setup()

        self.assertEqual(str(e.exception), 'Invalid linkage in Trajectory traj: Cannot link final '
                                           'value of "gam" in ascent to initial value of "gam" in '
                                           'descent. Values on both sides of the '
                                           'linkage are fixed and the linkage is enforced via constraint. Either link '
                                           'the variables via connection or make the variables design variables on at '
                                           'least one side of the connection.')

    @require_pyoptsparse(optimizer='SLSQP')
    def test_link_fixed_states_final_to_final(self):
        """ Test that linking phases with states that are fixed at the linkage point raises an exception. """

        import openmdao.api as om

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
        ascent.add_state('h', fix_initial=True, fix_final=True, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2')
        ascent.add_parameter('mass', targets=['m'], units='kg')

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
        descent.add_state('h', fix_initial=True, fix_final=True, units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2')
        descent.add_parameter('mass', targets=['m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False)

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
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'gam'], connected=False)

        traj.add_linkage_constraint(phase_a='ascent', phase_b='descent', var_a='h', var_b='h',
                                    loc_a='final', loc_b='final')

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        with self.assertRaises(ValueError) as e:
            p.setup()

        self.assertEqual(str(e.exception), 'Invalid linkage in Trajectory traj: Cannot link final '
                                           'value of "h" in ascent to final value of "h" in '
                                           'descent. Values on both sides of the '
                                           'linkage are fixed and the linkage is enforced via constraint. Either link '
                                           'the variables via connection or make the variables design variables on at '
                                           'least one side of the connection.')

    @require_pyoptsparse(optimizer='SLSQP')
    def test_link_fixed_states_initial_to_initial(self):
        """ Test that linking phases with states that are fixed at the linkage point raises an exception. """

        import openmdao.api as om

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
        ascent.add_state('h', fix_initial=True, fix_final=True, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2')
        ascent.add_parameter('mass', targets=['m'], units='kg')

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
        descent.add_state('h', fix_initial=True, fix_final=True, units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2')
        descent.add_parameter('mass', targets=['m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False)

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
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'gam'], connected=False)

        traj.add_linkage_constraint(phase_a='ascent', phase_b='descent', var_a='h', var_b='h',
                                    loc_a='initial', loc_b='initial')

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        with self.assertRaises(ValueError) as e:
            p.setup()

        self.assertEqual(str(e.exception), 'Invalid linkage in Trajectory traj: Cannot link initial '
                                           'value of "h" in ascent to initial value of "h" in '
                                           'descent. Values on both sides of the '
                                           'linkage are fixed and the linkage is enforced via constraint. Either link '
                                           'the variables via connection or make the variables design variables on at '
                                           'least one side of the connection.')

    @require_pyoptsparse(optimizer='SLSQP')
    def test_link_fixed_times_final_to_initial(self):
        """ Test that linking phases with times that are fixed at the linkage point raises an exception. """

        import openmdao.api as om

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
        ascent.set_time_options(fix_initial=True, fix_duration=True, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, units='m', rate_source='r_dot')
        ascent.add_state('h', fix_initial=True, fix_final=False, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2')
        ascent.add_parameter('mass', targets=['m'], units='kg')

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000, shape=(1,))

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(fix_initial=True, fix_duration=True, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', fix_initial=False, fix_final=True, units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, units='m/s', rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2')
        descent.add_parameter('mass', targets=['m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False)

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
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'h', 'gam'], connected=False)

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        with self.assertRaises(ValueError) as e:
            p.setup()

        self.assertEqual(str(e.exception), 'Invalid linkage in Trajectory traj: Cannot link final '
                                           'value of "time" in ascent to initial value of "time" in '
                                           'descent. Values on both sides of the '
                                           'linkage are fixed and the linkage is enforced via constraint. Either link '
                                           'the variables via connection or make the variables design variables on at '
                                           'least one side of the connection.')

    @require_pyoptsparse(optimizer='SLSQP')
    def test_link_bounded_times_final_to_initial(self):
        """ Test that linking phases with times that are fixed at the linkage point raises an exception. """
        import openmdao.api as om

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
        ascent.set_time_options(fix_initial=True, duration_bounds=(10, 10), duration_ref=100, units='s')
        ascent.add_state('r', fix_initial=True, fix_final=False, units='m', rate_source='r_dot')
        ascent.add_state('h', fix_initial=True, fix_final=False, units='m', rate_source='h_dot')
        ascent.add_state('gam', fix_initial=False, fix_final=True, units='rad', rate_source='gam_dot')
        ascent.add_state('v', fix_initial=False, fix_final=False, rate_source='v_dot')

        ascent.add_parameter('S', targets=['S'], units='m**2')
        ascent.add_parameter('mass', targets=['m'], units='kg')

        # Limit the muzzle energy
        ascent.add_boundary_constraint('ke', loc='initial', units='J',
                                       upper=400000, lower=0, ref=100000, shape=(1,))

        # Second Phase (descent)
        transcription = dm.GaussLobatto(num_segments=5, order=3, compressed=True)
        descent = dm.Phase(ode_class=CannonballODE, transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ascent.
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(10, 10), duration_bounds=(10, 10),
                                 duration_ref=100, units='s')
        descent.add_state('r', units='m', rate_source='r_dot')
        descent.add_state('h', fix_initial=False, fix_final=True, units='m', rate_source='h_dot')
        descent.add_state('gam', fix_initial=False, fix_final=False, units='rad', rate_source='gam_dot')
        descent.add_state('v', fix_initial=False, fix_final=False, rate_source='v_dot')

        descent.add_parameter('S', targets=['S'], units='m**2')
        descent.add_parameter('mass', targets=['m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_parameter('CD',
                           targets={'ascent': ['CD'], 'descent': ['CD']},
                           val=0.5, units=None, opt=False)

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
        traj.link_phases(phases=['ascent', 'descent'], vars=['time', 'r', 'h', 'gam'], connected=False)

        traj.add_linkage_constraint('ascent', 'descent', 'ke', 'ke',
                                    ref=100000, connected=False)

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.parameters:m')
        p.model.connect('size_comp.S', 'traj.parameters:S')

        with self.assertRaises(ValueError) as e:
            p.setup()

        self.assertEqual(str(e.exception), 'Invalid linkage in Trajectory traj: Cannot link final '
                                           'value of "time" in ascent to initial value of "time" in '
                                           'descent. Values on both sides of the '
                                           'linkage are fixed and the linkage is enforced via constraint. Either link '
                                           'the variables via connection or make the variables design variables on at '
                                           'least one side of the connection.')

    def test_link_parameters_fixed_to_fixed(self):
        """ Linking non-opt parameters across phases by connection should work.
         But linking them via constraint should not (no freedom to move them to satisfy the constraint).
         """
        import openmdao.api as om
        import dymos as dm
        from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

        p = om.Problem()

        traj = dm.Trajectory()
        p.model.add_subsystem('traj', subsys=traj)

        # First Phase (burn)
        burn1 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=4, order=3))

        traj.add_phase('burn1', burn1)

        burn1.set_time_options(fix_initial=True, duration_bounds=(.5, 10), units='TU')
        burn1.add_state('r', fix_initial=True, fix_final=False,
                        rate_source='r_dot', units='DU')
        burn1.add_state('theta', fix_initial=True, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn1.add_state('vr', fix_initial=True, fix_final=False,
                        rate_source='vr_dot', units='DU/TU')
        burn1.add_state('vt', fix_initial=True, fix_final=False,
                        rate_source='vt_dot', units='DU/TU')
        burn1.add_state('accel', fix_initial=True, fix_final=False,
                        rate_source='at_dot', units='DU/TU**2')
        burn1.add_state('deltav', fix_initial=True, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn1.add_control('u1',  rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn1.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Second Phase (Coast)

        coast = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=10, order=3))

        traj.add_phase('coast', coast)

        coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), duration_ref=10, units='TU')
        coast.add_state('r', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='r_dot', units='DU')
        coast.add_state('theta', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='theta_dot', units='rad')
        coast.add_state('vr', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vr_dot', units='DU/TU')
        coast.add_state('vt', fix_initial=False, fix_final=False, defect_scaler=100.0,
                        rate_source='vt_dot', units='DU/TU')
        coast.add_state('accel', fix_initial=True, fix_final=True, ref=1.0E-12, defect_ref=1.0E-12,
                        rate_source='at_dot', units='DU/TU**2')
        coast.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        coast.add_control('u1', opt=False, val=0.0, units='deg')
        coast.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        # Third Phase (burn)

        burn2 = dm.Phase(ode_class=FiniteBurnODE, transcription=dm.GaussLobatto(num_segments=3, order=3))

        traj.add_phase('burn2', burn2)

        burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10), initial_ref=10, units='TU')
        burn2.add_state('r', fix_initial=False, fix_final=True,
                        rate_source='r_dot', units='DU')
        burn2.add_state('theta', fix_initial=False, fix_final=False,
                        rate_source='theta_dot', units='rad')
        burn2.add_state('vr', fix_initial=False, fix_final=True,
                        rate_source='vr_dot', units='DU/TU')
        burn2.add_state('vt', fix_initial=False, fix_final=True,
                        rate_source='vt_dot', units='DU/TU')
        burn2.add_state('accel', fix_initial=False, fix_final=False, defect_ref=1.0E-6,
                        rate_source='at_dot', units='DU/TU**2')
        burn2.add_state('deltav', fix_initial=False, fix_final=False,
                        rate_source='deltav_dot', units='DU/TU')
        burn2.add_control('u1', rate_continuity=True, rate2_continuity=True,
                          units='deg', scaler=0.01, lower=-30, upper=30)
        burn2.add_parameter('c', opt=False, val=1.5, units='DU/TU')

        burn2.add_objective('deltav', loc='final')

        # Link Phases
        traj.link_phases(phases=['burn1', 'coast', 'burn2'],
                         vars=['time', 'r', 'theta', 'vr', 'vt', 'deltav', 'c'])

        with self.assertRaises(ValueError) as e:
            p.setup()

        self.assertEqual(str(e.exception), 'Invalid linkage in Trajectory traj: Cannot link final value of "c" in '
                                           'burn1 to initial value of "c" in coast. Values on both sides of the '
                                           'linkage are fixed and the linkage is enforced via constraint. Either link '
                                           'the variables via connection or make the variables design variables on at '
                                           'least one side of the connection.')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
