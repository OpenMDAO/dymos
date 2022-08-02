import unittest

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
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

        p.set_val('traj.parameters:CD', 0.5)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')

        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)

        exp_out = traj.simulate()

        print('optimal radius: {0:6.4f} m '.format(p.get_val('radius',
                                                             units='m')[0]))
        print('cannonball mass: {0:6.4f} kg '.format(p.get_val('size_comp.mass',
                                                               units='kg')[0]))
        print('launch angle: {0:6.4f} '
              'deg '.format(p.get_val('traj.ascent.timeseries.states:gam',  units='deg')[0, 0]))
        print('maximum range: {0:6.4f} '
              'm '.format(p.get_val('traj.descent.timeseries.states:r')[-1, 0]))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        time_imp = {'ascent': p.get_val('traj.ascent.timeseries.time'),
                    'descent': p.get_val('traj.descent.timeseries.time')}

        time_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.time'),
                    'descent': exp_out.get_val('traj.descent.timeseries.time')}

        r_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:r'),
                 'descent': p.get_val('traj.descent.timeseries.states:r')}

        r_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:r'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:r')}

        h_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:h'),
                 'descent': p.get_val('traj.descent.timeseries.states:h')}

        h_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:h'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:h')}

        axes.plot(r_imp['ascent'], h_imp['ascent'], 'bo')

        axes.plot(r_imp['descent'], h_imp['descent'], 'ro')

        axes.plot(r_exp['ascent'], h_exp['ascent'], 'b--')

        axes.plot(r_exp['descent'], h_exp['descent'], 'r--')

        axes.set_xlabel('range (m)')
        axes.set_ylabel('altitude (m)')

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
        states = ['r', 'h', 'v', 'gam']
        for i, state in enumerate(states):
            x_imp = {'ascent': p.get_val('traj.ascent.timeseries.states:{0}'.format(state)),
                     'descent': p.get_val('traj.descent.timeseries.states:{0}'.format(state))}

            x_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.states:{0}'.format(state)),
                     'descent': exp_out.get_val('traj.descent.timeseries.states:{0}'.format(state))}

            axes[i].set_ylabel(state)

            axes[i].plot(time_imp['ascent'], x_imp['ascent'], 'bo')
            axes[i].plot(time_imp['descent'], x_imp['descent'], 'ro')
            axes[i].plot(time_exp['ascent'], x_exp['ascent'], 'b--')
            axes[i].plot(time_exp['descent'], x_exp['descent'], 'r--')

        params = ['CD', 'mass', 'S']
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 6))
        for i, param in enumerate(params):
            p_imp = {
                'ascent': p.get_val('traj.ascent.timeseries.parameters:{0}'.format(param)),
                'descent': p.get_val('traj.descent.timeseries.parameters:{0}'.format(param))}

            p_exp = {'ascent': exp_out.get_val('traj.ascent.timeseries.'
                                               'parameters:{0}'.format(param)),
                     'descent': exp_out.get_val('traj.descent.timeseries.'
                                                'parameters:{0}'.format(param))}

            axes[i].set_ylabel(param)

            axes[i].plot(time_imp['ascent'], p_imp['ascent'], 'bo')
            axes[i].plot(time_imp['descent'], p_imp['descent'], 'ro')
            axes[i].plot(time_exp['ascent'], p_exp['ascent'], 'b--')
            axes[i].plot(time_exp['descent'], p_exp['descent'], 'r--')

        plt.show()

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

        p.set_val('traj.parameters:CD', 0.5)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')

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

        p.set_val('traj.parameters:CD', 0.5)

        p.set_val('traj.ascent.t_initial', 0.0)
        p.set_val('traj.ascent.t_duration', 10.0)

        p.set_val('traj.ascent.states:r', ascent.interp('r', [0, 100]))
        p.set_val('traj.ascent.states:h', ascent.interp('h', [0, 100]))
        p.set_val('traj.ascent.states:v', ascent.interp('v', [200, 150]))
        p.set_val('traj.ascent.states:gam', ascent.interp('gam', [25, 0]), units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interp('r', [100, 200]))
        p.set_val('traj.descent.states:h', descent.interp('h', [100, 0]))
        p.set_val('traj.descent.states:v', descent.interp('v', [150, 200]))
        p.set_val('traj.descent.states:gam', descent.interp('gam', [0, -45]), units='deg')

        dm.run_problem(p)

        assert_near_equal(p.get_val('traj.descent.states:r')[-1],
                          3183.25, tolerance=1.0E-2)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
