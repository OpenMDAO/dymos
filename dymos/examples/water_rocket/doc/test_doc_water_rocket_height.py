import unittest

import matplotlib.pyplot as plt
# plt.switch_backend('Agg')


class TestWaterRocketForDocs(unittest.TestCase):

    def test_water_rocket_for_docs(self):
        import numpy as np

        import openmdao.api as om
        from openmdao.utils.assert_utils import assert_near_equal

        import dymos as dm
        from dymos.examples.cannonball.size_comp import CannonballSizeComp
        from dymos.examples.cannonball.cannonball_phase import CannonballPhase
        from dymos.examples.water_rocket.water_propulsion_ode import WaterPropulsionODE

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.declare_coloring()

        external_params = p.model.add_subsystem('external_params', om.IndepVarComp())

        external_params.add_output('radius', val=0.10, units='m')
        external_params.add_output('dens', val=7.87, units='g/cm**3')

        external_params.add_design_var('radius', lower=0.01, upper=0.10, ref0=0.01, ref=0.10)

        p.model.add_subsystem('size_comp', CannonballSizeComp())

        traj = p.model.add_subsystem('traj', dm.Trajectory())

        # Propelled ascent
        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        propelled_ascent = CannonballPhase(ode_class=WaterPropulsionODE,
                                           transcription=transcription)

        # Add states specific for the propelled ascent
        propelled_ascent.add_state('p', units='N/m**2', rate_source='water_engine.pdot',
                                   targets=['water_engine.p'])
        propelled_ascent.add_state('V_w', units='m**3', rate_source='water_engine.Vdot',
                                   targets=['water_engine.V_w', 'mass_adder.V_w'])

        propelled_ascent = traj.add_phase('propelled_ascent', propelled_ascent)

        # All initial states except flight path angle are fixed
        # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
        # Final water volume is fixed (we will set it to zero so that phase ends when bottle empties)
        propelled_ascent.set_time_options(fix_initial=True, duration_bounds=(0, 0.5), duration_ref=0.1, units='s')
        propelled_ascent.set_state_options('r', fix_initial=True, fix_final=False)
        propelled_ascent.set_state_options('h', fix_initial=True, fix_final=False)
        propelled_ascent.set_state_options('gam', fix_initial=True, fix_final=False)
        propelled_ascent.set_state_options('v', fix_initial=True, fix_final=False)
        propelled_ascent.set_state_options('V_w', fix_initial=False, fix_final=True)
        propelled_ascent.set_state_options('p', fix_initial=True, fix_final=False)

        propelled_ascent.add_input_parameter('S', targets=['aero.S'], units='m**2')
        propelled_ascent.add_input_parameter('m_empty', targets=['mass_adder.m_empty'], units='kg')
        propelled_ascent.add_input_parameter('V_b', targets=['water_engine.V_b'], units='m**3')

        propelled_ascent.add_timeseries_output('water_engine.F','T') 

        # Ballistic ascent
        transcription = dm.Radau(num_segments=5, order=3, compressed=False)
        ballistic_ascent = CannonballPhase(transcription=transcription)

        ballistic_ascent = traj.add_phase('ballistic_ascent', ballistic_ascent)

        # All initial states except flight path angle are free (they will be
        # linked to the final stages of propelled_ascent).
        # Final flight path angle is fixed (we will set it to zero so that the
        # phase ends at apogee)
        ballistic_ascent.set_time_options(fix_initial=False, initial_bounds=(0,1), duration_bounds=(0, 10), duration_ref=1, units='s')
        ballistic_ascent.set_state_options('r', fix_initial=False, fix_final=False)
        ballistic_ascent.set_state_options('h', fix_initial=False, fix_final=False)
        ballistic_ascent.set_state_options('gam', fix_initial=False, fix_final=True)
        ballistic_ascent.set_state_options('v', fix_initial=False, fix_final=False)

        ballistic_ascent.add_input_parameter('S', targets=['aero.S'], units='m**2')
        ballistic_ascent.add_input_parameter('m_empty', targets=['eom.m'], units='kg')

        ballistic_ascent.add_objective('h', loc='final', scaler=-1.0)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['propelled_ascent', 'ballistic_ascent'], vars=['*'])

        # Add internally-managed design parameters to the trajectory.
        traj.add_design_parameter('CD',
                                  targets={'propelled_ascent': ['aero.CD'],
                                           'ballistic_ascent': ['aero.CD'],
                                           },
                                  val=0.5, units=None, opt=False)
        traj.add_design_parameter('CL',
                                  targets={'propelled_ascent': ['aero.CL'],
                                           'ballistic_ascent': ['aero.CL'],
                                           },
                                  val=0.0, units=None, opt=False)
        traj.add_design_parameter('T',
                                  targets={'ballistic_ascent': ['eom.T']},
                                  val=0.0, units='N', opt=False)
        traj.add_design_parameter('alpha',
                                  targets={'propelled_ascent': ['eom.alpha'],
                                           'ballistic_ascent': ['eom.alpha'],
                                           },
                                  val=0.0, units='deg', opt=False)

        # Add externally-provided design parameters to the trajectory.
        # In this case, we connect 'm' to pre-existing input parameters named 'mass' in each phase.
        traj.add_input_parameter('m_empty', units='kg', val=0.01,
                                 targets={'propelled_ascent': 'm_empty',
                                          'ballistic_ascent': 'm_empty',
                                          })
        traj.add_input_parameter('V_b', units='m**3', val=2e-3,
                                 targets={'propelled_ascent': 'V_b'})

        # In this case, by omitting targets, we're connecting these parameters to parameters
        # with the same name in each phase.
        traj.add_input_parameter('S', units='m**2', val=0.005)
        traj.add_design_parameter('A_out', units='m**2', val=np.pi*13e-3**2/4.,
                                 targets={'propelled_ascent': ['water_engine.A_out']})

        # Issue Connections
        p.model.connect('external_params.radius', 'size_comp.radius')
        p.model.connect('external_params.dens', 'size_comp.dens')

        p.model.connect('size_comp.mass', 'traj.input_parameters:m_empty')
        p.model.connect('size_comp.S', 'traj.input_parameters:S')

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.driver.add_recorder(om.SqliteRecorder('ex_water_rocket.db'))

        p.setup()

        # Set Initial Guesses
        p.set_val('external_params.radius', 0.05, units='m')
        p.set_val('external_params.dens', 7.87, units='g/cm**3')

        p.set_val('traj.design_parameters:CD', 0.5)
        p.set_val('traj.design_parameters:CL', 0.0)
        p.set_val('traj.design_parameters:T', 0.0)

        p.set_val('traj.propelled_ascent.t_initial', 0.0)
        p.set_val('traj.propelled_ascent.t_duration', 0.3)

        p.set_val('traj.propelled_ascent.states:r',
                  propelled_ascent.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('traj.propelled_ascent.states:h',
                  propelled_ascent.interpolate(ys=[0, 10], nodes='state_input'))
        #set initial value for velocity as non-zero to avoid undefined EOM
        p.set_val('traj.propelled_ascent.states:v',
                  propelled_ascent.interpolate(ys=[1e-3, 10], nodes='state_input'))
        p.set_val('traj.propelled_ascent.states:gam',
                  propelled_ascent.interpolate(ys=[90, 90], nodes='state_input'),
                  units='deg')
        p.set_val('traj.propelled_ascent.states:V_w',
                  propelled_ascent.interpolate(ys=[1e-3, 0], nodes='state_input'),
                  units='m**3')
        p.set_val('traj.propelled_ascent.states:p',
                  propelled_ascent.interpolate(ys=[5.5e5, 1e5], nodes='state_input'),
                  units='N/m**2')

        p.set_val('traj.ballistic_ascent.t_initial', 0)
        p.set_val('traj.ballistic_ascent.t_duration', 1)

        p.set_val('traj.ballistic_ascent.states:r',
                  ballistic_ascent.interpolate(ys=[0, 0], nodes='state_input'))
        p.set_val('traj.ballistic_ascent.states:h',
                  ballistic_ascent.interpolate(ys=[10, 100], nodes='state_input'))
        p.set_val('traj.ballistic_ascent.states:v',
                  ballistic_ascent.interpolate(ys=[10, 1e-3], nodes='state_input'))
        p.set_val('traj.ballistic_ascent.states:gam',
                  ballistic_ascent.interpolate(ys=[90, 0], nodes='state_input'),
                  units='deg')

        #p.run_model()
        dm.run_problem(p)

        exp_out = traj.simulate()

        plt.figure()
        plt.plot(exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                 exp_out.get_val('traj.propelled_ascent.timeseries.states:p'))
        plt.plot(p.get_val('traj.propelled_ascent.timeseries.time'),
                 p.get_val('traj.propelled_ascent.timeseries.states:p'),'o')
        plt.title('p')
        plt.figure()
        plt.plot(exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                 exp_out.get_val('traj.propelled_ascent.timeseries.states:V_w'))
        plt.plot(p.get_val('traj.propelled_ascent.timeseries.time'),
                 p.get_val('traj.propelled_ascent.timeseries.states:V_w'),'o')
        plt.title('V_w')
        plt.figure()
        plt.plot(exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                 exp_out.get_val('traj.propelled_ascent.timeseries.states:v'))
        plt.plot(p.get_val('traj.propelled_ascent.timeseries.time'),
                 p.get_val('traj.propelled_ascent.timeseries.states:v'),'o')
        plt.title('v')
        plt.figure()
        plt.plot(exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                 exp_out.get_val('traj.propelled_ascent.timeseries.states:h'))
        plt.plot(p.get_val('traj.propelled_ascent.timeseries.time'),
                 p.get_val('traj.propelled_ascent.timeseries.states:h'),'o')
        plt.title('h')
        plt.figure()
        plt.plot(exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                 exp_out.get_val('traj.propelled_ascent.timeseries.states:gam'))
        plt.plot(p.get_val('traj.propelled_ascent.timeseries.time'),
                 p.get_val('traj.propelled_ascent.timeseries.states:gam'),'o')
        plt.title('gam')
        plt.show()

        print('optimal radius: {0:6.4f} m '.format(p.get_val('external_params.radius',
                                                             units='m')[0]))
        print('cannonball mass: {0:6.4f} kg '.format(p.get_val('size_comp.mass',
                                                               units='kg')[0]))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        time_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.time'),
                    'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.time')}

        time_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.time'),
                    'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.time')}

        r_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:r'),
                 'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.states:r')}

        r_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:r'),
                 'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.states:r')}

        h_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:h'),
                 'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.states:h')}

        h_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:h'),
                 'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.states:h')}

        axes.plot(r_imp['ballistic_ascent'], h_imp['ballistic_ascent'], 'bo')

        axes.plot(r_imp['propelled_ascent'], h_imp['propelled_ascent'], 'ro')

        axes.plot(r_exp['ballistic_ascent'], h_exp['ballistic_ascent'], 'b--')

        axes.plot(r_exp['propelled_ascent'], h_exp['propelled_ascent'], 'r--')

        axes.set_xlabel('range (m)')
        axes.set_ylabel('altitude (m)')

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
        states = ['r', 'h', 'v', 'gam']
        for i, state in enumerate(states):
            x_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:{0}'.format(state)),
                     'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.states:{0}'.format(state))}

            x_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:{0}'.format(state)),
                     'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.states:{0}'.format(state))}

            axes[i].set_ylabel(state)

            axes[i].plot(time_imp['ballistic_ascent'], x_imp['ballistic_ascent'], 'bo')
            axes[i].plot(time_imp['propelled_ascent'], x_imp['propelled_ascent'], 'ro')
            axes[i].plot(time_exp['ballistic_ascent'], x_exp['ballistic_ascent'], 'b--')
            axes[i].plot(time_exp['propelled_ascent'], x_exp['propelled_ascent'], 'r--')

        params = ['CL', 'CD', 'T', 'alpha', 'm_empty', 'S']
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6))
        for i, param in enumerate(params):
            p_imp = {
                'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.input_parameters:{0}'.format(param)),
                'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.input_parameters:{0}'.format(param) if param !='T'
                                            else 'traj.propelled_ascent.timeseries.{0}'.format(param))}

            p_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.'
                                               'input_parameters:{0}'.format(param)),
                     'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.'
                                                         'input_parameters:{0}'.format(param) if param != 'T'
                                                         else 'traj.propelled_ascent.timeseries.{0}'.format(param))}

            axes[i].set_ylabel(param)

            axes[i].plot(time_imp['ballistic_ascent'], p_imp['ballistic_ascent'], 'bo')
            axes[i].plot(time_imp['propelled_ascent'], p_imp['propelled_ascent'], 'ro')
            axes[i].plot(time_exp['ballistic_ascent'], p_exp['ballistic_ascent'], 'b--')
            axes[i].plot(time_exp['propelled_ascent'], p_exp['propelled_ascent'], 'r--')

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
