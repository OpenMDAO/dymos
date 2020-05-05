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
        propelled_ascent.set_state_options('V_w', fix_initial=True, fix_final=True)
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

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['propelled_ascent', 'ballistic_ascent'], vars=['*'])


        # Ballistic descent
        transcription = dm.Radau(num_segments=5, order=3, compressed=True)
        descent = CannonballPhase(transcription=transcription)

        traj.add_phase('descent', descent)

        # All initial states and time are free (they will be linked to the final states of ballistic_ascent).
        # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
        descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                                 duration_ref=10, units='s')
        descent.add_state('r', )
        descent.add_state('h', fix_initial=False, fix_final=True)
        descent.add_state('gam', fix_initial=False, fix_final=False)
        descent.add_state('v', fix_initial=False, fix_final=False)

        descent.add_input_parameter('S', targets=['aero.S'], units='m**2')
        descent.add_input_parameter('mass', targets=['eom.m', 'kinetic_energy.m'], units='kg')

        descent.add_objective('r', loc='final', scaler=-1.0)

        # Add internally-managed design parameters to the trajectory.
        traj.add_design_parameter('CD',
                                  targets={'propelled_ascent': ['aero.CD'],
                                           'ballistic_ascent': ['aero.CD'],
                                           'descent': ['aero.CD']},
                                  val=0.5, units=None, opt=False)
        traj.add_design_parameter('CL',
                                  targets={'propelled_ascent': ['aero.CL'],
                                           'ballistic_ascent': ['aero.CL'],
                                           'descent': ['aero.CL']},
                                  val=0.0, units=None, opt=False)
        traj.add_design_parameter('T',
                                  targets={'ballistic_ascent': ['eom.T'],
                                           'descent': ['eom.T']},
                                  val=0.0, units='N', opt=False)
        traj.add_design_parameter('alpha',
                                  targets={'propelled_ascent': ['eom.alpha'],
                                           'ballistic_ascent': ['eom.alpha'],
                                           'descent': ['eom.alpha']},
                                  val=0.0, units='deg', opt=False)

        traj.add_design_parameter('m_empty', units='kg', val=0.1,
                                  targets={'propelled_ascent': 'm_empty',
                                           'ballistic_ascent': 'm_empty',
                                           'descent': 'mass'},
                                  opt=False)
        traj.add_design_parameter('V_b', units='m**3', val=2e-3,
                                 targets={'propelled_ascent': 'V_b'},
                                 opt=False)

        traj.add_design_parameter('S', units='m**2', val=0.005, opt=False)
        traj.add_design_parameter('A_out', units='m**2', val=np.pi*13e-3**2/4.,
                                 targets={'propelled_ascent': ['water_engine.A_out']},
                                 opt=False)

        # Link Phases (link time and all state variables)
        traj.link_phases(phases=['ballistic_ascent', 'descent'], vars=['*'])

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        p.driver.add_recorder(om.SqliteRecorder('ex_water_rocket.db'))

        p.setup()

        # Set Initial Guesses
        p.set_val('traj.design_parameters:CD', 0.0)
        p.set_val('traj.design_parameters:CL', 0.0)
        p.set_val('traj.design_parameters:T', 0.0)

        p.set_val('traj.propelled_ascent.t_initial', 0.0)
        p.set_val('traj.propelled_ascent.t_duration', 0.3)

        p.set_val('traj.propelled_ascent.states:r',
                  propelled_ascent.interpolate(ys=[0, 3], nodes='state_input'))
        p.set_val('traj.propelled_ascent.states:h',
                  propelled_ascent.interpolate(ys=[0, 10], nodes='state_input'))
        #set initial value for velocity as non-zero to avoid undefined EOM
        p.set_val('traj.propelled_ascent.states:v',
                  propelled_ascent.interpolate(ys=[10, 100], nodes='state_input'))
        p.set_val('traj.propelled_ascent.states:gam',
                  propelled_ascent.interpolate(ys=[80, 80], nodes='state_input'),
                  units='deg')
        p.set_val('traj.propelled_ascent.states:V_w',
                  propelled_ascent.interpolate(ys=[1e-3, 0], nodes='state_input'),
                  units='m**3')
        p.set_val('traj.propelled_ascent.states:p',
                  propelled_ascent.interpolate(ys=[5.5e5, 1e5], nodes='state_input'),
                  units='N/m**2')

        p.set_val('traj.ballistic_ascent.t_initial', 0.3)
        p.set_val('traj.ballistic_ascent.t_duration', 5)

        p.set_val('traj.ballistic_ascent.states:r',
                  ballistic_ascent.interpolate(ys=[0, 10], nodes='state_input'))
        p.set_val('traj.ballistic_ascent.states:h',
                  ballistic_ascent.interpolate(ys=[10, 100], nodes='state_input'))
        p.set_val('traj.ballistic_ascent.states:v',
                  ballistic_ascent.interpolate(ys=[60, 20], nodes='state_input'))
        p.set_val('traj.ballistic_ascent.states:gam',
                  ballistic_ascent.interpolate(ys=[80, 0], nodes='state_input'),
                  units='deg')

        p.set_val('traj.descent.t_initial', 10.0)
        p.set_val('traj.descent.t_duration', 10.0)

        p.set_val('traj.descent.states:r', descent.interpolate(ys=[10, 20], nodes='state_input'))
        p.set_val('traj.descent.states:h', descent.interpolate(ys=[10, 0], nodes='state_input'))
        p.set_val('traj.descent.states:v', descent.interpolate(ys=[20, 60], nodes='state_input'))
        p.set_val('traj.descent.states:gam', descent.interpolate(ys=[0, -45], nodes='state_input'),
                  units='deg')

        p.run_model()
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

        print('launch angle: {0:6.4f} '
              'deg '.format(p.get_val('traj.ballistic_ascent.timeseries.states:gam',  units='deg')[0, 0]))
        print('maximum range: {0:6.4f} '
              'm '.format(p.get_val('traj.descent.timeseries.states:r')[-1, 0]))

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        time_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.time'),
                    'descent': p.get_val('traj.descent.timeseries.time')}

        time_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.time'),
                    'descent': exp_out.get_val('traj.descent.timeseries.time')}

        r_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:r'),
                 'descent': p.get_val('traj.descent.timeseries.states:r')}

        r_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:r'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:r')}

        h_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:h'),
                 'descent': p.get_val('traj.descent.timeseries.states:h')}

        h_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:h'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:h')}

        axes.plot(r_imp['ballistic_ascent'], h_imp['ballistic_ascent'], 'bo')

        axes.plot(r_imp['descent'], h_imp['descent'], 'ro')

        axes.plot(r_exp['ballistic_ascent'], h_exp['ballistic_ascent'], 'b--')

        axes.plot(r_exp['descent'], h_exp['descent'], 'r--')

        axes.set_xlabel('range (m)')
        axes.set_ylabel('altitude (m)')

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 6))
        states = ['r', 'h', 'v', 'gam']
        for i, state in enumerate(states):
            x_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:{0}'.format(state)),
                     'descent': p.get_val('traj.descent.timeseries.states:{0}'.format(state))}

            x_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:{0}'.format(state)),
                     'descent': exp_out.get_val('traj.descent.timeseries.states:{0}'.format(state))}

            axes[i].set_ylabel(state)

            axes[i].plot(time_imp['ballistic_ascent'], x_imp['ballistic_ascent'], 'bo')
            axes[i].plot(time_imp['descent'], x_imp['descent'], 'ro')
            axes[i].plot(time_exp['ballistic_ascent'], x_exp['ballistic_ascent'], 'b--')
            axes[i].plot(time_exp['descent'], x_exp['descent'], 'r--')

        plt.show()
        params = ['CL', 'CD', 'T', 'alpha', 'mass', 'S']
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6))
        for i, param in enumerate(params):
            p_imp = {
                'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.input_parameters:{0}'.format(param)),
                'descent': p.get_val('traj.descent.timeseries.input_parameters:{0}'.format(param))}

            p_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.'
                                               'input_parameters:{0}'.format(param)),
                     'descent': exp_out.get_val('traj.descent.timeseries.'
                                                'input_parameters:{0}'.format(param))}

            axes[i].set_ylabel(param)

            axes[i].plot(time_imp['ballistic_ascent'], p_imp['ballistic_ascent'], 'bo')
            axes[i].plot(time_imp['descent'], p_imp['descent'], 'ro')
            axes[i].plot(time_exp['ballistic_ascent'], p_exp['ballistic_ascent'], 'b--')
            axes[i].plot(time_exp['descent'], p_exp['descent'], 'r--')

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    TestWaterRocketForDocs().test_water_rocket_for_docs()
