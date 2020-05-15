import unittest

import matplotlib as mpl
import matplotlib.pyplot as plt
# plt.switch_backend('Agg')

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import dymos as dm
from dymos.examples.cannonball.size_comp import CannonballSizeComp
from dymos.examples.cannonball.cannonball_phase import CannonballPhase
from dymos.examples.water_rocket.water_propulsion_ode import WaterPropulsionODE


transcription = dm.GaussLobatto(num_segments=1, order=21, compressed=False)
#transcription = dm.GaussLobatto(num_segments=3, order=7, compressed=False)
#transcription = dm.GaussLobatto(num_segments=7, order=3, compressed=False)
def new_propelled_ascent_phase():
    propelled_ascent = CannonballPhase(ode_class=WaterPropulsionODE,
                                       transcription=transcription)

    # Add states specific for the propelled ascent
    propelled_ascent.add_state('p', units='bar', rate_source='water_engine.pdot',
                               targets=['water_engine.p'])
    propelled_ascent.add_state('V_w', units='m**3', ref=1e-3, rate_source='water_engine.Vdot',
                               targets=['water_engine.V_w', 'mass_adder.V_w'])

    # All initial states except flight path angle and water volume are fixed
    # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
    # Final water volume is fixed (we will set it to zero so that phase ends when bottle empties)
    propelled_ascent.set_time_options(
        fix_initial=True, duration_bounds=(0, 0.5), duration_ref=0.1, units='s')
    propelled_ascent.set_state_options(
        'r', fix_initial=True, fix_final=False)
    propelled_ascent.set_state_options(
        'h', fix_initial=True, fix_final=False)
    propelled_ascent.set_state_options(
        'gam', fix_initial=False, fix_final=False, lower=0, upper=89, units='deg')
    propelled_ascent.set_state_options(
        'v', fix_initial=True, fix_final=False)
    propelled_ascent.set_state_options(
        'V_w', fix_initial=False, fix_final=True)
    propelled_ascent.set_state_options(
        'p', fix_initial=True, fix_final=False)

    propelled_ascent.add_input_parameter(
        'S', targets=['aero.S'], units='m**2')
    propelled_ascent.add_input_parameter(
        'm_empty', targets=['mass_adder.m_empty'], units='kg')
    propelled_ascent.add_input_parameter(
        'V_b', targets=['water_engine.V_b'], units='m**3')

    propelled_ascent.add_timeseries_output('water_engine.F', 'T', units='N')

    return propelled_ascent


def new_ballistic_ascent_phase():
    ballistic_ascent = CannonballPhase(transcription=transcription)

    # All initial states are free (they will be  linked to the final stages of propelled_ascent).
    # Final flight path angle is fixed (we will set it to zero so that the phase ends at apogee)
    ballistic_ascent.set_time_options(
        fix_initial=False, initial_bounds=(0, 1), duration_bounds=(0, 10), 
        duration_ref=1, units='s')
    ballistic_ascent.set_state_options(
        'r', fix_initial=False, fix_final=False)
    ballistic_ascent.set_state_options(
        'h', fix_initial=False, fix_final=False)
    ballistic_ascent.set_state_options(
        'gam', fix_initial=False, fix_final=True, upper=89, units='deg')
    ballistic_ascent.set_state_options(
        'v', fix_initial=False, fix_final=False)

    ballistic_ascent.add_input_parameter(
        'S', targets=['aero.S'], units='m**2')
    ballistic_ascent.add_input_parameter(
        'm_empty', targets=['eom.m'], units='kg')

    return ballistic_ascent


def new_descent_phase():
    descent = CannonballPhase(transcription=transcription)

    # All initial states and time are free (they will be linked to the final states of ballistic_ascent).
    # Final altitude is fixed (we will set it to zero so that the phase ends at ground impact)
    descent.set_time_options(initial_bounds=(.5, 100), duration_bounds=(.5, 100),
                             duration_ref=10, units='s')
    descent.add_state('r', )
    descent.add_state('h', fix_initial=False, fix_final=True)
    descent.add_state('gam', fix_initial=False, fix_final=False, units='deg')
    descent.add_state('v', fix_initial=False, fix_final=False)

    descent.add_input_parameter('S', targets=['aero.S'], units='m**2')
    descent.add_input_parameter('mass', targets=['eom.m', 'kinetic_energy.m'], units='kg')

    return descent

def new_water_rocket_trajectory():
    traj = dm.Trajectory()

    # Add phases to trajectory
    propelled_ascent = traj.add_phase('propelled_ascent', new_propelled_ascent_phase())
    ballistic_ascent = traj.add_phase('ballistic_ascent', new_ballistic_ascent_phase())
    descent = traj.add_phase('descent', new_descent_phase())

    # Link phases
    traj.link_phases(phases=['propelled_ascent', 'ballistic_ascent'], vars=['*'])
    traj.link_phases(phases=['ballistic_ascent', 'descent'], vars=['*'])

    # Set objective function
    # NOTE: only one objective function must be commented out at any time
    # Use this line to optimize for height
    #ballistic_ascent.add_objective('h', loc='final', scaler=-1.0)
    # Use this line to optimize for range
    descent.add_objective('r', loc='final', scaler=-1.0)

    # Add design parameters to the trajectory.
    traj.add_design_parameter('CD',
                              targets={'propelled_ascent': ['aero.CD'],
                                       'ballistic_ascent': ['aero.CD'],
                                       'descent': ['aero.CD']},
                              val=0.3450, units=None, opt=False)
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

    traj.add_design_parameter('m_empty', units='kg', val=0.15,
                              targets={'propelled_ascent': 'm_empty',
                                       'ballistic_ascent': 'm_empty',
                                       'descent': 'mass'},
                              lower=0, upper=1, ref=0.1,
                              opt=True)
    traj.add_design_parameter('V_b', units='m**3', val=2e-3,
                             targets={'propelled_ascent': 'V_b'},
                             opt=False)

    traj.add_design_parameter('S', units='m**2', val=np.pi*106e-3**2/4, opt=False)
    traj.add_design_parameter('A_out', units='m**2', val=np.pi*22e-3**2/4.,
                             targets={'propelled_ascent': ['water_engine.A_out']},
                             opt=False)
    traj.add_design_parameter('k', units=None, val=1.2, opt=False,
            targets={'propelled_ascent': ['water_engine.k']})

    return traj, (propelled_ascent, ballistic_ascent, descent)


class TestWaterRocketForDocs(unittest.TestCase):

    def test_water_rocket_for_docs(self):

        p = om.Problem(model=om.Group())

        p.driver = om.ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['maxiter'] = 200
        p.driver.options['tol'] = 1e-8
        #p.driver.options['debug_print'] = ['nl_cons']
        p.driver.declare_coloring()

        traj, (propelled_ascent, ballistic_ascent, descent) = new_water_rocket_trajectory()
        traj = p.model.add_subsystem('traj', traj)

        # Finish Problem Setup
        p.model.linear_solver = om.DirectSolver()

        #p.driver.add_recorder(om.SqliteRecorder('ex_water_rocket.db'))

        p.setup()

        # Set Initial Guesses
        p.set_val('traj.propelled_ascent.t_initial', 0.0)
        p.set_val('traj.propelled_ascent.t_duration', 0.3)

        p.set_val('traj.propelled_ascent.states:r',
                  propelled_ascent.interpolate(ys=[0, 3], nodes='state_input'))
        p.set_val('traj.propelled_ascent.states:h',
                  propelled_ascent.interpolate(ys=[0, 10], nodes='state_input'))
        #set initial value for velocity as non-zero to avoid undefined EOM
        p.set_val('traj.propelled_ascent.states:v',
                  propelled_ascent.interpolate(ys=[0.1, 100], nodes='state_input'))
        p.set_val('traj.propelled_ascent.states:gam',
                  propelled_ascent.interpolate(ys=[80, 80], nodes='state_input'),
                  units='deg')
        p.set_val('traj.propelled_ascent.states:V_w',
                  propelled_ascent.interpolate(ys=[1e-3, 0], nodes='state_input'),
                  units='m**3')
        p.set_val('traj.propelled_ascent.states:p',
                  propelled_ascent.interpolate(ys=[6.5, 3.5], nodes='state_input'),
                  units='bar')

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

        p.run_driver()
        self.print_results(p)

        exp_out = traj.simulate(times_per_seg=200)

        self.plot_trajectory(p, exp_out)

    def print_results(self, p):
        print('launch angle: {0:6.4f} '
              'deg '.format(p.get_val('traj.propelled_ascent.timeseries.states:gam',  units='deg')[0, 0]))
        print('Flight angle at end of propulsion: {0:6.4f} '
              'deg '.format(p.get_val('traj.propelled_ascent.timeseries.states:gam',  units='deg')[-1, 0]))
        print('empty mass: {0:6.4f} '
              'kg '.format(p.get_val('traj.design_parameters:m_empty')[0,0]))
        print('water volume: {0:6.4f} '
                'L '.format(p.get_val('traj.propelled_ascent.timeseries.states:V_w', 'L')[0,0]))
        print('maximum range: {0:6.4f} '
              'm '.format(p.get_val('traj.descent.timeseries.states:r')[-1, 0]))
        print('maximum height: {0:6.4f} '
              'm '.format(p.get_val('traj.ballistic_ascent.timeseries.states:h')[-1, 0]))
        print('maximum velocity: {0:6.4f} '
              'm/s '.format(p.get_val('traj.propelled_ascent.timeseries.states:v')[-1, 0]))

    def plot_trajectory(self, p, exp_out):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3, 8))

        time_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.time'),
                    'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.time'),
                    'descent': p.get_val('traj.descent.timeseries.time')}

        time_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.time'),
                    'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.time'),
                    'descent': exp_out.get_val('traj.descent.timeseries.time')}

        r_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:r'),
                 'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.states:r'),
                 'descent': p.get_val('traj.descent.timeseries.states:r')}

        r_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:r'),
                 'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.states:r'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:r')}

        h_imp = {'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.states:h'),
                 'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.states:h'),
                 'descent': p.get_val('traj.descent.timeseries.states:h')}

        h_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.states:h'),
                 'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.states:h'),
                 'descent': exp_out.get_val('traj.descent.timeseries.states:h')}

        axes.plot(r_imp['propelled_ascent'], h_imp['propelled_ascent'], 'ro', markerfacecolor='None')
        axes.plot(r_imp['ballistic_ascent'], h_imp['ballistic_ascent'], 'mo', markerfacecolor='None')
        axes.plot(r_imp['descent'], h_imp['descent'], 'bo', markerfacecolor='None')

        axes.plot(r_exp['propelled_ascent'], h_exp['propelled_ascent'], 'r-')
        axes.plot(r_exp['ballistic_ascent'], h_exp['ballistic_ascent'], 'm-')
        axes.plot(r_exp['descent'], h_exp['descent'], 'b-')

        axes.set_xlabel('r (m)')
        axes.set_ylabel('h (m)')
        axes.set_aspect('equal', 'box')
        fig.tight_layout()
        fig.savefig('path.pdf', bbox_inches = 'tight')

        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4, 8), sharex=True)
        states = ['r', 'h', 'v', 'gam']
        units =  ['m', 'm', 'm/s', 'deg']
        phases = ['propelled_ascent', 'ballistic_ascent', 'descent']
        
        x_imp = {phase: {state: p.get_val(f"traj.{phase}.timeseries.states:{state}", unit) for state, unit in zip(states,units)} for phase in phases}
        x_exp = {phase: {state: exp_out.get_val(f"traj.{phase}.timeseries.states:{state}", unit) for state, unit in zip(states,units)} for phase in phases}

        for i, (state, unit) in enumerate(zip(states, units)):
            axes[i].set_ylabel(f"{state} ({unit})" if state != 'gam' else f'$\gamma$ ({unit})')

            axes[i].plot(time_imp['propelled_ascent'], x_imp['propelled_ascent'][state], 'ro', markerfacecolor='None')
            axes[i].plot(time_imp['ballistic_ascent'], x_imp['ballistic_ascent'][state], 'mo', markerfacecolor='None')
            axes[i].plot(time_imp['descent'], x_imp['descent'][state], 'bo', markerfacecolor='None')
            axes[i].plot(time_exp['propelled_ascent'], x_exp['propelled_ascent'][state], 'r-', label='Propelled Ascent')
            axes[i].plot(time_exp['ballistic_ascent'], x_exp['ballistic_ascent'][state], 'm-', label='Ballistic Ascent')
            axes[i].plot(time_exp['descent'], x_exp['descent'][state], 'b-', label='Descent')

            if state == 'gam':
                axes[i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[1, 1.5, 3, 4.5, 6, 9, 10]))

        axes[i].set_xlabel('t (s)')
        axes[0].legend()

        fig.tight_layout()
        fig.savefig('states.pdf', dpi=600)
        
        params = ['CL', 'CD', 'T', 'alpha', 'S']
        fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(12, 6))
        for i, param in enumerate(params):
            p_imp = {
                'ballistic_ascent': p.get_val('traj.ballistic_ascent.timeseries.input_parameters:{0}'.format(param)),
                'propelled_ascent': p.get_val('traj.propelled_ascent.timeseries.input_parameters:{0}'.format(param) if param !='T'
                                            else 'traj.propelled_ascent.timeseries.{0}'.format(param)),
                'descent': p.get_val('traj.descent.timeseries.input_parameters:{0}'.format(param))}

            p_exp = {'ballistic_ascent': exp_out.get_val('traj.ballistic_ascent.timeseries.'
                                               'input_parameters:{0}'.format(param)),
                     'propelled_ascent': exp_out.get_val('traj.propelled_ascent.timeseries.'
                                                         'input_parameters:{0}'.format(param) if param != 'T'
                                                         else 'traj.propelled_ascent.timeseries.{0}'.format(param)),
                     'descent': exp_out.get_val('traj.descent.timeseries.'
                                                'input_parameters:{0}'.format(param))}

            axes[i].set_ylabel(param)

            axes[i].plot(time_imp['propelled_ascent'], p_imp['propelled_ascent'], 'ro')
            axes[i].plot(time_imp['ballistic_ascent'], p_imp['ballistic_ascent'], 'mo')
            axes[i].plot(time_imp['descent'], p_imp['descent'], 'bo')
            axes[i].plot(time_exp['propelled_ascent'], p_exp['propelled_ascent'], 'r--')
            axes[i].plot(time_exp['ballistic_ascent'], p_exp['ballistic_ascent'], 'm--')
            axes[i].plot(time_exp['descent'], p_exp['descent'], 'b--')

        #Plot propelled ascent states

        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(4,6))
        t_imp = p.get_val('traj.propelled_ascent.time', 's')
        t_exp = exp_out.get_val('traj.propelled_ascent.time', 's')

        ax[0].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:p', 'bar'), 'ro', markerfacecolor='None')
        ax[0].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:p', 'bar'), 'r-')
        ax[0].set_ylabel('p (bar)')
        ax[0].set_ylim(bottom=0)

        ax[1].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:V_w', 'L'), 'ro', markerfacecolor='None')
        ax[1].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:V_w', 'L'), 'r-')
        ax[1].set_ylabel('$V_w$ (L)')
        ax[1].set_ylim(0, p.get_val('traj.design_parameters:V_b', 'L'))

        ax[2].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.T', 'N'), 'ro', markerfacecolor='None')
        ax[2].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.T', 'N'), 'r-')
        ax[2].set_ylabel('T (N)')
        ax[2].set_ylim(bottom=0)

        ax[3].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:v', 'm/s'), 'ro', markerfacecolor='None')
        ax[3].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:v', 'm/s'), 'r-')
        ax[3].set_ylabel('v (m/s)')
        ax[3].set_ylim(bottom=0)
 
        ax[4].plot(t_imp, p.get_val('traj.propelled_ascent.timeseries.states:gam', 'deg'), 'ro', markerfacecolor='None')
        ax[4].plot(t_exp, exp_out.get_val('traj.propelled_ascent.timeseries.states:gam', 'deg'), 'r-')
        ax[4].set_ylabel('$\gamma$ (deg)')
        ax[4].yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins='auto', steps=[1, 1.5, 3, 4.5, 6, 9, 10]))

        ax[-1].set_xlabel('t (s)')

        fig.tight_layout()
        fig.savefig('propelled_ascent.pdf', dpi=600)

        plt.show()


if __name__ == '__main__':  # pragma: no cover
    TestWaterRocketForDocs().test_water_rocket_for_docs()
