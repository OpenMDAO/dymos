from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DirectSolver

from dymos import Phase
from dymos.phases.components.phase_linkage_comp import PhaseLinkageComp
from dymos.examples.finite_burn_orbit_raise.finite_burn_eom import FiniteBurnODE

OPTIMIZER = 'SNOPT'
SHOW_PLOTS = True


def two_burn_orbit_raise_problem(transcription='gauss-lobatto', num_segments=8,
                                 transcription_order=3, compressed=True):
    p = Problem(model=Group())

    if OPTIMIZER == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER
        p.driver.options['dynamic_simul_derivs'] = True
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
        p.driver.opt_settings['Major optimality tolerance'] = 1.0E-6
        p.driver.opt_settings['iSumm'] = 6
    else:
        p.driver = ScipyOptimizeDriver()
        p.driver.options['dynamic_simul_derivs'] = True

    # First Phase (burn)

    burn1 = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=num_segments,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('burn1', burn1)

    burn1.set_time_options(opt_initial=False, duration_bounds=(.5, 10))
    burn1.set_state_options('x1', fix_initial=True, fix_final=False)
    burn1.set_state_options('x2', fix_initial=True, fix_final=False)
    burn1.set_state_options('x3', fix_initial=True, fix_final=False)
    burn1.set_state_options('x4', fix_initial=True, fix_final=False)
    burn1.set_state_options('x5', fix_initial=True, fix_final=False)
    burn1.set_state_options('deltav', fix_initial=True, fix_final=False)
    burn1.add_control('u1', rate_continuity=True, units='deg', lower=-179.9, upper=179.9)
    burn1.add_design_parameter('c', opt=False, val=1.5)

    # Second Phase (Coast)

    coast = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=num_segments,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('coast', coast)

    coast.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
    coast.set_state_options('x1', fix_initial=False, fix_final=False)
    coast.set_state_options('x2', fix_initial=False, fix_final=False)
    coast.set_state_options('x3', fix_initial=False, fix_final=False)
    coast.set_state_options('x4', fix_initial=False, fix_final=False)
    coast.set_state_options('x5', fix_initial=True, fix_final=False)
    coast.set_state_options('deltav', fix_initial=False, fix_final=False)
    coast.add_control('u1', opt=False, val=0.0, units='deg', lower=-179.9, upper=179.9)
    coast.add_design_parameter('c', opt=False, val=1.5)

    # Third Phase (burn)

    burn2 = Phase(transcription,
                  ode_class=FiniteBurnODE,
                  num_segments=num_segments,
                  transcription_order=transcription_order,
                  compressed=compressed)

    p.model.add_subsystem('burn2', burn2)

    burn2.set_time_options(initial_bounds=(0.5, 20), duration_bounds=(.5, 10))
    burn2.set_state_options('x1', fix_initial=False, fix_final=True)
    burn2.set_state_options('x2', fix_initial=False, fix_final=False)
    burn2.set_state_options('x3', fix_initial=False, fix_final=True)
    burn2.set_state_options('x4', fix_initial=False, fix_final=True)
    burn2.set_state_options('x5', fix_initial=False, fix_final=False)
    burn2.set_state_options('deltav', fix_initial=False, fix_final=False)
    burn2.add_control('u1', rate_continuity=True, units='deg', lower=-179.9, upper=179.9)
    burn2.add_design_parameter('c', opt=False, val=1.5)

    burn2.add_objective('deltav', loc='final')

    # Link Phases
    linkage_comp = p.model.add_subsystem('linkages', subsys=PhaseLinkageComp())
    linkage_comp.add_linkage(name='L01', vars=['t', 'x1', 'x2', 'x3', 'x4', 'deltav'])
    linkage_comp.add_linkage(name='L12', vars=['t', 'x1', 'x2', 'x3', 'x4', 'deltav'])
    linkage_comp.add_linkage(name='L01', vars=['x5'])

    # Time Continuity
    p.model.connect('burn1.time++', 'linkages.L01_t:lhs')
    p.model.connect('coast.time--', 'linkages.L01_t:rhs')

    p.model.connect('coast.time++', 'linkages.L12_t:lhs')
    p.model.connect('burn2.time--', 'linkages.L12_t:rhs')

    # Position and velocity continuity
    for state in ['x1', 'x2', 'x3', 'x4', 'deltav']:
        p.model.connect('burn1.states:{0}++'.format(state), 'linkages.L01_{0}:lhs'.format(state))
        p.model.connect('coast.states:{0}--'.format(state), 'linkages.L01_{0}:rhs'.format(state))

        p.model.connect('coast.states:{0}++'.format(state), 'linkages.L12_{0}:lhs'.format(state))
        p.model.connect('burn2.states:{0}--'.format(state), 'linkages.L12_{0}:rhs'.format(state))

    # Thrust/weight continuity between the burn phases
    p.model.connect('burn1.states:x5++', 'linkages.L01_x5:lhs')
    p.model.connect('burn2.states:x5--', 'linkages.L01_x5:rhs')

    # Finish Problem Setup

    p.model.options['assembled_jac_type'] = 'csc'
    p.model.linear_solver = DirectSolver(assemble_jac=True)

    p.setup(mode='fwd', check=True)

    # Set Initial Guesses

    p.set_val('burn1.t_initial', value=0.0)
    p.set_val('burn1.t_duration', value=2.25)

    p.set_val('burn1.states:x1', value=burn1.interpolate(ys=[1, 1.5], nodes='state_input'))
    p.set_val('burn1.states:x2', value=burn1.interpolate(ys=[0, 1.7], nodes='state_input'))
    p.set_val('burn1.states:x3', value=burn1.interpolate(ys=[0, 0], nodes='state_input'))
    p.set_val('burn1.states:x4', value=burn1.interpolate(ys=[1, 1], nodes='state_input'))
    p.set_val('burn1.states:x5', value=burn1.interpolate(ys=[0.1, 0], nodes='state_input'))
    p.set_val('burn1.states:deltav', value=burn1.interpolate(ys=[0, 0.1], nodes='state_input'))
    p.set_val('burn1.controls:u1', value=burn1.interpolate(ys=[-3.5, 13.0], nodes='control_input'))
    p.set_val('burn1.design_parameters:c', value=1.5)

    p.set_val('coast.t_initial', value=2.25)
    p.set_val('coast.t_duration', value=7.0)

    p.set_val('coast.states:x1', value=burn1.interpolate(ys=[1.3, 1.5], nodes='state_input'))
    p.set_val('coast.states:x2', value=burn1.interpolate(ys=[2.1767, 1.7], nodes='state_input'))
    p.set_val('coast.states:x3', value=burn1.interpolate(ys=[0.3285, 0], nodes='state_input'))
    p.set_val('coast.states:x4', value=burn1.interpolate(ys=[0.97, 1], nodes='state_input'))
    p.set_val('coast.states:x5', value=burn1.interpolate(ys=[0.11, 0], nodes='state_input'))
    p.set_val('coast.design_parameters:c', value=1.5)

    p.set_val('burn2.t_initial', value=9.25)
    p.set_val('burn2.t_duration', value=1.75)

    p.set_val('burn2.states:x1', value=burn2.interpolate(ys=[1, 3], nodes='state_input'))
    p.set_val('burn2.states:x2', value=burn2.interpolate(ys=[0, 4.0], nodes='state_input'))
    p.set_val('burn2.states:x3', value=burn2.interpolate(ys=[0, 0], nodes='state_input'))
    p.set_val('burn2.states:x4', value=burn2.interpolate(ys=[1, 1], nodes='state_input'))
    p.set_val('burn2.states:x5', value=burn2.interpolate(ys=[0.1, 0], nodes='state_input'))
    p.set_val('burn2.states:deltav', value=burn2.interpolate(ys=[0.1, 0.2], nodes='state_input'))
    p.set_val('burn2.controls:u1', value=burn2.interpolate(ys=[1, 1], nodes='control_input'))
    p.set_val('burn2.design_parameters:c', value=1.5)

    p.run_model()
    p.run_driver()

    # Plot results
    if SHOW_PLOTS:
        burn1_exp_out = burn1.simulate(times=np.linspace(
            p['burn1.t_initial'], p['burn1.t_initial'] + p['burn1.t_duration'], 50))

        coast_exp_out = coast.simulate(times=np.linspace(
            p['coast.t_initial'], p['coast.t_initial'] + p['coast.t_duration'], 50))

        burn2_exp_out = burn2.simulate(times=np.linspace(
            p['burn2.t_initial'], p['burn2.t_initial'] + p['burn2.t_duration'], 50))

        fig, ax = plt.subplots()
        fig.suptitle('Two Burn Orbit Raise Solution')

        for (phase, phase_exp_out) in [(burn1, burn1_exp_out), (coast, coast_exp_out), (burn2, burn2_exp_out)]:
            x_imp = phase.get_values('pos_x', nodes='all')
            y_imp = phase.get_values('pos_y', nodes='all')

            x_exp = phase_exp_out.get_values('pos_x')
            y_exp = phase_exp_out.get_values('pos_y')

            print(phase_exp_out.get_values('x1')[-1])
            print(phase_exp_out.get_values('x2')[-1])
            print(phase_exp_out.get_values('x3')[-1])
            print(phase_exp_out.get_values('x4')[-1])
            print(phase_exp_out.get_values('x5')[-1])
            print()

            ax.plot(x_imp, y_imp, 'ro', label='implicit')
            ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_xlabel('x (DU)')
        ax.set_ylabel('y (DU)')
        ax.grid(True)
        ax.legend(loc='upper right')

        plt.show()

    return p


if __name__ == '__main__':
    two_burn_orbit_raise_problem(transcription='gauss-lobatto', num_segments=20,
                                 transcription_order=3, compressed=False)
