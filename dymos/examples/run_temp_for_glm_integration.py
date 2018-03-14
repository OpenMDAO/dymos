from __future__ import print_function, division, absolute_import

import numpy as np

import matplotlib.pyplot as plt

from openmdao.api import Problem, Group, pyOptSparseDriver, ScipyOptimizeDriver, DenseJacobian,\
    CSCJacobian, CSRJacobian, DirectSolver

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE

OPTIMIZER = 'SNOPT'
SHOW_PLOTS = False


def brachistochrone_min_time(transcription='gauss-lobatto', top_level_jacobian='csc'):
    p = Problem(model=Group())

    if OPTIMIZER == 'SNOPT':
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Verify level'] = 3
    else:
        p.driver = ScipyOptimizeDriver()

    transcription = 'glm'
    formulation = 'optimizer-based'
    # formulation = 'solver-based'
    # formulation = 'time-marching'
    method_name = 'GaussLegendre6'
    method_name = 'ImplicitMidpoint'
    method_name = 'Lobatto4'
    phase = Phase(transcription,
                  ode_class=BrachistochroneODE,
                  num_segments=10, #8,
                  formulation=formulation,
                  method_name=method_name)

    p.model.add_subsystem('phase0', phase)

    phase.set_time_options(initial_bounds=(0, 0), duration_bounds=(.5, 10))

    if 0:
        phase.set_state_options('x', fix_initial=True, fix_final=True)
        phase.set_state_options('y', fix_initial=True, fix_final=True)
        phase.set_state_options('v', fix_initial=True)
    elif 1:
        phase.add_boundary_constraint('x', loc='initial', equals=0.)
        phase.add_boundary_constraint('x', loc='final', equals=10.)
        phase.add_boundary_constraint('y', loc='initial', equals=10.)
        phase.add_boundary_constraint('y', loc='final', equals=5.)
        phase.add_boundary_constraint('v', loc='initial', equals=0.)
    else:
        phase.add_boundary_constraint('x', loc='initial', equals=0.)
        phase.add_boundary_constraint('x', loc='final', equals=2.)
        phase.add_boundary_constraint('y', loc='initial', equals=0.)
        phase.add_boundary_constraint('y', loc='final', equals=-2.)
        phase.add_boundary_constraint('v', loc='initial', equals=0.)

    phase.add_control('theta', units='deg', dynamic=True,
                      rate_continuity=True, lower=0.01, upper=179.9)

    phase.add_control('g', units='m/s**2', dynamic=False, opt=False, val=9.80665)

    # Minimize time at the end of the phase
    phase.add_objective('time', loc='final', scaler=10)

    # top_level_jacobian = 'dense'
    # # top_level_jacobian = 'csc'
    # top_level_jacobian = 'default'
    # if top_level_jacobian.lower() == 'csc':
    #     p.model.jacobian = CSCJacobian()
    # elif top_level_jacobian.lower() == 'dense':
    #     p.model.jacobian = DenseJacobian()
    # elif top_level_jacobian.lower() == 'csr':
    #     p.model.jacobian = CSRJacobian()
    # else:
    #     pass
    #
    # p.model.linear_solver = DirectSolver()

    p.setup(check=True)

    p['phase0.t_initial'] = 0.0
    p['phase0.t_duration'] = 2.0

    if 1:
        p['phase0.states:x'] = phase.interpolate(ys=[0, 10], nodes='all')
        p['phase0.states:y'] = phase.interpolate(ys=[10, 5], nodes='all')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 9.9], nodes='all')
    else:
        p['phase0.states:x'] = phase.interpolate(ys=[0, 2], nodes='all')
        p['phase0.states:y'] = phase.interpolate(ys=[0, -2], nodes='all')
        p['phase0.states:v'] = phase.interpolate(ys=[0, 2], nodes='all')

    p['phase0.controls:theta'] = phase.interpolate(ys=[5, 100.5], nodes='all')

    p.run_model()

    p.run_driver()

    import matplotlib.pyplot as plt
    plt.subplot(2, 1, 1)
    plt.plot(p['phase0.out_states:x'], p['phase0.out_states:y'])
    plt.subplot(2, 1, 2)
    plt.plot(p['phase0.controls:theta'])
    plt.show()

    print(p['phase0.states:x'])
    print(p['phase0.states:y'])
    print(p['phase0.states:v'])
    print(p['phase0.ozone.times'])
    exit()

    # ----------------------------------------------------------------------------------------

    exp_out = phase.simulate(times=np.linspace(p['phase0.t_initial'], p['phase0.t_duration'], 50))

    # Plot results
    if SHOW_PLOTS:
        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = phase.get_values('x', nodes='all')
        y_imp = phase.get_values('y', nodes='all')

        x_exp = exp_out.get_values('x')
        y_exp = exp_out.get_values('y')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend(loc='upper right')

        fig, ax = plt.subplots()
        fig.suptitle('Brachistochrone Solution')

        x_imp = phase.get_values('time', nodes='all')
        y_imp = phase.get_values('theta_rate2', nodes='all')

        x_exp = exp_out.get_values('time')
        y_exp = exp_out.get_values('theta_rate2')

        ax.plot(x_imp, y_imp, 'ro', label='implicit')
        ax.plot(x_exp, y_exp, 'b-', label='explicit')

        ax.set_xlabel('time (s)')
        ax.set_ylabel('theta rate2 (rad/s**2)')
        ax.grid(True)
        ax.legend(loc='lower right')

        plt.show()

    return p


if __name__ == '__main__':
    from six import iteritems
    prob = brachistochrone_min_time()

    phase = prob.model.phase0

    subsystems_ordered = phase._subsystems_allprocs
    print([sys.name for sys in subsystems_ordered])

    input_srcs = phase._conn_global_abs_in2out

    connections = {}

    for tgt, src in iteritems(input_srcs):
        if src is None:
            continue
        src_rel = src.replace('phase0.', '')
        if src_rel in connections:
            connections[src_rel].append(tgt.replace('phase0.', ''))
        else:
            connections[src_rel] = [tgt.replace('phase0.', '')]

    # connections = {
    #     tgt: src for tgt, src in iteritems(input_srcs) if src is not None
    # }

    print(connections)
    exit()

    from pyxdsm.XDSM import XDSM

    opt = 'Optimization'
    solver = 'MDA'
    comp = 'Analysis'
    group = 'Metamodel'
    func = 'Function'

    x = XDSM()

    for system in subsystems_ordered:
        x.add_system(system.name, comp, r'{0}'.format(system.name.replace('_', ' ')))

    for src, tgts in iteritems(connections):
        for tgt in tgts:
            x.connect(src.split('.')[0], tgt.split('.')[0], 'x')

    x.write('brach_xdsm', build=True)
