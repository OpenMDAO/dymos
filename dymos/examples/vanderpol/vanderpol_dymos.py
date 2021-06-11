import openmdao.api as om
import dymos as dm
from dymos.examples.vanderpol.vanderpol_ode import VanderpolODE


def vanderpol(transcription='gauss-lobatto', num_segments=40, transcription_order=3,
              compressed=True, optimizer='SLSQP', use_pyoptsparse=False, delay=0.0, distrib=True,
              solve_segments=False):
    """Dymos problem definition for optimal control of a Van der Pol oscillator"""

    # define the OpenMDAO problem
    p = om.Problem(model=om.Group())

    if not use_pyoptsparse:
        p.driver = om.ScipyOptimizeDriver()
    else:
        p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = optimizer
    if use_pyoptsparse:
        if optimizer == 'SNOPT':
            p.driver.opt_settings['iSumm'] = 6  # show detailed SNOPT output
        elif optimizer == 'IPOPT':
            p.driver.opt_settings['print_level'] = 4
    p.driver.declare_coloring()

    # define a Trajectory object and add to model
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)

    # define a Transcription
    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed,
                            solve_segments=solve_segments)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed,
                     solve_segments=solve_segments)

    # define a Phase as specified above and add to Phase
    phase = dm.Phase(ode_class=VanderpolODE, transcription=t,
                     ode_init_kwargs={'delay': delay, 'distrib': distrib})
    traj.add_phase(name='phase0', phase=phase)

    t_final = 15
    phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=t_final, units='s')

    # set the State time options
    phase.add_state('x0', fix_initial=False, fix_final=False,
                    rate_source='x0dot',
                    units='V/s',
                    targets='x0')  # target required because x0 is an input
    phase.add_state('x1', fix_initial=False, fix_final=False,
                    rate_source='x1dot',
                    units='V',
                    targets='x1')  # target required because x1 is an input
    phase.add_state('J', fix_initial=False, fix_final=False,
                    rate_source='Jdot',
                    units=None)

    # define the control
    phase.add_control(name='u', units=None, lower=-0.75, upper=1.0, continuity=True,
                      rate_continuity=True, targets='u')  # target required because u is an input

    # add constraints
    phase.add_boundary_constraint('x0', loc='initial', equals=1.0)
    phase.add_boundary_constraint('x1', loc='initial', equals=1.0)
    phase.add_boundary_constraint('J', loc='initial', equals=0.0)

    phase.add_boundary_constraint('x0', loc='final', equals=0.0)
    phase.add_boundary_constraint('x1', loc='final', equals=0.0)

    # define objective to minimize
    phase.add_objective('J', loc='final')

    # setup the problem
    p.setup(check=True)

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = t_final

    # add a linearly interpolated initial guess for the state and control curves
    p['traj.phase0.states:x0'] = phase.interp('x0', [1, 0])
    p['traj.phase0.states:x1'] = phase.interp('x1', [1, 0])
    p['traj.phase0.states:J'] = phase.interp('J', [0, 1])
    p['traj.phase0.controls:u'] = phase.interp('u', [-0.75, -0.75])

    return p


if __name__ == '__main__':
    # just set up the problem, test it elsewhere
    p = vanderpol(transcription='radau-ps', num_segments=30, transcription_order=3,
                  compressed=True, optimizer='SLSQP', delay=0.005, distrib=True, use_pyoptsparse=True)

    dm.run_problem(p, run_driver=True, simulate=False)
