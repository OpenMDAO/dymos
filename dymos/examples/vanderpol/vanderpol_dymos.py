import openmdao.api as om
import dymos as dm
from dymos.examples.vanderpol.vanderpol_ode import vanderpol_ode, vanderpol_ode_group


def vanderpol(transcription='gauss-lobatto', num_segments=8, transcription_order=3,
              compressed=True, optimizer='SLSQP', use_pyoptsparse=False, delay=None):
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
            p.driver.opt_settings['print_level'] = 5
    p.driver.declare_coloring()

    # define a Trajectory object and add to model
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', subsys=traj)

    # define a Transcription
    if transcription == 'gauss-lobatto':
        t = dm.GaussLobatto(num_segments=num_segments,
                            order=transcription_order,
                            compressed=compressed)
    elif transcription == 'radau-ps':
        t = dm.Radau(num_segments=num_segments,
                     order=transcription_order,
                     compressed=compressed)
    elif transcription == 'runge-kutta':
        t = dm.RungeKutta(num_segments=num_segments,
                          order=transcription_order,
                          compressed=compressed)

    # define a Phase as specified above and add to Phase
    if not delay:
        phase = dm.Phase(ode_class=vanderpol_ode, transcription=t)
    else:
        phase = dm.Phase(ode_class=vanderpol_ode_group, transcription=t)  # distributed component group
    traj.add_phase(name='phase0', phase=phase)

    t_final = 15.0
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

    # TODO - Dymos API will soon provide a way to specify this.
    # the linear solver used to compute derivatives is not working on MPI, so switch to LinearRunOnce
    for phase in traj._phases.values():
        phase.linear_solver = om.LinearRunOnce()

    p['traj.phase0.t_initial'] = 0.0
    p['traj.phase0.t_duration'] = t_final

    # add a linearly interpolated initial guess for the state and control curves
    p['traj.phase0.states:x0'] = phase.interpolate(ys=[1, 0], nodes='state_input')
    p['traj.phase0.states:x1'] = phase.interpolate(ys=[1, 0], nodes='state_input')
    p['traj.phase0.states:J'] = phase.interpolate(ys=[0, 1], nodes='state_input')
    p['traj.phase0.controls:u'] = phase.interpolate(ys=[-0.75, -0.75], nodes='control_input')

    p.final_setup()

    # debugging helpers:
    # om.n2(p)       # show n2 diagram
    #
    # with np.printoptions(linewidth=1024):  # display partials for manual checking
    #     p.check_partials(compact_print=True)

    return p


if __name__ == '__main__':
    # just set up the problem, test it elsewhere
    p = vanderpol(transcription='gauss-lobatto', num_segments=75, transcription_order=3,
                  compressed=True, optimizer='SLSQP')
