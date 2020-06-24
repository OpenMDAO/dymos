import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def vanderpol_dymos_plots(p):
    """ Plot results of a given Dymos VanDerPol ODE problem """

    # check validity by using scipy.integrate.solve_ivp to integrate the solution
    sim_out = p.model.traj.simulate()

    # plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))

    # state variable x1
    axes[0, 0].plot(p.get_val('traj.phase0.timeseries.time'),
                    p.get_val('traj.phase0.timeseries.states:x1'),
                    'ro', label='solution')

    axes[0, 0].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                    sim_out.get_val('traj.phase0.timeseries.states:x1'),
                    'b-', label='simulation')

    axes[0, 0].set_xlabel('time (s)')
    axes[0, 0].set_ylabel('x1 (V)')
    axes[0, 0].legend()
    axes[0, 0].grid()

    # state variable x0
    axes[0, 1].plot(p.get_val('traj.phase0.timeseries.time'),
                    p.get_val('traj.phase0.timeseries.states:x0'),
                    'ro', label='solution')

    axes[0, 1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                    sim_out.get_val('traj.phase0.timeseries.states:x0'),
                    'b-', label='simulation')

    axes[0, 1].set_xlabel('time (s)')
    axes[0, 1].set_ylabel('x0 (V/s)')
    axes[0, 1].legend()
    axes[0, 1].grid()

    # state variable x1 vs x0
    axes[1, 0].plot(p.get_val('traj.phase0.timeseries.states:x0'),
                    p.get_val('traj.phase0.timeseries.states:x1'),
                    'ro', label='solution')

    axes[1, 0].plot(sim_out.get_val('traj.phase0.timeseries.states:x0'),
                    sim_out.get_val('traj.phase0.timeseries.states:x1'),
                    'b-', label='simulation')

    axes[1, 0].set_xlabel('x0 (V/s)')
    axes[1, 0].set_ylabel('x1 (V)')
    axes[1, 0].legend()
    axes[1, 0].grid()

    # control variable u
    axes[1, 1].plot(p.get_val('traj.phase0.timeseries.time'),
                    p.get_val('traj.phase0.timeseries.controls:u'),
                    'ro', label='solution')

    axes[1, 1].plot(sim_out.get_val('traj.phase0.timeseries.time'),
                    sim_out.get_val('traj.phase0.timeseries.controls:u'),
                    'b-', label='simulation')

    axes[1, 1].set_xlabel('time (s)')
    axes[1, 1].set_ylabel('control u')
    axes[1, 1].legend()
    axes[1, 1].grid()

    plt.show()
