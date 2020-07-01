import unittest
from dymos.utils.doc_utils import save_for_docs


class TestDocProjectile(unittest.TestCase):

    @save_for_docs
    def test_ivp(self):
        import openmdao.api as om
        import dymos as dm
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')  # disable plotting to the screen

        from projectile_ode import ProjectileODE

        # Instnatiate an OpenMDAO Problem instance.
        prob = om.Problem()

        # Instantiate a Dymos Trajectory and add it to the Problem model.
        traj = dm.Trajectory()
        prob.model.add_subsystem('traj', traj)

        # Instantiate a Phase and add it to the Trajectory.
        # Here the transcription is necessary but not particularly relevant.
        phase = dm.Phase(ode_class=ProjectileODE, transcription=dm.Radau(num_segments=10))
        traj.add_phase('phase0', phase)

        # Tell Dymos the states to be propagated using the given ODE.
        phase.add_state('x', rate_source='x_dot', targets=None, units='m')
        phase.add_state('y', rate_source='y_dot', targets=None, units='m')
        phase.add_state('vx', rate_source='vx_dot', targets=['vx'], units='m/s')
        phase.add_state('vy', rate_source='vy_dot', targets=['vy'], units='m/s')

        # Setup the OpenMDAO problem
        prob.setup()

        # Assign values to the times and states
        prob.set_val('traj.phase0.t_initial', 0.0)
        prob.set_val('traj.phase0.t_duration', 15.0)

        prob.set_val('traj.phase0.states:x', 0.0)
        prob.set_val('traj.phase0.states:y', 0.0)
        prob.set_val('traj.phase0.states:vx', 100.0)
        prob.set_val('traj.phase0.states:vy', 100.0)

        # Perform a single execution of the model (executing the model is required before simulation).
        prob.run_model()

        # Perform an explicit simulation of our ODE from the initial conditions.
        sim_out = traj.simulate()

        # Plot the state values obtained from the phase timeseries objects in the simulation output.
        t_sol = prob.get_val('traj.phase0.timeseries.time')
        t_sim = sim_out.get_val('traj.phase0.timeseries.time')

        fig, axes = plt.subplots(4, 1)
        for i, state in enumerate(['x', 'y', 'vx', 'vy']):
            sol = axes[i].plot(t_sol, prob.get_val(f'traj.phase0.timeseries.states:{state}'), 'o')
            sim = axes[i].plot(t_sim, sim_out.get_val(f'traj.phase0.timeseries.states:{state}'), '-')
            axes[i].set_ylabel(state)
        axes[3].set_xlabel('time (s)')
        fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
        plt.tight_layout()
        plt.show()

    @save_for_docs
    def test_ivp_solve_segments(self):
        import openmdao.api as om
        import dymos as dm
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')  # disable plotting to the screen

        from projectile_ode import ProjectileODE

        # Instnatiate an OpenMDAO Problem instance.
        prob = om.Problem()

        # We need an optimization driver.  To solve this simple problem ScipyOptimizerDriver will work.
        prob.driver = om.ScipyOptimizeDriver()

        # Instantiate a Dymos Trajectory and add it to the Problem model.
        traj = dm.Trajectory()
        prob.model.add_subsystem('traj', traj)

        # Instantiate a Phase and add it to the Trajectory.
        # Here the transcription is necessary but not particularly relevant.
        phase = dm.Phase(ode_class=ProjectileODE, transcription=dm.Radau(num_segments=10, solve_segments=True))
        traj.add_phase('phase0', phase)

        # Tell Dymos that the time duration of the Phase should not be fixed (its a design variable for
        # the optimization). The duration of a phase may be negative but it should never be zero.
        phase.set_time_options(fix_initial=True, duration_bounds=(5, 50))

        # Tell Dymos the states to be propagated using the given ODE.
        phase.add_state('x', fix_initial=True, targets=None, rate_source='x_dot', units='m')
        phase.add_state('y', fix_initial=True, targets=None, rate_source='y_dot', units='m')
        phase.add_state('vx', fix_initial=True, targets=['vx'], rate_source='vx_dot', units='m/s')
        phase.add_state('vy', fix_initial=True, targets=['vy'], rate_source='vy_dot', units='m/s')

        phase.add_boundary_constraint('vy', loc='final', upper=0.0)

        # Since we're using an optimization driver, an objective is required.
        # We'll minimize the final time in this case.
        phase.add_objective('time', loc='final')

        # Setup the OpenMDAO problem
        prob.setup()

        # Assign values to the times and states
        prob.set_val('traj.phase0.t_initial', 0.0)
        prob.set_val('traj.phase0.t_duration', 15.0)

        prob.set_val('traj.phase0.states:x', 0.0)
        prob.set_val('traj.phase0.states:y', 0.0)
        prob.set_val('traj.phase0.states:vx', 100.0)
        prob.set_val('traj.phase0.states:vy', 100.0)

        # Now we're using the optimization driver to iteratively run the model and vary the
        # phase duration until the final y value is 0.
        prob.run_model()

        # Perform an explicit simulation of our ODE from the initial conditions.
        sim_out = traj.simulate()

        # Plot the state values obtained from the phase timeseries objects in the simulation output.
        t_sol = prob.get_val('traj.phase0.timeseries.time')
        t_sim = sim_out.get_val('traj.phase0.timeseries.time')

        fig, axes = plt.subplots(4, 1)
        for i, state in enumerate(['x', 'y', 'vx', 'vy']):
            sol = axes[i].plot(t_sol, prob.get_val(f'traj.phase0.timeseries.states:{state}'), 'o')
            sim = axes[i].plot(t_sim, sim_out.get_val(f'traj.phase0.timeseries.states:{state}'), '-')
            axes[i].set_ylabel(state)
        axes[3].set_xlabel('time (s)')
        fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
        plt.tight_layout()
        plt.show()

    @save_for_docs
    def test_bvp_driver_derivs(self):
        import openmdao.api as om
        import dymos as dm
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')  # disable plotting to the screen

        from projectile_ode_with_partials import ProjectileODE

        # Instnatiate an OpenMDAO Problem instance.
        prob = om.Problem()

        # We need an optimization driver.  To solve this simple problem ScipyOptimizerDriver will work.
        prob.driver = om.ScipyOptimizeDriver()

        # Instantiate a Dymos Trajectory and add it to the Problem model.
        traj = dm.Trajectory()
        prob.model.add_subsystem('traj', traj)

        # Instantiate a Phase and add it to the Trajectory.
        # Here the transcription is necessary but not particularly relevant.
        phase = dm.Phase(ode_class=ProjectileODE, transcription=dm.Radau(num_segments=10))
        traj.add_phase('phase0', phase)

        # Tell Dymos that the time duration of the Phase should not be fixed (its a design variable
        # for the optimization). The duration of a phase may be negative but it should never be zero.
        phase.set_time_options(fix_initial=True, duration_bounds=(5, 50))

        # Tell Dymos the states to be propagated using the given ODE.
        phase.add_state('x', fix_initial=True, targets=None, rate_source='x_dot', units='m')
        phase.add_state('y', fix_initial=True, fix_final=True, targets=None, rate_source='y_dot', units='m')
        phase.add_state('vx', fix_initial=True, targets=['vx'], rate_source='vx_dot', units='m/s')
        phase.add_state('vy', fix_initial=True, targets=['vy'], rate_source='vy_dot', units='m/s')

        # Since we're using an optimization driver, an objective is required.
        # We'll minimize the final time in this case.
        phase.add_objective('time', loc='final')

        # Setup the OpenMDAO problem
        prob.setup()

        # Assign values to the times and states
        prob.set_val('traj.phase0.t_initial', 0.0)
        prob.set_val('traj.phase0.t_duration', 15.0)

        # Note we're now using the phase interpolate method to linearly interpolate values
        # of x and vy onto the "state input" nodes.
        # Without doing so, SLSQP is unable to converge the solution, but more capable optimizers
        # like SNOPT and IPOPT will work in this case.
        prob.set_val('traj.phase0.states:x', phase.interpolate(ys=(0, 100), nodes='state_input'))
        prob.set_val('traj.phase0.states:y', 0.0)
        prob.set_val('traj.phase0.states:vx', 100.0)
        prob.set_val('traj.phase0.states:vy', phase.interpolate(ys=(100, -100), nodes='state_input'))

        # Now we're using the optimization driver to iteratively run the model and vary the
        # phase duration until the final y value is 0.
        prob.run_driver()

        # Perform an explicit simulation of our ODE from the initial conditions.
        sim_out = traj.simulate()

        # Plot the state values obtained from the phase timeseries objects in the simulation output.
        t_sol = prob.get_val('traj.phase0.timeseries.time')
        t_sim = sim_out.get_val('traj.phase0.timeseries.time')

        fig, axes = plt.subplots(4, 1)
        for i, state in enumerate(['x', 'y', 'vx', 'vy']):
            sol = axes[i].plot(t_sol, prob.get_val(f'traj.phase0.timeseries.states:{state}'), 'o')
            sim = axes[i].plot(t_sim, sim_out.get_val(f'traj.phase0.timeseries.states:{state}'), '-')
            axes[i].set_ylabel(state)
        axes[3].set_xlabel('time (s)')
        fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
