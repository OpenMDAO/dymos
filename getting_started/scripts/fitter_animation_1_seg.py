import unittest
from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.interpolate import LagrangeBarycentricInterpolant

import numpy as np
import openmdao.api as om
import dymos as dm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.switch_backend('Agg')  # disable plotting to the screen

from dymos.examples.oscillator.doc.oscillator_ode import OscillatorODE

NUM_SEG = 1
ORDER = 5
NUM_FRAMES = 20
ORDER = 3

# Instantiate an OpenMDAO Problem instance.
prob = om.Problem()

# We need an optimization driver.  To solve this simple problem ScipyOptimizerDriver will work.
prob.driver = om.ScipyOptimizeDriver()

# Instantiate a Dymos Trajectory and add it to the Problem model.
traj = dm.Trajectory()
prob.model.add_subsystem('traj', traj)

# Instantiate a Phase and add it to the Trajectory.
phase = dm.Phase(ode_class=OscillatorODE, transcription=dm.Radau(num_segments=NUM_SEG, order=ORDER, compressed=False))
traj.add_phase('phase0', phase)

# Tell Dymos that the duration of the phase is bounded.
phase.set_time_options(fix_initial=True, fix_duration=True)

# Tell Dymos the states to be propagated using the given ODE.
phase.add_state('x', fix_initial=True, rate_source='v', targets=['x'], units='m')
phase.add_state('v', fix_initial=True, rate_source='v_dot', targets=['v'], units='m/s')

# The spring constant, damping coefficient, and mass are inputs to the system that are constant throughout the phase.
phase.add_parameter('k', units='N/m', targets=['k'])
phase.add_parameter('c', units='N*s/m', targets=['c'])
phase.add_parameter('m', units='kg', targets=['m'])

# secondary "dense" timeseries
phase.add_timeseries('timeseries2', transcription=dm.Radau(num_segments=NUM_SEG, order=41, compressed=False))

# Since we're using an optimization driver, an objective is required.  We'll minimize the final time in this case.
phase.add_objective('time', loc='final')

# Setup the OpenMDAO problem
prob.setup()

# Assign values to the times and states
prob.set_val('traj.phase0.t_initial', 0.0)
prob.set_val('traj.phase0.t_duration', 5.0)

prob.set_val('traj.phase0.states:x', 10.0)
prob.set_val('traj.phase0.states:v', 0.0)

prob.set_val('traj.phase0.parameters:k', 1.0)
prob.set_val('traj.phase0.parameters:c', 0.5)
prob.set_val('traj.phase0.parameters:m', 1.0)

starts= {}

starts['x'] = prob.get_val('traj.phase0.states:x')
starts['v'] = prob.get_val('traj.phase0.states:v')

# Now we're using the optimization driver to iteratively run the model and vary the
# phase duration until the final y value is 0.
prob.run_driver()

# Perform an explicit simulation of our ODE from the initial conditions.
sim_out = traj.simulate(times_per_seg=50)

# Plot the state values obtained from the phase timeseries objects in the simulation output.
t_sol = prob.get_val('traj.phase0.timeseries.time')
t_sim = sim_out.get_val('traj.phase0.timeseries.time')
t_dense = prob.get_val('traj.phase0.timeseries2.time')

all_idxs = phase.options['transcription'].grid_data.subset_segment_indices['all']

states = ['x', 'v']
solutions = {}
histories = {}

for i, state in enumerate(states):
    state_sol = prob.get_val(f'traj.phase0.timeseries.states:{state}')
    state_sim = sim_out.get_val(f'traj.phase0.timeseries.states:{state}')
    state_dense = prob.get_val(f'traj.phase0.timeseries2.states:{state}')
    # sol = axes[i].plot(t_sol, state_sol, 'o')
    # sim = axes[i].plot(t_sim, state_sim, '-')
    # dense = axes[i].plot(t_dense, state_dense, '--', color='#CCCCCC')
    # axes[i].set_ylabel(state)
    solutions[state] = state_sol
    histories[state] = np.linspace(starts[state], solutions[state], NUM_FRAMES)

fig, axes = plt.subplots(len(states), 1)

for i in range(NUM_FRAMES):
    for j, state in enumerate(states):
        prob.set_val(f'traj.phase0.states:{state}', histories[state][i, ...])

    prob.run_model()

    for j, state in enumerate(states):
        state_sol = prob.get_val(f'traj.phase0.timeseries.states:{state}')
        state_dense = prob.get_val(f'traj.phase0.timeseries2.states:{state}')
        sol = axes[j].plot(t_sol, state_sol, 'o')
        dense = axes[j].plot(t_dense, state_dense, '--', color='#CCCCCC')

t_data = []
x_data = []
v_data = []

fig, axes = plt.subplots(len(states), 1)

x_sol_line, = axes[0].plot(t_data, x_data, 'o')
v_sol_line, = axes[1].plot(t_data, v_data, 'o')
# x_dense_line, = axes[0].plot(t_data, x_data, '--')
# v_dense_line, = axes[1].plot(t_data, v_data, '--')
lines = x_sol_line, v_sol_line

def init():
    return lines

def update(frame):
    # xdata.append(frame)
    # ydata.append(np.sin(frame))

    x_sol_line, x_dense_line, v_sol_line, v_dense_line = lines

    for j, state in enumerate(states):
        prob.set_val(f'traj.phase0.states:{state}', histories[state][frame, ...])

    prob.run_model()

    x_sol = prob.get_val('traj.phase0.timeseries.states:x')
    # x_dense = prob.get_val('traj.phase0.timeseries2.states:x')

    v_sol = prob.get_val('traj.phase0.timeseries.states:v')
    # v_dense = prob.get_val('traj.phase0.timeseries2.states:v')

    x_sol_line.set_ydata(x_sol)
    # x_dense_line.set_ydata(x_dense)
    v_sol_line.set_ydata(v_sol)
    # v_dense_line.set_ydata(v_dense)

    return x_sol_line, v_sol_line
    # return

ani = FuncAnimation(fig, update, frames=range(NUM_FRAMES),
                    init_func=init, blit=True)

plt.show()

    # for j in range(NUM_SEG):
    #     start_idx, end_idx = all_idxs[j, :]
    #     t_seg_start = t_sol.ravel()[start_idx]
    #     t_seg_end = t_sol.ravel()[end_idx - 1]
        # print(t_seg_start, t_seg_end)
        # print(state_sol[start_idx:end_idx])
        # print(t_seg_start, t_seg_end)
        # nodes_stau = phase.options['transcription'].grid_data.node_stau[start_idx:end_idx]
        # lbi = LagrangeBarycentricInterpolant(nodes=nodes_stau, shape=(1,))
        # # print(nodes_stau)
        # lbi.setup(t_seg_start, t_seg_end, state_sol[start_idx:end_idx])
        # t_interp = np.linspace(t_seg_start, t_seg_end, 100)
        # with np.printoptions(linewidth=1024, edgeitems=500):
        #     sol_interp = lbi.eval(t_interp)
        #     soldot_interp = lbi.eval_deriv(0.0)
        # axes[i].plot(t_interp, sol_interp, 'k-')



# axes[-1].set_xlabel('time (s)')
# # fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
# plt.tight_layout()
# plt.show()
fig, axes = plt.subplots(len(states), 1)
for i, state in enumerate(states):
    state_sol = prob.get_val(f'traj.phase0.timeseries.states:{state}')
    state_sim = sim_out.get_val(f'traj.phase0.timeseries.states:{state}')
    sol = axes[i].plot(t_sol, state_sol, 'o')
    # sim = axes[i].plot(t_sim, state_sim, '-')
    axes[i].set_ylabel(state)

    for j in range(NUM_SEG):
        start_idx, end_idx = all_idxs[j, :]
        t_seg_start = t_sol.ravel()[start_idx]
        t_seg_end = t_sol.ravel()[end_idx - 1]
        # print(t_seg_start, t_seg_end)
        # print(state_sol[start_idx:end_idx])
        # print(t_seg_start, t_seg_end)
        nodes_stau = phase.options['transcription'].grid_data.node_stau[start_idx:end_idx]
        lbi = LagrangeBarycentricInterpolant(nodes=nodes_stau, shape=(1,))
        # print(nodes_stau)
        lbi.setup(t_seg_start, t_seg_end, state_sol[start_idx:end_idx])
        t_interp = np.linspace(t_seg_start, t_seg_end, 100)
        with np.printoptions(linewidth=1024, edgeitems=500):
            sol_interp = lbi.eval(t_interp)
        axes[i].plot(t_interp, sol_interp, 'k-')



axes[-1].set_xlabel('time (s)')
# fig.legend((sol[0], sim[0]), ('solution', 'simulation'), 'lower right', ncol=2)
plt.tight_layout()
plt.show()
