import unittest
from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.interpolate import LagrangeBarycentricInterpolant

import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.utils.lagrange import lagrange_matrices
from dymos.utils.hermite import hermite_matrices
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.switch_backend('Agg')  # disable plotting to the screen

from dymos.examples.oscillator.doc.oscillator_ode import OscillatorODE

NUM_SEG = 1
ORDER = 3


class FallEOM(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_output('vy_dot', shape=(nn,), units='m/s**2')
        self.add_output('y_dot', shape=(nn,), units='m/s')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='y_dot', wrt='vy', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        vy = inputs['vy']
        outputs['y_dot'] = vy
        outputs['vy_dot'] = -9.80665

# Instantiate an OpenMDAO Problem instance.
prob = om.Problem()

# We need an optimization driver.  To solve this simple problem ScipyOptimizerDriver will work.
prob.driver = om.ScipyOptimizeDriver()
prob.driver.add_recorder(om.SqliteRecorder('collocation_figures.db'))
prob.driver.recording_options['includes'] = ['*dstau*', '*rhs_col*', '*staterate*', '*timeseries*', '*timeseries2*', '*rhs_disc*']

# Instantiate a Dymos Trajectory and add it to the Problem model.
traj = dm.Trajectory()
prob.model.add_subsystem('traj', traj)

# Instantiate a Phase and add it to the Trajectory.
phase = dm.Phase(ode_class=FallEOM, transcription=dm.GaussLobatto(num_segments=NUM_SEG, order=ORDER, compressed=False))
traj.add_phase('phase0', phase)

# Tell Dymos that the duration of the phase is bounded.
phase.set_time_options(fix_initial=True, fix_duration=False)

# Tell Dymos the states to be propagated using the given ODE.
phase.add_state('y', fix_initial=True, fix_final=True, rate_source='y_dot', targets=None, units='m')
phase.add_state('vy', fix_initial=True, rate_source='vy_dot', targets=['vy'], units='m/s')

# Since we're using an optimization driver, an objective is required.  We'll minimize the final time in this case.
phase.add_objective('time', loc='final')

phase.add_timeseries('timeseries2', transcription=dm.GaussLobatto(num_segments=10, order=3))

# Setup the OpenMDAO problem
prob.setup()

# Assign values to the times and states
prob.set_val('traj.phase0.t_initial', 0.0)
prob.set_val('traj.phase0.t_duration', 10.0)

prob.set_val('traj.phase0.states:y', phase.interpolate(ys=[100.0, 0.0], nodes='state_input'))
prob.set_val('traj.phase0.states:vy', phase.interpolate(ys=[0, -50], nodes='state_input'))


# Now we're using the optimization driver to iteratively run the model and vary the
# phase duration until the final y value is 0.
prob.run_driver()

# prob.model.list_outputs(print_arrays=True)

state_disc_idxs = phase.options['transcription'].grid_data.subset_node_indices['state_disc']
col_idxs = phase.options['transcription'].grid_data.subset_node_indices['col']

#
# Initialize the plots
#
fig, axes = plt.subplots(nrows=2, ncols=1)

axes[0].set_ylabel('$y$ (m)')
axes[1].set_ylabel('$v_y$ (m/s)')
axes[1].set_xlabel('time (s)')

cr = om.CaseReader('collocation_figures.db')


def plot_tangent(x, y, slope, ax, dx=1, *args, **kwargs):
    x = x.ravel()
    y = y.ravel()
    slope = slope.ravel()
    x0 = x - dx
    xf = x + dx
    y0 = y - slope * dx
    yf = y + slope * dx
    ax.plot((x0, xf), (y0, yf), *args, **kwargs)

for i, case_name in enumerate(cr.list_cases(source='driver', out_stream=None)):
    case = cr.get_case(case_name)
    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].set_xlim(-1, 11)
    axes[1].set_xlim(-1, 11)
    axes[0].set_ylim(-50, 150)
    axes[1].set_ylim(-60, 10)

    axes[0].set_ylabel('$y$ (m)')
    axes[1].set_ylabel('$v_y$ (m/s)')
    axes[1].set_xlabel('time (s)')

    #
    # Get the dense values from the secondary timeseries
    #

    # time_dense = case.get_val('traj.phase0.timeseries2.time')
    # y_dense = case.get_val('traj.phase0.timeseries2.states:y')
    # vy_dense = case.get_val('traj.phase0.timeseries2.states:vy')
    #
    # axes[0].plot(time_dense, y_dense, linestyle=':', color='#CCCCCC')
    # axes[1].plot(time_dense, vy_dense, linestyle=':', color='#CCCCCC')

    #
    # Get the node values
    #

    time = case.get_val('traj.phase0.timeseries.time')
    y = case.get_val('traj.phase0.timeseries.states:y')
    vy = case.get_val('traj.phase0.timeseries.states:vy')

    idxs = state_disc_idxs
    y_disc = y[idxs]
    vy_disc = vy[idxs]
    axes[0].plot(time[idxs], y_disc, 'o', ms=8)
    axes[1].plot(time[idxs], vy_disc, 'o', ms=8)

    idxs = col_idxs
    y_col = y[idxs]
    vy_col = vy[idxs]
    axes[0].plot(time[idxs], y_col, marker=(3, 0, 0), ms=8)
    axes[1].plot(time[idxs], vy_col, marker=(3, 0, 0), ms=8)

    #
    # Get the ODE outputs and the interpolated state rates
    #
    prob.model.list_outputs(includes=['*dstau*', '*staterate_col*', '*rhs_col*', '*rhs_disc*', '*defect*'], prom_name=True, print_arrays=True, units=True)
    yprime_col = case.get_val('traj.phase0.state_interp.staterate_col:y')
    vyprime_col = case.get_val('traj.phase0.state_interp.staterate_col:vy')

    ydot_disc = case.get_val('traj.phase0.rhs_disc.y_dot')
    vydot_disc = case.get_val('traj.phase0.rhs_disc.vy_dot')

    ydot_col = case.get_val('traj.phase0.rhs_col.y_dot')
    vydot_col = case.get_val('traj.phase0.rhs_col.vy_dot')

    plot_tangent(time[idxs, 0], y[idxs, 0], yprime_col, axes[0], color='b')
    plot_tangent(time[idxs, 0], y[idxs, 0], ydot_col, axes[0], color='r')

    plot_tangent(time[idxs, 0], vy[idxs, 0], yprime_col, axes[1], color='b')
    plot_tangent(time[idxs, 0], vy[idxs, 0], ydot_col, axes[1], color='r')

    #
    # Plot the interpolating polynomial
    #
    node_locs = phase.options['transcription'].grid_data.node_stau
    p = LagrangeBarycentricInterpolant(node_locs, shape=(1,))
    p.setup(time[0], time[-1], y)

    time_dense = np.linspace(time[0], time[-1], 100)
    y_dense = np.zeros(100)
    vy_dense = np.zeros(100)
    for i, t in enumerate(time_dense):
        y_dense[i] = p.eval(t)

    p.setup(time[0], time[-1], vy)
    for i, t in enumerate(time_dense):
        vy_dense[i] = p.eval(t)

    axes[0].plot(time_dense, y_dense, ':', color='#cccccc', zorder=-1000)
    axes[1].plot(time_dense, vy_dense, ':', color='#cccccc', zorder=-1000)

    # #
    # # Build hermite matrices
    # #
    # time_dense = np.linspace(time[0], time[-1], 100)
    # dt_dstau = case.get_val('traj.phase0.dt_dstau')[0]
    # A_i, B_i, A_d, B_d = hermite_matrices(time[state_disc_idxs], time_dense)
    # y_dense = np.dot(A_i, y_disc) + dt_dstau * np.dot(B_i, ydot_disc)
    # vy_dense = np.dot(A_i, vy_disc) + dt_dstau * np.dot(B_i, vydot_disc)
    # axes[0].plot(time_dense, y_dense, '-', color='#CCCCCC')
    # axes[1].plot(time_dense, vy_dense, '-', color='#CCCCCC')

    print('y_slope =', (y[-1] - y[0]) / (time[-1] - time[0]))
    print('vy_slope =', (vy[-1] - vy[0]) / (time[-1] - time[0]))






    # axes[0].quiver(time[idxs], y[idxs], [1], yprime_col, angles='xy', scale_units='xy', scale=1)
    # axes[0].quiver(time[idxs], y[idxs], [1], ydot_col, angles='xy', scale_units='xy', scale=1)
    #
    # axes[1].quiver(time[idxs], vy[idxs], [1], vyprime_col, angles='xy', scale_units='xy', scale=1)
    # axes[1].quiver(time[idxs], vy[idxs], [1], vydot_col, angles='xy', scale_units='xy', scale=1)

plt.show()


