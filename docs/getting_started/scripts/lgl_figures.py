import unittest
from dymos.utils.doc_utils import save_for_docs
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.interpolate import LagrangeBarycentricInterpolant

import numpy as np
import openmdao.api as om
import dymos as dm
from dymos.utils.lagrange import lagrange_matrices
from dymos.utils.hermite import hermite_matrices
from dymos.utils.lgl import lgl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.switch_backend('Agg')  # disable plotting to the screen

from dymos.examples.oscillator.doc.oscillator_ode import OscillatorODE

NUM_SEG = 1
ORDER = 3




def plot_tangent(x, y, slope, ax, dx=1, scale=1.0, *args, **kwargs):
    x = x.ravel()
    y = y.ravel()
    slope = slope.ravel()
    x0 = x - dx * scale
    xf = x + dx * scale
    y0 = y - slope * dx * scale
    yf = y + slope * dx * scale
    ax.plot((x0, xf), (y0, yf), *args, **kwargs)


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
prob.set_val('traj.phase0.t_duration', 2.0)

prob.set_val('traj.phase0.states:y', phase.interp('y', [100.0, 0.0]))
prob.set_val('traj.phase0.states:vy', phase.interp('vy', [0, -50]))


# Now we're using the optimization driver to iteratively run the model and vary the
# phase duration until the final y value is 0.
prob.run_driver()

state_disc_idxs = phase.options['transcription'].grid_data.subset_node_indices['state_disc']
col_idxs = phase.options['transcription'].grid_data.subset_node_indices['col']

cr = om.CaseReader('collocation_figures.db')

for i, case_name in enumerate(cr.list_cases()):

    case = cr.get_case(case_name)
    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].set_xlim(-.5, 5)
    axes[1].set_xlim(-.5, 5)
    axes[0].set_ylim(-50, 150)
    axes[1].set_ylim(-60, 10)

    axes[0].set_ylabel('$y$ (m)')
    axes[1].set_ylabel('$v_y$ (m/s)')
    axes[1].set_xlabel('time (s)')

    # Plot the initial state
    time = case.get_val('traj.phase0.timeseries.time')
    y = case.get_val('traj.phase0.timeseries.states:y')
    vy = case.get_val('traj.phase0.timeseries.states:vy')
    dt_dstau = case.get_val('traj.phase0.dt_dstau')

    # Plot the rates at the discretization nodes

    idxs = state_disc_idxs
    y_disc = y[state_disc_idxs]
    vy_disc = vy[state_disc_idxs]
    axes[0].plot(time[state_disc_idxs], y_disc, 'o', ms=8)
    axes[1].plot(time[state_disc_idxs], vy_disc, 'o', ms=8)

    if i == 0:
        plt.savefig('lgl_animation_0.png')

    ydot = case.get_val('traj.phase0.timeseries.state_rates:y')
    vydot = case.get_val('traj.phase0.timeseries.state_rates:vy')

    plot_tangent(time[state_disc_idxs], y[state_disc_idxs], ydot[state_disc_idxs], axes[0], color='b', scale=.25)
    plot_tangent(time[state_disc_idxs], vy[state_disc_idxs], vydot[state_disc_idxs], axes[1], color='b', scale=.25)

    if i == 0:
        plt.savefig('lgl_animation_1.png')

    # Plot the interpolating polynomials
    t_dense = np.linspace(time[0], time[-1], 100)
    A_i, B_i, A_d, B_d = hermite_matrices(time[state_disc_idxs], t_dense)
    y_dense = (A_i.dot(y[state_disc_idxs]) + B_i.dot(ydot[state_disc_idxs]))
    vy_dense = A_i.dot(vy[state_disc_idxs]) + B_i.dot(vydot[state_disc_idxs])

    axes[0].plot(t_dense, y_dense, ms=None, ls=':')
    axes[1].plot(t_dense, vy_dense, ms=None, ls=':')

    # Plot the values and rates at the collocation nodes
    tau_s, _ = lgl(3)
    tau_disc = tau_s[0::2]
    tau_col = tau_s[1::2]
    A_i, B_i, A_d, B_d = hermite_matrices(tau_disc, tau_col)

    y_col = B_i.dot(ydot[state_disc_idxs]) * dt_dstau[col_idxs] + A_i.dot(y[state_disc_idxs])
    vy_col = B_i.dot(vydot[state_disc_idxs]) * dt_dstau[col_idxs] + A_i.dot(vy[state_disc_idxs])

    yprime_col = A_d.dot(y[state_disc_idxs]) / dt_dstau + B_d.dot(ydot[state_disc_idxs])
    vyprime_col = A_d.dot(vy[state_disc_idxs]) / dt_dstau + B_d.dot(vydot[state_disc_idxs])

    axes[0].plot(time[col_idxs], y_col, marker=(3, 0, 0), ms=8)
    axes[1].plot(time[col_idxs], vy_col, marker=(3, 0, 0), ms=8)

    plot_tangent(time[col_idxs], y[col_idxs], yprime_col, axes[0], color='b', scale=.25)
    plot_tangent(time[col_idxs], vy[col_idxs], vyprime_col, axes[1], color='b', scale=.25)

    if i == 0:
        plt.savefig('lgl_animation_2.png')

    # Plot the collocation node ODE state rates
    plot_tangent(time[col_idxs], y[col_idxs], ydot[col_idxs], axes[0], color='r', scale=.25)
    plot_tangent(time[col_idxs], vy[col_idxs], vydot[col_idxs], axes[1], color='r', scale=.25)

    if i == 0:
        plt.savefig('lgl_animation_3.png')

    plt.savefig(f'lgl_solution_{i}.png')

