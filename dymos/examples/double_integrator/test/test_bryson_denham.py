
import openmdao.api as om

import dymos as dm
from dymos.examples.double_integrator.test.bryson_denham_ode import BrysonDenhamODE


optimizer = 'SLSQP'
transcription = 'radau-ps'
compressed = False

p = om.Problem(model=om.Group())
p.driver = om.ScipyOptimizeDriver()
# p.driver.options['optimizer'] = optimizer
# if optimizer == 'SNOPT':
#     p.driver.opt_settings['Major iterations limit'] = 200
#     p.driver.opt_settings['iSumm'] = 6
# elif optimizer == 'SLSQP':
#     p.driver.opt_settings['MAXIT'] = 50
# elif optimizer == 'IPOPT':
#     p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
#     # p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
#     p.driver.opt_settings['print_level'] = 5
#     p.driver.opt_settings['linear_solver'] = 'mumps'
p.driver.declare_coloring()

if transcription == 'gauss-lobatto':
    t = dm.GaussLobatto(num_segments=10, order=3, compressed=compressed)
elif transcription == "radau-ps":
    t = dm.Radau(num_segments=10, order=3, compressed=compressed)
else:
    raise ValueError('invalid transcription')

traj = p.model.add_subsystem('traj', dm.Trajectory())

phase = traj.add_phase('phase0', dm.Phase(ode_class=BrysonDenhamODE, transcription=t))

phase.set_time_options(fix_initial=True, fix_duration=True, units='s')

phase.add_state('x', fix_initial=True, fix_final=True, rate_source='v', units='m')
phase.add_state('v', fix_initial=True, fix_final=True, rate_source='u', units='m/s')
phase.add_state('obj', fix_initial=True, fix_final=False, rate_source='usq', units='m**2/s**3')

phase.add_control('u', units='m/s**2', scaler=0.01, targets=['u'], continuity=False,
                  rate_continuity=False, rate2_continuity=False)

phase.add_path_constraint('x', upper=1/9)

# Minimize the integral of the square of the control
phase.add_objective('obj', loc='final', scaler=1)

p.model.linear_solver = om.DirectSolver()

p.setup(check=True)

p['traj.phase0.t_initial'] = 0.0
p['traj.phase0.t_duration'] = 1.0

p['traj.phase0.states:x'] = phase.interpolate(ys=[0, 0], nodes='state_input')
p['traj.phase0.states:obj'] = phase.interpolate(ys=[0, 0], nodes='state_input')
p['traj.phase0.states:v'] = phase.interpolate(ys=[1, -1], nodes='state_input')
p['traj.phase0.controls:u'] = phase.interpolate(ys=[0, 0], nodes='control_input')

p.run_driver()

t = p.get_val('traj.phase0.timeseries.time')
x = p.get_val('traj.phase0.timeseries.states:x')
v = p.get_val('traj.phase0.timeseries.states:v')
u = p.get_val('traj.phase0.timeseries.controls:u')
obj = p.get_val('traj.phase0.timeseries.states:obj')

exp_out = p.model.traj.simulate()

t_exp = exp_out.get_val('traj.phase0.timeseries.time')
x_exp = exp_out.get_val('traj.phase0.timeseries.states:x')
v_exp = exp_out.get_val('traj.phase0.timeseries.states:v')
u_exp = exp_out.get_val('traj.phase0.timeseries.controls:u')
obj_exp = exp_out.get_val('traj.phase0.timeseries.states:obj')

import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1)
axes[0].plot(t, x, marker='.', linestyle='none')
axes[1].plot(t, v, marker='.', linestyle='none')
axes[2].plot(t, u, marker='.', linestyle='none')
axes[3].plot(t, obj, marker='.', linestyle='none')

axes[0].plot(t_exp, x_exp, linestyle='-', color='k')
axes[1].plot(t_exp, v_exp, linestyle='-', color='k')
axes[2].plot(t_exp, u_exp, linestyle='-', color='k')
axes[3].plot(t_exp, obj_exp, linestyle='-', color='k')

plt.show()
