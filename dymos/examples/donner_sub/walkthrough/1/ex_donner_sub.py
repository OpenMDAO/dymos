import openmdao.api as om
import dymos as dm


from dymos.examples.donner_sub.donner_sub_ode import DonnerSubODE

p = om.Problem(model=om.Group())

traj = dm.Trajectory()

phase = dm.Phase(ode_class=DonnerSubODE, transcription=dm.GaussLobatto(num_segments=30, order=3, compressed=False))

phase.set_time_options(units=None, targets=['time'], fix_initial=True, duration_bounds=(0.1, 100))
phase.add_state('lat', rate_source='dlat_dt', targets=['lat'], fix_initial=True, fix_final=True)
phase.add_state('lon', rate_source='dlon_dt', targets=['lon'], fix_initial=True, fix_final=True)

phase.add_design_parameter('speed', targets=['speed'], opt=True, upper=5, lower=0.1)
phase.add_control('heading', targets=['heading'], units='rad')

phase.add_path_constraint('sub_range', lower=0)

phase.add_timeseries_output('r_ship', shape=(1,))
phase.add_timeseries_output('sub_range', shape=(1,))

phase.add_objective(name='speed', loc='initial')

traj.add_phase('phase0', phase=phase)
p.model.add_subsystem('traj', traj)

p.driver = om.pyOptSparseDriver()
p.driver.options['optimizer'] = 'SNOPT'
p.driver.opt_settings['iSumm'] = 6
p.driver.declare_coloring()

p.setup(force_alloc_complex=True)

p.set_val('traj.phase0.t_initial', value=0)
p.set_val('traj.phase0.t_duration', value=1.0)
p.set_val('traj.phase0.states:lat', value=phase.interpolate(ys=[0, 0], nodes='state_input'))
p.set_val('traj.phase0.states:lon', value=phase.interpolate(ys=[-1, 1], nodes='state_input'))
p.set_val('traj.phase0.controls:heading', value=phase.interpolate(ys=[90, 90], nodes='control_input'), units='deg')
p.set_val('traj.phase0.design_parameters:speed', value=2.0)

p.run_driver()

exp_out = traj.simulate()

import matplotlib.pyplot as plt

speed = p.get_val('traj.phase0.timeseries.design_parameters:speed')

lat = p.get_val('traj.phase0.timeseries.states:lat')
lon = p.get_val('traj.phase0.timeseries.states:lon')

lat_x = exp_out.get_val('traj.phase0.timeseries.states:lat')
lon_x = exp_out.get_val('traj.phase0.timeseries.states:lon')

fig, ax = plt.subplots(1, 1)
ax.set_aspect('equal')
ax.plot(lon, lat, 'ro')
ax.plot(lon_x, lat_x, 'k-')
ax.text(0, -0.1, f'speed = {speed[0, 0]:6.4f}')
plt.show()
