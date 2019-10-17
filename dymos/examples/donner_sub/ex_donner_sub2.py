import numpy as np
import openmdao.api as om
import dymos as dm


from dymos.examples.donner_sub.donner_sub_ode2 import DonnerSubODE


def solve_donner_multisub_problem(num_subs=1):

    p = om.Problem(model=om.Group())

    traj = dm.Trajectory()

    phase = dm.Phase(ode_class=DonnerSubODE,
                     ode_init_kwargs={'num_subs': num_subs},
                     transcription=dm.Radau(num_segments=30, order=3, compressed=False))

    phase.set_time_options(units=None, targets=['time'], fix_initial=True, duration_bounds=(0.001, 100))
    phase.add_state('lat', rate_source='dlat_dt', targets=['lat'], fix_initial=True, fix_final=True, defect_ref=0.01)
    phase.add_state('lon', rate_source='dlon_dt', targets=['lon'], fix_initial=True, fix_final=True, defect_ref=0.01)

    phase.add_design_parameter('speed', targets=['speed'], opt=True, lower=0.1)
    phase.add_control('heading', targets=['heading'], units='deg', lower=0, upper=180, rate_continuity=True)

    for i in range(num_subs):
        phase.add_path_constraint(f'sub_{i}_range', lower=0, scaler=1)
        phase.add_timeseries_output(f'ship_radius_{i}', shape=(1,))
        phase.add_timeseries_output(f'sub_{i}_range', shape=(1,))
    # phase.add_path_constraint(f'sub_{num_subs-1}_range', lower=0)

    phase.add_objective(name='speed', loc='initial')

    traj.add_phase('phase0', phase=phase)
    p.model.add_subsystem('traj', traj)

    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.opt_settings['iSumm'] = 6
    p.driver.declare_coloring()

    print('recording to', f'donner_multisub_{num_subs - 1}.sql')
    recorder = om.SqliteRecorder(f'donner_multisub_{num_subs - 1}.sql')
    p.driver.add_recorder(recorder)

    p.setup()

    p.set_val('traj.phase0.t_initial', value=0)
    p.set_val('traj.phase0.t_duration', value=0.2)
    p.set_val('traj.phase0.states:lat', value=phase.interpolate(ys=[0, 0], nodes='state_input'))
    p.set_val('traj.phase0.states:lon', value=phase.interpolate(ys=[-1, 1], nodes='state_input'))
    p.set_val('traj.phase0.controls:heading', value=phase.interpolate(ys=[100, 80], nodes='control_input'), units='deg')
    p.set_val('traj.phase0.design_parameters:speed', value=2.0)

    if num_subs > 1:
        cr = om.CaseReader(f'donner_multisub_{num_subs - 2}.sql')
        dm.load_case(p, cr.get_case(-1))

    p.run_driver()

    return p.get_val('traj.phase0.design_parameters:speed')[0, 0]

    # exp_out = traj.simulate()
    #
    # import matplotlib.pyplot as plt
    #
    # speed = p.get_val('traj.phase0.timeseries.design_parameters:speed')
    #
    # lat = p.get_val('traj.phase0.timeseries.states:lat')
    # lon = p.get_val('traj.phase0.timeseries.states:lon')
    #
    # lat_x = exp_out.get_val('traj.phase0.timeseries.states:lat')
    # lon_x = exp_out.get_val('traj.phase0.timeseries.states:lon')
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.set_aspect('equal')
    # ax.plot(lon, lat, 'ro')
    # ax.plot(lon_x, lat_x, 'k-')
    # ax.text(0, -0.1, f'speed = {speed[0, 0]:6.4f}')
    # plt.show()


if __name__ == '__main__':
    n = 6
    speeds = []
    for i in range(n):
        speeds.append(solve_donner_multisub_problem(num_subs=i+1))

    poly = np.polyfit(np.arange(n)+1, speeds, deg=1)
    print(poly)

    import matplotlib.pyplot as plt
    plt.plot(np.arange(n)+1, speeds, marker='o', linestyle=None)

    x_fit = np.linspace(1, 6, 100)
    y_fit = np.poly1d(poly)(x_fit)
    plt.plot(x_fit, y_fit, 'k--')


    plt.xlabel('number of subs')
    plt.ylabel('minimum speed')
    plt.show()
