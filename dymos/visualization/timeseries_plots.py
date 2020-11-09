import os
import warnings

import numpy as np

import openmdao.api as om

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import matplotlib.pyplot as plt


def get_timeseries_units(cr, varname):
    root_children = cr.problem_metadata['tree']['children']

    traj_name = varname.split('.')[0]
    for item in root_children:
        if item['name'] == traj_name:
            traj_dict = item
            break

    # get phases
    for item in traj_dict['children']:
        if item['name'] == 'phases':
            phases_dict = item
            break

    # get the phase we want
    phase_name = varname.split('.')[1]
    for item in phases_dict['children']:
        if item['name'] == phase_name:
            phase_dict = item
            break

    # get the timeseries
    for item in phase_dict['children']:
        if item['name'] == 'timeseries':
            timeseries_dict = item
            break

    # get the variable
    for item in timeseries_dict['children']:
        if item['name'] == 'states:' + varname.split(':')[-1] :
            var_units = item['units']
        if item['name'] == 'time' :
            time_units = item['units']


    return time_units, var_units

def timeseries_plots(recorder_filename, plot_simulation=False, simulation_record_file=None, plot_dir="plots"):
    plot_dir_path = os.path.join(os.getcwd(), plot_dir)
    if not os.path.isdir(plot_dir_path):
        os.mkdir(plot_dir_path)

    plt.switch_backend('Agg')


    # Check for possible user error. If you provide a simulation_record_file but plot_simulation is not True, not plot of simulation
    if not plot_simulation and simulation_record_file is not None:
        warnings.warn('Setting simulation_record_file but not setting plot_simulation will not result in plotting simulation data')

    # If user wants plot of simulation, need to provide a source of the data in a recorder file
    if plot_simulation and simulation_record_file is None:
        raise ValueError('If plot_simulation is True, simulation_record_file must be path to simulation case recorder file, not None')

    figsize = (10, 8)

    cr = om.CaseReader(recorder_filename)

    # qqq why did Rob use "final" to get the case ?
    system_solution_cases = cr.list_cases('problem')
    last_solution_case = cr.get_case(system_solution_cases[-1])

    if plot_simulation:
        cr_simulate = om.CaseReader(simulation_record_file)
        system_simulation_cases = cr_simulate.list_cases('problem')
        last_simulation_case = cr_simulate.get_case(system_simulation_cases[-1])

    # Loop over all the variables and make individual plots
    for varname_full in last_solution_case.outputs.keys():
        if "timeseries.states:" in varname_full:
            # get solution values
            time_name = ".".join(varname_full.split(".")[:-1])+".time"
            var_val = last_solution_case.outputs[varname_full]
            time_val = last_solution_case.outputs[time_name]

            # get simulation values, if plotting simulation
            if plot_simulation:
                var_val_simulate = last_simulation_case.outputs[varname_full]
                time_val_simulate = last_simulation_case.outputs[time_name]

            # get plot labels
            varname = varname_full.split(':')[-1]
            time_units, var_units = get_timeseries_units(cr,varname_full)
            time_label = f'time ({time_units})'
            var_label = f'{varname} ({var_units})'
            title = 'Timeseries ' + ".".join(varname_full.split(".")[:2])

            # plot values
            fig, ax = plt.subplots()
            ax.plot(time_val, var_val, 'ro', label='solution')
            if plot_simulation:
                ax.plot(time_val_simulate, var_val_simulate, 'b-', label='simulation')

            # add labels, title, and legend
            ax.set_xlabel(time_label)
            ax.set_ylabel(var_label)
            fig.suptitle(title)
            fig.legend(loc='lower center', ncol=2, shadow=True)
            plt.subplots_adjust(bottom=0.17, top=0.9)

            # save to file
            plot_file_path = os.path.join(plot_dir_path, f'test_{varname}.png' )
            plt.savefig(plot_file_path)
