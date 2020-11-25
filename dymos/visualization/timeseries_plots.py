import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import openmdao.api as om


def _get_phases_node_in_problem_metadata(node, path=""):
    """
    Find the phases node in the Problem metadata hierarchy.

    There is one node in the hierarchy that has the name 'phases'. Finding this
    node will be used to find information about all the phases in the model.
    This is a recursive function.

    Parameters
    ----------
    node : list
        Node in Problem metadata hierarchy
    path : str
        The dotted string path name to the node in the Problem hierarchy. Used
        recursively to build up the path to a node

    Returns
    -------
    node, node_path : tuple of a list and a string
        Returns the node and path name to the node, if found. Otherwise, returns (None, None)
    """
    for item in node:
        if item['name'] == 'phases':
            return item, f'{path}'
        else:
            if 'children' in item:
                children = item['children']
                name = item['name']
                if path:
                    new_path = f'{path}:{name}'
                else:
                    new_path = name
                phases_node, phase_node_path = _get_phases_node_in_problem_metadata(children,
                                                                                    new_path)
                if phases_node:
                    return phases_node, phase_node_path
    return None, None


def _get_timeseries_units(cr, varname):
    """
    Get the units for a timeseries variable and also the time associated with that variable.

    Parameters
    ----------
    cr : CaseRecorder
        The CaseRecorder that contains the problem metadata
    varname : str
        The CaseRecorder that contains the problem metadata

    Returns
    -------
    units : tuple of two strings
        The units for the requested timeseries variable and also the time associated with that
        variable
    """
    root_children = cr.problem_metadata['tree']['children']

    # The `phases` node can be anywhere in the hierarchy so need to recurse
    phases_node = _get_phases_node_in_problem_metadata(root_children)

    time_units, var_units = None, None

    # get the phase child
    phase_name = varname.split('.')[0]
    for item in phases_node['children']:
        if item['name'] == phase_name:
            phase_dict = item
            break

    # get the timeseries child
    for item in phase_dict['children']:
        if item['name'] == 'timeseries':
            timeseries_dict = item
            break

    # get the variable child
    for item in timeseries_dict['children']:
        if item['name'] == 'states:' + varname.split(':')[-1]:
            var_units = item['units']
        if item['name'] == 'time':
            time_units = item['units']

    if var_units is None:
        raise RuntimeError(f'Units not found for variable: {varname}')
    if time_units is None:
        raise RuntimeError(f'Time units not found for variable: {varname}')

    return time_units, var_units

def timeseries_plots(solution_recorder_filename, simulation_record_file=None,
                     plot_dir="plots"):
    """
    Given timeseries data from case recorder files, make separate plots of each variable
    and store the plot files in the directory indicated by the variable plot_dir

    Parameters
    ----------
    solution_recorder_filename : str
        The path to the case recorder file containing solution data
    simulation_record_file : str or None (default:None)
        The path to the case recorder file containing simulation data. If not None,
        this implies that the data from it should be plotted
    plot_dir : str
        The path to the directory to which the plot files will be written
    """

    # get ready to generate plot files
    plot_dir_path = os.path.join(os.getcwd(), plot_dir)
    if not os.path.isdir(plot_dir_path):
        os.mkdir(plot_dir_path)

    cr = om.CaseReader(solution_recorder_filename)

    solution_cases = cr.list_cases('problem')
    last_solution_case = cr.get_case(solution_cases[-1])

    # If plotting simulation results, get the values for those variables
    if simulation_record_file:
        cr_simulate = om.CaseReader(simulation_record_file)
        system_simulation_cases = cr_simulate.list_cases('problem')
        last_simulation_case = cr_simulate.get_case(system_simulation_cases[-1])

    # we will use the problem metadata to get information about the phases and units
    root_children = cr.problem_metadata['tree']['children']

    # We have to key off the phases node. It will tell us
    #  what the prefix is for the variable names before the phases part of the variables
    #  and also let us know what the phase names are
    phases_node, phases_node_path = _get_phases_node_in_problem_metadata(root_children)

    # get phase names
    phase_names = [phase['name'] for phase in phases_node['children']]

    # Get the units and var names along the way
    var_units = {}
    time_units = {}
    for phase_node in phases_node['children']:  # Hopefully all phases have the same vars
        for phase_node_child in phase_node['children']:  # find the timeseries node
            if phase_node_child['name'] == 'timeseries':
                timeseries_node = phase_node_child
                break
        # get the time units first so they can be associated with all the variables
        for timeseries_node_child in timeseries_node['children']:
            if timeseries_node_child['name'] == 'time':
                units_for_time = timeseries_node_child['units']
                break
        for timeseries_node_child in timeseries_node['children']:
            varname = timeseries_node_child['name'][len('states:'):]
            if timeseries_node_child['name'].startswith("states:"):
                var_units[varname] = timeseries_node_child['units']
                time_units[varname] = units_for_time
    varnames = list(var_units.keys())

    # Check to see if there is anything to plot
    if len(varnames) == 0:
        warnings.warn('There are no timeseries variables to plot', RuntimeWarning)
        return

    # get ready to plot
    plt.switch_backend('Agg')
    # use a colormap with 20 values
    cm = plt.cm.get_cmap('tab20')

    for ivar, var_name in enumerate(varnames):
        # start a new plot
        fig, ax = plt.subplots()

        # Get the labels
        time_label = f'time ({time_units[var_name]})'
        var_label = f'{var_name} ({var_units[var_name]})'
        title = f'Timeseries: timeseries.states:{var_name}'

        # add labels, title, and legend
        ax.set_xlabel(time_label)
        ax.set_ylabel(var_label)
        fig.suptitle(title)

        # Plot each phase
        for iphase, phase_name in enumerate(phase_names):
            if phases_node_path:
                var_name_full = f'{phases_node_path}.{phase_name}.timeseries.states:{var_name}'
                time_name = f'{phases_node_path}.{phase_name}.timeseries.time'
            else:
                var_name_full = f'{phase_name}.timeseries.states:{var_name}'
                time_name = f'{phase_name}.timeseries.time'

            # Get values
            var_val = last_solution_case.outputs[var_name_full]
            time_val = last_solution_case.outputs[time_name]

            # Plot the data
            color = cm.colors[iphase % 20]

            ax.plot(time_val, var_val, marker='o', linestyle='None', label='solution', color=color)

            # get simulation values, if plotting simulation
            if simulation_record_file:
                # if the phases_node_path is empty, need to pre-pend names with "sim_traj."
                #   as that is pre-pended in Trajectory.simulate code
                sim_prefix = "" if phases_node_path else "sim_traj."
                var_val_simulate = last_simulation_case.outputs[sim_prefix + var_name_full]
                time_val_simulate = last_simulation_case.outputs[sim_prefix + time_name]
                ax.plot(time_val_simulate, var_val_simulate, linestyle='--', label='simulation',
                        color=color)

        # Create two legends
        #   Solution/Simulation legend
        solution_line = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                      label='Solution')
        if simulation_record_file:
            simulation_line = mlines.Line2D([], [], color='black', linestyle='--',
                                            label='Simulation')
            sol_sim_legend = plt.legend(handles=[solution_line, simulation_line],
                                        loc='upper center', bbox_to_anchor=(0., -0.12), shadow=True)
        else:
            sol_sim_legend = plt.legend(handles=[solution_line],
                                        loc='upper center', bbox_to_anchor=(0., -0.12),
                                        shadow=True)
        plt.gca().add_artist(sol_sim_legend)

        #   Phases legend
        handles = []
        for iphase, phase_name in enumerate(phase_names):
            patch = mpatches.Patch(color=cm.colors[iphase], label=phase_name)
            handles.append(patch)
        plt.legend(handles=handles, loc='upper right', ncol=len(phase_names), shadow=True,
                   bbox_to_anchor=(1.1, -0.12), title='Phases')

        plt.subplots_adjust(bottom=0.23, top=0.9)

        # save to file
        plot_file_path = os.path.join(plot_dir_path, f'test_{var_name}.png')
        plt.savefig(plot_file_path)
