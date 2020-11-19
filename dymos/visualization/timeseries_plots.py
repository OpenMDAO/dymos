import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import openmdao.api as om

def _get_phases_node_in_problem_metadata(metadata,path=""):
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
        The units for the requested timeseries variable and also the time associated with that variable
    """
    for item in metadata:
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
                phases_node, phase_node_path  = _get_phases_node_in_problem_metadata(children,new_path)
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
        The units for the requested timeseries variable and also the time associated with that variable
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
        if item['name'] == 'states:' + varname.split(':')[-1] :
            var_units = item['units']
        if item['name'] == 'time' :
            time_units = item['units']

    if var_units is None:
        raise RuntimeError(f'Units not found for variable: {varname}')
    if time_units is None:
        raise RuntimeError(f'Time units not found for variable: {varname}')

    return time_units, var_units

def timeseries_plots(recorder_filename, plot_simulation=False, simulation_record_file=None, plot_dir="plots"):
    """
    Get the units for a timeseries variable and also the time associated with that variable.

    Parameters
    ----------
    recorder_filename : str
        The path to the CaseRecorder file containing solution data
    plot_simulation : bool(False)
        If True, then plot simulation results
    simulation_record_file : str
        The path to the CaseRecorder file containing simulation data
    plot_dir : str
        The path to the directory to which the plot files will be written
    """

    # get ready to generate plot files
    plot_dir_path = os.path.join(os.getcwd(), plot_dir)
    if not os.path.isdir(plot_dir_path):
        os.mkdir(plot_dir_path)

    plt.switch_backend('Agg')

    # Check for possible user error.
    # If provided a simulation_record_file but plot_simulation is not True, not plot of simulation
    if not plot_simulation and simulation_record_file is not None:
        warnings.warn('Setting simulation_record_file but not setting plot_simulation will not result in plotting simulation data')

    # If user wants plot of simulation, need to provide a source of the data in a recorder file
    if plot_simulation and simulation_record_file is None:
        raise ValueError('If plot_simulation is True, simulation_record_file must be path to simulation case recorder file, not None')

    cr = om.CaseReader(recorder_filename)

    # qqq why did Rob use "final" to get the case ?
    system_solution_cases = cr.list_cases('problem')
    last_solution_case = cr.get_case(system_solution_cases[-1])

    # We have to key off of the phases node. It will tell us
    #  what the prefix is for the variable names before the phases part of the variables
    #  and also let us know what the phase names are
    root_children = cr.problem_metadata['tree']['children']
    phases_node, phases_node_path = _get_phases_node_in_problem_metadata(root_children)

    if plot_simulation:
        cr_simulate = om.CaseReader(simulation_record_file)
        system_simulation_cases = cr_simulate.list_cases('problem')
        last_simulation_case = cr_simulate.get_case(system_simulation_cases[-1])

    # Need to figure out how many phases and also check for multiple trajectories
    # Give a warning if multiple trajectories
    varnames = last_solution_case.outputs.keys()
    full_timeseries_varnames = list(filter(lambda v: "timeseries.states:" in v, varnames))

    # new
    phase_names = []
    for phase in phases_node['children']:
        phase_names.append(phase['name'])

    # new get var names and units
    var_units = {}
    time_units = {}
    for phase_node in phases_node['children']: # Hopefully all phases have the same vars
        for phase_node_child in phase_node['children']: # find the timeseries node
            if phase_node_child['name'] == 'timeseries':
                timeseries_node = phase_node_child
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

    # Check to see if there is anything to plot ?? qqq
    if len(varnames) == 0:
        warnings.warn('There are no timeseries variables to plot', RuntimeWarning)
        return

    # # look for prefixes
    # prefixes = set()
    # no_prefix_timeseries_varnames = []
    # for v in full_timeseries_varnames:
    #     prefix = ".".join(v.split(".")[:-3])
    #     prefixes.add(prefix)
    #     no_prefix_timeseries_varname = v[len(prefix):]
    #     no_prefix_timeseries_varnames.append(no_prefix_timeseries_varname)


    # Check to see the length of the variables to see if the trajectory is in the
    #   var name but I don't think I can count on that ! It all depends on the group structure

    # So really just need to look at the 4 elements of this. Cannot assume anything before that
    #    'trajectory.phase.timeseries.states:h'

    # # if len(prefixes) > 1:
    # #     warnings.warn('Cannot plot mulitple trajectories', RuntimeWarning)
    # #
    # # prefix = prefixes.pop()
    # #
    # # if prefix != "":
    # #     prefix += "."
    # #
    # # Just look at these
    # timeseries_varnames = []
    # for v in full_timeseries_varnames:
    #     phase = v.split('.')[-3:-2]
    #     var_name = v.split(':')[-1]
    #     timeseries_varnames.append(f'{phase}.timeseries.states:{var_name}')
    #



    # timeseries_varnames = [v.split('.')[-3:]  for v in timeseries_varnames]

    # need to see what the prefix is on those. If more than one, we cannot handle it qqq
    #
    # trajectories = list(set([ v.split(".")[0] for v in timeseries_varnames]))
    # if len(trajectories) > 1:
    #     warnings.warn('Cannot plot mulitple trajectories', RuntimeWarning)

    # trajectory = trajectories[0]

    cm = plt.cm.get_cmap('tab20')

    # timeseries_varnames have this pattern
    #    'phase.timeseries.states:h'
    # get phases
    # phase_names = list(set([ v.split(".")[0] for v in no_prefix_timeseries_varnames]))

    # Need to figure out the variable names
    # var_names = set([ v.split(".")[4] for v in timeseries_varnames])
    # var_names = set([ v.split(":")[1] for v in timeseries_varnames])

    for ivar, var_name in enumerate(varnames):
        # start a new plot
        fig, ax = plt.subplots()

        # Get the labels using the first and possibly only phase
        phase_name = phase_names[0]
        var_name_full = f'{phases_node_path}:{phase_name}.timeseries.states:{var_name}'
        # time_units, var_units = _get_timeseries_units(cr, var_name_full)
        v_units = var_units[var_name]
        t_units = time_units[var_name]
        time_label = f'time ({t_units})'
        var_label = f'{var_name} ({v_units})'
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
            # time_name = ".".join(var_name_full.split(".")[:-1])+".time"
            var_val = last_solution_case.outputs[var_name_full]
            time_val = last_solution_case.outputs[time_name]

            # Plot the data
            # get simulation values, if plotting simulation
            if plot_simulation:

                # if the prefix is empty, need to pre-pend names with "sim_traj."
                sim_prefix = "" if phases_node_path else "sim_traj."
                var_val_simulate = last_simulation_case.outputs[sim_prefix + var_name_full]
                time_val_simulate = last_simulation_case.outputs[sim_prefix + time_name]

            ax.plot(time_val, var_val, marker='o', linestyle='None', label='solution', color=cm.colors[iphase])
            if plot_simulation:
                # ax.plot(time_val_simulate, var_val_simulate, f'{color}--', label='simulation')
                ax.plot(time_val_simulate, var_val_simulate, linestyle='--', label='simulation', color=cm.colors[iphase])


        # Create two legends
        # Solution/Simulation legend
        solution_line = mlines.Line2D([], [], color='black', marker='o', linestyle='None', label='Solution')
        if plot_simulation:
            simulation_line = mlines.Line2D([], [], color='black', linestyle='--', label='Simulation')
            sol_sim_legend = plt.legend(handles=[solution_line, simulation_line], loc='upper center', bbox_to_anchor=(0.,-0.12), shadow=True)
        else:
            sol_sim_legend = plt.legend(handles=[solution_line],
                                        loc='upper center', bbox_to_anchor=(0., -0.12), shadow=True)
        plt.gca().add_artist(sol_sim_legend)

        # Phases legend
        handles = []
        for iphase, phase_name in enumerate(phase_names):
            patch = mpatches.Patch(color=cm.colors[iphase], label=phase_name)
            handles.append(patch)
        plt.legend(handles=handles,loc='upper right', ncol=len(phase_names), shadow=True, bbox_to_anchor=(1.1,-0.12), title='Phases')

        plt.subplots_adjust(bottom=0.23, top=0.9)

        # save to file
        plot_file_path = os.path.join(plot_dir_path, f'test_{var_name}.png' )
        plt.savefig(plot_file_path)
