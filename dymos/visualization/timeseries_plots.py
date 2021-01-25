import os
import warnings

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import openmdao.api as om
from dymos.options import options as dymos_options


def _get_phases_node_in_problem_metadata(node, path=""):
    """
    Find the phases node in the Problem metadata hierarchy.

    There is one node in the hierarchy that has the name 'phases'. Finding this
    node will be used to find information about all the phases in the model.
    This is a recursive function.

    Parameters
    ----------
    node : list
        Node in Problem metadata hierarchy.
    path : str
        The dotted string path name to the node in the Problem hierarchy. Used
        recursively to build up the path to a node.

    Returns
    -------
    tuple of a list and a string
        Returns the node and path name to the node, if found. Otherwise, returns (None, None).
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


def _mpl_timeseries_plots(varnames, time_units, var_units, phase_names, phases_node_path,
                          last_solution_case, last_simulation_case, plot_dir_path):
    # get ready to plot
    backend_save = plt.get_backend()
    plt.switch_backend('Agg')
    # use a colormap with 20 values
    cm = plt.cm.get_cmap('tab20')

    for ivar, var_name in enumerate(varnames):
        # start a new plot
        fig, ax = plt.subplots()

        # Get the labels
        time_label = f'time ({time_units[var_name]})'
        var_label = f'{var_name} ({var_units[var_name]})'
        title = f'timeseries.{var_name}'

        # add labels, title, and legend
        ax.set_xlabel(time_label)
        ax.set_ylabel(var_label)
        fig.suptitle(title)

        # Plot each phase
        for iphase, phase_name in enumerate(phase_names):
            if phases_node_path:
                var_name_full = f'{phases_node_path}.{phase_name}.timeseries.{var_name}'
                time_name = f'{phases_node_path}.{phase_name}.timeseries.time'
            else:
                var_name_full = f'{phase_name}.timeseries.{var_name}'
                time_name = f'{phase_name}.timeseries.time'

            # Get values
            if var_name_full not in last_solution_case.outputs:
                continue

            var_val = last_solution_case.outputs[var_name_full]
            time_val = last_solution_case.outputs[time_name]

            # Plot the data
            color = cm.colors[iphase % 20]

            ax.plot(time_val, var_val, marker='o', linestyle='None', label='solution', color=color)

            # get simulation values, if plotting simulation
            if last_simulation_case:
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
        if last_simulation_case:
            simulation_line = mlines.Line2D([], [], color='black', linestyle='--',
                                            label='Simulation')
            sol_sim_legend = plt.legend(handles=[solution_line, simulation_line],
                                        loc='upper left', bbox_to_anchor=(-0.3, -0.12), shadow=True)
        else:
            sol_sim_legend = plt.legend(handles=[solution_line],
                                        loc='upper left', bbox_to_anchor=(-0.3, -0.12),
                                        shadow=True)
        plt.gca().add_artist(sol_sim_legend)

        #   Phases legend
        handles = []
        for iphase, phase_name in enumerate(phase_names):
            patch = mpatches.Patch(color=cm.colors[iphase], label=phase_name)
            handles.append(patch)
        plt.legend(handles=handles, loc='upper right', ncol=len(phase_names), shadow=True,
                   bbox_to_anchor=(1.15, -0.12), title='Phases')

        plt.subplots_adjust(bottom=0.23, top=0.9, left=0.2)

        # save to file
        plot_file_path = os.path.join(plot_dir_path, f'{var_name.replace(":","_")}.png')
        plt.savefig(plot_file_path)

    plt.switch_backend(backend_save)


def _bokeh_timeseries_plots(varnames, time_units, var_units, phase_names, phases_node_path,
                            last_solution_case, last_simulation_case, plot_dir_path, num_cols=2,
                            bg_fill_color='#282828', grid_line_color='#666666', open_browser=False):
    from bokeh.io import output_notebook, output_file, save, show
    from bokeh.layouts import gridplot, column, row, grid, layout
    from bokeh.models import Legend, LegendItem
    from bokeh.plotting import figure
    import bokeh.palettes as bp

    if dymos_options['notebook_mode']:
        output_notebook()
    else:
        output_file(os.path.join(plot_dir_path, 'plots.html'))

    # Prune the edges from the color map
    cmap = bp.turbo(len(phase_names) + 2)[1:-1]
    figures = []
    colors = {}
    sol_plots = {}
    sim_plots = {}

    # Get the minimum and maximum times in any phase, so when we plot a variable that only exists
    # in a few phases, it is plotted against the entire time range.
    min_time = 1.0E21
    max_time = -1.0E21

    for iphase, phase_name in enumerate(phase_names):
        if phases_node_path:
            time_name = f'{phases_node_path}.{phase_name}.timeseries.time'
        else:
            time_name = f'{phase_name}.timeseries.time'
        min_time = min(min_time, np.min(last_solution_case.outputs[time_name]))
        max_time = max(max_time, np.max(last_solution_case.outputs[time_name]))
        colors[phase_name] = cmap[iphase]

    for ivar, var_name in enumerate(varnames):
        # Get the labels
        time_label = f'time ({time_units[var_name]})'
        var_label = f'{var_name} ({var_units[var_name]})'
        title = f'timeseries.{var_name}'

        # add labels, title, and legend
        padding = 0.05 * (max_time - min_time)
        fig = figure(title=title, background_fill_color=bg_fill_color,
                     x_range=(min_time - padding, max_time + padding), plot_width=180,
                     plot_height=180)
        fig.xaxis.axis_label = time_label
        fig.yaxis.axis_label = var_label
        fig.xgrid.grid_line_color = grid_line_color
        fig.ygrid.grid_line_color = grid_line_color

        # Plot each phase
        for iphase, phase_name in enumerate(phase_names):
            sol_color = cmap[iphase]
            sim_color = cmap[iphase]

            if phases_node_path:
                var_name_full = f'{phases_node_path}.{phase_name}.timeseries.{var_name}'
                time_name = f'{phases_node_path}.{phase_name}.timeseries.time'
            else:
                var_name_full = f'{phase_name}.timeseries.{var_name}'
                time_name = f'{phase_name}.timeseries.time'

            # Get values
            if var_name_full not in last_solution_case.outputs:
                continue

            var_val = last_solution_case.outputs[var_name_full]
            time_val = last_solution_case.outputs[time_name]

            for idxs, i in np.ndenumerate(np.zeros(var_val.shape[1:])):
                var_val_i = var_val[:, idxs].ravel()
                sol_plots[phase_name] = fig.circle(time_val.ravel(), var_val_i, size=5,
                                                   color=sol_color, name='sol:' + phase_name)

            # get simulation values, if plotting simulation
            if last_simulation_case:
                # if the phases_node_path is empty, need to pre-pend names with "sim_traj."
                #   as that is pre-pended in Trajectory.simulate code
                sim_prefix = '' if phases_node_path else 'sim_traj.'
                var_val_simulate = last_simulation_case.outputs[sim_prefix + var_name_full]
                time_val_simulate = last_simulation_case.outputs[sim_prefix + time_name]
                for idxs, i in np.ndenumerate(np.zeros(var_val_simulate.shape[1:])):
                    var_val_i = var_val_simulate[:, idxs].ravel()
                    sim_plots[phase_name] = fig.line(time_val_simulate.ravel(), var_val_i,
                                                     line_dash='solid', line_width=0.5, color=sim_color,
                                                     name='sim:' + phase_name)
        figures.append(fig)

    # Implement a single legend for all figures using the example here:
    # https://stackoverflow.com/a/56825812/754536

    # ## Use a dummy figure for the LEGEND
    dum_fig = figure(outline_line_alpha=0, toolbar_location=None,
                     background_fill_color=bg_fill_color, plot_width=250, max_width=250)

    # set the components of the figure invisible
    for fig_component in [dum_fig.grid, dum_fig.ygrid, dum_fig.xaxis, dum_fig.yaxis]:
        fig_component.visible = False

    # The glyphs referred by the legend need to be present in the figure that holds the legend,
    # so we must add them to the figure renderers.
    sol_legend_items = [(phase_name + ' solution', [dum_fig.circle([0], [0],
                                                                   size=5,
                                                                   color=colors[phase_name],
                                                                   tags=['sol:' + phase_name])]) for phase_name in phase_names]
    sim_legend_items = [(phase_name + ' simulation', [dum_fig.line([0], [0],
                                                                   line_dash='solid',
                                                                   line_width=0.5,
                                                                   color=colors[phase_name],
                                                                   tags=['sim:' + phase_name])]) for phase_name in phase_names]
    legend_items = [j for i in zip(sol_legend_items, sim_legend_items) for j in i]

    # # set the figure range outside of the range of all glyphs
    dum_fig.x_range.end = 1005
    dum_fig.x_range.start = 1000

    legend = Legend(click_policy='hide', location='top_left', border_line_alpha=0, items=legend_items,
                    background_fill_alpha=0.0, label_text_color='white', label_width=120, spacing=10)

    dum_fig.add_layout(legend, place='center')

    gd = gridplot(figures, ncols=num_cols, sizing_mode='scale_both')

    plots = gridplot([[gd, column(dum_fig, sizing_mode='stretch_height')]],
                     toolbar_location=None,
                     sizing_mode='scale_both')

    if dymos_options['notebook_mode'] or open_browser:
        show(plots)
    else:
        save(plots)


def timeseries_plots(solution_recorder_filename, simulation_record_file=None, plot_dir="plots"):
    """
    Create plots of the timeseries.

    Given timeseries data from case recorder files, make separate plots of each variable
    and store the plot files in the directory indicated by the variable plot_dir

    Parameters
    ----------
    solution_recorder_filename : str
        The path to the case recorder file containing solution data.
    simulation_record_file : str or None (default:None)
        The path to the case recorder file containing simulation data. If not None,
        this implies that the data from it should be plotted.
    plot_dir : str
        The path to the directory to which the plot files will be written.
    """

    # get ready to generate plot files
    plot_dir_path = os.path.join(os.getcwd(), plot_dir)
    if not os.path.isdir(plot_dir_path):
        os.mkdir(plot_dir_path)

    cr = om.CaseReader(solution_recorder_filename)

    # get outputs from the solution
    solution_cases = cr.list_cases('problem')
    last_solution_case = cr.get_case(solution_cases[-1])

    # If plotting simulation results, get the values for those variables
    if simulation_record_file:
        cr_simulate = om.CaseReader(simulation_record_file)
        system_simulation_cases = cr_simulate.list_cases('problem')
        last_simulation_case = cr_simulate.get_case(system_simulation_cases[-1])
    else:
        last_simulation_case = None

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
            # plot everything in the timeseries except input_values. Also not time since that
            #   is the independent variable in these plot
            if not timeseries_node_child['name'].startswith("input_values:")\
                    and not timeseries_node_child['name'] == 'time':
                varname = timeseries_node_child['name']
                var_units[varname] = timeseries_node_child['units']
                time_units[varname] = units_for_time
    varnames = list(var_units.keys())

    # Check to see if there is anything to plot
    if len(varnames) == 0:
        warnings.warn('There are no timeseries variables to plot', RuntimeWarning)
        return

    if dymos_options['plots'] == 'bokeh':
        _bokeh_timeseries_plots(varnames, time_units, var_units, phase_names, phases_node_path,
                                last_solution_case, last_simulation_case, plot_dir_path)
    elif dymos_options['plots'] == 'matplotlib':
        _mpl_timeseries_plots(varnames, time_units, var_units, phase_names, phases_node_path,
                              last_solution_case, last_simulation_case, plot_dir_path)
    else:
        raise ValueError(f'Unknown plotting option: {dymos_options["plots"]}')
