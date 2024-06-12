from collections import ChainMap
import datetime
from pathlib import Path
import os.path

from dymos.trajectory.trajectory import Trajectory
from dymos.phase.phase import Phase

try:
    from bokeh.io import save
    from bokeh.layouts import column, grid, row
    from bokeh.models import Legend, DataTable, Div, ColumnDataSource, TableColumn, \
        TabPanel, Tabs, CheckboxButtonGroup, CustomJS, MultiChoice
    from bokeh.plotting import figure, curdoc
    import bokeh.palettes as bp
    import bokeh.resources as bokeh_resources
    _NO_BOKEH = False
except ImportError:
    _NO_BOKEH = True

import openmdao.api as om
from openmdao.utils.units import conversion_to_base_units
from openmdao.utils.mpi import MPI
from openmdao.recorders.sqlite_recorder import META_KEY_SEP


# Javascript Callback when the solution/simulation checkbox buttons are toggled
# args: (figures)
_js_show_figures = """
const phases_to_show = phase_select.value;
const kinds_to_show = sol_sim_toggle.active;
var figures = figures;
var renderer;
var renderer_phase;

function show_renderer(renderer, phases_to_show, kinds_to_show) {
    var tags = renderer.tags;
    for(var k=0; k < tags.length; k++) {
        if (tags[k].substring(0, 6) == 'phase:') {
            renderer_phase = tags[k].substring(6);
            break;
        }
    }
    return ((tags.includes('sol') && kinds_to_show.includes(0)) ||
            (tags.includes('sim') && kinds_to_show.includes(1))) &&
           phases_to_show.includes(renderer_phase);
}

for (var i = 0; i < figures.length; i++) {
    if (figures[i]) {
        for (var j=0; j < figures[i].renderers.length; j++) {
            renderer = figures[i].renderers[j];
            // Get the phase with which this renderer is associated
            renderer.visible = show_renderer(renderer, phases_to_show, kinds_to_show);
        }
    }
}
"""


def _meta_tree_subsys_iter(tree, recurse=True, cls=None, path=None):
    """
    Yield a generator of local subsystems of this system.

    Parameters
    ----------
    recurse : bool
        If True, iterate over the whole tree under this system.
    cls : None, str, or Sequence
        The class of the nodes to be iterated
    path : The absolute path of the given tree.

    Yields
    ------
    type or None
    """
    _cls = [cls] if isinstance(cls, str) else cls

    for s in tree['children']:
        if s['type'] != 'subsystem':
            continue
        else:
            s['path'] = f'{path}.{s["name"]}' if path else s['name']

        if cls is None or s['class'] in _cls:
            yield s
        if recurse:
            for child in _meta_tree_subsys_iter(s, recurse=True, cls=_cls, path=s['path']):
                yield child


def _get_model_options_from_cr(cr, syspath, run_number=None):
    """Retrieve model options for the given system from the given case reader.

    The the options for system stored in the given case reader.
    If there is more than one set of model options, this function returns the last recorded ones.

    Parameters
    ----------
    cr : CaseReader
        The CaseReader instance holding the data.
    syspath : str
        Pathname of system whose options are requested.
    run_number : int or None
        Run_driver or run_model iteration to inspect, if given. If None, return the last available.

    Returns
    -------
    dict{system: {key: val}}
        The model options dictionary for the given system.
    """
    if not cr._system_options:
        raise RuntimeError('System options were not recorded.')

    # need to handle edge case for v11 recording
    if cr._format_version < 12:
        SEP = '_'
    else:
        SEP = META_KEY_SEP

    for key in cr._system_options:
        if key.find(SEP) > 0:
            name, num = key.rsplit(SEP, 1)
        else:
            name = key
            num = 0

        # Get the component associated with the highest run number
        if name == syspath and (run_number is None or run_number == int(num)):
            return cr._system_options[key]['component_options']
    else:
        raise KeyError(f'No options found for system {syspath}')


def _get_traj_and_phases_from_problem(problem, rank: int = 0):
    """Retrieve a dictionary tree structure of all trajectories and phases in the problem.

    Parameters
    ----------
    problem : Problem
        The problem being searched for trajectories and phases.
    rank : int
        Under MPI, the returned trajectory tree is only valid on this rank.

    Returns
    -------
    trajs : dict
        A dictionary for each trajectory containing its parameter options dictionary and
        a subdictionary of phases and their associated options. Under MPI, this dictionary
    """
    comm_rank = 0 if MPI is None else MPI.COMM_WORLD.rank
    traj_options = _gather_system_options(problem.model, sys_cls=Trajectory, rank=rank)
    phase_options = _gather_system_options(problem.model, sys_cls=Phase, rank=rank)

    trajs = {}

    if comm_rank == 0:
        for traj_path, toptions in traj_options.items():
            trajs[traj_path] = {'phases': {},
                                'parameter_options': toptions['parameter_options'],
                                'name': traj_path}
            for phase_path, poptions in phase_options.items():
                if phase_path.startswith(f'{traj_path}.'):
                    trajs[traj_path]['phases'][phase_path] = {'name': phase_path.split('.')[-1]}
                    for opt in ('time_options', 'parameter_options', 'state_options',
                                'control_options'):
                        trajs[traj_path]['phases'][phase_path][opt] = poptions[opt]

    return trajs


# TODO: Enable this function when system options are correctly stored under MPI
def _get_trajs_and_phases_from_cr(cr, problem=None):  # pragma: no cover
    """Retrieve dictionaries of the trajectories and phases from the given case reader and problem.

    Due to a bug in OpenMDAO, model options may not be available from the case reader. In this case,
    use the problem to obtain them.

    Parameters
    ----------
    cr : CaseReader
        The CaseReader from which the model tree should be loaded.
    problem : Problem, optional
        The Problem instance from which system options should be loaded as a fallback.

    Returns
    -------
    _type_
        _description_
    """
    trajs = {}
    traj_cls = 'dymos.trajectory.trajectory:Trajectory'
    phase_cls = ['dymos.phase.phase:Phase',
                 'dymos.phase.phase:AnalyticPhase',
                 'dymos.phase.phase:SimualationPhase']

    traj_nodes = [n for n in _meta_tree_subsys_iter(cr.problem_metadata['tree'], cls=traj_cls)]
    phase_nodes = [n for n in _meta_tree_subsys_iter(cr.problem_metadata['tree'], cls=phase_cls)]

    for tn in traj_nodes:
        phase_nodes = [n for n in _meta_tree_subsys_iter(tn, cls=phase_cls, path=tn['path'])]
        traj_path = tn['path']
        trajs[traj_path] = {'phases': {}, 'parameter_options': {}, 'name': tn['name']}

        traj_options = _get_model_options_from_cr(cr=cr, syspath=traj_path)

        if MPI and MPI.COMM_WORLD.rank == 0:
            for param_name, param_options in traj_options['parameter_options'].items():
                trajs[traj_path]['parameter_options'][param_name] = param_options
            for pn in phase_nodes:
                phase_path = pn['path']
                phase_options = _get_model_options_from_cr(cr=cr, syspath=phase_path)
                phase_meta = trajs[traj_path]['phases'][phase_path] = {'time_options': None,
                                                                       'parameter_options': None,
                                                                       'state_options': None,
                                                                       'control_options': None,
                                                                       'name': pn['name']}
                phase_meta['time_options'] = phase_options['time_options']
                phase_meta['parameter_options'] = phase_options['parameter_options']
                phase_meta['state_options'] = phase_options['state_options']
                phase_meta['control_options'] = phase_options['control_options']

    return trajs


def _load_data_sources(traj_and_phase_meta=None, solution_record_file=None, simulation_record_file=None):
    """
    Load the data for the timeseries plots from the given solution and record files.

    Parameters
    ----------
    traj_and_phase_meta : dict
        A tree dictionary structure that contains the trajectories, their child phases,
        and associated options.
    solution_record_file : str
        The path to the solution record file.
    sim_record_file : str
        The path to the corresponding simulation record file.

    Returns
    -------
    dict
        A dictionary containing parameter data, solution timeseries data, simulation timeseries data, and
        units for each timeseries output.
    """
    data_dict = {}

    if Path(solution_record_file).is_file():
        sol_cr = om.CaseReader(solution_record_file)
        sol_case = sol_cr.get_case('final')
        abs2prom_map = sol_cr.problem_metadata['abs2prom']
    else:
        sol_case = None

    if Path(simulation_record_file).is_file():
        sim_cr = om.CaseReader(simulation_record_file)
        sim_case = sim_cr.get_case('final')
        abs2prom_map = sim_cr.problem_metadata['abs2prom']
    else:
        sim_case = None

    source_case = sol_case or sim_case or None
    outputs = {abs_path: meta for abs_path, meta in source_case.list_outputs(out_stream=None, units=True)}

    if source_case is None:
        om.issue_warning('No recorded data provided. Trajectory results report will not be created.')
        return

    if sol_case:
        sol_outputs = {abs_path: meta for abs_path, meta in
                       sol_case.list_outputs(out_stream=None, units=True) if 'timeseries' in abs_path}
    else:
        sol_outputs = None

    if sim_case:
        sim_outputs = {abs_path: meta for abs_path, meta in
                       sim_case.list_outputs(out_stream=None, units=True) if 'timeseries' in abs_path}
    else:
        sim_outputs = None

    source_case = sol_case or sim_case
    outputs = sol_outputs or sim_outputs

    if sim_outputs is not None and sol_outputs is not None:
        if not set(sim_outputs.keys()).issubset(sol_outputs.keys()):
            om.issue_warning('Simulation file does not contain the same outputs as the solution '
                             'file. Skipping plotting of simulation timeseries data.')
            sim_case = None

    for traj_path, traj_data in traj_and_phase_meta.items():
        traj_params = traj_data['parameter_options']
        traj_name = traj_data['name']
        data_dict[traj_data['name']] = {'param_data_by_phase': {},
                                        'sol_data_by_phase': {},
                                        'sim_data_by_phase': {},
                                        'timeseries_units': {}}

        for phase_path, phase_data in traj_data['phases'].items():
            phase_name = phase_data['name']
            phase_param_data = \
                data_dict[traj_name]['param_data_by_phase'][phase_name] = \
                {'param': [], 'val': [], 'units': []}

            phase_sol_data = data_dict[traj_data['name']]['sol_data_by_phase'][phase_name] = {}
            phase_sim_data = data_dict[traj_data['name']]['sim_data_by_phase'][phase_name] = {}

            # param_outputs = {op: meta for op, meta in outputs.items()
            #                  if op.startswith(f'{phase_path}.param_comp.parameter_vals:')}
            ts_outputs = {op: meta for op, meta in outputs.items()
                          if op.startswith(f'{phase_path}.timeseries.')}

            # Populate the phase parameter data
            phase_params = traj_and_phase_meta[traj_path]['phases'][phase_path]['parameter_options']
            for param_name in sorted(phase_params, key=str.casefold):
                units = phase_params[param_name]['units']
                param_path = f'{traj_path}.{phase_name}.parameter_vals:{param_name}'
                phase_param_data['param'].append(param_name)
                phase_param_data['units'].append(units)
                phase_param_data['val'].append(source_case.get_val(param_path, units=units))

            # Find the "largest" unit used for any timeseries output across all phases
            ts_units_dict = data_dict[traj_data['name']]['timeseries_units']
            for abs_name in sorted(ts_outputs.keys(), key=str.casefold):
                meta = ts_outputs[abs_name]
                prom_name = meta['prom_name']
                var_name = prom_name.split('.')[-1]

                if var_name not in ts_units_dict:
                    ts_units_dict[var_name] = meta['units']
                else:
                    _, new_conv_factor = conversion_to_base_units(meta['units'])
                    _, old_conv_factor = conversion_to_base_units(ts_units_dict[var_name])
                    if new_conv_factor < old_conv_factor:
                        ts_units_dict[var_name] = meta['units']

            # Populate the phase timeseries data
            for output_name in sorted(ts_outputs.keys(), key=str.casefold):
                meta = ts_outputs[output_name]
                prom_name = meta['prom_name']
                var_name = prom_name.split('.')[-1]

                if sol_case:
                    phase_sol_data[var_name] = sol_case.get_val(prom_name, units=ts_units_dict[var_name])
                if sim_case:
                    phase_sim_data[var_name] = sim_case.get_val(prom_name, units=ts_units_dict[var_name])

    return data_dict


def _gather_system_options(model, sys_cls=None, rank=0):
    """Retreive system options for systems of the given class and/or pathname.

    Parameters
    ----------
    model : System
        The root system from which model options are being gathered.
    sys_cls : class, optional
        The class of system for which we want to retrieve options, or None if we
        should look for all systems regardless of class.
    rank : int
        The rank onto which the system options shoud be gathered. The retured
        dictionary of system options will only be valid on this rank.
    """
    comm_size = 1 if MPI is None else MPI.COMM_WORLD.size
    comm_rank = 0 if MPI is None else MPI.COMM_WORLD.rank

    system_options = {}
    for subsys in model.system_iter(include_self=True, recurse=True, typ=sys_cls):
        system_options[subsys.pathname] = subsys.options

    if comm_size > 1:
        gathered = MPI.COMM_WORLD.gather(system_options, rank)

        if comm_rank == 0:
            system_options.update(dict(ChainMap(*gathered)))

    return system_options


def make_timeseries_report(prob, solution_record_file=None, simulation_record_file=None,
                           x_name='time', ncols=2, margin=10, theme='light_minimal'):
    """
    Create the bokeh-based timeseries results report.

    Parameters
    ----------
    prob : om.Problem
        The problem instance for which the timeseries plots are being created.
    solution_record_file : str
        The path to the solution record file, if available.
    simulation_record_file : str
        The path to the simulation record file, if available.
    x_name : str
        Name of the horizontal axis variable in the timeseries.
    ncols : int
        The number of columns of timeseries output plots.
    margin : int
        A margin to be placed between the plot figures.
    theme : str
        A valid bokeh theme name to style the report.
    """
    comm_rank = 0 if MPI is None else MPI.COMM_WORLD.rank
    report_dir = Path(prob.get_reports_dir()) if prob is not None else Path(os.getcwd())
    if not report_dir.exists():
        report_dir = Path(os.getcwd())

    if prob is not None:
        # Retrieve system information fom the problem if available.
        traj_data = _get_traj_and_phases_from_problem(prob)

    # The rest of the traj report generation occurs on rank 0
    if comm_rank == 0:

        # For the primary timeseries in each phase in each trajectory, build a set of the pathnames
        # to be plotted.
        source_data = _load_data_sources(traj_data, solution_record_file, simulation_record_file)

        # Colors of each phase in the plot. Start with the bright colors followed by the faded ones.
        if not _NO_BOKEH:
            colors = bp.d3['Category20'][20][0::2] + bp.d3['Category20'][20][1::2]
            curdoc().theme = theme

        for traj_path, traj_data in source_data.items():
            traj_name = traj_path.split('.')[-1]
            report_filename = f'{traj_path}_results_report.html'
            report_path = report_dir / report_filename
            if _NO_BOKEH:
                with open(report_path, 'wb') as f:
                    f.write("<html>\n<head>\n<title> \nError: bokeh not available</title>\n</head> <body>\n"
                            "This report requires bokeh but bokeh was not available in this python installation.\n"
                            "</body></html>".encode())
                continue

            param_tables = []
            phase_names = [k.split('.')[-1] for k in traj_data['param_data_by_phase']]
            for phase_name, param_data in traj_data['param_data_by_phase'].items():
                # Make the parameter table
                columns = [
                    TableColumn(field='param', title='Parameter'),
                    TableColumn(field='val', title='Value'),
                    TableColumn(field='units', title='Units'),
                ]
                param_tables.append(DataTable(source=ColumnDataSource(param_data),
                                              columns=columns,
                                              index_position=None,
                                              height=30*len(param_data['param']),
                                              sizing_mode='stretch_both'))

            # Plot the timeseries
            ts_units_dict = source_data[traj_name]['timeseries_units']

            figures = []
            x_range = None

            for var_name in sorted(ts_units_dict.keys(), key=str.casefold):
                fig_kwargs = {'x_range': x_range} if x_range is not None else {}

                tool_tips = [(f'{x_name}', '$x'), (f'{var_name}', '$y')]

                fig = figure(tools='pan,box_zoom,xwheel_zoom,hover,undo,reset,save',
                             tooltips=tool_tips,
                             x_axis_label=f'{x_name} ({ts_units_dict[x_name]})',
                             y_axis_label=f'{var_name} ({ts_units_dict[var_name]})',
                             toolbar_location='above',
                             sizing_mode='stretch_both',
                             min_height=250, max_height=300,
                             margin=margin,
                             **fig_kwargs)
                fig.xaxis.axis_label_text_font_size = '10pt'
                fig.yaxis.axis_label_text_font_size = '10pt'
                fig.toolbar.autohide = True
                legend_data = []
                if x_range is None:
                    x_range = fig.x_range
                for i, phase_name in enumerate(phase_names):
                    color = colors[i % 20]
                    sol_data = source_data[traj_name]['sol_data_by_phase'][phase_name]
                    sim_data = source_data[traj_name]['sim_data_by_phase'][phase_name]
                    sol_source = ColumnDataSource(sol_data)
                    sim_source = ColumnDataSource(sim_data)
                    if x_name in sol_data and var_name in sol_data:
                        legend_items = []
                        if sol_data:
                            sol_plot = fig.scatter(x='time', y=var_name, source=sol_source,
                                                   color=color, size=5)
                            sol_plot.tags.extend(['sol', f'phase:{phase_name}'])
                            legend_items.append(sol_plot)
                        if sim_data:
                            sim_plot = fig.line(x='time', y=var_name, source=sim_source, color=color)
                            sim_plot.tags.extend(['sim', f'phase:{phase_name}'])
                            legend_items.append(sim_plot)
                        legend_data.append((phase_name, legend_items))

                legend = Legend(items=legend_data, location='center', label_text_font_size='8pt')
                fig.add_layout(legend, 'right')
                figures.append(fig)

            # Since we're putting figures in two columns, make sure we have an even number of things to put in the layout.
            if len(figures) % 2 == 1:
                figures.append(None)

            param_panels = [TabPanel(child=table, title=f'{phase_names[i]} parameters')
                            for i, table in enumerate(param_tables)]

            sol_sim_toggle = CheckboxButtonGroup(labels=['Solution', 'Simulation'], active=[0, 1])

            sol_sim_row = row(children=[Div(text='Display data:', sizing_mode='stretch_height'),
                                        sol_sim_toggle],
                              sizing_mode='stretch_both',
                              max_height=50)

            phase_select = MultiChoice(options=[phase_name for phase_name in phase_names],
                                       value=[phase_name for phase_name in phase_names],
                                       sizing_mode='stretch_both',
                                       min_width=400, min_height=50)

            phase_select_row = row(children=[Div(text='Plot phases:'), phase_select],
                                   sizing_mode='stretch_width')

            figures_grid = grid(children=figures, ncols=ncols, sizing_mode='stretch_both')

            ts_layout = column(children=[sol_sim_row, phase_select_row, figures_grid],
                               sizing_mode='stretch_both')

            tab_panes = Tabs(tabs=[TabPanel(child=ts_layout, title='Timeseries')] + param_panels,
                             sizing_mode='stretch_both',
                             active=0)

            summary = rf'Results of {prob._name}<br>Creation Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

            report_layout = column(children=[Div(text=summary), tab_panes], sizing_mode='stretch_both')

            # Assign callbacks

            sol_sim_toggle.js_on_change("active",
                                        CustomJS(code=_js_show_figures,
                                                 args=dict(figures=figures,
                                                           sol_sim_toggle=sol_sim_toggle,
                                                           phase_select=phase_select)))

            phase_select.js_on_change("value",
                                      CustomJS(code=_js_show_figures,
                                               args=dict(figures=figures,
                                                         sol_sim_toggle=sol_sim_toggle,
                                                         phase_select=phase_select)))

            # Save

            save(report_layout, filename=report_path, title=f'trajectory results for {traj_name}',
                 resources=bokeh_resources.INLINE)
