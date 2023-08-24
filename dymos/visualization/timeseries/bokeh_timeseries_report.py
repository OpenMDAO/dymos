import datetime
from pathlib import Path
import os.path

try:
    from bokeh.io import output_notebook, output_file, save, show
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
import dymos as dm


_js_show_renderer = """
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

"""

# Javascript Callback when the solution/simulation checkbox buttons are toggled
# args: (figures)
_SOL_SIM_TOGGLE_JS = """
// Loop through figures and toggle the visibility of the renderers
const active = cb_obj.active;
var figures = figures;
var renderer;

for (var i = 0; i < figures.length; i++) {
    if (figures[i]) {
        for (var j =0; j < figures[i].renderers.length; j++) {
            renderer = figures[i].renderers[j]
            if (renderer.tags.includes('sol')) {
                renderer.visible = active.includes(0);
            }
            else if (renderer.tags.includes('sim')) {
                renderer.visible = active.includes(1);
            }
        }
    }
}
"""

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


def _meta_tree_subsys_iter(tree, recurse=True, cls=None):
    """
    Yield a generator of local subsystems of this system.

    Parameters
    ----------
    recurse : bool
        If True, iterate over the whole tree under this system.
    cls : None, str, or Sequence
        The class of the nodes to be iterated

    Yields
    ------
    type or None
    """
    _cls = [cls] if isinstance(cls, str) else cls

    for s in tree['children']:
        if s['type'] != 'subsystem':
            continue
        if cls is None or s['class'] in _cls:
            yield s
        if recurse:
            for child in _meta_tree_subsys_iter(s, recurse=True, cls=_cls):
                yield child


def _load_data_sources(prob, solution_record_file=None, simulation_record_file=None):

    data_dict = {}

    if Path(solution_record_file).is_file():
        sol_cr = om.CaseReader(solution_record_file)
        sol_case = sol_cr.get_case('final')
        outputs = {name: meta for name, meta in sol_case.list_outputs(units=True, out_stream=None)}
        abs2prom_map = sol_cr.problem_metadata['abs2prom']
    else:
        sol_case = None

    if Path(simulation_record_file).is_file():
        sim_cr = om.CaseReader(simulation_record_file)
        sim_case = sim_cr.get_case('final')
        outputs = {name: meta for name, meta in sim_case.list_outputs(units=True, out_stream=None)}
        abs2prom_map = sim_cr.problem_metadata['abs2prom']
    else:
        sim_case = None

    if sol_cr is None and sim_cr is None:
        om.issue_warning('No recorded data provided. Trajectory results report will not be created.')
        return

    for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
        traj_name = traj.pathname.split('.')[-1]
        data_dict[traj_name] = {'param_data_by_phase': {},
                                'sol_data_by_phase': {},
                                'sim_data_by_phase': {},
                                'timeseries_units': {}}

        for phase_name, phase in traj._phases.items():

            data_dict[traj_name]['param_data_by_phase'][phase_name] = {'param': [], 'val': [], 'units': []}
            phase_sol_data = data_dict[traj_name]['sol_data_by_phase'][phase_name] = {}
            phase_sim_data = data_dict[traj_name]['sim_data_by_phase'][phase_name] = {}
            ts_units_dict = data_dict[traj_name]['timeseries_units']

            param_outputs = {op: meta for op, meta in outputs.items()
                             if op.startswith(f'{phase.pathname}.param_comp.parameter_vals')}
            param_case = sol_case if sol_case else sim_case

            for output_name in sorted(param_outputs.keys(), key=str.casefold):
                meta = param_outputs[output_name]
                param_dict = data_dict[traj_name]['param_data_by_phase'][phase_name]

                prom_name = abs2prom_map['output'][output_name]
                param_name = output_name.replace(f'{phase.pathname}.param_comp.parameter_vals:', '', 1)

                param_dict['param'].append(param_name)
                param_dict['units'].append(meta['units'])
                param_dict['val'].append(param_case.get_val(prom_name, units=meta['units']))

            ts_outputs = {op: meta for op, meta in outputs.items() if op.startswith(f'{phase.pathname}.timeseries')}

            # Find the "largest" unit used for any timeseries output across all phases
            for output_name in sorted(ts_outputs.keys(), key=str.casefold):
                meta = ts_outputs[output_name]
                prom_name = abs2prom_map['output'][output_name]
                var_name = prom_name.split('.')[-1]

                if var_name not in ts_units_dict:
                    ts_units_dict[var_name] = meta['units']
                else:
                    _, new_conv_factor = conversion_to_base_units(meta['units'])
                    _, old_conv_factor = conversion_to_base_units(ts_units_dict[var_name])
                    if new_conv_factor < old_conv_factor:
                        ts_units_dict[var_name] = meta['units']

        # Now a second pass through the phases since we know the units in which to plot
        # each timeseries variable output.
        for phase_name, phase in traj._phases.items():

            phase_sol_data = data_dict[traj_name]['sol_data_by_phase'][phase_name] = {}
            phase_sim_data = data_dict[traj_name]['sim_data_by_phase'][phase_name] = {}
            ts_units_dict = data_dict[traj_name]['timeseries_units']

            ts_outputs = {op: meta for op, meta in outputs.items() if op.startswith(f'{phase.pathname}.timeseries')}

            for output_name in sorted(ts_outputs.keys(), key=str.casefold):
                meta = ts_outputs[output_name]
                prom_name = abs2prom_map['output'][output_name]
                var_name = prom_name.split('.')[-1]

                if sol_case:
                    phase_sol_data[var_name] = sol_case.get_val(prom_name, units=ts_units_dict[var_name])
                if sim_case:
                    phase_sim_data[var_name] = sim_case.get_val(prom_name, units=ts_units_dict[var_name])

    return data_dict


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
    # For the primary timeseries in each phase in each trajectory, build a set of the pathnames
    # to be plotted.
    source_data = _load_data_sources(prob, solution_record_file, simulation_record_file)

    # Colors of each phase in the plot. Start with the bright colors followed by the faded ones.
    if not _NO_BOKEH:
        colors = bp.d3['Category20'][20][0::2] + bp.d3['Category20'][20][1::2]
        curdoc().theme = theme

    for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
        traj_name = traj.pathname.split('.')[-1]
        report_filename = f'{traj.pathname}_results_report.html'
        report_dir = Path(prob.get_reports_dir())
        report_path = report_dir / report_filename
        if not os.path.isdir(report_dir):
            om.issue_warning(f'Reports directory not available. {report_path} will not be created.')
            continue
        if _NO_BOKEH:
            with open(report_path, 'wb') as f:
                f.write("<html>\n<head>\n<title> \nError: bokeh not available</title>\n</head> <body>\n"
                        "This report requires bokeh but bokeh was not available in this python installation.\n"
                        "</body></html>".encode())
            continue

        param_tables = []
        phase_names = []

        for phase_name, phase in traj._phases.items():

            phase_names.append(phase_name)

            # Make the parameter table
            source = ColumnDataSource(source_data[traj_name]['param_data_by_phase'][phase_name])
            columns = [
                TableColumn(field='param', title='Parameter'),
                TableColumn(field='val', title='Value'),
                TableColumn(field='units', title='Units'),
            ]
            param_tables.append(DataTable(source=source, columns=columns, index_position=None,
                                          height=30*len(source.data['param']), sizing_mode='stretch_both'))

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
                        sol_plot = fig.circle(x='time', y=var_name, source=sol_source, color=color)
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

        ts_layout = column(children=[sol_sim_row,
                                     phase_select_row,
                                     figures_grid],
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
