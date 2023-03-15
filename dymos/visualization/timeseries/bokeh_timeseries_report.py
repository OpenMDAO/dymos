import datetime
from pathlib import Path

from bokeh.io import output_notebook, output_file, save, show
from bokeh.layouts import gridplot, column, grid, GridBox, layout, row
from bokeh.models import Legend, DataTable, Div, ColumnDataSource, Paragraph, TableColumn, TabPanel, Tabs
from bokeh.plotting import figure, curdoc
import bokeh.palettes as bp

import dymos as dm
from dymos.options import options as dymos_options

import openmdao.api as om
import openmdao.utils.reports_system as rptsys


_default_timeseries_report_filename = 'dymos_results_{traj_name}.html'


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


def _load_data_sources(prob, solution_record_file, simulation_record_file):

    data_dict = {}

    sol_cr = om.CaseReader(solution_record_file)
    sol_case = sol_cr.get_case('final')
    sim_case = om.CaseReader(simulation_record_file).get_case('final')
    sol_outputs = {name: meta for name, meta in sol_case.list_outputs(units=True, out_stream=None)}

    abs2prom_map = sol_cr.problem_metadata['abs2prom']

    for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
        traj_name = traj.pathname.split('.')[-1]
        data_dict[traj_name] = {'param_data_by_phase': {},
                                'sol_data_by_phase': {},
                                'sim_data_by_phase': {},
                                'timeseries_units': {}}

        for phase in traj.system_iter(include_self=True, recurse=True, typ=dm.Phase):
            phase_name = phase.pathname.split('.')[-1]

            data_dict[traj_name]['param_data_by_phase'][phase_name] = {'param': [], 'val': [], 'units': []}
            phase_sol_data = data_dict[traj_name]['sol_data_by_phase'][phase_name] = {}
            phase_sim_data = data_dict[traj_name]['sim_data_by_phase'][phase_name] = {}
            ts_units_dict = data_dict[traj_name]['timeseries_units']

            param_outputs = {op: meta for op, meta in sol_outputs.items() if op.startswith(f'{phase.pathname}.param_comp.parameter_vals')}

            for output_name, meta in dict(sorted(param_outputs.items())).items():
                param_dict = data_dict[traj_name]['param_data_by_phase'][phase_name]

                prom_name = abs2prom_map['output'][output_name]
                param_name = prom_name.split(':')[-1]

                param_dict['param'].append(param_name)
                param_dict['units'].append(meta['units'])
                param_dict['val'].append(sol_case.get_val(prom_name, units=meta['units']))

            ts_outputs = {op: meta for op, meta in sol_outputs.items() if op.startswith(f'{phase.pathname}.timeseries')}

            for output_name, meta in ts_outputs.items():

                prom_name = abs2prom_map['output'][output_name]
                var_name = prom_name.split('.')[-1]

                if meta['units'] not in ts_units_dict:
                    ts_units_dict[var_name] = meta['units']
                phase_sol_data[var_name] = sol_case.get_val(prom_name, units=meta['units'])
                phase_sim_data[var_name] = sim_case.get_val(prom_name, units=meta['units'])

    return data_dict

def make_timeseries_report(prob, solution_record_file=None, simulation_record_file=None, solution_history=False, x_name='time',
                           ncols=2, min_fig_height=250, max_fig_height=300, margin=10, theme='light_minimal'):
    """

    Parameters
    ----------
    prob
    solution_record_file
    simulation_record_file
    solution_history
    x_name : str
        Name of the horizontal axis variable in the timeseries.

    Returns
    -------

    """
    # For the primary timeseries in each phase in each trajectory, build a set of the pathnames
    # to be plotted.
    source_data = _load_data_sources(prob, solution_record_file, simulation_record_file)

    # Colors of each phase in the plot. Start with the bright colors followed by the faded ones.
    colors = bp.d3['Category20'][20][0::2] + bp.d3['Category20'][20][1::2]

    curdoc().theme = theme

    for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
        traj_name = traj.pathname.split('.')[-1]
        report_filename = f'dymos_traj_report_{traj.pathname}.html'
        report_path = str(Path(prob.get_reports_dir()) / report_filename)

        param_tables = []
        phase_names = []

        for phase in traj.system_iter(include_self=True, recurse=True, typ=dm.Phase):
            phase_name = phase.pathname.split('.')[-1]
            phase_names.append(phase_name)

            # Make the parameter table
            source = ColumnDataSource(source_data[traj_name]['param_data_by_phase'][phase_name])
            columns = [
                TableColumn(field='param', title=f'{phase_name} Parameters'),
                TableColumn(field='val', title='Value'),
                TableColumn(field='units', title='Units'),
            ]
            param_tables.append(DataTable(source=source, columns=columns, index_position=None, sizing_mode='stretch_width'))

        # Plot the timeseries
        ts_units_dict = source_data[traj_name]['timeseries_units']

        figures = []
        x_range = None

        for var_name in ts_units_dict.keys():
            fig_kwargs = {'x_range': x_range} if x_range is not None else {}
            fig = figure(tools='pan,box_zoom,xwheel_zoom,undo,reset,save',
                         x_axis_label=f'{x_name} ({ts_units_dict[x_name]})',
                         y_axis_label=f'{var_name} ({ts_units_dict[var_name]})',
                         toolbar_location='above',
                         sizing_mode='stretch_both',
                         min_height=250, max_height=300,
                         margin=margin,
                         **fig_kwargs)
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
                    sol_plot = fig.circle(x='time', y=var_name, source=sol_source, color=color)
                    sim_plot = fig.line(x='time', y=var_name, source=sim_source, color=color)
                    legend_data.append((phase_name, [sol_plot, sim_plot]))

            legend = Legend(items=legend_data, location='center')
            fig.add_layout(legend, 'right')
            figures.append(fig)

        if len(figures) % 2 == 1:
            figures.append(None)

        # Put the DataTables in a GridBox so we can control the spacing
        gbc = []
        row = 0
        col = 0
        for i, table in enumerate(param_tables):
            gbc.append((table, row, col, 1, 1))
            col += 1
            if col >= ncols:
                col = 0
                row += 1
        param_panel = GridBox(children=gbc, spacing=50, sizing_mode='stretch_both')

        timeseries_panel = grid(children=figures, ncols=ncols, sizing_mode='stretch_both')

        tab_panes = Tabs(tabs=[TabPanel(child=timeseries_panel, title='Timeseries'),
                               TabPanel(child=param_panel, title='Parameters')],
                         sizing_mode='stretch_both',
                         active=0)

        summary = rf'Results of {prob._name}<br>Creation Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        report_layout = column(children=[Div(text=summary), tab_panes], sizing_mode='stretch_both')

        save(report_layout, filename=report_path, title=f'trajectory results for {traj_name}')



    # for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
    #     report_filename = f'{traj.pathname}_{_default_timeseries_report_filename}'
    #     report_path = str(Path(prob.get_reports_dir()) / report_filename)
    #     output_file(report_path)
    #     for phase in traj.system_iter(include_self=True, recurse=True, typ=dm.Phase):
    #         phase_name = phase.pathname.split()[-1]
    #
    #         # if param_comp:
            #
            #     param_outputs = param_comp.list_outputs(prom_name=True, units=True, out_stream=None)
            #
            #
            #     for abs_name, units in param_units_by_phase.items():
            #         param_vals_by_phase[phase_name][f'sol::{prom_name}'] = sol_case.get_val(prom_name, units=units)
            #         param_vals_by_phase[phase_name][f'sim::{prom_name}'] = sim_case.get_val(prom_name, units=units)

            # timeseries_sys = phase._get_subsystem('timeseries_sys')
            # if timeseries_sys:
            #     param_units_by_phase = {meta['prom_name']: meta['units'] for op, meta in timeseries_sys.list_outputs(prom_name=True,
            #                                                                                            units=True,
            #                                                                                            out_stream=None)}
            #     for prom_name, units in param_units_by_phase.items():
            #         timeseries_vals_by_phase[phase_name][f'sol::{prom_name}'] = sol_case.get_val(prom_name, units=units)
            #         timeseries_vals_by_phase[phase_name][f'sim::{prom_name}'] = sim_case.get_val(prom_name, units=units)


            # timeseries_sys = phase._get_subsystem('timeseries')
            # if timeseries_sys:
            #     output_data = timeseries_sys.list_outputs(prom_name=True, units=True, val=True, out_stream=None)
            #     timeseries_vals_by_phase[phase_name] = {meta['prom_name']: meta['val'] for _, meta in output_data}
            #     timeseries_units_by_phase[phase_name] = {meta['prom_name']: meta['units'] for _, meta in output_data}
            #



            # for param_comp in traj.system_iter(include_self=True, recurse=True, typ=dm.Phase):

            # print(phase.pathname)

            # for path, meta in phase.list_inputs(out_stream=None, prom_name=True, units=True, val=True):
            #     if meta['prom_name'].startswith('parameters:'):
            #         parameters_by_phase[phase_name][meta['prom_name']] = {'val': meta['val'], 'units': meta['units']}
            #
            # for path, meta in phase.timeseries.list_outputs(out_stream=None, prom_name=True, units=True):
            #     if not meta['prom_name'].startswith('parameters:'):
            #         timeseries_by_phase[phase_name][meta['prom_name']] = {'val': meta['val'], 'units': meta['units']}

if __name__ == '__main__':
    import openmdao.api as om
    cr = om.CaseReader('/Users/rfalck/Projects/dymos.git/dymos/examples/balanced_field/doc/dymos_solution.db')
    # print(cr.problem_metadata['tree'])
    # print(cr.problem_metadata['tree'])
    # for traj_tree in _meta_tree_subsys_iter(cr.problem_metadata['tree'], recurse=True, cls='Trajectory'):
    #     for phase_tree in _meta_tree_subsys_iter(traj_tree, recurse=True, cls=['Phase', 'AnalyticPhase']):
    #         print(phase_tree['name'])
    #         timeseries_meta = [child for child in phase_tree['children'] if child['name'] == 'timeseries'][0]
    #         print(timeseries_meta)




