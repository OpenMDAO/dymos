from pathlib import Path

from bokeh.io import output_notebook, output_file, save, show
from bokeh.layouts import gridplot, column
from bokeh.models import Legend
from bokeh.plotting import figure
import bokeh.palettes as bp

import dymos as dm
from dymos.options import options as dymos_options

import openmdao.utils.reports_system as rptsys


_default_timeseries_report_filename = 'timeseries_report.html'


def _meta_tree_subsys_iter(tree, recurse=True, cls=None):
    """
    Yield a generator of local subsystems of this system.

    Parameters
    ----------
    include_self : bool
        If True, include this system in the iteration.
    recurse : bool
        If True, iterate over the whole tree under this system.
    typ : str or None
        The type of the nodes to be iterated.
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


def make_timeseries_report(prob, solution_record_file=None, simulation_record_file=None, solution_history=False):
    """

    Parameters
    ----------
    prob
    solution_record_file
    simulation_record_file
    solution_history

    Returns
    -------

    """
    # For the primary timeseries in each phase in each trajectory, build a set of the pathnames
    # to be plotted.
    parameters_by_phase = {}
    timeseries_by_phase = {}

    for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
        report_filename = f'{traj.pathname}_{_default_timeseries_report_filename}'
        report_path = str(Path(prob.get_reports_dir()) / report_filename)
        output_file(report_path)
        for phase in traj.system_iter(include_self=True, recurse=True, typ=dm.Phase):
            phase_name = phase.pathname.split()[-1]
            parameters_by_phase[phase_name] = {}
            timeseries_by_phase[phase_name] = {}

            for path, meta in phase.list_inputs(out_stream=None, prom_name=True, units=True, val=True):
                if meta['prom_name'].startswith('parameters:'):
                    parameters_by_phase[phase_name][meta['prom_name']] = {'val': meta['val'], 'units': meta['units']}

            for path, meta in phase.timeseries.list_outputs(out_stream=None, prom_name=True, units=True):
                if not meta['prom_name'].startswith('parameters:'):
                    timeseries_by_phase[phase_name][meta['prom_name']] = {'val': meta['val'], 'units': meta['units']}

if __name__ == '__main__':
    import openmdao.api as om
    cr = om.CaseReader('/Users/rfalck/Projects/dymos.git/dymos/examples/balanced_field/test/dymos_solution.db')
    for traj_tree in _meta_tree_subsys_iter(cr.problem_metadata['tree'], recurse=True, cls='Trajectory'):
        for phase_tree in _meta_tree_subsys_iter(traj_tree, recurse=True, cls=['Phase', 'AnalyticPhase']):
            print(phase_tree['name'])
            timeseries_meta = [child for child in phase_tree['children'] if child['name'] == 'timeseries'][0]
            print(timeseries_meta)




