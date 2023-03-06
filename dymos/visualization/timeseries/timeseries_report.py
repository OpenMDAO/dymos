import dymos as dm
from collections import OrderedDict
import inspect
import re
from pathlib import Path
from openmdao.visualization.htmlpp import HtmlPreprocessor
import openmdao.utils.reports_system as rptsys

_default_timeseries_report_title = 'Dymos Timeseries Report'
_default_timeseries_report_filename = 'timeseries_report.html'

def _run_timeseries_report(prob):
    """ Function invoked by the reports system """

    # Find all Trajectory objects in the Problem. Usually, there's only one
    for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
        report_filename = f'{traj.pathname}_{_default_timeseries_report_filename}'
        report_path = str(Path(prob.get_reports_dir()) / report_filename)
        create_timeseries_report(traj, report_path)


# def _timeseries_report_register():
#     rptsys.register_report('dymos.timeseries', _run_timeseries_report, _default_timeseries_report_title,
#                            'prob', 'run_driver', 'post')
#     rptsys.register_report('dymos.timeseries', _run_timeseries_report, _default_timeseries_report_title,
#                            'prob', 'run_model', 'post')
#     rptsys._default_reports.append('dymos.timeseries')
