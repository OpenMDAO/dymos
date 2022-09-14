import dymos as dm
from collections import OrderedDict
import inspect
from pathlib import Path
from openmdao.visualization.htmlpp import HtmlPreprocessor
import openmdao.utils.reports_system as rptsys

CBN = 'children_by_name'
_default_linkage_report_title = 'Dymos Linkage Report'
_default_linkage_report_filename = 'linkage_report.html'


def create_linkage_report(traj, output_file: str = _default_linkage_report_filename,
                          show_all_vars=False,
                          title=_default_linkage_report_title, embedded=False):
    """
    Create a tree based on the trajectory and linkages, then export as HTML.

    Parameters
    ----------
    traj : Trajectory
        The Dymos Trajectory containing the phase and linkage data.
    output_file : str
        Where to write the HTML file with the report diagram.
    show_all_vars : bool
        If True, display all parameters and controls even if not linked.
    title : str
        The title to display in the HTML document.
    embedded : bool
        Whether the file is to be included in another. If True, leave out some tags.
    """
    model_data = _trajectory_to_dict(traj)
    model_data['connections_list'] = _linkages_to_list(traj, model_data)
    _convert_dicts_to_lists(model_data['tree'], show_all_vars)

    html_vars = {
        'title': title,
        'embeddable': "non-embedded-diagram",
        'sw_version': 0.1,
        'model_data': model_data
    }

    import openmdao
    openmdao_dir = Path(inspect.getfile(openmdao)).parent
    vis_dir = openmdao_dir / 'visualization' / 'n2_viewer'

    dymos_dir = Path(inspect.getfile(dm)).parent
    reports_dir = dymos_dir / 'visualization' / 'linkage'

    HtmlPreprocessor(str(reports_dir / 'report_template.html'), output_file,
                     search_path=[str(vis_dir), str(reports_dir)], allow_overwrite=True,
                     var_dict=html_vars, verbose=False).run()


def _is_fixed(var_name: str, class_name: str, phase, loc: str):
    """
    Determine whether a variable is fixed or not.

    Parameters
    ----------
    var_name : str
        Identifier of the variable as known to the phase.
    class_name : str
        The type of variable.
    phase : Phase
        The phase where the variable is found.
    loc : str
        Either 'initial' or 'final'.

    Returns
    -------
    bool
        True if the variable is fixed, otherwise False.
    """
    fixed = False

    if class_name == 'time':
        fixed = phase.is_time_fixed(loc)
    elif class_name == 'state':
        fixed = phase.is_state_fixed(var_name, loc)
    elif class_name in {'input_control', 'indep_control'}:
        fixed = phase.is_control_fixed(var_name, loc)
    elif class_name in {'input_polynomial_control', 'indep_polynomial_control'}:
        fixed = phase.is_polynomial_control_fixed(var_name, loc)
    else:
        fixed = True

    return bool(fixed)


def _tree_var(var_name, phase, loc, var_name_prefix=""):
    """ Create a dict to represent a variable in the tree. """
    class_name = phase.classify_var(var_name)

    return {
        'name': f"{var_name_prefix}{var_name}",
        'type': 'variable',
        'class': str(class_name),
        'fixed': _is_fixed(var_name, class_name, phase, loc)
    }


def _trajectory_to_dict(traj):
    """
    Iterate over the variables in the trajectory's phases and create a hierarchical structure.

    Parameters
    ----------
    traj : Trajectory
        The trajectory containing the phases to find variables in.

    Returns
    -------
    dict
        A dictionary in the form: root -> phases -> conditions (initial/final) -> variables.
    """
    model_data = {
        'tree': {
            'name': 'Trajectory',
            'type': 'root',
            CBN: OrderedDict()
        }
    }

    tree = model_data['tree']

    for phase_name, phase in traj._phases.items():
        tree_phase = {
            'name': str(phase_name),
            'type': 'phase',
            CBN: OrderedDict({
                'initial': {'name': 'initial', 'type': 'condition', CBN: OrderedDict()},
                'final': {'name': 'final', 'type': 'condition', CBN: OrderedDict()}
            })
        }

        tree[CBN][phase_name] = tree_phase

        condition_children = OrderedDict({
            'initial': tree_phase[CBN]['initial'][CBN],
            'final': tree_phase[CBN]['final'][CBN]
        })

        # Time
        for loc, child in condition_children.items():
            child['time'] = _tree_var('time', phase, loc)

        # States
        for state_name in phase.state_options:
            for loc, child in condition_children.items():
                child[state_name] = _tree_var(state_name, phase, loc, 'states:')

        # Controls
        for control_name in phase.control_options:
            for loc, child in condition_children.items():
                child[control_name] = _tree_var(control_name, phase, loc, 'controls:')

        # Polynomial Controls
        for pc_name in phase.polynomial_control_options:
            for loc, child in condition_children.items():
                child[pc_name] = _tree_var(pc_name, phase, loc, 'polynomial_controls:')

        # Parameters
        for param_name in phase.parameter_options:
            for loc, child in condition_children.items():
                child[param_name] = _tree_var(param_name, phase, loc, 'parameters:')

    return model_data


def _linkages_to_list(traj, model_data):
    """
    Extract variables from the trajectory's linkages dict and put them in an array.

    Also flag in the model_data dict whether a variable is linked and/or fixed.

    Parameters
    ----------
    traj : Trajectory
        The trajectory with the linkages structure.
    model_data : dict
        Data extracted from the trajectory's phases.

    Returns
    -------
    list
        Dict objects with src, tgt, src_fixed, and tgt_fixed keys.
    """
    linkages = []
    tree = model_data['tree']

    for phase_pair, var_dict in traj._linkages.items():
        phase_name_a, phase_name_b = phase_pair

        for var_pair, options in var_dict.items():
            var_a, var_b = var_pair
            loc_a = options['loc_a']
            loc_b = options['loc_b']

            tree_var_a = tree[CBN][phase_name_a][CBN][loc_a][CBN][var_a]
            tree_var_b = tree[CBN][phase_name_b][CBN][loc_b][CBN][var_b]

            tree_var_a['linked'] = tree_var_b['linked'] = True
            tree_var_a['connected'] = tree_var_b['connected'] = options['connected']

            linkages.append({
                'src': f'{phase_name_a}.{loc_a}.{tree_var_a["name"]}',
                'src_fixed': tree_var_a['fixed'],
                'tgt': f'{phase_name_b}.{loc_b}.{tree_var_b["name"]}',
                'tgt_fixed': tree_var_b['fixed']
            })

    return linkages


def _display_child(child, show_all_vars):
    """ Determine whether the object should be included in the diagram. """
    if show_all_vars is True:
        return True

    if CBN in child or 'children' in child:
        return True

    if 'linked' in child:
        return True

    if child['class'] == 'time' or child['class'] == 'state':
        return True

    return False


def _convert_dicts_to_lists(tree_dict, show_all_vars):
    """ Convert all children_by_name dicts to lists. """
    if CBN in tree_dict:
        tree_dict['children'] = []
        for child in tree_dict[CBN].values():
            _convert_dicts_to_lists(child, show_all_vars)
            if _display_child(child, show_all_vars):
                tree_dict['children'].append(child)

        tree_dict.pop(CBN)


def _run_linkage_report(prob):
    """ Function invoked by the reports system """

    # Find all Trajectory objects in the Problem. Usually, there's only one
    for sysname, sysinfo in prob.model._subsystems_allprocs.items():
        if isinstance(sysinfo.system, dm.Trajectory):
            traj = sysinfo.system
            # Only create a report for a trajectory with multiple phases
            if len(traj._phases) > 1:
                report_filename = f'{sysname}_{_default_linkage_report_filename}'
                report_path = str(Path(prob.get_reports_dir()) / report_filename)
                create_linkage_report(traj, report_path)


def _linkage_report_register():
    rptsys.register_report('dymos.linkage', _run_linkage_report, _default_linkage_report_title,
                           'Problem', 'final_setup', 'post')
    rptsys._default_reports.append('dymos.linkage')
