import dymos as dm
from collections import OrderedDict
import inspect
import re
from pathlib import Path
from openmdao.visualization.htmlpp import HtmlPreprocessor
import openmdao.utils.reports_system as rptsys

CBN = 'children_by_name'
_default_linkage_report_title = 'Dymos Linkage Report'
_default_linkage_report_filename = 'linkage_report.html'


def create_linkage_report(traj, output_file: str = _default_linkage_report_filename,
                          show_all_vars=True,
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
    model_data['connections_list'].extend(_parameter_connections(traj))
    for c in model_data['connections_list']:
        print(c['src'],'->',c['tgt'])

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
        Either 'initial' or 'final' for non-parameters.

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


def _tree_var(var_name, phase, loc = None, var_name_prefix=""):
    """
    Create a dict to represent a variable in the tree.
    
    Parameters
    ----------
    var_name : str
        The name of the variable.
    phase : dymos.Phase
        Reference to the Phase object.
    loc : str
        Either 'initial' or 'final' for non-parameters.
    var_name_prefix : str
        A string to prepend to the variable name.

    Returns
    -------
    dict
        Information about the variable represented in the report.
    """
    class_name = str(phase.classify_var(var_name))
    opt = phase.parameter_options[var_name]['opt'] if class_name == 'parameter' else None
    
    return {
        'name': f"{var_name_prefix}{var_name}",
        'type': 'variable',
        'class': class_name,
        'fixed': _is_fixed(var_name, class_name, phase, loc),
        'paramOpt': opt
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

    tree[CBN]['params'] = {
        'name': 'params',
        'type': 'phase',
        CBN: OrderedDict()
    }

    for param_name, param in traj.parameter_options.items():
        tree[CBN]['params'][CBN][param_name] = {
            'name': str(param_name),
            'type': 'variable',
            'class': 'parameter',
            'fixed': False,
            'paramOpt': traj.parameter_options[param_name]['opt']
        }

    for phase_name, phase in traj._phases.items():
        tree_phase = {
            'name': str(phase_name),
            'type': 'phase',
            CBN: OrderedDict({
                'params': {'name': 'params', 'type': 'condition', CBN: OrderedDict()},
                'initial': {'name': 'initial', 'type': 'condition', CBN: OrderedDict()},
                'final': {'name': 'final', 'type': 'condition', CBN: OrderedDict()}
            })
        }

        tree[CBN][phase_name] = tree_phase

        params_children = tree_phase[CBN]['params'][CBN]
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
        for param_name, param in phase.parameter_options.items():
            params_children[param_name] = _tree_var(param_name, phase)
    
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

            try:
                tree_var_a = tree[CBN][phase_name_a][CBN][loc_a][CBN][var_a]
            except KeyError:
                if var_a in tree[CBN][phase_name_a][CBN]['params'][CBN]:
                    loc_a = 'params'
                    tree_var_a = tree[CBN][phase_name_a][CBN][loc_a][CBN][var_a]
                else:
                    # When linking two ODE outputs, it's possible for the variable to not show up in the tree.
                    # For example, see the 'ke' linkage in
                    # TestTwoPhaseCannonballODEOutputLinkage.test_traj_param_target_unspecified_units.
                    # For now, let these linkages go undisplayed in the report.
                    continue

            try:
                tree_var_b = tree[CBN][phase_name_b][CBN][loc_b][CBN][var_b]
            except KeyError:
                if var_b in tree[CBN][phase_name_b][CBN]['params'][CBN]:
                    loc_b = 'params'
                    tree_var_b = tree[CBN][phase_name_b][CBN][loc_b][CBN][var_b]
                else:
                    continue


            tree_var_a['linked'] = tree_var_b['linked'] = True
            tree_var_a['connected'] = tree_var_b['connected'] = options['connected']

            linkages.append({
                'src': f'{phase_name_a}.{loc_a}.{tree_var_a["name"]}',
                'src_fixed': tree_var_a['fixed'],
                'tgt': f'{phase_name_b}.{loc_b}.{tree_var_b["name"]}',
                'tgt_fixed': tree_var_b['fixed']
            })

    return linkages


def _conn_name_to_path(name):
    tokens = re.split(r'\W+', name)
    if tokens[1] == 'param_comp':
        return f'params.{tokens.pop()}'

    if tokens[3] == 'param_comp':
        return f'{tokens[2]}.params.{tokens.pop()}'

    return name


def _is_param_conn(name):
    return re.match(r'.*param_comp.*', name) and not re.match(r'.*timeseries.*', name)


def _parameter_connections(traj):
    allconn = traj._problem_meta['model_ref']()._conn_global_abs_in2out
    param_conns = []

    for tgt, src in allconn.items():
        if _is_param_conn(src) and _is_param_conn(tgt):
            param_conns.append({
                'src': _conn_name_to_path(src),
                'src_fixed': False,
                'tgt': _conn_name_to_path(tgt),
                'tgt_fixed': False
            })

    return param_conns


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
            # Only create a report for a trajectory with linkages
            if traj._linkages:
                report_filename = f'{sysname}_{_default_linkage_report_filename}'
                report_path = str(Path(prob.get_reports_dir()) / report_filename)
                create_linkage_report(traj, report_path)


def _linkage_report_register():
    rptsys.register_report('dymos.linkage', _run_linkage_report, _default_linkage_report_title,
                           'Problem', 'final_setup', 'post')
    rptsys._default_reports.append('dymos.linkage')
