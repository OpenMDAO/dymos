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


def _create_model_data(traj):
    """
    Creates the model_data dictionary for the given trajectory.

    Parameters
    ----------
    traj : A dymos Trajectory object.

    Returns
    -------
    dict
        The generated model_data.
    """
    model_data = _trajectory_to_dict(traj)
    model_data['connections_list'] = _linkages_to_list(traj, model_data)
    model_data['connections_list'].extend(_parameter_connections(traj, model_data))
    return model_data


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
    model_data = _create_model_data(traj)

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
    elif class_name == 'control':
        fixed = phase.is_control_fixed(var_name, loc)
    else:
        fixed = True

    return bool(fixed)


def _tree_var(var_name, phase, loc=None, var_name_prefix=""):
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
            'paramOpt': param['opt'],
            'connected': True
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
        for var_pair, options in var_dict.items():
            skip = False
            loc = [options['loc_a'], options['loc_b']]
            tree_var = []

            for i in range(2):
                phase_cbn = tree[CBN][phase_pair[i]][CBN]
                if var_pair[i] not in phase_cbn[loc[i]][CBN]:
                    if var_pair[i] in phase_cbn['params'][CBN]:
                        # Variable is in the parameters area rather than initial or final condition
                        loc[i] = 'params'
                    else:
                        # When linking two ODE outputs, it's possible for the variable to not show up in the tree.
                        # For example, see the 'ke' linkage in
                        # TestTwoPhaseCannonballODEOutputLinkage.test_traj_param_target_unspecified_units.
                        # For now, let these linkages go undisplayed in the report.
                        skip = True

                if skip is False:
                    tree_var.append(phase_cbn[loc[i]][CBN][var_pair[i]])

            if skip is True:
                continue

            tree_var[0]['linked'] = tree_var[1]['linked'] = True
            tree_var[0]['connected'] = tree_var[1]['connected'] = options['connected']

            linkages.append({
                'src': f'{phase_pair[0]}.{loc[0]}.{tree_var[0]["name"]}',
                'src_fixed': tree_var[0]['fixed'],
                'tgt': f'{phase_pair[1]}.{loc[1]}.{tree_var[1]["name"]}',
                'tgt_fixed': tree_var[1]['fixed']
            })

    return linkages


def _conn_name_to_path(name):
    """ Convert the full name of the connection to a format the diagram uses. """
    if re.search(r'.*param_comp.parameter_vals.*|.*param_comp.parameters.*', name):
        # Special handling in case the name contains colons
        tokens = re.split(r'\.', name)
        last_name = tokens[-1]
        param_name = re.split(r':', last_name, 1)
        tokens[-1] = param_name[0]
        tokens.append(param_name[1])
    else:
        tokens = re.split(r'\W+', name)

    if len(tokens) > 0:
        if tokens[1] == 'param_comp':
            return f'params.{tokens.pop()}'

        # Example: traj.linkages.v1_to_vr:alpha -> v1_to_vr.alpha
        if tokens[1] == 'linkages':
            return f'{tokens[2]}.{tokens[3]}'

        if len(tokens) > 3 and tokens[3] == 'param_comp':
            return f'{tokens[2]}.params.{tokens.pop()}'

    return name


def _is_param_conn(name) -> bool:
    """ Determine if the specified connection involves a parameter. """
    return re.match(r'.*param_comp\.parameters:.*', name) \
        and not re.match(r'.*param_comp\.parameter_vals:.*', name)


def _is_ignored_conn(name) -> bool:
    """ Determine if an connection endpoint should be ignored for this diagram. """
    return re.match(r'.*timeseries.*', name) or re.match(r'.*_auto_ivc.*', name)


def _var_ref_from_path(tree, path: str):
    """ Find a reference into the tree from a path string. """
    tokens = re.split(r'\.', path)

    if (len(tokens) < 2):
        return None

    refpath = tree

    try:
        for t in tokens:
            refpath = refpath[CBN][t]
    except (KeyError):
        return None

    return refpath


def _get_fixed_val(tree, path: str) -> bool:
    """ Get the value of the fixed property for the specified path. """
    ref = _var_ref_from_path(tree, path)
    fixed = False if (ref is None or 'fixed' not in ref) else ref['fixed']

    return fixed


def _parameter_connections(traj, model_data):
    """ Find all parameter-to-parameter connections. """
    allconn = traj._problem_meta['model_ref']()._conn_global_abs_in2out
    tree = model_data['tree']
    param_conns = []

    for tgt, src in allconn.items():
        is_ignored = (_is_ignored_conn(src) or _is_ignored_conn(tgt))
        is_param_conn = (_is_param_conn(src) or _is_param_conn(tgt))
        if not is_ignored and is_param_conn:
            src_path = _conn_name_to_path(src)
            tgt_path = _conn_name_to_path(tgt)
            param_conns.append({
                'src': src_path,
                'src_fixed': _get_fixed_val(tree, src_path),
                'tgt': tgt_path,
                'tgt_fixed': _get_fixed_val(tree, tgt_path)
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
    for traj in prob.model.system_iter(include_self=True, recurse=True, typ=dm.Trajectory):
        # Only create a report for a trajectory with linkages
        if traj._linkages:
            report_filename = f'{traj.pathname}_{_default_linkage_report_filename}'
            report_path = str(Path(prob.get_reports_dir()) / report_filename)
            create_linkage_report(traj, report_path)


def _linkage_report_register():
    rptsys.register_report('dymos.linkage', _run_linkage_report, _default_linkage_report_title,
                           'Problem', 'final_setup', 'post')
    rptsys._default_reports.append('dymos.linkage')
