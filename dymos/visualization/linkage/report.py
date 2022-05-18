import dymos as dm
from openmdao.visualization.htmlpp import HtmlPreprocessor
from collections import OrderedDict
import json
import inspect 
import os

CBN = 'children_by_name'

def create_linkage_report(traj, output_file: str='linkage_report.html',
                          show_all_vars=False,
                          title='Dymos Linkage Report', embedded=False):
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
    openmdao_dir = os.path.dirname(inspect.getfile(openmdao))
    vis_dir = os.path.join(openmdao_dir, "visualization/n2_viewer")
    
    dymos_dir = os.path.dirname(inspect.getfile(dm))
    reports_dir = os.path.join(dymos_dir, "visualization/linkage")
    # pretty = json.dumps(model_data, indent=2)
    # print(pretty)

    HtmlPreprocessor(os.path.join(reports_dir, "report_template.html"), output_file,
                    search_path=[vis_dir, reports_dir], allow_overwrite=True, var_dict=html_vars,
                    verbose=False).run()

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
                'initial': { 'name': 'initial', 'type': 'condition', CBN: OrderedDict() },
                'final': { 'name': 'final', 'type': 'condition', CBN: OrderedDict() }
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
        for state_name, s in phase.state_options.items():
            for loc, child in condition_children.items():
                child[state_name] = _tree_var(state_name, phase, loc, 'states:')

        # Controls
        for control_name, c in phase.control_options.items():
            for loc, child in condition_children.items():
                child[control_name] =  _tree_var(control_name, phase, loc, 'controls:')

        # Polynomial Controls
        for pc_name, pc in phase.polynomial_control_options.items():
            for loc, child in condition_children.items():
                child[pc_name] = _tree_var(pc_name, phase, loc, 'polynomial_controls:')

        # Parameters
        for param_name, par in phase.parameter_options.items():
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
        for child_name, child in tree_dict[CBN].items():
            _convert_dicts_to_lists(child, show_all_vars)
            if _display_child(child, show_all_vars):
                tree_dict['children'].append(child)

        tree_dict.pop(CBN)

        
