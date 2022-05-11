import dymos as dm
from collections import OrderedDict
import json

def create_linkage_report(traj):
    model_data = _trajectory_to_dict(traj)
    model_data['connections'] = _linkages_to_list(traj, model_data)
    _convert_dicts_to_lists(model_data['tree'])

    pretty = json.dumps(model_data, indent=2)
    print(pretty)

def _tree_var(var_name, var_class):
    return {
        'name': var_name,
        'type': 'variable',
        'class': var_class
    }

def _trajectory_to_dict(traj):
    model_data = {
        'tree': {
            'name': 'root',
            'type': 'group',
            'children_by_name': OrderedDict()
        }
    }

    tree = model_data['tree']

    for pname, p in traj._phases.items():
        tree_phase = {
            'name': pname,
            'type': 'phase',
            'children_by_name': OrderedDict({
                'initial': { 'name': 'initial', 'type': 'condition', 'children_by_name': OrderedDict() },
                'final': { 'name': 'final', 'type': 'condition', 'children_by_name': OrderedDict() }
            })
        }

        tree['children_by_name'][pname] = tree_phase

        condition_children = [ tree_phase['children_by_name']['initial']['children_by_name'],
            tree_phase['children_by_name']['final']['children_by_name'] ]

        # Time
        time_child = _tree_var('time', p.classify_var('time'))
        for child in condition_children:
            child['time'] = time_child

        # States
        for sname, s in p.state_options.items():
            for child in condition_children:
                child[sname] = _tree_var(f'states:{sname}', p.classify_var(sname))

        # Controls
        for cname, c in p.control_options.items():
            for child in condition_children:
                child[cname] =  _tree_var(f'controls:{cname}', p.classify_var(cname))

        # Polynomial Controls
        for pcname, pc in p.polynomial_control_options.items():
            for child in condition_children:
                child[pcname] = _tree_var(f'polynomial_controls:{pcname}', p.classify_var(pcname))

        # Parameters
        for parname, par in p.parameter_options.items():
            for child in condition_children:
                child[parname] = _tree_var(f'parameters:{parname}', p.classify_var(parname))

    return model_data

def _is_fixed(var_name, class_name, phase, loc):
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

def _linkages_to_list(traj, model_data):
    linkages = []
    tree = model_data['tree']

    for phase_pair, var_dict in traj._linkages.items():
        phase_name_a, phase_name_b = phase_pair

        phase_a = traj._get_subsystem(f'phases.{phase_name_a}')
        phase_b = traj._get_subsystem(f'phases.{phase_name_b}')

        for var_pair, options in var_dict.items():
            var_a, var_b = var_pair
            loc_a = options['loc_a']
            loc_b = options['loc_b']

            tree_var_a = tree['children_by_name'][phase_name_a]['children_by_name'][loc_a]['children_by_name'][var_a]
            tree_var_b = tree['children_by_name'][phase_name_b]['children_by_name'][loc_b]['children_by_name'][var_b]

            tree_var_a['linked'] = tree_var_b['linked'] = True

            class_a = phase_a.classify_var(var_a)
            class_b = phase_b.classify_var(var_b)

            tree_var_a['connected'] = tree_var_b['connected'] = options['connected']
            tree_var_a['fixed'] = _is_fixed(var_a, class_a, phase_a, loc_a)
            tree_var_b['fixed'] = _is_fixed(var_b, class_b, phase_b, loc_b)

            linkages.append({
                'src': tree_var_a['name'],
                'src_fixed': tree_var_a['fixed'],
                'tgt': tree_var_b['name'],
                'tgt_fixed': tree_var_b['fixed']
            })

    return linkages

def _convert_dicts_to_lists(tree_dict):
    if 'children_by_name' in tree_dict:
        tree_dict['children'] = []
        for child_name, child in tree_dict['children_by_name'].items():
            _convert_dicts_to_lists(child)
            tree_dict['children'].append(child)

        tree_dict.pop('children_by_name')

        
