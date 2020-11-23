"""
Generic utilities for use by the grid refinement schemes.
"""
import numpy as np
import openmdao.api as om

from ..phase.phase import Phase
from ..utils.lagrange import lagrange_matrices
from ..utils.misc import get_rate_units
from ..utils.introspection import get_targets

from scipy.linalg import block_diag

from ..transcriptions import GaussLobatto, Radau

from .grid_refinement_ode_system import GridRefinementODESystem


def interpolation_lagrange_matrix(old_grid, new_grid):
    """
    Evaluate lagrange matrix to interpolate state and control values from the solved grid onto the new grid

    Parameters
    ----------
    old_grid: GridData
        The GridData object representing the grid on which the problem has been solved
    new_grid: GridData
        The GridData object representing the new, higher-order grid

    Returns
    -------
    L: np.ndarray
        The lagrange interpolation matrix

    """
    L_blocks = []
    D_blocks = []

    for iseg in range(old_grid.num_segments):
        i1, i2 = old_grid.subset_segment_indices['all'][iseg, :]
        indices = old_grid.subset_node_indices['all'][i1:i2]
        nodes_given = old_grid.node_stau[indices]

        i1, i2 = new_grid.subset_segment_indices['all'][iseg, :]
        indices = new_grid.subset_node_indices['all'][i1:i2]
        nodes_eval = new_grid.node_stau[indices]

        L_block, D_block = lagrange_matrices(nodes_given, nodes_eval)

        L_blocks.append(L_block)
        D_blocks.append(D_block)

    L = block_diag(*L_blocks)
    D = block_diag(*D_blocks)

    return L, D


def integration_matrix(grid):
    """
    Evaluate the Integration matrix of the given grid.

    Parameters
    ----------
    grid: GridData
        The GridData object representing the grid on which the integration matrix is to be evaluated

    Returns
    -------
    I: np.ndarray
        The integration matrix used to propagate initial states over segments

    """
    I_blocks = []

    for iseg in range(grid.num_segments):
        i1, i2 = grid.subset_segment_indices['all'][iseg, :]
        indices = grid.subset_node_indices['all'][i1:i2]
        nodes_given = grid.node_stau[indices]

        i1, i2 = grid.subset_segment_indices['all'][iseg, :]
        indices = grid.subset_node_indices['all'][i1:i2]
        nodes_eval = grid.node_stau[indices][1:]

        _, D_block = lagrange_matrices(nodes_given, nodes_eval)
        I_block = np.linalg.inv(D_block[:, 1:])
        I_blocks.append(I_block)

    I = block_diag(*I_blocks)

    return I


def eval_ode_on_grid(phase, transcription):
    """
    Evaluate the ODE from the given phase at all nodes of the given transcription.

    Parameters
    ----------
    phase : Phase
        A Phase object which has been executed and whose ODE is to be run at all nodes
        of the given transcription.
    transcription : Radau or GaussLobatto transcription instance
        The transcription object at which to execute the ODE of the given phase at all nodes.

    Returns
    -------
    x : dict of (str: np.ndarray)
        A dictionary of the state values from the phase interpolated to the new transcription.
    u : dict of (str: np.ndarray)
        A dictionary of the control values from the phase interpolated to the new transcription.
    p : dict of (str: np.ndarray)
        A dictionary of the polynomial control values from the phase interpolated to the new transcription.
    f : dict of (str: np.ndarray)
        A dictionary of the state rates computed in the phase's ODE at the new transcription points.

    """
    x = {}
    u = {}
    u_rate = {}
    u_rate2 = {}
    p = {}
    p_rate = {}
    p_rate2 = {}
    param = {}
    f = {}

    # Build the interpolation matrix which interpolates from all nodes on the old grid to
    # all nodes on the new grid.
    grid_data = transcription.grid_data
    L, _ = interpolation_lagrange_matrix(old_grid=phase.options['transcription'].grid_data,
                                         new_grid=grid_data)

    # Create a new problem for the grid_refinement
    # For this test, use the same grid as the original problem.
    p_refine = om.Problem(model=om.Group())
    grid_refinement_system = GridRefinementODESystem(grid_data=grid_data,
                                                     time=phase.time_options,
                                                     states=phase.state_options,
                                                     controls=phase.control_options,
                                                     polynomial_controls=phase.polynomial_control_options,
                                                     parameters=phase.parameter_options,
                                                     ode_class=phase.options['ode_class'],
                                                     ode_init_kwargs=phase.options[
                                                         'ode_init_kwargs'])
    p_refine.model.add_subsystem('grid_refinement_system', grid_refinement_system, promotes=['*'])
    p_refine.setup()

    # Set the values in the refinement problem using the outputs from the first
    ode = p_refine.model.grid_refinement_system.ode

    t_prev = phase.get_val('timeseries.time', units=phase.time_options['units'])
    t_phase_prev = phase.get_val('timeseries.time_phase', units=phase.time_options['units'])
    t_initial = np.repeat(t_prev[0, 0], repeats=transcription.grid_data.num_nodes, axis=0)
    t_duration = np.repeat(t_prev[-1, 0], repeats=transcription.grid_data.num_nodes, axis=0)
    t = np.dot(L, t_prev)
    t_phase = np.dot(L, t_phase_prev)
    targets = get_targets(ode, 'time', phase.time_options['targets'])
    time_phase_targets = get_targets(ode, 'time_phase', phase.time_options['time_phase_targets'])
    t_initial_targets = get_targets(ode, 't_initial', phase.time_options['t_initial_targets'])
    t_duration_targets = get_targets(ode, 't_duration', phase.time_options['t_duration_targets'])
    if targets:
        p_refine.set_val(f'time', t)
    if time_phase_targets:
        p_refine.set_val(f'time_phase', t_phase)
    if t_initial_targets:
        p_refine.set_val(f't_initial', t_initial)
    if t_duration_targets:
        p_refine.set_val(f't_duration', t_duration)

    for name, options in phase.state_options.items():
        x_prev = phase.get_val(f'timeseries.states:{name}', units=options['units'])
        x[name] = np.dot(L, x_prev)
        targets = get_targets(ode, name, options['targets'])
        if targets:
            p_refine.set_val(f'states:{name}', x[name])

    for name, options in phase.control_options.items():
        targets = get_targets(ode, name, options['targets'])
        rate_targets = get_targets(ode, f'{name}_rate', options['rate_targets'])
        rate2_targets = get_targets(ode, f'{name}_rate12', options['rate2_targets'])

        u_prev = phase.get_val(f'timeseries.controls:{name}', units=options['units'])
        u[name] = np.dot(L, u_prev)
        if targets:
            p_refine.set_val(f'controls:{name}', u[name])

        u_rate_prev = phase.get_val(f'timeseries.control_rates:{name}_rate')
        u_rate[name] = np.dot(L, u_rate_prev)
        if rate_targets:
            p_refine.set_val(f'control_rates:{name}_rate', u_rate[name])

        u_rate2_prev = phase.get_val(f'timeseries.control_rates:{name}_rate2')
        u_rate2[name] = np.dot(L, u_rate2_prev)
        if rate2_targets:
            p_refine.set_val(f'control_rates:{name}_rate2', u_rate2[name])

    for name, options in phase.polynomial_control_options.items():
        targets = get_targets(ode, name, options['targets'])
        rate_targets = get_targets(ode, f'{name}_rate', options['rate_targets'])
        rate2_targets = get_targets(ode, f'{name}_rate2', options['rate2_targets'])

        p_prev = phase.get_val(f'timeseries.polynomial_controls:{name}', units=options['units'])
        p[name] = np.dot(L, p_prev)
        if targets:
            p_refine.set_val(f'polynomial_controls:{name}', p[name])

        p_rate_prev = phase.get_val(f'timeseries.polynomial_control_rates:{name}_rate')
        p_rate[name] = np.dot(L, p_rate_prev)
        if rate_targets:
            p_refine.set_val(f'polynomial_control_rates:{name}_rate', p_rate[name])

        p_rate2_prev = phase.get_val(f'timeseries.polynomial_control_rates:{name}_rate2')
        p_rate2[name] = np.dot(L, p_rate2_prev)
        if rate2_targets:
            p_refine.set_val(f'polynomial_control_rates:{name}_rate2', p_rate2[name])

    # Configure the parameters
    for name, options in phase.parameter_options.items():
        targets = get_targets(ode, name, options['targets'])
        # The value of the parameter at one node
        d = phase.get_val(f'timeseries.parameters:{name}', units=options['units'])[0, ...]
        # Duplicate along the first axis by the number of nodes in the new transcription
        param[name] = np.atleast_2d(np.repeat(d, repeats=transcription.grid_data.num_nodes, axis=0))
        if targets:
            p_refine.set_val(f'parameters:{name}', param[name], units=options['units'])

    # Execute the model
    p_refine.run_model()

    # Assign the state rates on the new grid to f
    for name, options in phase.state_options.items():
        rate_units = get_rate_units(options['units'], phase.time_options['units'])
        rate_source = options['rate_source']
        rate_source_class = phase.classify_var(rate_source)
        if rate_source_class in {'time'}:
            src_units = phase.time_options['units']
            f[name] = om.convert_units(t, src_units, rate_units)
        elif rate_source_class in {'time_phase'}:
            src_units = phase.time_options['units']
            f[name] = om.convert_units(t_phase, src_units, rate_units)
        elif rate_source_class in {'state'}:
            src_units = phase.state_options[rate_source]['units']
            f[name] = om.convert_units(x[rate_source], src_units, rate_units)
        elif rate_source_class in {'input_control', 'indep_control'}:
            src_units = phase.control_options[rate_source]['units']
            f[name] = om.convert_units(u[rate_source], src_units, rate_units)
        elif rate_source_class in {'control_rate'}:
            u_name = rate_source[:-5]
            u_units = phase.control_options[u_name]['units']
            src_units = get_rate_units(u_units, phase.time_options['units'], deriv=1)
            f[name] = om.convert_units(u_rate[rate_source], src_units, rate_units)
        elif rate_source_class in {'control_rate2'}:
            u_name = rate_source[:-6]
            u_units = phase.control_options[u_name]['units']
            src_units = get_rate_units(u_units, phase.time_options['units'], deriv=2)
            f[name] = om.convert_units(u_rate2[rate_source], src_units, rate_units)
        elif rate_source_class in {'input_polynomial_control', 'indep_polynomial_control'}:
            src_units = phase.polynomial_control_options[rate_source]['units']
            f[name] = om.convert_units(p[rate_source], src_units, rate_units)
        elif rate_source_class in {'polynomial_control_rate'}:
            pc_name = rate_source[:-5]
            pc_units = phase.polynomial_control_options[pc_name]['units']
            src_units = get_rate_units(pc_units, phase.time_options['units'], deriv=1)
            f[name] = om.convert_units(p_rate[rate_source], src_units, rate_units)
        elif rate_source_class in {'polynomial_control_rate2'}:
            pc_name = rate_source[:-6]
            pc_units = phase.polynomial_control_options[pc_name]['units']
            src_units = get_rate_units(pc_units, phase.time_options['units'], deriv=2)
            f[name] = om.convert_units(p_rate2[rate_source], src_units, rate_units)
        elif rate_source_class in {'parameter'}:
            src_units = phase.parameter_options[rate_source]['units']
            f[name] = om.convert_units(param[rate_source], src_units, rate_units)
        elif rate_source_class in {'ode'}:
            f[name] = np.atleast_2d(p_refine.get_val(f'ode.{rate_source}', units=rate_units))

        if options['shape'] == (1,):
            f[name] = f[name].T

    return x, u, p, f


def compute_state_quadratures(x_hat, f_hat, t_duration, transcription):
    """
    Compute the integral of the given states at each node in the given transcription.

    This estimation of the states is computed with a quadrature.

    Parameters
    ----------
    x_hat : dict of (str: float)
        The interpolated state values at the nodes for each state.
    f_hat : dict of (str: float)
        State rates computed using the interpolated state values at each node.
    t_duration : float
        The time duration of the phase.
    transcription : Radau or GaussLobatto
        The transcription instance used to compute f_hat.

    Returns
    -------
    x_prime : dict
        A dictionary keyed by state name containing the estimated state values at each node,
        computed using a quadrature.

    """
    x_prime = {}
    gd = transcription.grid_data

    # Build the integration matrix which integrates state values at all nodes on the new grid.
    I = integration_matrix(gd)

    left_end_idxs = gd.subset_node_indices['segment_ends'][0::2]
    all_idxs = gd.subset_node_indices['all']
    not_left_end_idxs = np.array(sorted(list(set(all_idxs).difference(set(left_end_idxs)))))

    dt_dstau = np.atleast_2d(0.5 * t_duration * gd.node_dptau_dstau[not_left_end_idxs]).T

    if x_hat.keys() != f_hat.keys():
        raise ValueError('x_hat and f_hat don\'t contain the same states.\n'
                         f'x_hat states are: {list(x_hat.keys())}\n'
                         f'f_hat states are: {list(f_hat.keys())}')

    for state_name in x_hat:
        x_prime[state_name] = np.zeros_like(x_hat[state_name])
        x_prime[state_name][left_end_idxs, ...] = x_hat[state_name][left_end_idxs, ...]
        nnps = np.array(gd.subset_num_nodes_per_segment['all']) - 1
        left_end_idxs_repeated = np.repeat(left_end_idxs, nnps)
        x_prime[state_name][not_left_end_idxs, ...] = \
            x_hat[state_name][left_end_idxs_repeated, ...] \
            + dt_dstau * np.dot(I, f_hat[state_name][not_left_end_idxs, ...])

    return x_prime


def check_error(phases):
    """
    Compute the error in every solved segment

    Returns
    -------
    need_refinement: dict
        Indicator for which segments of the given phase require grid refinement

    """
    refine_results = {}

    for phase_path, phase in phases.items():
        refine_results[phase_path] = {}

        # Save the original grid to the refine results
        tx = phase.options['transcription']
        gd = tx.grid_data
        numseg = gd.num_segments

        refine_results[phase_path]['num_segments'] = numseg
        refine_results[phase_path]['order'] = gd.transcription_order
        refine_results[phase_path]['segment_ends'] = gd.segment_ends
        refine_results[phase_path]['need_refinement'] = np.zeros(numseg, dtype=bool)
        refine_results[phase_path]['max_rel_error'] = np.zeros(numseg, dtype=float)  # Eq. 21
        refine_results[phase_path]['error_state'] = ['' for _ in phase.state_options]

        # Instantiate a new phase as a copy of the old one, but first up the transcription order
        # by 1 for Radau and by 2 for Gauss-Lobatto
        new_num_segments = tx.options['num_segments']
        new_segment_ends = tx.options['segment_ends']
        new_compressed = tx.options['compressed']
        if isinstance(tx, GaussLobatto):
            new_order = tx.options['order'] + 2
            new_tx = GaussLobatto(num_segments=new_num_segments, order=new_order,
                                  segment_ends=new_segment_ends, compressed=new_compressed)
        elif isinstance(tx, Radau):
            new_order = tx.options['order'] + 1
            new_tx = Radau(num_segments=new_num_segments, order=new_order,
                           segment_ends=new_segment_ends, compressed=new_compressed)
        else:
            # Only refine GuassLobatto or Radau transcription phases
            continue

        # Let x be the interpolated states on the new transcription
        # Let f by the evaluated state rates given the interpolation of the states and controls
        # onto the new grid.
        x, _, _, f = eval_ode_on_grid(phase=phase, transcription=new_tx)

        # x_hat is the state value at each node computed using a quadrature
        # from the initial state value in each segment and the computed state rates
        # at each node in the new transcription.
        x_hat = compute_state_quadratures(x, f, phase.get_val('t_duration'), new_tx)

        E = {}  # The absolute error computed in each state at each node (Eq. 20 pt 1)
        e = {}  # The relative error computed in each state at each node (Eq. 20 pt 2)

        for state_name, options in phase.state_options.items():
            E[state_name] = np.abs(x_hat[state_name] - x[state_name])  # Equation 20.1
            e[state_name] = np.zeros_like(E[state_name])               # Equation 20.2
            for k in range(numseg):
                i1, i2 = new_tx.grid_data.subset_segment_indices['all'][k, :]
                k_idxs = new_tx.grid_data.subset_node_indices['all'][i1:i2]
                e[state_name][k_idxs, ...] = E[state_name][k_idxs] \
                    / (1.0 + np.max(np.abs(x[state_name][k_idxs])))
                if np.any(np.max(e[state_name][k_idxs]) > refine_results[phase_path]['max_rel_error'][k]):
                    refine_results[phase_path]['max_rel_error'][k] = np.max(e[state_name][k_idxs])
                    refine_results[phase_path]['error_state'] = state_name
                    if refine_results[phase_path]['max_rel_error'][k] > phase.refine_options['tolerance']:
                        refine_results[phase_path]['need_refinement'][k] = True

    return refine_results
