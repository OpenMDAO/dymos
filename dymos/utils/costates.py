"""
Estimation of the costates of collocation phases from the Lagrange multipliers of the NLP.
"""
import numpy as np

import dymos as dm
from dymos.utils.birkhoff import birkhoff_matrix
from dymos.utils.cgl import cgl
from dymos.utils.lagrange import lagrange_matrices
from dymos.utils.lgl import lgl


def _find_constraint(driver, phase, suffix):
    """
    Find the driver constraint of a phase from the end of the path of its source.

    Parameters
    ----------
    driver : om.Driver
        The driver of the problem.
    phase : dm.Phase
        The phase which owns the constraint.
    suffix : str
        The end of the absolute path of the constrained output.

    Returns
    -------
    name : str or None
        The name of the constraint in the driver, or None if there is no such constraint.
    meta : dict or None
        The metadata of the constraint.
    """
    prefix = phase.pathname + '.'
    for name, meta in driver._cons.items():
        if meta['source'].startswith(prefix) and meta['source'].endswith(suffix):
            return name, meta
    return None, None


def _multiplier_array(con_multipliers, name, meta, shape):
    """
    Return the multipliers of a constraint as an array of the shape of the constraint.

    Parameters
    ----------
    con_multipliers : dict
        The multipliers of the active constraints from Driver.compute_lagrange_multipliers.
    name : str
        The name of the constraint in the driver.
    meta : dict
        The metadata of the constraint.
    shape : tuple
        The shape of the constraint.

    Returns
    -------
    ndarray
        The multipliers, with zeros for the entries which are not active.
    """
    mult = np.zeros(meta['size'])
    if name in con_multipliers:
        idxs = np.asarray(con_multipliers[name]['indices']).ravel()
        mult[idxs] = np.asarray(con_multipliers[name]['multipliers']).ravel()
    return mult.reshape(shape)


def _birkhoff_weights(gd):
    """
    Return the quadrature weights of the nodes of a Birkhoff grid.

    The weights of a segment are the last row of its Birkhoff matrix, which integrates the
    state rate over the segment. For a Chebyshev grid they differ from gd.node_weight.

    Parameters
    ----------
    gd : GridData
        The grid of a Birkhoff transcription.

    Returns
    -------
    ndarray
        The weights of all nodes, segment by segment.
    """
    weights = []
    for iseg in range(gd.num_segments):
        n = gd.subset_num_nodes_per_segment['all'][iseg]
        tau, w = lgl(n) if gd.grid_type == 'lgl' else cgl(n)
        weights.append(birkhoff_matrix(tau, w, grid_type=gd.grid_type)[-1, :])
    return np.concatenate(weights)


def estimate_costates(problem, feas_tol=1.0E-6):
    """
    Estimate the costates of the collocation phases of a converged problem.

    The costates are recovered from the Lagrange multipliers of the defect constraints
    (covector mapping). The multipliers come from Driver.compute_lagrange_multipliers, so the
    optimizer does not have to provide them.

    The Hamiltonian is taken as H = lambda^T f, with an integrated cost, if any, as one of the
    states. With this convention lambda(t) is the sensitivity of the optimal objective to a
    change of the state at time t, and the costate of an integrated cost state is 1.

    Phases with a Radau, GaussLobatto or Birkhoff transcription are supported. States whose
    defects are not constraints of the driver (solve_segments) are left out.

    Parameters
    ----------
    problem : om.Problem
        A problem whose driver has converged to an optimum.
    feas_tol : float
        The feasibility tolerance which decides whether a constraint is active.

    Returns
    -------
    dict
        For each phase, keyed by its pathname, a dict with the keys
        'time', the time at the nodes where the costates are estimated;
        'node_idxs', the indices of these nodes among all nodes of the phase;
        'costates', a dict from state name to an array of shape (num_nodes,) + shape;
        'initial_costates' and 'final_costates', dicts from state name to the costate at the
        start and at the end of the phase (empty for GaussLobatto).
    """
    driver = problem.driver

    # The dense solve is used because the default sparse solve of OpenMDAO does not give
    # the multipliers to full accuracy (OpenMDAO issue 3859).
    _, con_multipliers = driver.compute_lagrange_multipliers(driver_scaling=False,
                                                             feas_tol=feas_tol,
                                                             use_sparse_solve=False)

    results = {}

    for phase in problem.model.system_iter(include_self=True, recurse=True, typ=dm.Phase):
        tx = phase.options['transcription']
        if not isinstance(tx, (dm.Radau, dm.GaussLobatto, dm.Birkhoff)):
            continue

        gd = tx.grid_data
        time_name = phase.time_options['name']
        time_all = phase.get_val(f'timeseries.{time_name}').ravel()

        if isinstance(tx, dm.Birkhoff):
            suffix = '.state_rate_defects:{}'
            node_idxs = gd.subset_node_indices['all']
            weights = _birkhoff_weights(gd)
        else:
            suffix = 'collocation_constraint.defects:{}'
            node_idxs = gd.subset_node_indices['col']
            weights = gd.node_weight[node_idxs]

        costates = {}
        initial_costates = {}
        final_costates = {}

        for state_name, options in phase.state_options.items():
            name, meta = _find_constraint(driver, phase, suffix.format(state_name))
            if name is None:
                continue

            state_shape = tuple(options['shape'])
            mult = _multiplier_array(con_multipliers, name, meta, (len(node_idxs),) + state_shape)

            # The Lagrangian of the NLP is J + mult^T defect and the defect at a node is
            # dx/dstau - f dt/dstau, so mult = -w * lambda.
            w = weights.reshape((-1,) + (1,) * len(state_shape))
            costates[state_name] = -mult / w

            if isinstance(tx, dm.Radau):
                # The last node of the phase is not a collocation node. Stationarity of the
                # Lagrangian with respect to the final state gives
                # lambda_f = -sum_k mult_k D[k, final] over the nodes of the last segment.
                i0, i1 = gd.segment_indices[gd.num_segments - 1]
                seg_col = np.intersect1d(np.arange(i0, i1), node_idxs) - i0
                seg_nodes = gd.node_stau[i0:i1]
                _, D = lagrange_matrices(seg_nodes, seg_nodes[seg_col])
                final_costates[state_name] = -np.tensordot(D[:, -1], mult[-len(seg_col):, ...],
                                                           axes=(0, 0))
                initial_costates[state_name] = costates[state_name][0, ...]

            elif isinstance(tx, dm.Birkhoff):
                # In each segment the state defects are
                #   X_i - x_a - (B V)_i = 0      (multipliers omega_i, one for each node)
                #   B[-1, :] V - x_b + x_a = 0   (multiplier omega_b)
                # Stationarity with respect to the final state x_b gives lambda_b = omega_b, and
                # the sensitivity to the initial state x_a gives lambda_a = omega_b - sum(omega).
                name_s, meta_s = _find_constraint(driver, phase, f'.state_defects:{state_name}')
                if name_s is not None:
                    nn = gd.subset_num_nodes_per_segment['all']
                    rows = sum(nn) + gd.num_segments
                    omega = _multiplier_array(con_multipliers, name_s, meta_s,
                                              (rows,) + state_shape)
                    first = omega[:nn[0] + 1, ...]
                    initial_costates[state_name] = first[-1, ...] - first[:-1, ...].sum(axis=0)
                    final_costates[state_name] = omega[-1, ...]

        results[phase.pathname] = {'time': time_all[node_idxs], 'node_idxs': node_idxs,
                                   'costates': costates, 'initial_costates': initial_costates,
                                   'final_costates': final_costates}

    return results
