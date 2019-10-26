from __future__ import division, print_function, absolute_import

from collections.abc import Iterable, Sequence
from ...transcriptions.grid_data import GridData
from ...transcriptions.common import TimeComp
from ...phase.phase import Phase
from ...utils.lagrange import lagrange_matrices

from scipy.linalg import block_diag

import numpy as np

import openmdao.api as om
from openmdao.core.system import System
import dymos as dm


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

    for iseg in range(old_grid.num_segments):
        i1, i2 = old_grid.subset_segment_indices['all'][iseg, :]
        indices = old_grid.subset_node_indices['all'][i1:i2]
        nodes_given = old_grid.node_stau[indices]

        i1, i2 = new_grid.subset_segment_indices['all'][iseg, :]
        indices = new_grid.subset_node_indices['all'][i1:i2]
        nodes_eval = new_grid.node_stau[indices]

        L_block, _ = lagrange_matrices(nodes_given, nodes_eval)

        L_blocks.append(L_block)

    L = block_diag(*L_blocks)

    return L


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


def split_segments(old_seg_ends, B):
    """
    Funcion to compute the new segment ends for the refined grid by splitting the necessary segments

    Parameters
    ----------
    old_seg_ends: np.array
        segment ends of grid on which the problem was solved

    B: np.array of ints
        Number of segments to be split into

    Returns
    -------
    new_segment_ends: np.array
        Segment ends of refined grid

    """
    new_segment_ends = []
    for q in range(0, B.size):
        new_ends = list(np.linspace(old_seg_ends[q], old_seg_ends[q + 1], B[q] + 1))
        new_segment_ends.extend(new_ends[:-1])
    new_segment_ends.extend([1])
    new_segment_ends = np.asarray(new_segment_ends)
    return new_segment_ends


class PHAdaptive:
    """
    Grid refinement object for the p-then-h grid refinement algorithm

    The error on a solved phase is evaluated. If error exceeds chosen tolerance, the grid is refined following the
    p-then-h refinement algorithm.
    Patterson, M. A., Hager, W. W., and Rao. A. V., “A ph Mesh Refinement Method for Optimal Control”,
    Optimal Control Applications and Methods, Vol. 36, No. 4, July - August 2015, pp. 398 - 421. DOI: 10.1002/oca2114

    """

    def __init__(self, phases):
        """
        Initialize and compute attributes

        Parameters
        ----------
        phases: Phase
            The Phase object representing the solved phase

        """
        self.phases = phases
        self.error = {}

    def check_error(self):
        """
        Compute the error in every solved segment

        Returns
        -------
        need_refinement: dict
            Indicator for which segments of the given phase require grid refinement

        """
        need_refinement = {}
        for phase_path, phase in self.phases.items():
            if not phase.refine_options['refine']:
                self.error[phase_path] = 'Not computed'
                need_refinement[phase_path] = False
                continue
            gd = phase.options['transcription'].grid_data
            num_nodes = gd.subset_num_nodes['all']
            numseg = gd.num_segments

            outputs = phase.list_outputs(units=False, out_stream=None)

            out_values_dict = {k: v['value'] for k, v in outputs}

            prom_to_abs_map = phase._var_allprocs_prom2abs_list['output']

            num_scalar_states = 0
            for state_name, options in phase.state_options.items():
                shape = options['shape']
                size = np.prod(shape)
                num_scalar_states += size

            x = np.zeros([num_nodes, num_scalar_states])
            f = np.zeros([num_nodes, num_scalar_states])
            c = 0

            # Obtain the solution on the current grid
            for state_name, options in phase.state_options.items():
                prom_name = f'timeseries.states:{state_name}'
                abs_name = prom_to_abs_map[prom_name][0]
                rate_source_prom_name = f"timeseries.state_rates:{state_name}"
                rate_abs_name = prom_to_abs_map[rate_source_prom_name][0]
                x[:, c] = out_values_dict[abs_name].ravel()
                f[:, c] = out_values_dict[rate_abs_name].ravel()
                c += 1

            # Obtain the solution on the new grid
            # interpolate x at t_hat
            new_order = gd.transcription_order + 1
            # Gauss-Lobatto does not allow even orders so increase order by 2 instead
            if gd.transcription == 'gauss-lobatto':
                new_order += 1
            new_grid = GridData(numseg, gd.transcription, new_order, gd.segment_ends, gd.compressed)
            left_end_idxs = new_grid.subset_node_indices['segment_ends'][0::2]
            left_end_idxs = np.append(left_end_idxs, new_grid.subset_num_nodes['all'] - 1)

            L = interpolation_lagrange_matrix(gd, new_grid)
            I = integration_matrix(new_grid)

            # Call the ODE at all nodes of the new grid
            x_hat, x_prime = self.eval_ode(phase, new_grid, L, I)
            E = {}
            e = {}
            err_over_states = {}
            for state_name, options in phase.state_options.items():
                E[state_name] = np.absolute(x_prime[state_name] - x_hat[state_name])
                for k in range(0, numseg):
                    e[state_name] = E[state_name]/(1 + np.max(x_hat[state_name][left_end_idxs[k]:left_end_idxs[k + 1]]))
                err_over_states[state_name] = np.zeros(numseg)

            for state_name, options in phase.state_options.items():
                for k in range(0, numseg):
                    err_over_states[state_name][k] = np.max(e[state_name][left_end_idxs[k]:left_end_idxs[k + 1]])

            self.error[phase_path] = np.zeros(numseg)
            need_refinement[phase_path] = np.zeros(numseg, dtype=bool)

            for state_name, options in phase.state_options.items():
                for k in range(0, numseg):
                    if err_over_states[state_name][k] > self.error[phase_path][k]:
                        self.error[phase_path][k] = err_over_states[state_name][k]
                        if self.error[phase_path][k] > phase.refine_options['tolerance']:
                            need_refinement[phase_path][k] = True

        return need_refinement

    def refine(self, need_refinement):
        """
        Compute the order, number of nodes, and segment ends required for the new grid
        and assigns them to the transcription of each phase.

        Parameters
        ----------
        need_refinement : dict
            A dictionary where each key is the path to a phase in the problem, and the
            associated value is a sequence of True/False indicating whether each segment
            in that phase needs refinement.

        Returns
        -------
        refined : dict
            A dictionary of phase paths : phases which were refined.

        """
        refined = {}
        for phase_path, need_refine in need_refinement.items():
            if not np.any(need_refine):
                continue
            refined[phase_path] = self.phases[phase_path]

            # Refinement is needed
            phase = self.phases[phase_path]
            gd = phase.options['transcription'].grid_data
            numseg = gd.num_segments

            refine_seg_idxs = np.where(need_refine)
            P = np.zeros(numseg)
            P[refine_seg_idxs] = np.log(self.error[phase_path][refine_seg_idxs] /
                                        phase.refine_options['tolerance']) / np.log(
                gd.transcription_order[refine_seg_idxs])
            P = np.ceil(P).astype(int)

            if gd.transcription == 'gauss-lobatto':
                odd_idxs = np.where(P % 2 != 0)
                P[odd_idxs] += 1

            new_order = gd.transcription_order + P
            B = np.ones(numseg, dtype=int)

            raise_order_idxs = np.where(gd.transcription_order + P <= phase.refine_options['max_order'])
            split_seg_idxs = np.where(gd.transcription_order + P > phase.refine_options['max_order'])

            new_order[raise_order_idxs] = gd.transcription_order[raise_order_idxs] + P[raise_order_idxs]
            new_order[split_seg_idxs] = phase.refine_options['min_order']

            B[split_seg_idxs] = np.around((gd.transcription_order[split_seg_idxs] +
                                           P[split_seg_idxs]) / phase.refine_options['min_order']).astype(int)

            new_order = np.repeat(new_order, repeats=B)
            new_num_segments = int(np.sum(B))
            new_segment_ends = split_segments(gd.segment_ends, B)

            T = phase.options['transcription']
            T.options['order'] = new_order
            T.options['num_segments'] = new_num_segments
            T.options['segment_ends'] = new_segment_ends
            T.init_grid()

        return refined

    def eval_ode(self, phase, grid, L, I):
        """
        Evaluate the phase ODE on the given grid.

        Parameters
        ----------
        phase : Phase
            The phase object whose ODE is to be evaluated at the given grid.
        grid : GridData
            The GridData object representing the grid at which the ODE is to be evaluated.
        L : np.ndarray
            The interpolation matrix used to obtain interpolated values for the states and controls
            on the given grid, using the existing values on the current grid.
        I : np.ndarray
            The integration matrix used to propagate the initial states of segments across the given grid

        Returns
        -------
        x_hat : dict
            Interpolated state values at all nodes of the given grid.
        x_prime : dict
            Evaluted state values at all nodes of the given grid from use of Integration matrix.

        """
        time_units = phase.time_options['units']
        grid_data = phase.options['transcription'].grid_data
        p = om.Problem(model=om.Group())
        ode_class = phase.options['ode_class']
        ode_init_kwargs = phase.options['ode_init_kwargs']
        ivc = om.IndepVarComp()
        p.model.add_subsystem('ivc', ivc, promotes_outputs=['*'])

        time_comp = TimeComp(num_nodes=grid.num_nodes, node_ptau=grid.node_ptau,
                             node_dptau_dstau=grid.node_dptau_dstau, units=time_units)

        p.model.add_subsystem('time', time_comp, promotes_outputs=['*'], promotes_inputs=['*'])

        p.model.add_subsystem('ode', subsys=ode_class(num_nodes=grid.num_nodes, **ode_init_kwargs))

        u = {}
        x = {}
        x_hat = {}
        u_hat = {}
        f_hat = {}
        x_prime = {}

        outputs = phase.list_outputs(units=False, out_stream=None)
        values_dict = {k: v['value'] for k, v in outputs}
        prom_to_abs_map = phase._var_allprocs_prom2abs_list['output']

        inputs = phase.list_inputs(units=False, out_stream=None)
        prom_to_abs_map.update(phase._var_allprocs_prom2abs_list['input'])
        values_dict.update({k: v['value'] for k, v in inputs})

        if phase.time_options['targets']:
            p.model.connect('time',
                            [f'ode.{t}' for t in phase.time_options['targets']])

        if phase.time_options['time_phase_targets']:
            p.model.connect('time_phase',
                            [f'ode.{t}'for t in phase.time_options['time_phase_targets']],
                            src_indices=grid_data.subset_node_indices['all'])

        if phase.time_options['t_initial_targets']:
            tgts = phase.time_options['t_initial_targets']
            p.model.connect('t_initial',
                            [f'ode.{t}' for t in tgts])

        if phase.time_options['t_duration_targets']:
            tgts = phase.time_options['t_duration_targets']
            p.model.connect('t_duration',
                            [f'ode.{t}' for t in tgts])

        for state_name, options in phase.state_options.items():
            prom_name = f'timeseries.states:{state_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            x[state_name] = values_dict[abs_name]
            x_hat[state_name] = np.dot(L, x[state_name])
            ivc.add_output(f'states:{state_name}', val=x_hat[state_name], units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'states:{state_name}', [f'ode.{tgt}' for tgt in options['targets']])

        for control_name, options in phase.control_options.items():
            prom_name = f'timeseries.controls:{control_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            u[control_name] = values_dict[abs_name]
            u_hat[control_name] = np.dot(L, u[control_name])
            ivc.add_output(f'controls:{control_name}', val=u_hat[control_name], units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'controls:{control_name}', [f'ode.{tgt}' for tgt in options['targets']])

        for dp_name, options in phase.design_parameter_options.items():
            prom_name = f'design_parameters:{dp_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            dp_val = values_dict[abs_name][0, ...]
            ivc.add_output(f'design_parameters:{dp_name}', val=dp_val, units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'design_parameters:{dp_name}',
                                [f'ode.{tgt}' for tgt in options['targets']],
                                src_indices=np.zeros(grid.num_nodes, dtype=int))

        for dp_name, options in phase.input_parameter_options.items():
            prom_name = f'input_parameters:{dp_name}'
            abs_name = prom_to_abs_map[prom_name][0]
            dp_val = values_dict[abs_name][0, ...]
            ivc.add_output(f'input_parameters:{dp_name}', val=dp_val, units=options['units'])
            if options['targets'] is not None:
                p.model.connect(f'input_parameters:{dp_name}',
                                [f'ode.{tgt}' for tgt in options['targets']],
                                src_indices=np.zeros(grid.num_nodes, dtype=int))

        p.setup()

        ti_prom_name = f't_initial'
        ti_abs_name = prom_to_abs_map[ti_prom_name][0]
        t_initial = values_dict[ti_abs_name]

        td_prom_name = f't_duration'
        td_abs_name = prom_to_abs_map[td_prom_name][0]
        t_duration = values_dict[td_abs_name]

        p.set_val('t_initial', t_initial)
        p.set_val('t_duration', t_duration)

        p.run_model()

        left_end_idxs = grid.subset_node_indices['segment_ends'][0::2]
        all_idxs = grid.subset_node_indices['all']
        not_left_end_idxs = np.array(sorted(list(set(all_idxs).difference(set(left_end_idxs)))))

        oodt_dstau = np.atleast_2d(p.get_val('dt_dstau')[not_left_end_idxs]).T

        for state_name, options in phase.state_options.items():
            rate_source = options['rate_source']
            f_hat[state_name] = np.atleast_2d(p.get_val(f'ode.{rate_source}'))
            if options['shape'] == (1,):
                f_hat[state_name] = f_hat[state_name].T
            x_prime[state_name] = np.zeros((grid.num_nodes,) + options['shape'])
            x_prime[state_name][left_end_idxs, ...] = x_hat[state_name][left_end_idxs, ...]
            nnps = np.array(grid.subset_num_nodes_per_segment['all']) - 1
            left_end_idxs_repeated = np.repeat(left_end_idxs, nnps)

            x_prime[state_name][not_left_end_idxs, ...] = \
                x_hat[state_name][left_end_idxs_repeated, ...] \
                + oodt_dstau * np.dot(I, f_hat[state_name][not_left_end_idxs, ...])

        return x_hat, x_prime
