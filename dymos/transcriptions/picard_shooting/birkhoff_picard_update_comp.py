import numpy as np
import scipy
import scipy.sparse as sp

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
# from dymos._options import options as dymos_options
from dymos.utils.birkhoff import birkhoff_matrix


class PicardUpdateComp(om.ExplicitComponent):
    """
    Class definition for the PicardUpdateComp.

    Given the initial state values (for forward propagation) or the final state values
    (for backward propagation), compute the next state value for picard iteration
    using a NonlinearBlockGS solver.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):
        """
        Declare component options.
        """
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self.options.declare(
            'state_options', types=dict,
            desc='Dictionary of state names/options for the phase')

        self.options.declare(
            'time_units', default=None, allow_none=True, types=str,
            desc='Units of time')

    def configure_io(self, phase):
        """
        I/O creation is delayed until configure so we can determine shape and units.

        Parameters
        ----------
        phase : Phase
            The phase object that contains this collocation comp.
        """
        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['all']
        num_segs = gd.num_segments
        time_units = self.options['time_units']
        state_options = self.options['state_options']

        B_blocks = []

        self._seg_repeats = gd.subset_num_nodes_per_segment['all']
        self._dx_dtdtau_jac_map = {}

        start_idx = 0
        for i in range(num_segs):
            nnps_i = gd.subset_num_nodes_per_segment['all'][i]
            tau_i = gd.node_stau[start_idx: start_idx + nnps_i]
            w_i = gd.node_weight[start_idx: start_idx + nnps_i]

            B_i = birkhoff_matrix(tau_i, w_i, grid_type=gd.grid_type)
            rs, cs = B_i.nonzero()
            B_i_sparse = sp.csr_matrix((B_i[rs, cs].ravel(), (rs, cs)), shape=B_i.shape)
            B_blocks.append(B_i_sparse)
            start_idx += nnps_i

        self._B = scipy.sparse.block_diag(B_blocks, format='csr')

        self.add_input('dt_dstau', units=self.options['time_units'], shape=(num_nodes,))

        self.var_names = var_names = {}
        for state_name, options in state_options.items():
            var_names[state_name] = {
                'f_computed': f'f_computed:{state_name}',
                'x_0': f'seg_initial_states:{state_name}',
                'x_f': f'seg_final_states:{state_name}',
                'x_a': f'initial_states:{state_name}',
                'x_b': f'final_states:{state_name}',
                'x_hat': f'states:{state_name}'
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            size = np.prod(shape)
            units = options['units']
            direction = options['solve_segments']

            rate_units = get_rate_units(units, time_units)
            var_names = self.var_names[state_name]

            self.add_input(
                name=var_names['f_computed'],
                shape=(num_nodes,) + shape,
                desc=f'Computed derivative of state {state_name} at the polynomial nodes',
                units=rate_units)

            self.add_output(
                name=var_names['x_hat'],
                shape=(num_nodes,) + shape,
                units=units
            )

            if direction == 'forward':
                self.add_input(
                    name=var_names['x_0'],
                    shape=(num_segs,) + shape,
                    desc=f'Initial value of state {state_name} in each segment',
                    units=units
                )
                self.add_output(
                    name=var_names['x_b'],
                    shape=(1,) + shape,
                    desc=f'Final value of state {state_name} in the phase',
                    units=units
                )

                num_seg = gd.num_segments
                nnps_last = gd.subset_num_nodes_per_segment['all'][num_seg - 1]

                # Derivatives of integrated state wrt dt_dstau at each node
                rs, cs = sp.kron(self._B, np.ones((size, 1), dtype=int), format='csr').nonzero()
                self.declare_partials(of=var_names['x_hat'],
                                      wrt='dt_dstau',
                                      rows=rs, cols=cs)

                # Derivatives of final state wrt dt_dstau
                last_seg_first_col = num_nodes - nnps_last
                rs = np.repeat(np.arange(size, dtype=int), nnps_last)
                cs = np.tile(np.arange(last_seg_first_col, last_seg_first_col + nnps_last, dtype=int), size)
                self.declare_partials(of=var_names['x_b'],
                                      wrt='dt_dstau',
                                      rows=rs, cols=cs)

                # Derivatives of integrated state wrt computed state rate
                rs, cs = sp.kron(self._B.multiply(np.ones((1, num_nodes))), sp.eye(size), format='csr').nonzero()
                self.declare_partials(of=var_names['x_hat'],
                                      wrt=var_names['f_computed'],
                                      rows=rs, cols=cs)

                # Derivativews of final state wrt computed state rate
                last_seg_first_col = size * (num_nodes - nnps_last)
                rs = rs[-size * nnps_last:]
                rs = np.asarray(rs, dtype=int) - rs[0]
                cs = cs[-size * nnps_last:]
                self.declare_partials(of=var_names['x_b'],
                                      wrt=var_names['f_computed'],
                                      rows=rs, cols=cs)

                # Derivatives of integrated state wrt seg initial value
                blocks = []
                for seg_i in range(gd.num_segments):
                    nnps_i = gd.subset_num_nodes_per_segment['all'][seg_i]
                    blocks.append(sp.kron(np.ones((nnps_i, 1)), sp.eye(size)))
                dxhat_dx0 = sp.block_diag(blocks)
                rs, cs, vals = sp.find(dxhat_dx0)
                self.declare_partials(of=var_names['x_hat'],
                                      wrt=var_names['x_0'],
                                      rows=rs, cols=cs, val=vals)

                # Derivatives of final state wrt seg initial value
                template = sp.lil_array((1, num_segs), dtype=int)
                template[0, -1] = 1
                template = sp.kron(template.tocsr(), sp.eye(size, dtype=int))
                rs, cs = template.nonzero()
                self.declare_partials(of=var_names['x_b'],
                                      wrt=var_names['x_0'],
                                      rows=rs, cols=cs, val=1.0)

                # Build a mapper to help pack the derivatives wrt dt_dstau
                if (size, direction) not in self._dx_dtdtau_jac_map:
                    phase_node_idx = 0
                    seg_row_idx0 = 0
                    self._dx_dtdtau_jac_map[size, direction] = {}
                    for seg_idx in range(gd.num_segments):
                        nnps = gd.subset_num_nodes_per_segment['all'][seg_idx]
                        for seg_node_idx in range(gd.subset_num_nodes_per_segment['all'][seg_idx]):
                            for state_idx in range(size):
                                rs = seg_row_idx0 + size + state_idx + np.arange((nnps - 1) * size, step=size, dtype=int)
                                cs = phase_node_idx * np.ones_like(rs, dtype=int)
                                self._dx_dtdtau_jac_map[size, direction][phase_node_idx, state_idx] = rs, cs

                            phase_node_idx += 1
                        seg_row_idx0 += nnps * size

            elif direction == 'backward':
                self.add_input(name=var_names['x_f'],
                               shape=(num_segs,) + shape,
                               desc=f'Final value of state {state_name} in each segment',
                               units=units)
                self.add_output(name=var_names['x_a'],
                                shape=(1,) + shape,
                                desc=f'Initial value of state {state_name} in the phase',
                                units=units)

                nnps_first = gd.subset_num_nodes_per_segment['all'][0]
                # Derivatives of integrated state wrt dt_dstau at each node
                rs, cs = sp.kron(self._B[::-1, ::-1], np.ones((size, 1), dtype=int), format='csr').nonzero()
                self.declare_partials(of=var_names['x_hat'],
                                      wrt='dt_dstau',
                                      rows=rs, cols=cs)

                # Derivatives of final state wrt dt_dstau
                self.declare_partials(of=var_names['x_a'],
                                      wrt='dt_dstau',
                                      rows=rs[:nnps_first * size], cols=cs[:nnps_first * size])

                nnps_first = gd.subset_num_nodes_per_segment['all'][0]
                rs, cs = sp.kron(self._B[::-1, ::-1].multiply(np.ones((1, num_nodes))), sp.eye(size), format='csr').nonzero()
                self.declare_partials(of=var_names['x_hat'],
                                      wrt=var_names['f_computed'],
                                      rows=rs, cols=cs)

                self.declare_partials(of=var_names['x_a'],
                                      wrt=var_names['f_computed'],
                                      rows=rs[:size * nnps_first],
                                      cols=cs[:size * nnps_first])

                # Derivatives of integrated state wrt seg final value
                blocks = []
                for seg_i in range(gd.num_segments):
                    nnps_i = gd.subset_num_nodes_per_segment['all'][seg_i]
                    blocks.append(sp.kron(np.ones((nnps_i, 1)), sp.eye(size)))
                dxhat_dx0 = sp.block_diag(blocks)
                rs, cs, vals = sp.find(dxhat_dx0)
                self.declare_partials(of=var_names['x_hat'],
                                      wrt=var_names['x_f'],
                                      rows=rs, cols=cs, val=vals)

                template = sp.lil_array((1, num_segs), dtype=int)
                template[0, 0] = 1
                template = sp.kron(template.tocsr(), sp.eye(size, dtype=int))
                rs, cs = template.nonzero()
                self.declare_partials(of=var_names['x_a'], wrt=var_names['x_f'],
                                      rows=rs, cols=cs, val=1.0)

                # Build a mapper to help pack the derivatives wrt dt_dstau
                if (size, direction) not in self._dx_dtdtau_jac_map:
                    phase_node_idx = 0
                    seg_row_idx0 = 0
                    self._dx_dtdtau_jac_map[size, direction] = {}
                    for seg_idx in range(gd.num_segments):
                        nnps = gd.subset_num_nodes_per_segment['all'][seg_idx]
                        for seg_node_idx in range(gd.subset_num_nodes_per_segment['all'][seg_idx]):
                            for state_idx in range(size):
                                rs = seg_row_idx0 + state_idx + np.arange((nnps - 1) * size, step=size, dtype=int)
                                cs = phase_node_idx * np.ones_like(rs, dtype=int)
                                self._dx_dtdtau_jac_map[size, direction][phase_node_idx, state_idx] = rs, cs

                            phase_node_idx += 1
                        seg_row_idx0 += nnps * size

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        """
        Compute component outputs.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        outputs : Vector
            Unscaled, dimensional output variables read via outputs[key].
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        discrete_outputs : dict or None
            If not None, dict containing discrete output values.
        """
        dt_dstau = np.atleast_2d(inputs['dt_dstau']).T
        nn = self.options['grid_data'].num_nodes

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]

            # Multiplication by B results in the integral in segment tau space,
            # so we need to convert f from being dx/dt to dx/dstau.
            f = np.einsum('i...,i...->i...', inputs[var_names['f_computed']], dt_dstau)
            f_flat = f.reshape(nn, -1)

            if options['solve_segments'] == 'forward':
                x_0 = inputs[var_names['x_0']]
                x_0_repeated = np.repeat(x_0, self._seg_repeats, axis=0)
                outputs[var_names['x_hat']] = x_0_repeated + (self._B @ f_flat).reshape(f.shape)
                outputs[var_names['x_b']][...] = outputs[var_names['x_hat']][-1, ...]

            elif options['solve_segments'] == 'backward':
                x_f = inputs[var_names['x_f']]
                x_f_repeated = np.repeat(x_f, self._seg_repeats, axis=0)
                outputs[var_names['x_hat']] = x_f_repeated - (self._B[::-1, ...] @ f_flat[::-1, ...]).reshape(f.shape)
                outputs[var_names['x_a']][...] = outputs[var_names['x_hat']][0, ...]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        """
        Compute component partials.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : dict
            Partial values to be returend in partials[of, wrt] format.
        discrete_inputs : dict or None
            If not None, dict containing discrete input values.
        """
        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['all']

        dt_dstau = np.atleast_2d(inputs['dt_dstau']).T

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]
            size = np.prod(options['shape'])

            x_a_name = var_names['x_a']
            x_b_name = var_names['x_b']
            x_name = var_names['x_hat']
            f_name = var_names['f_computed']

            # Multiplication by B results in the integral in tau space,
            # so we need to convert f from being dx/dt to dx/dtau.
            f_t = inputs[var_names['f_computed']]

            direction = options['solve_segments']

            if direction == 'forward':

                # Patials of the integrated state wrt the computed state rates
                dx_df = sp.kron(self._B.multiply(dt_dstau.T), sp.eye(size), format='csr')
                partials[x_name, f_name] = dx_df.data

                # Partials of the final state wrt the computed state rates
                nnps_last = gd.subset_num_nodes_per_segment['all'][-1]
                partials[x_b_name, f_name] = dx_df.data[-size * nnps_last:]

                B_kron = sp.kron(self._B, np.ones((size, 1)), format='csr')

                f_t_flat = f_t.reshape(num_nodes, -1)
                M_f = sp.lil_matrix(B_kron.shape)
                for (phase_node_idx, state_idx), (rs, cs) in self._dx_dtdtau_jac_map[size, direction].items():
                    M_f[rs, cs] = f_t_flat[phase_node_idx, state_idx]

                # Convert M_f to COO format
                M_f_coo = M_f.tocoo()
                M_f_dict = {(r, c): v for r, c, v in zip(M_f_coo.row, M_f_coo.col, M_f_coo.data)}

                # Multiply B_kron.data with corresponding M_f values
                partials[x_name, 'dt_dstau'] = np.array([
                    B_kron.data[i] * M_f_dict.get((r, c), 0.0)
                    for i, (r, c) in enumerate(zip(*B_kron.nonzero()))])
                partials[x_b_name, 'dt_dstau'] = partials[x_name, 'dt_dstau'][-nnps_last * size:]

            elif options['solve_segments'] == 'backward':
                B_flip = self._B[::-1, ::-1]
                dt_dstau_flip = dt_dstau[::-1, ...]
                nnps_first = gd.subset_num_nodes_per_segment['all'][0]

                dx_df = -sp.kron(self._B[::-1, ::-1].multiply(dt_dstau_flip.T), sp.eye(size), format='csr')
                partials[x_name, f_name] = dx_df.data

                # partials[x_name, f_name] = -(B_flip.multiply(dt_dstau_flip.T)).todense()[:-1, ::-1].ravel()
                partials[x_a_name, f_name] = partials[x_name, f_name][:nnps_first * size]

                B_flip_kron = sp.kron(B_flip, np.ones((size, 1)), format='csr')
                f_t_flat = f_t.reshape(num_nodes, -1)
                M_f = sp.lil_matrix(B_flip_kron.shape)
                for (phase_node_idx, state_idx), (rs, cs) in self._dx_dtdtau_jac_map[size, direction].items():
                    M_f[rs, cs] = f_t_flat[phase_node_idx, state_idx]

                # Convert M_f to COO format
                M_f_coo = M_f.tocoo()
                M_f_dict = {(r, c): v for r, c, v in zip(M_f_coo.row, M_f_coo.col, M_f_coo.data)}

                # Multiply B_kron.data with corresponding M_f values
                partials[x_name, 'dt_dstau'] = -np.array([
                    B_flip_kron.data[i] * M_f_dict.get((r, c), 0.0)
                    for i, (r, c) in enumerate(zip(*B_flip_kron.nonzero()))])

                # partials[x_name, 'dt_dstau'] = -(B_flip_kron.multiply(f_t[0, 0])).data
                partials[x_a_name, 'dt_dstau'] = partials[x_name, 'dt_dstau'][:nnps_first * size]
