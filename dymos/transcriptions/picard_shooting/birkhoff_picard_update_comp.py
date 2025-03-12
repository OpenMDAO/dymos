import numpy as np
import scipy
import scipy.sparse as sp

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos._options import options as dymos_options
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
        self._no_check_partials = not dymos_options['include_check_partials']

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

        start_idx = 0
        for i in range(num_segs):
            nnps_i = gd.subset_num_nodes_per_segment['all'][i]
            tau_i = gd.node_stau[start_idx: start_idx + nnps_i]
            w_i = gd.node_weight[start_idx: start_idx + nnps_i]

            B_i = birkhoff_matrix(tau_i, w_i, grid_type=gd.grid_type)
            B_blocks.append(B_i)
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

            rate_units = get_rate_units(units, time_units)
            var_names = self.var_names[state_name]

            rate_source = options['rate_source']
            rate_source_type = phase.classify_var(rate_source)

            if rate_source_type == 'state':
                var_names['f_computed'] = f'state_val:{rate_source}'
            else:
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

            if options['solve_segments'] == 'forward':
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
                # The first row of these matrices are zero.
                rs = np.repeat(np.arange(1, num_nodes, dtype=int), num_nodes)
                cs = np.tile(np.arange(num_nodes, dtype=int), num_nodes - 1)
                self.declare_partials(of=var_names['x_hat'],
                                      wrt='dt_dstau',
                                      rows=rs, cols=cs)
                self.declare_partials(of=var_names['x_hat'],
                                      wrt=var_names['f_computed'],
                                      rows=rs, cols=cs)
                self.declare_partials(of=var_names['x_b'],
                                      wrt='dt_dstau')
                self.declare_partials(of=var_names['x_b'],
                                      wrt=var_names['f_computed'])

                rs = np.arange(num_nodes, dtype=int)
                cs = np.repeat(np.arange(num_segs, dtype=int), self._seg_repeats)
                self.declare_partials(of=var_names['x_hat'],
                                      wrt=var_names['x_0'],
                                      rows=rs, cols=cs, val=1.0)
                template = sp.lil_array((1, num_segs), dtype=int)
                template[0, -1] = 1
                template = sp.kron(template.tocsr(), sp.eye(size, dtype=int))
                rs, cs = template.nonzero()
                self.declare_partials(of=var_names['x_b'],
                                      wrt=var_names['x_0'],
                                      rows=rs, cols=cs, val=1.0)

            elif options['solve_segments'] == 'backward':
                self.add_input(name=var_names['x_f'],
                               shape=(num_segs,) + shape,
                               desc=f'Final value of state {state_name} in each segment',
                               units=units)
                self.add_output(name=var_names['x_a'],
                                shape=(1,) + shape,
                                desc=f'Initial value of state {state_name} in the phase',
                                units=units)
                rs = np.repeat(np.arange(num_nodes - 1, dtype=int), num_nodes)
                cs = np.tile(np.arange(num_nodes, dtype=int), num_nodes - 1)
                self.declare_partials(of=var_names['x_hat'], wrt='dt_dstau',
                                      rows=rs, cols=cs)
                self.declare_partials(of=var_names['x_hat'], wrt=var_names['f_computed'],
                                      rows=rs, cols=cs)
                self.declare_partials(of=var_names['x_a'],
                                      wrt='dt_dstau')
                self.declare_partials(of=var_names['x_a'],
                                      wrt=var_names['f_computed'])
                rs = np.arange(num_nodes, dtype=int)
                cs = np.repeat(np.arange(num_segs, dtype=int), self._seg_repeats)
                self.declare_partials(of=var_names['x_hat'],
                                      wrt=var_names['x_f'],
                                      rows=rs, cols=cs, val=1.0)
                template = sp.lil_array((1, num_segs), dtype=int)
                template[0, 0] = 1
                template = sp.kron(template.tocsr(), sp.eye(size, dtype=int))
                rs, cs = template.nonzero()
                self.declare_partials(of=var_names['x_a'], wrt=var_names['x_f'],
                                      rows=rs, cols=cs, val=1.0)

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

            x_a_name = var_names['x_a']
            x_b_name = var_names['x_b']
            x_name = var_names['x_hat']
            f_name = var_names['f_computed']

            # Multiplication by B results in the integral in tau space,
            # so we need to convert f from being dx/dt to dx/dtau.
            f_t = inputs[var_names['f_computed']]

            if options['solve_segments'] == 'forward':
                partials[x_name, f_name] = (self._B.multiply(dt_dstau.T)).todense()[1:, ...].ravel()
                partials[x_b_name, f_name] = partials[x_name, f_name][-num_nodes:]
                partials[x_name, 'dt_dstau'] = (self._B.multiply(f_t.T)).todense()[1:, ...].ravel()
                partials[x_b_name, 'dt_dstau'] = partials[x_name, 'dt_dstau'][-num_nodes:]

            elif options['solve_segments'] == 'backward':
                B_flip = self._B[::-1, ...]
                dt_dstau_flip = dt_dstau[::-1, ...]

                partials[x_name, f_name] = -(B_flip.multiply(dt_dstau_flip.T)).todense()[:-1, ::-1].ravel()
                partials[x_a_name, f_name] = partials[x_name, f_name][:num_nodes]
                partials[x_name, 'dt_dstau'] = -(B_flip.multiply(f_t[::-1, ...].T)).todense()[:-1, ::-1].ravel()
                partials[x_a_name, 'dt_dstau'] = partials[x_name, 'dt_dstau'][:num_nodes]
