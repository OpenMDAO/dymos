import numpy as np
import scipy

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos._options import options as dymos_options
from dymos.utils.lgl import lgl
from dymos.utils.cgl import cgl
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
        num_nodes = gd.subset_num_nodes['col']
        num_segs = gd.num_segments
        time_units = self.options['time_units']
        state_options = self.options['state_options']

        B_blocks = []

        for i in range(num_segs):
            nnps_i = gd.subset_num_nodes_per_segment['all'][i]
            if gd.grid_type == 'lgl':
                tau_i, w_i = lgl(nnps_i)
            elif gd.grid_type == 'cgl':
                tau_i, w_i = cgl(nnps_i)
            else:
                raise ValueError('invalid grid type')

            B_i = birkhoff_matrix(tau_i, w_i, grid_type=gd.grid_type)

            B_blocks.append(B_i)

        self._B = scipy.linalg.block_diag(*B_blocks)

        self.add_input('dt_dstau', units=self.options['time_units'], shape=(num_nodes,))

        # self.add_input('dt_dstau', units=time_units, shape=(gd.subset_num_nodes['col'],))
        self.var_names = var_names = {}
        for state_name, options in state_options.items():
            var_names[state_name] = {
                'f_computed': f'state_rates:{state_name}',
                'current_state': f'state_val:{state_name}',
                'state_initial_value': f'initial_states:{state_name}',
                'state_final_value': f'final_states:{state_name}',
                'next_state': f'states:{state_name}'
            }

        for state_name, options in state_options.items():
            shape = options['shape']
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

            self.add_input(
                name=var_names['current_state'],
                shape=(num_nodes,) + shape,
                desc=f'Value of the state {state_name} at the polynomial nodes',
                units=units
            )

            if options['solve_segments'] == 'forward':
                self.add_input(
                    name=var_names['state_initial_value'],
                    shape=(1,) + shape,
                    desc=f'Desired initial value of state {state_name}',
                    units=units
                )
                self.add_output(
                    name=var_names['state_final_value'],
                    shape=(1,) + shape,
                    desc=f'Estimated final value of state {state_name}',
                    units=units
                )
                self.declare_partials(of=var_names['state_final_value'],
                                      wrt='dt_dstau')
                self.declare_partials(of=var_names['state_final_value'],
                                      wrt=var_names['state_initial_value'],
                                      val=1.0)
                self.declare_partials(of=var_names['state_final_value'],
                                      wrt=var_names['f_computed'])
                self.declare_partials(of=var_names['next_state'],
                                      wrt=var_names['state_initial_value'],
                                      val=1.0)

                # The first row of these matrices are zero.
                rs = np.repeat(np.arange(1, num_nodes, dtype=int), num_nodes)
                cs = np.tile(np.arange(num_nodes, dtype=int), num_nodes - 1)

                self.declare_partials(of=var_names['next_state'],
                                      wrt=var_names['f_computed'],
                                      rows=rs, cols=cs)

                self.declare_partials(of=var_names['next_state'],
                                      wrt='dt_dstau',
                                      rows=rs, cols=cs)


            elif options['solve_segments'] == 'backward':
                self.add_input(
                    name=var_names['state_final_value'],
                    shape=(1,) + shape,
                    desc=f'Estimated final value of state {state_name}',
                    units=units
                )
                self.add_output(
                    name=var_names['state_initial_value'],
                    shape=(1,) + shape,
                    desc=f'Desired initial value of state {state_name}',
                    units=units
                )
                self.declare_partials(of=var_names['state_initial_value'],
                                      wrt='dt_dstau')
                self.declare_partials(of=var_names['state_initial_value'],
                                      wrt=var_names['state_final_value'],
                                      val=1.0)
                self.declare_partials(of=var_names['state_initial_value'],
                                      wrt=var_names['f_computed'])
                self.declare_partials(of=var_names['next_state'],
                                      wrt=var_names['state_final_value'],
                                      val=1.0)

                # The last row of these matrices are zero.
                rs = np.repeat(np.arange(0, num_nodes-1, dtype=int), num_nodes)
                cs = np.tile(np.arange(num_nodes, dtype=int), num_nodes - 1)

                self.declare_partials(of=var_names['next_state'],
                                      wrt=var_names['f_computed'],
                                      rows=rs, cols=cs)

                self.declare_partials(of=var_names['next_state'],
                                      wrt='dt_dstau',
                                      rows=rs, cols=cs)

            self.add_output(
                name=var_names['next_state'],
                shape=(num_nodes,) + shape,
                units=units
            )






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

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]

            # Multiplication by B results in the integral in tau space,
            # so we need to convert f from being dx/dt to dx/dtau.
            f = np.einsum('i...,i...->i...', inputs[var_names['f_computed']], dt_dstau)

            if options['solve_segments'] == 'forward':
                x_a = inputs[var_names['state_initial_value']]
                outputs[var_names['next_state']] = x_a + np.einsum('ij,jk...->ik...', self._B, f)
                outputs[var_names['state_final_value']][...] = outputs[var_names['next_state']][-1, ...]

            elif options['solve_segments'] == 'backward':
                x_b = inputs[var_names['state_final_value']]
                outputs[var_names['next_state']] = x_b - np.einsum('ij,jk...->ik...', self._B[::-1, ...], f[::-1, ...])
                outputs[var_names['state_initial_value']][...] = outputs[var_names['next_state']][0, ...]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        gd = self.options['grid_data']
        num_nodes = gd.subset_num_nodes['col']

        dt_dstau = np.atleast_2d(inputs['dt_dstau']).T

        for state_name, options in self.options['state_options'].items():
            var_names = self.var_names[state_name]

            x_a_name = var_names['state_initial_value']
            x_b_name = var_names['state_final_value']
            x_name = var_names['next_state']
            f_name = var_names['f_computed']

            # Multiplication by B results in the integral in tau space,
            # so we need to convert f from being dx/dt to dx/dtau.
            f_t = inputs[var_names['f_computed']]

            if options['solve_segments'] == 'forward':
                # x = x_a + np.einsum('ij,jk...->ik...', self._B, f)
                partials[x_name, f_name] = (self._B * dt_dstau.T)[1:, ...].ravel()
                partials[x_b_name, f_name] = partials[x_name, f_name][-num_nodes:]
                partials[x_name, 'dt_dstau'] = (self._B * f_t.T)[1:, ...].ravel()
                partials[x_b_name, 'dt_dstau'] = partials[x_name, 'dt_dstau'][-num_nodes:]

            elif options['solve_segments'] == 'backward':
                # x = x_b - np.einsum('ij,jk...->ik...', self._B[::-1, ...], f[::-1, ...])
                B_flip = self._B[::-1, ...]
                dt_dstau_flip = dt_dstau[::-1, ...]

                partials[x_name, f_name] = -(B_flip * dt_dstau_flip.T)[:-1, ::-1].ravel()

                # B_flip = self._B[::-1, :-1]
                # partials[x_name, f_name] = -(B_flip * dt_dstau[:-1].T).ravel()

                partials[x_a_name, f_name] = partials[x_name, f_name][:num_nodes]
                partials[x_name, 'dt_dstau'] = -(self._B[::-1, ...] * f_t[::-1, ...].T)[:-1, ::-1].ravel()
                partials[x_a_name, 'dt_dstau'] = partials[x_name, 'dt_dstau'][:num_nodes]
