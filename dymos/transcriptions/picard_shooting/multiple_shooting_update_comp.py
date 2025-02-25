import numpy as np
import scipy.sparse as sp

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos._options import options as dymos_options


class MultipleShootingUpdateComp(om.ExplicitComponent):
    """
    Class definition for the PicardUpdateComp.

    Given the initial state values (for forward propagation) or the final state values
    (for backward propagation), compute the next state value for picard iteration
    using a NonlinearBlockGS solver.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.

    Attributes
    ----------
    _no_check_partials : bool
        If True, this component will be excluded when partials are checked.
    _M_fwd : array-like
        A selection matrix that pulls appropriate values from x into x_0
        when in forward propagation mode.
    _M_bkwd : array-like
        A selection matrix that pulls appropriate values from x into x_f
        when in backwards propagation mode.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']
        self._M_fwd = None
        self._M_bkwd = None

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
        nodes_per_seg = gd.subset_num_nodes_per_segment['all']

        # Construct the forward and backward mapping matrices
        rows = np.empty(0, dtype=int)
        cols = np.empty(0, dtype=int)
        data = np.empty(0, dtype=int)
        seg_start_nodes = gd.subset_node_indices['segment_ends'][::2]
        seg_end_nodes = gd.subset_node_indices['segment_ends'][1::2]

        for iseg in range(1, gd.num_segments):
            rows_i = np.arange(seg_start_nodes[iseg], seg_start_nodes[iseg] + nodes_per_seg[iseg], dtype=int)
            cols_i = seg_end_nodes[iseg-1] * np.ones(nodes_per_seg[iseg])
            data_i = np.ones(nodes_per_seg[iseg])

            rows = np.concatenate((rows, rows_i))
            cols = np.concatenate((cols, cols_i))
            data = np.concatenate((data, data_i))

        self._M_fwd = sp.coo_array((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

        rows = np.empty(0, dtype=int)
        cols = np.empty(0, dtype=int)
        data = np.empty(0, dtype=int)
        seg_end_nodes = gd.subset_node_indices['segment_ends'][1::2]
        num_nodes_first_seg = gd.subset_num_nodes_per_segment['all'][0]
        num_nodes_last_seg = gd.subset_num_nodes_per_segment['all'][-1]

        for iseg in range(gd.num_segments - 1):
            rows_i = np.arange(seg_start_nodes[iseg], seg_start_nodes[iseg] + nodes_per_seg[iseg], dtype=int)
            cols_i = seg_start_nodes[iseg+1] * np.ones(nodes_per_seg[iseg])
            data_i = np.ones(nodes_per_seg[iseg])

            rows = np.concatenate((rows, rows_i))
            cols = np.concatenate((cols, cols_i))
            data = np.concatenate((data, data_i))

        self._M_bkwd = sp.coo_array((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

        self._var_names = var_names = {}
        for state_name, options in state_options.items():
            var_names[state_name] = {
                'x_a': f'initial_states:{state_name}',
                'x_b': f'final_states:{state_name}',
                'x_0': f'seg_initial_states:{state_name}',
                'x_f': f'seg_final_states:{state_name}',
                'x': f'states:{state_name}'
            }

        for state_name, options in state_options.items():
            shape = options['shape']
            units = options['units']
            size = np.prod(shape)

            rate_units = get_rate_units(units, time_units)
            var_names = self._var_names[state_name]

            self.add_input(
                name=var_names['x'],
                shape=(num_nodes,) + shape,
                desc='State value at the nodes.',
                units=rate_units)

            if options['solve_segments'] == 'forward':

                self.add_input(
                    name=var_names['x_a'],
                    shape=(1,) + shape,
                    desc=f'Desired initial value of state {state_name} in phase.',
                    units=units
                )
                self.add_output(
                    name=var_names['x_0'],
                    shape=(num_nodes,) + shape,
                    val=0.0,
                    desc=f'Given initial value of state {state_name} in each segment.',
                    units=units
                )

                ar_size_x_nn0 = np.arange(num_nodes_first_seg * size, dtype=int)
                self.declare_partials(of=var_names['x_0'],
                                      wrt=var_names['x_a'],
                                      rows=ar_size_x_nn0,
                                      cols=np.zeros_like(ar_size_x_nn0),
                                      val=1.0)

                if num_segs > 1:
                    rs, cs = self._M_fwd.nonzero()
                    self.declare_partials(of=var_names['x_0'],
                                        wrt=var_names['x'],
                                        rows=rs, cols=cs,
                                        val=self._M_fwd.data)

            elif options['solve_segments'] == 'backward':

                self.add_input(
                    name=var_names['x_b'],
                    shape=(1,) + shape,
                    desc=f'Desired initial value of state {state_name} in phase.',
                    units=units
                )

                self.add_output(
                    name=var_names['x_f'],
                    shape=(num_nodes,) + shape,
                    val=0.0,
                    desc=f'Given final value of state {state_name} in each segment.',
                    units=units
                )

                ar_size_x_nnf = np.arange(num_nodes_last_seg * size, dtype=int)
                rs = size * (num_nodes - num_nodes_last_seg) + ar_size_x_nnf
                self.declare_partials(of=var_names['x_f'],
                                      wrt=var_names['x_b'],
                                      rows=rs,
                                      cols=np.zeros(num_nodes_last_seg * size),
                                      val=1.0)

                if num_segs > 1:
                    rs, cs = self._M_bkwd.nonzero()
                    self.declare_partials(of=var_names['x_f'],
                                          wrt=var_names['x'],
                                          rows=rs, cols=cs,
                                          val=self._M_bkwd)


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
        gd = self.options['grid_data']
        nn = gd.num_nodes

        for state_name, options in self.options['state_options'].items():
            var_names = self._var_names[state_name]
            x = inputs[var_names['x']]
            x_flat = x.reshape(nn, -1)

            if options['solve_segments'] == 'forward':
                x_a = inputs[var_names['x_a']]
                outputs[var_names['x_0']] = (self._M_fwd @ x_flat).reshape(x.shape)
                outputs[var_names['x_0']][:gd.subset_num_nodes_per_segment['all'][0]] = x_a

            elif options['solve_segments'] == 'backward':
                x_b = inputs[var_names['x_b']]
                outputs[var_names['x_f']] = (self._M_bkwd @ x_flat).reshape(x.shape)
                outputs[var_names['x_f']][-gd.subset_num_nodes_per_segment['all'][-1]:] = x_b

            else:
                raise ValueError(f'{self.msginfo}: Invalid direction of integration: {options["solve_segments"]}')
