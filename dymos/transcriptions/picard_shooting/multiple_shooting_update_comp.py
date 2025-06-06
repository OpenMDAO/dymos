import numpy as np
import scipy.sparse as sp

import openmdao.api as om

from dymos.transcriptions.grid_data import GridData
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
        state_options = self.options['state_options']

        # Construct the forward and backward mapping matrices
        seg_start_nodes = gd.subset_node_indices['segment_ends'][::2]
        seg_end_nodes = gd.subset_node_indices['segment_ends'][1::2]

        M_fwd = sp.lil_array((num_segs, num_nodes))
        for iseg in range(1, gd.num_segments):
            # The ith row of M_fwd contains in the column pertaining to first node in the (i-1)th segment
            M_fwd[iseg, seg_end_nodes[iseg-1]] = 1.

        M_bkwd = sp.lil_array((num_segs, num_nodes))
        for iseg in range(gd.num_segments-1):
            # The ith row of M_bkwd contains in the column pertaining to first node in the (i+1)th segment
            M_bkwd[iseg, seg_start_nodes[iseg+1]] = 1.

        self._M_fwd = {}
        self._M_bkwd = {}

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

            var_names = self._var_names[state_name]

            self.add_input(
                name=var_names['x'],
                shape=(num_nodes,) + shape,
                desc='State value at the nodes.',
                units=units)

            if options['solve_segments'] == 'forward':
                self._M_fwd[state_name] = M_fwd

                self.add_input(
                    name=var_names['x_a'],
                    shape=(1,) + shape,
                    desc=f'Desired initial value of state {state_name} in phase.',
                    units=units
                )
                self.add_output(
                    name=var_names['x_0'],
                    shape=(num_segs,) + shape,
                    val=0.0,
                    desc=f'Given initial value of state {state_name} in each segment.',
                    units=units
                )

                rs, cs, data = sp.find(sp.eye(size, dtype=int))
                self.declare_partials(of=var_names['x_0'],
                                      wrt=var_names['x_a'],
                                      rows=rs,
                                      cols=cs,
                                      val=data)

                if num_segs > 1:
                    rs, cs, data = sp.find(sp.kron(self._M_fwd[state_name], sp.eye(size), format='csr'))
                    self.declare_partials(of=var_names['x_0'],
                                          wrt=var_names['x'],
                                          rows=rs, cols=cs,
                                          val=data)

            elif options['solve_segments'] == 'backward':
                self._M_bkwd[state_name] = M_bkwd

                self.add_input(
                    name=var_names['x_b'],
                    shape=(1,) + shape,
                    desc=f'Desired initial value of state {state_name} in phase.',
                    units=units
                )

                self.add_output(
                    name=var_names['x_f'],
                    shape=(num_segs,) + shape,
                    val=0.0,
                    desc=f'Given final value of state {state_name} in each segment.',
                    units=units
                )

                ar_size_x_nnf = (num_segs - 1) * size + np.arange(size, dtype=int)
                self.declare_partials(of=var_names['x_f'],
                                      wrt=var_names['x_b'],
                                      rows=ar_size_x_nnf,
                                      cols=np.zeros_like(ar_size_x_nnf),
                                      val=1.0)

                if num_segs > 1:
                    rs, cs, data = sp.find(sp.kron(M_bkwd, sp.eye(size), format='csr'))
                    self.declare_partials(of=var_names['x_f'],
                                          wrt=var_names['x'],
                                          rows=rs, cols=cs,
                                          val=data)

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
                M_fwd = self._M_fwd[state_name]
                x_a = inputs[var_names['x_a']]
                outputs[var_names['x_0']] = M_fwd @ x_flat
                outputs[var_names['x_0']][0, ...] = x_a[0, ...]
                # print(x_flat)
                # print(x_a)
                # print(outputs[var_names['x_0']])

            elif options['solve_segments'] == 'backward':
                M_bkwd = self._M_bkwd[state_name]
                x_b = inputs[var_names['x_b']]
                outputs[var_names['x_f']] = M_bkwd @ x_flat
                outputs[var_names['x_f']][-1, ...] = x_b

            else:
                raise ValueError(f'{self.msginfo}: Invalid direction of integration: {options["solve_segments"]}')
