from __future__ import division, absolute_import, print_function

from six import iteritems, string_types
import numpy as np

from scipy.linalg import block_diag

from dymos.phases.grid_data import GridData
from dymos.utils.misc import get_rate_units
from dymos.utils.lagrange import lagrange_matrices

from openmdao.api import ExplicitComponent


class DynamicInterpComp(ExplicitComponent):

    def __init__(self, **kwargs):
        super(DynamicInterpComp, self).__init__(**kwargs)

        self._L = None
        self._D = None

    def initialize(self):

        self.options.declare('grid_data', types=GridData,
                             desc='Container object for grid info')

        self.options.declare('control_options', types=dict,
                             desc='Dictionary of control names/options for the phase')

        self.options.declare('time_units', default=None, allow_none=True,
                             types=string_types,
                             desc='Units of the integration variable')

        self.options.declare('normalized_times', types=np.ndarray,
                             desc='Array of timesteps for the ODE solver.')

        self.options.declare(
            'segment_times', types=list,
            desc='Ranges of timesteps corresponding to each segment.'
        )

    def setup(self):
        time_units = self.options['time_units']

        grid_data = self.options['grid_data']
        num_nodes = grid_data.num_nodes

        normalized_times = self.options['normalized_times']
        num_timesteps = normalized_times.shape[0]

        control_options = self.options['control_options']

        for dynamic_name, opts in iteritems(control_options):
            shape = opts['shape']
            units = opts['units']

            if opts['dynamic']:
                self.add_input(
                    name='dynamic_nodes:{0}'.format(dynamic_name),
                    shape=(num_nodes,) + shape,
                    desc='Values of dynamic entry {0} at all nodes'.format(dynamic_name),
                    units=units)

                self.add_output(
                    name='dynamic_ts:{0}'.format(dynamic_name),
                    shape=(num_timesteps,) + shape,
                    units=units,
                    desc='Interpolated value of dynamic '
                         'entry {0} at timesteps'.format(dynamic_name))

        L_blocks = []
        segment_times = self.options['segment_times']
        for iseg in range(grid_data.num_segments):
            i1, i2 = grid_data.subset_segment_indices['all'][iseg, :]
            indices = grid_data.subset_node_indices['all'][i1:i2]
            nodes_given = grid_data.node_stau[indices]

            ts_start, ts_end = segment_times[iseg]
            segment_tsteps = normalized_times[ts_start:ts_end]
            timesteps_stau = (2.0 * segment_tsteps - (segment_tsteps[-1] + segment_tsteps[0])) \
                / (segment_tsteps[-1] - segment_tsteps[0])

            L_block, _ = lagrange_matrices(nodes_given, timesteps_stau)

            L_blocks.append(L_block)

        self._L = block_diag(*L_blocks)

        # Setup partials

        self.jacs = {'L': {}}
        self.matrices = {'L': self._L}

        for dynamic_name, opts in iteritems(control_options):
            if opts['dynamic']:
                shape = opts['shape']
                m = np.prod(shape)

                for key in self.jacs:
                    jac = np.zeros((num_timesteps, m, num_nodes, m))
                    for i in range(m):
                        jac[:, i, :, i] = self.matrices[key]
                    jac = jac.reshape((num_timesteps * m, num_nodes * m), order='C')
                    self.jacs[key][dynamic_name] = jac

                self.declare_partials(
                    of='dynamic_ts:{0}'.format(dynamic_name),
                    wrt='dynamic_nodes:{0}'.format(dynamic_name),
                    val=self.jacs['L'][dynamic_name]
                )

    def compute(self, inputs, outputs):
        control_options = self.options['control_options']
        for dynamic_name, options in iteritems(control_options):
            if options['dynamic']:
                outputs['dynamic_ts:' + dynamic_name] = \
                    np.dot(self._L, inputs['dynamic_nodes:' + dynamic_name])
