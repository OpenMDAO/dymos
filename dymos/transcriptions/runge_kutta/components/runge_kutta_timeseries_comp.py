from __future__ import print_function, division, absolute_import

import numpy as np
from scipy.linalg import block_diag

from ...common.timeseries_output_comp import TimeseriesOutputCompBase
from ....transcriptions.grid_data import GridData
from dymos.utils.lagrange import lagrange_matrices


class RungeKuttaTimeseriesOutputComp(TimeseriesOutputCompBase):

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        igd = self.options['input_grid_data']
        ogd = self.options['output_grid_data']
        output_subset = self.options['output_subset']

        if ogd is None:
            ogd = igd

        # Build the interpolation matrix which maps from the input grid to the output grid.
        # Rather than a single phase-wide interpolating polynomial, map each segment.
        # To do this, find the nodes in the output grid which fall in each segment of the input
        # grid.  Then build a Lagrange interpolating polynomial for that segment
        L_blocks = []
        output_nodes_ptau = list(ogd.node_ptau[ogd.subset_node_indices[output_subset]])

        for iseg in range(igd.num_segments):
            i1, i2 = igd.segment_indices[iseg]
            iptau_segi = np.take(igd.node_ptau, (i1, i2-1))
            istau_segi = np.take(igd.node_stau, (i1, i2-1))

            if ogd is igd:
                o1, o2 = ogd.segment_indices[iseg]
                optau_segi = output_nodes_ptau[o1:o2]
            else:
                # Get the indices of output_nodes_ptau that fall within iptau_segi
                output_idxs = np.where(np.logical_and(output_nodes_ptau >= iptau_segi[0],
                                                      output_nodes_ptau <= iptau_segi[-1]))[0]
                print(iptau_segi)
                print(output_nodes_ptau)
                print(output_idxs)
                exit(0)
                # Get only the unique indices
                # If the output grid has a segment boundary at the same place as the input grid, this
                # prevents two points at the end of the segment from being included in the indices.
                # output_idxs = \
                # np.unique(np.asarray(output_nodes_ptau)[output_idxs], return_index=True)[1]
                # optau_segi = np.asarray(output_nodes_ptau)[output_idxs]
                # output_nodes_ptau = [node for idx, node in enumerate(output_nodes_ptau) if
                #                      idx not in output_idxs]
                # raise NotImplementedError('RK transcription does not interpolate to other grids')

            # Now get the output nodes which fall in iseg in iseg's segment tau space.
            ostau_segi = 2.0 * (optau_segi - iptau_segi[0]) / (iptau_segi[-1] - iptau_segi[0]) - 1

            # Create the interpolation matrix and add it to the blocks
            L, _ = lagrange_matrices(istau_segi, ostau_segi)
            L_blocks.append(L)

        self.interpolation_matrix = block_diag(*L_blocks)
        r, c = np.nonzero(self.interpolation_matrix)

        output_num_nodes, input_num_nodes = self.interpolation_matrix.shape

        for (name, kwargs) in self._timeseries_outputs:

            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            input_name = 'input_values:{0}'.format(name)
            self.add_input(input_name,
                           shape=(input_num_nodes,) + kwargs['shape'],
                           **input_kwargs)

            output_name = name
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (output_num_nodes,) + kwargs['shape']
            self.add_output(output_name, **output_kwargs)

            self._vars.append((input_name, output_name, kwargs['shape']))

            size = np.prod(kwargs['shape'])
            val_jac = np.zeros((output_num_nodes, size, input_num_nodes, size))

            for i in range(size):
                val_jac[:, i, :, i] = self.interpolation_matrix

            val_jac = val_jac.reshape((output_num_nodes * size, input_num_nodes * size),
                                      order='C')

            val_jac_rows, val_jac_cols = np.where(val_jac != 0)

            rs, cs = val_jac_rows, val_jac_cols
            self.declare_partials(of=output_name,
                                  wrt=input_name,
                                  rows=rs, cols=cs, val=val_jac[rs, cs])

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = np.tensordot(self.interpolation_matrix, inputs[input_name],
                                                axes=(1, 0))
