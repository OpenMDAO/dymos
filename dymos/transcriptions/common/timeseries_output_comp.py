from __future__ import division, print_function

from six import iteritems

import numpy as np
import openmdao.api as om
from scipy.linalg import block_diag

from dymos.transcriptions.grid_data import GridData
from dymos.utils.lagrange import lagrange_matrices


class TimeseriesOutputCompBase(om.ExplicitComponent):
    """
    TimeseriesOutputComp collects variable values from the phase and provides them in chronological
    order as outputs.  Some phase types don't internally have access to a contiguous array of all
    values of a given variable in the phase.  For instance, the GaussLobatto pseudospectral has
    separate arrays of variable values at discretization and collocation nodes.  These values
    need to be interleaved to provide a time series.  Pseudospectral techniques provide timeseries
    data at 'all' nodes, while ExplicitPhase provides values at the step boundaries.
    """

    def initialize(self):

        self._timeseries_outputs = []

        self._vars = []

        self.options.declare('input_grid_data',
                             types=GridData,
                             desc='Container object for grid on which inputs are provided.')

        self.options.declare('output_grid_data',
                             types=GridData,
                             allow_none=True,
                             default=None,
                             desc='Container object for grid on which outputs are interpolated.')

        self.options.declare('output_subset',
                             types=str,
                             default='all',
                             desc='Name of the node subset at which outputs are desired.')

    def _add_timeseries_output(self, name, var_class, shape=(1,), units=None, desc='',
                               distributed=False):
        """
        Add a final constraint to this component

        Parameters
        ----------
        name : str
            name of the variable in this component's namespace.
        var_class : str
            The 'class' of the variable as given by phase.classify_var.  One of 'time', 'state',
            'indep_control', 'input_control', 'design_parameter', 'input_parameter',
            'control_rate', 'control_rate2', or 'ode'.
        shape : int or tuple or list or None
            Shape of this variable, only required if val is not an array.
            Default is None.
        units : str or None
            Units in which the output variables will be provided to the component during execution.
            Default is None, which means it has no units.
        desc : str
            description of the timeseries output variable.
        distributed : bool
            If True, this variable is distributed across multiple processes.
        """
        src_all = var_class in ['time', 'time_phase', 'indep_control', 'input_control',
                                'control_rate', 'control_rate2', 'indep_polynomial_control',
                                'input_polynomial_control', 'polynomial_control_rate',
                                'polynomial_control_rate2', 'design_parameter', 'input_parameter',
                                'traj_parameter']
        kwargs = {'shape': shape, 'units': units, 'desc': desc, 'src_all': src_all,
                  'distributed': distributed}
        self._timeseries_outputs.append((name, kwargs))


class PseudospectralTimeseriesOutputComp(TimeseriesOutputCompBase):

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        igd = self.options['input_grid_data']
        ogd = self.options['output_grid_data']
        output_subset = self.options['output_subset']

        if ogd is None:
            ogd = igd

        input_num_nodes = igd.num_nodes
        output_num_nodes = ogd.subset_num_nodes[output_subset]

        # Build the interpolation matrix which maps from the input grid to the output grid.
        # Rather than a single phase-wide interpolating polynomial, map each segment.
        # To do this, find the nodes in the output grid which fall in each segment of the input
        # grid.  Then build a Lagrange interpolating polynomial for that segment
        L_blocks = []
        output_nodes_ptau = list(ogd.node_ptau[ogd.subset_node_indices[output_subset]])
        for iseg in range(igd.num_segments):
            i1, i2 = igd.segment_indices[iseg]
            iptau_segi = igd.node_ptau[i1:i2]
            istau_segi = igd.node_stau[i1:i2]

            # The indices of the output grid that fall within this segment of the input grid
            output_idxs = np.where(np.logical_and(output_nodes_ptau >= iptau_segi[0],
                                                  output_nodes_ptau <= iptau_segi[-1]))[0]
            # Get only the unique indices
            # If the output grid has a segment boundary at the same place as the input grid, this
            # prevents two points at the end of the segment from being included in the indices.
            output_idxs = np.unique(np.asarray(output_nodes_ptau)[output_idxs], return_index=True)[1]
            optau_segi = np.asarray(output_nodes_ptau)[output_idxs]
            output_nodes_ptau = [node for idx, node in enumerate(output_nodes_ptau) if idx not in output_idxs]

            # Now get the output nodes which fall in iseg in iseg's segment tau space.
            ostau_segi = 2.0 * (optau_segi - iptau_segi[0]) / (iptau_segi[-1] - iptau_segi[0]) - 1

            # Create the interpolation matrix and add it to the blocks
            L, _ = lagrange_matrices(istau_segi, ostau_segi)
            L_blocks.append(L)

        self.interpolation_matrix = block_diag(*L_blocks)

        for (name, kwargs) in self._timeseries_outputs:
            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            input_name = 'input_values:{0}'.format(name)
            shape = kwargs['shape']

            print(self.pathname, input_num_nodes, output_num_nodes, self.interpolation_matrix.shape)

            self.add_input(input_name,
                           shape=(input_num_nodes,) + shape,
                           **input_kwargs)

            output_name = name
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (output_num_nodes,) + kwargs['shape']
            self.add_output(output_name, **output_kwargs)

            self._vars.append((input_name, output_name, shape))

            size = np.prod(shape)
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


class ExplicitTimeseriesOutputComp(TimeseriesOutputCompBase):

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        gd = self.options['grid_data']
        total_num_steps = np.sum(gd.num_steps_per_segment) + gd.num_segments
        self._vars = {}

        for (name, kwargs) in self._timeseries_outputs:

            self._vars[name] = {'inputs': [],
                                'output': '',
                                'dest_indices': []}

            output_name = name
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (total_num_steps,) + kwargs['shape']
            size = np.prod(kwargs['shape'])
            self.add_output(output_name, **output_kwargs)

            idx0 = 0
            for iseg in range(gd.num_segments):
                num_steps = gd.num_steps_per_segment[iseg]
                idx1 = idx0 + num_steps + 1
                input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
                input_name = 'seg_{0}_values:{1}'.format(iseg, name)

                self.add_input(input_name,
                               shape=(num_steps + 1,) + kwargs['shape'],
                               **input_kwargs)

                self._vars[name]['inputs'].append(input_name)
                self._vars[name]['dest_indices'].append((idx0, idx1))

                ar = np.arange((num_steps + 1) * size, dtype=int)

                self.declare_partials(
                    of=output_name,
                    wrt=input_name,
                    rows=idx0 * size + ar,
                    cols=ar,
                    val=1.0)

                idx0 = idx1

            self._vars[name]['output'] = output_name

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for name, options in iteritems(self._vars):
            for i, input_name in enumerate(options['inputs']):
                idx0, idx1 = options['dest_indices'][i]
                outputs[options['output']][idx0:idx1, ...] = inputs[input_name]
