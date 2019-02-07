from __future__ import division, print_function

from six import iteritems

import numpy as np
from openmdao.api import ExplicitComponent

from dymos.phases.grid_data import GridData
from dymos.utils.constants import INF_BOUND


class TimeseriesOutputCompBase(ExplicitComponent):
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
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

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
                                'control_rate', 'control_rate2', 'design_parameter',
                                'input_parameter', 'traj_parameter']
        kwargs = {'shape': shape, 'units': units, 'desc': desc, 'src_all': src_all,
                  'distributed': distributed}
        self._timeseries_outputs.append((name, kwargs))


class GaussLobattoTimeseriesOutputComp(TimeseriesOutputCompBase):

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        grid_data = self.options['grid_data']

        num_nodes = grid_data.num_nodes
        num_state_disc_nodes = grid_data.subset_num_nodes['state_disc']
        num_col_nodes = grid_data.subset_num_nodes['col']
        for (name, kwargs) in self._timeseries_outputs:
            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            shape = kwargs['shape']
            if kwargs['src_all']:
                all_input_name = 'all_values:{0}'.format(name)
                disc_input_name = col_input_name = ''
                self.add_input(all_input_name,
                               shape=(num_nodes,) + shape,
                               **input_kwargs)
            else:
                all_input_name = ''
                disc_input_name = 'disc_values:{0}'.format(name)
                col_input_name = 'col_values:{0}'.format(name)

                self.add_input(disc_input_name,
                               shape=(num_state_disc_nodes,) + shape,
                               **input_kwargs)

                self.add_input(col_input_name,
                               shape=(num_col_nodes,) + shape,
                               **input_kwargs)

            output_name = name
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (num_nodes,) + shape
            self.add_output(output_name, **output_kwargs)

            self._vars.append((disc_input_name, col_input_name, all_input_name,
                               kwargs['src_all'], output_name, shape))

            # constraint_kwargs = {k: kwargs.get(k, None)
            #                      for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
            #                                'scaler', 'indices', 'linear')}
            # self.add_constraint(output_name, **constraint_kwargs)

            # Setup partials
            if kwargs['src_all']:
                all_shape = (num_nodes,) + shape
                var_size = np.prod(shape)
                all_size = np.prod(all_shape)

                all_row_starts = grid_data.subset_node_indices['all'] * var_size
                all_rows = []
                for i in all_row_starts:
                    all_rows.extend(range(i, i + var_size))
                all_rows = np.asarray(all_rows, dtype=int)

                self.declare_partials(
                    of=output_name,
                    wrt=all_input_name,
                    dependent=True,
                    rows=all_rows,
                    cols=np.arange(all_size),
                    val=1.0)
            else:
                disc_shape = (num_state_disc_nodes,) + shape
                col_shape = (num_col_nodes,) + shape

                var_size = np.prod(shape)
                disc_size = np.prod(disc_shape)
                col_size = np.prod(col_shape)

                state_disc_row_starts = grid_data.subset_node_indices['state_disc'] * var_size
                disc_rows = []
                for i in state_disc_row_starts:
                    disc_rows.extend(range(i, i + var_size))
                disc_rows = np.asarray(disc_rows, dtype=int)

                self.declare_partials(
                    of=output_name,
                    wrt=disc_input_name,
                    dependent=True,
                    rows=disc_rows,
                    cols=np.arange(disc_size),
                    val=1.0)

                col_row_starts = grid_data.subset_node_indices['col'] * var_size
                col_rows = []
                for i in col_row_starts:
                    col_rows.extend(range(i, i + var_size))
                col_rows = np.asarray(col_rows, dtype=int)

                self.declare_partials(
                    of=output_name,
                    wrt=col_input_name,
                    dependent=True,
                    rows=col_rows,
                    cols=np.arange(col_size),
                    val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        disc_indices = self.options['grid_data'].subset_node_indices['state_disc']
        col_indices = self.options['grid_data'].subset_node_indices['col']
        for (disc_input_name, col_input_name, all_inp_name, src_all, output_name, _) in self._vars:
            if src_all:
                outputs[output_name] = inputs[all_inp_name]
            else:
                outputs[output_name][disc_indices] = inputs[disc_input_name]
                outputs[output_name][col_indices] = inputs[col_input_name]


class RadauTimeseriesOutputComp(TimeseriesOutputCompBase):

    def setup(self):
        """
        Define the independent variables as output variables.
        """
        grid_data = self.options['grid_data']
        num_nodes = grid_data.num_nodes

        for (name, kwargs) in self._timeseries_outputs:

            input_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            input_name = 'all_values:{0}'.format(name)
            self.add_input(input_name,
                           shape=(num_nodes,) + kwargs['shape'],
                           **input_kwargs)

            output_name = name
            output_kwargs = {k: kwargs[k] for k in ('units', 'desc')}
            output_kwargs['shape'] = (num_nodes,) + kwargs['shape']
            self.add_output(output_name, **output_kwargs)

            # constraint_kwargs = {k: kwargs.get(k, None)
            #                      for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
            #                                'scaler', 'indices', 'linear')}
            # self.add_constraint(output_name, **constraint_kwargs)

            self._vars.append((input_name, output_name, kwargs['shape']))

            # Setup partials

            all_shape = (num_nodes,) + kwargs['shape']
            var_size = np.prod(kwargs['shape'])
            all_size = np.prod(all_shape)

            all_row_starts = grid_data.subset_node_indices['all'] * var_size
            all_rows = []
            for i in all_row_starts:
                all_rows.extend(range(i, i + var_size))
            all_rows = np.asarray(all_rows, dtype=int)

            self.declare_partials(
                of=output_name,
                wrt=input_name,
                dependent=True,
                rows=all_rows,
                cols=np.arange(all_size),
                val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for (input_name, output_name, _) in self._vars:
            outputs[output_name] = inputs[input_name]


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

            # constraint_kwargs = {k: kwargs.get(k, None)
            #                      for k in ('lower', 'upper', 'equals', 'ref', 'ref0', 'adder',
            #                                'scaler', 'indices', 'linear')}
            # self.add_constraint(output_name, **constraint_kwargs)

            self._vars[name]['output'] = output_name

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for name, options in iteritems(self._vars):
            for i, input_name in enumerate(options['inputs']):
                idx0, idx1 = options['dest_indices'][i]
                outputs[options['output']][idx0:idx1, ...] = inputs[input_name]
