import numpy as np
import openmdao.api as om

from ...grid_data import GridData


class GaussLobattoInterleaveComp(om.ExplicitComponent):
    r""" Provides a contiguous output at all nodes for inputs which are only known at
    state discretiation or collocation nodes.
    """

    def initialize(self):

        self.vars = {}
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

    def add_var(self, name, shape, units):
        """
        Add a variable to be interleaved.  In general these need to be variables whose values are
        stored separately for state discretization or collocation nodes (such as states
        or ODE outputs).

        Parameters
        ----------
        name : str
            The name of variable as it should appear in the outputs of the
            component ('interleave_comp.all_values:{name}').
        shape : tuple
            The shape of the variable at each instance in time.
        units : str
            The units of the variable.
        """
        self.vars[name] = {'shape': shape, 'units': units}

    def setup(self):

        num_disc_nodes = self.options['grid_data'].subset_num_nodes['state_disc']
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']
        num_nodes = self.options['grid_data'].subset_num_nodes['all']

        self._varnames = {}

        for name, options in self.vars.items():
            shape = options['shape']
            units = options['units']
            size = np.prod(shape)

            self._varnames[name] = {}
            self._varnames[name]['disc'] = 'disc_values:{0}'.format(name)
            self._varnames[name]['col'] = 'col_values:{0}'.format(name)
            self._varnames[name]['all'] = 'all_values:{0}'.format(name)

            self.add_input(
                name=self._varnames[name]['disc'],
                shape=(num_disc_nodes,) + shape,
                desc='Values of {0} at discretization nodes'.format(name),
                units=units)

            self.add_input(
                name=self._varnames[name]['col'],
                shape=(num_col_nodes,) + shape,
                desc='Values of {0} at collocation nodes'.format(name),
                units=units)

            self.add_output(
                name=self._varnames[name]['all'],
                shape=(num_nodes,) + shape,
                desc='Values of {0} at all nodes'.format(name),
                units=units)

            start_rows = self.options['grid_data'].subset_node_indices['state_disc'] * size
            r = (start_rows[:, np.newaxis] + np.arange(size, dtype=int)).ravel()
            c = np.arange(size * num_disc_nodes, dtype=int)

            self.declare_partials(of=self._varnames[name]['all'],
                                  wrt=self._varnames[name]['disc'],
                                  rows=r, cols=c, val=1.0)

            start_rows = self.options['grid_data'].subset_node_indices['col'] * size
            r = (start_rows[:, np.newaxis] + np.arange(size, dtype=int)).ravel()
            c = np.arange(size * num_col_nodes, dtype=int)

            self.declare_partials(of=self._varnames[name]['all'],
                                  wrt=self._varnames[name]['col'],
                                  rows=r, cols=c, val=1.0)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        disc_idxs = self.options['grid_data'].subset_node_indices['disc']
        col_idxs = self.options['grid_data'].subset_node_indices['col']

        for name in self.vars:
            outputs[self._varnames[name]['all']][disc_idxs] = inputs[self._varnames[name]['disc']]
            outputs[self._varnames[name]['all']][col_idxs] = inputs[self._varnames[name]['col']]
