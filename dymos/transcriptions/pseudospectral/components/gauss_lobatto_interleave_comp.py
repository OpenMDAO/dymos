import numpy as np
import openmdao.api as om

from ...grid_data import GridData
from ....options import options as dymos_options


class GaussLobattoInterleaveComp(om.ExplicitComponent):
    r""" Provides a contiguous output at all nodes for inputs which are only known at
    state discretiation or collocation nodes.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._no_check_partials = not dymos_options['include_check_partials']

    def initialize(self):

        self._varnames = {}
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

        Returns
        -------
        success : bool
            True if the variable was added to the interleave comp, False if not due to it already
            being there.
        """
        if name in self._varnames:
            return False

        num_disc_nodes = self.options['grid_data'].subset_num_nodes['state_disc']
        num_col_nodes = self.options['grid_data'].subset_num_nodes['col']
        num_nodes = self.options['grid_data'].subset_num_nodes['all']

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

        return True

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        disc_idxs = self.options['grid_data'].subset_node_indices['disc']
        col_idxs = self.options['grid_data'].subset_node_indices['col']

        for name, varnames in self._varnames.items():
            outputs[varnames['all']][disc_idxs] = inputs[varnames['disc']]
            outputs[varnames['all']][col_idxs] = inputs[varnames['col']]
