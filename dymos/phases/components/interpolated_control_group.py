from __future__ import division, print_function, absolute_import

import numpy as np
from six import iteritems, string_types

from openmdao.api import Group, ExplicitComponent

from ..grid_data import GridData
from ...utils.lgl import lgl
from ...utils.lagrange import lagrange_matrices


class LGLInterpolatedControlComp(ExplicitComponent):
    """
    Component which interpolates controls as a single polynomial across the entire phase.
    """

    def initialize(self):
        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

        self._interp_controls = {}
        self._matrices = {}

    def add_interpolated_control(self, name, options):
        self._interp_controls[name] = options

    def setup(self):

        gd = self.options['grid_data']
        eval_nodes = gd.node_ptau

        for name, options in iteritems(self._interp_controls):
            disc_nodes = lgl(options['num_points'])

            L_de, D_de = lagrange_matrices(disc_nodes, eval_nodes)
            _, D_dd = lagrange_matrices(disc_nodes, disc_nodes)
            D2_de = np.dot(D_de, D_dd)

            self._matrices[name] = L_de, D_de, D2_de

            self.add_input('controls:{0}'.format(name))
            self.add_output('control_values:{0}'.format(name))
            self.add_output('control_rates:{0}_rate'.format(name))
            self.add_output('control_rates:{0}_rate2'.format(name))



class InterpolatedControlGroup(Group):

    def initialize(self):
        self.options.declare(
            'control_options', types=dict,
            desc='Dictionary of options for the dynamic controls')
        self.options.declare(
            'time_units', default=None, allow_none=True, types=string_types,
            desc='Units of time')
        self.options.declare(
            'grid_data', types=GridData,
            desc='Container object for grid info')

    def setup(self):
