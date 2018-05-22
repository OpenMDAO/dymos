from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent
from six import string_types

from dymos.phases.grid_data import GridData


class TimeComp(ExplicitComponent):

    def initialize(self):
        # Required
        self.options.declare('grid_data', types=GridData, desc='Container object for grid info')

        # Optional
        self.options.declare('units', default=None, allow_none=True, types=string_types,
                             desc='Units of time (or the integration variable)')

        self.options.declare('internal', default=True, types=bool, desc='If True, compute dt_dstau')

    def setup(self):
        time_units = self.options['units']
        node_ptau = self.options['grid_data'].node_ptau

        self.add_input('t_initial', val=0., units=time_units)
        self.add_input('t_duration', val=1., units=time_units)
        self.add_output('time', units=time_units, shape=len(node_ptau))
        if self.options['internal']:
            self.add_output('dt_dstau', units=time_units, shape=len(node_ptau))

        # Setup partials
        nn = self.options['grid_data'].num_nodes
        rs = np.arange(nn)
        cs = np.zeros(nn)

        self.declare_partials(of='time', wrt='t_initial', rows=rs, cols=cs, val=1.0)

        self.declare_partials(of='time', wrt='t_duration', rows=rs, cols=cs, val=1.0)

        self.declare_partials(of='dt_dstau', wrt='t_duration', rows=rs, cols=cs, val=1.0)

    def compute(self, inputs, outputs):
        node_ptau = self.options['grid_data'].node_ptau
        node_dptau_dstau = self.options['grid_data'].node_dptau_dstau

        t_initial = inputs['t_initial']
        t_duration = inputs['t_duration']

        outputs['time'][:] = t_initial + 0.5 * (node_ptau + 1) * t_duration

        if self.options['internal']:
            outputs['dt_dstau'][:] = 0.5 * t_duration * node_dptau_dstau

    def compute_partials(self, inputs, jacobian):
        node_ptau = self.options['grid_data'].node_ptau
        node_dptau_dstau = self.options['grid_data'].node_dptau_dstau

        jacobian['time', 't_duration'] = 0.5 * (node_ptau + 1)

        if self.options['internal']:
            jacobian['dt_dstau', 't_duration'] = 0.5 * node_dptau_dstau
