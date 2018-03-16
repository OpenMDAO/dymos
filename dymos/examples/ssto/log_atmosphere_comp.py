from __future__ import print_function, division, absolute_import

import numpy as np

from openmdao.api import ExplicitComponent


class LogAtmosphereComp(ExplicitComponent):
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)
        self.metadata.declare('rho_ref', types=float, default=1.225,
                              desc='reference density, kg/m**3')
        self.metadata.declare('h_scale', types=float, default=8.44E3,
                              desc='reference altitude, m')

    def setup(self):
        nn = self.metadata['num_nodes']

        self.add_input('y', val=np.zeros(nn), desc='altitude', units='m')

        self.add_output('rho', val=np.zeros(nn), desc='density', units='kg/m**3')

        # Setup partials
        arange = np.arange(self.metadata['num_nodes'])
        self.declare_partials(of='rho', wrt='y', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        rho_ref = self.metadata['rho_ref']
        h_scale = self.metadata['h_scale']
        y = inputs['y']
        outputs['rho'] = rho_ref * np.exp(-y / h_scale)

    def compute_partials(self, inputs, jacobian):
        rho_ref = self.metadata['rho_ref']
        h_scale = self.metadata['h_scale']
        y = inputs['y']
        jacobian['rho', 'y'] = -(rho_ref / h_scale) * np.exp(-y / h_scale)
