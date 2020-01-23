import numpy as np

import openmdao.api as om


class LogAtmosphereComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('rho_ref', types=float, default=1.225,
                             desc='reference density, kg/m**3')
        self.options.declare('h_scale', types=float, default=8.44E3,
                             desc='reference altitude, m')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('y', val=np.zeros(nn), desc='altitude', units='m')

        self.add_output('rho', val=np.zeros(nn), desc='density', units='kg/m**3')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='rho', wrt='y', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        rho_ref = self.options['rho_ref']
        h_scale = self.options['h_scale']
        y = inputs['y']
        outputs['rho'] = rho_ref * np.exp(-y / h_scale)

    def compute_partials(self, inputs, jacobian):
        rho_ref = self.options['rho_ref']
        h_scale = self.options['h_scale']
        y = inputs['y']
        jacobian['rho', 'y'] = -(rho_ref / h_scale) * np.exp(-y / h_scale)
