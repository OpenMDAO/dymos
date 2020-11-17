import numpy as np
import openmdao.api as om


class AerodynamicsComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('rho', val=np.ones(nn), desc='atmospheric density', units='kg/m**3')
        self.add_input('v', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input('S', val=np.ones(nn), desc='aerodynamic reference area', units='m**2')
        self.add_input('CL', val=np.ones(nn), desc='lift coefficient', units=None)
        self.add_input('CD0', val=np.ones(nn), desc='zero-lift drag coefficient', units=None)
        self.add_input('K', val=np.ones(nn), desc='drag coefficient', units=None)

        self.add_output('q', val=np.ones(nn), desc='dynamic pressure', units='Pa')
        self.add_output('L', val=np.ones(nn), desc='lift', units='N')
        self.add_output('D', val=np.ones(nn), desc='drag', units='N')

        ar = np.arange(nn)

        self.declare_partials('L', 'rho', rows=ar, cols=ar)
        self.declare_partials('L', 'v', rows=ar, cols=ar)
        self.declare_partials('L', 'S', rows=ar, cols=ar)
        self.declare_partials('L', 'CL', rows=ar, cols=ar)

        self.declare_partials('D', 'rho', rows=ar, cols=ar)
        self.declare_partials('D', 'v', rows=ar, cols=ar)
        self.declare_partials('D', 'S', rows=ar, cols=ar)
        self.declare_partials('D', 'CD', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        rho = inputs['rho']
        v = inputs['v']
        S = inputs['S']
        CL = inputs['CL']
        CD0 = inputs['CD0']
        K = inputs['K']

        outputs['q'] = q = 0.5 * rho * v ** 2
        outputs['L'] = q * S * CL
        outputs['D'] = q * S * (CD0 + K * CL ** 2)

    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        v = inputs['v']
        S = inputs['S']
        CL = inputs['CL']
        CD = inputs['CD']

        q = 0.5 * rho * v ** 2

        partials['L', 'rho'] = 0.5 * S * CL * v ** 2
        partials['L', 'v'] = rho * v * S * CL
        partials['L', 'S'] = q * CL
        partials['L', 'CL'] = q * S

        partials['D', 'rho'] = 0.5 * S * CD * v ** 2
        partials['D', 'v'] = rho * v * S * CD
        partials['D', 'S'] = q * CD
        partials['D', 'CD0'] =
        partials['D', 'K'] =
        partials['D', 'CL'] =
