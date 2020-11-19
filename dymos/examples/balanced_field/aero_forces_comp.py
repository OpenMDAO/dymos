import numpy as np
import openmdao.api as om


class AeroForcesComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('rho', val=np.ones(nn), desc='atmospheric density', units='kg/m**3')
        self.add_input('v', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CL', val=np.ones(nn), desc='lift coefficient', units=None)
        self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)
        self.add_input('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)

        self.add_output('q', val=np.ones(nn), desc='dynamic pressure', units='Pa')
        self.add_output('L', val=np.ones(nn), desc='lift', units='N')
        self.add_output('D', val=np.ones(nn), desc='drag', units='N')

        ar = np.arange(nn)

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_summary=False, show_sparsity=False)
        self.declare_partials(of='L', wrt=['rho', 'v', 'S', 'CL'], method='cs')
        self.declare_partials(of='D', wrt=['rho', 'v', 'S', 'CL', 'CD0', 'K'], method='cs')

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

    # def compute_partials(self, inputs, partials):
    #     rho = inputs['rho']
    #     v = inputs['v']
    #     S = inputs['S']
    #     CL = inputs['CL']
    #     CD = inputs['CD']
    #
    #     q = 0.5 * rho * v ** 2
    #
    #     partials['L', 'rho'] = 0.5 * S * CL * v ** 2
    #     partials['L', 'v'] = rho * v * S * CL
    #     partials['L', 'S'] = q * CL
    #     partials['L', 'CL'] = q * S
    #
    #     partials['D', 'rho'] = 0.5 * S * CD * v ** 2
    #     partials['D', 'v'] = rho * v * S * CD
    #     partials['D', 'S'] = q * CD
    #     partials['D', 'CD0'] = q * S
    #     partials['D', 'K'] = q * S * CL ** 2
    #     partials['D', 'CL'] = 2 * q * S * K * CL
