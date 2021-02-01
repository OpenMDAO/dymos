import numpy as np
import openmdao.api as om


class AeroForcesCompGroundroll(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('rho', val=1.225 * np.ones(nn), desc='atmospheric density', units='kg/m**3')
        self.add_input('v', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)

        self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
        self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
        self.add_input('span', val=35.7, desc='Wingspan', units='m')
        self.add_input('h', val=np.ones(nn), desc='altitude', units='m')
        self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
        self.add_input('m', val=np.ones(nn), desc='aircraft mass', units='kg')

        self.add_input('alpha', val=np.ones(nn), desc='angle of attack', units='deg')
        self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
        self.add_input('CL_max', val=2.0, desc='maximum lift coefficient', units=None)
        self.add_input('alpha_max', val=10, desc='angle of attack at CL_max', units='deg')

        self.add_input(name='T', val=120101.98, desc='thrust', units='N')

        self.add_input(name='mu_r', val=0.05, desc='runway friction coefficient', units=None)

        self.add_output(name='v_dot', val=np.ones(nn), desc='rate of change of velocity magnitude', units='m/s**2')

        self.add_output(name='r_dot', val=np.ones(nn), desc='rate of change of range', units='m/s')

        self.add_output(name='F_r', val=np.ones(nn), desc='runway normal force', units='N')

        self.add_output('CL', val=np.ones(nn), desc='lift coefficient', units=None)

        self.declare_coloring(wrt='*', method='cs')
        self.declare_partials(of='K', wrt='*', method='cs')

        self.add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)
        self.add_output('q', val=np.ones(nn), desc='dynamic pressure', units='Pa')
        self.add_output('L', val=np.ones(nn), desc='lift', units='N')
        self.add_output('D', val=np.ones(nn), desc='drag', units='N')

        self.add_output('W', val=np.ones(nn), desc='aircraft weight', units='N')
        self.add_output('v_stall', val=np.ones(nn), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', val=np.ones(nn), desc='stall speed ratio', units=None)

        ar = np.arange(nn)

        self.declare_coloring(wrt='*', method='cs')
        # self.declare_partials(of='L', wrt=['rho', 'v', 'S', 'CL'], method='cs')
        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        g = 9.80665
        rho = inputs['rho']
        v = inputs['v']
        S = inputs['S']
        CD0 = inputs['CD0']

        h = inputs['h']
        h_w = inputs['h_w']
        span = inputs['span']
        AR = inputs['AR']
        e = inputs['e']
        b = span / 2.0

        CL0 = inputs['CL0']
        alpha = inputs['alpha']
        alpha_max = inputs['alpha_max']
        CL_max = inputs['CL_max']
        m = inputs['m']

        T = inputs['T']
        mu_r = inputs['mu_r']

        W = outputs['W'] = m * g
        outputs['v_stall'] = np.sqrt(2 * W / rho / S / CL_max)
        outputs['v_over_v_stall'] = v / outputs['v_stall']

        CL = outputs['CL'] = CL0 + (alpha / alpha_max) * (CL_max - CL0)

        K_nom = 1.0 / (np.pi * AR * e)
        K = outputs['K'] = K_nom * 33 * ((h + h_w) / b)**1.5 / (1.0 + 33 * ((h + h_w) / b)**1.5)

        q = outputs['q'] = 0.5 * rho * v ** 2
        L = outputs['L'] = q * S * CL
        D = outputs['D'] = q * S * (CD0 + K * CL ** 2)

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        F_r = outputs['F_r'] = m * g - L * calpha - T * salpha
        outputs['v_dot'] = (T * calpha - D - F_r * mu_r) / m

        outputs['r_dot'] = v

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
