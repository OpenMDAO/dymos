import numpy as np
from scipy.interpolate import interp1d

import openmdao.api as om
from dymos.models.atmosphere.atmos_1976 import USatm1976Data


english_to_metric_rho = om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
english_to_metric_h = om.unit_conversion('ft', 'm')[0]
rho_interp = interp1d(np.array(USatm1976Data.alt*english_to_metric_h, dtype=complex),
                      np.array(USatm1976Data.rho*english_to_metric_rho, dtype=complex), kind='linear')


class CannonballODE(om.ExplicitComponent):
    """
    Cannonball ODE assuming flat earth and accounting for air resistance
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # static parameters
        self.add_input('m', units='kg')
        self.add_input('S', units='m**2')
        # 0.5 good assumption for a sphere
        self.add_input('CD', 0.5)

        # time varying inputs
        self.add_input('h', units='m', shape=nn)
        self.add_input('v', units='m/s', shape=nn)
        self.add_input('gam', units='rad', shape=nn)

        # state rates
        self.add_output('v_dot', shape=nn, units='m/s**2')
        self.add_output('gam_dot', shape=nn, units='rad/s')
        self.add_output('h_dot', shape=nn, units='m/s')
        self.add_output('r_dot', shape=nn, units='m/s')
        self.add_output('ke', shape=nn, units='J')

        # Ask OpenMDAO to compute the partial derivatives using complex-step
        # with a partial coloring algorithm for improved performance, and use
        # coloring to determine the sparsity pattern.
        self.declare_coloring(wrt='*', method='cs')

    def compute(self, inputs, outputs):

        gam = inputs['gam']
        v = inputs['v']
        h = inputs['h']
        m = inputs['m']
        S = inputs['S']
        CD = inputs['CD']

        GRAVITY = 9.80665  # m/s**2

        # handle complex-step gracefully from the interpolant
        if np.iscomplexobj(h):
            rho = rho_interp(inputs['h'])
        else:
            rho = rho_interp(inputs['h']).real

        q = 0.5*rho*inputs['v']**2
        qS = q * S
        D = qS * CD
        cgam = np.cos(gam)
        sgam = np.sin(gam)
        outputs['v_dot'] = - D/m-GRAVITY*sgam
        outputs['gam_dot'] = -(GRAVITY/v)*cgam
        outputs['h_dot'] = v*sgam
        outputs['r_dot'] = v*cgam
        outputs['ke'] = 0.5*m*v**2
