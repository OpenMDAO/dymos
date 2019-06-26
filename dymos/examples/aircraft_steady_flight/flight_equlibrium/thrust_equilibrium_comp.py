import numpy as np

import openmdao.api as om


class ThrustEquilibriumComp(om.ExplicitComponent):
    """
    Compute the rates of TAS and flight path angle required to match a given
    flight condition.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

        self._g = 9.80665

    def setup(self):
        n = self.options['num_nodes']

        # Parameter inputs
        self.add_input(name='CD', shape=(n,), desc='drag coefficient', units=None)
        self.add_input(name='W_total', shape=(n,), desc='total weight', units='N')
        self.add_input(name='alpha', shape=(n,), desc='angle of attack', units='rad')
        self.add_input(name='gam', shape=(n,), desc='flight path angle', units='rad')
        self.add_input(name='q', shape=(n,), desc='dynamic pressure', units='Pa')
        self.add_input(name='S', shape=(n,), desc='reference area', units='m**2')

        self.add_output(name='CT', shape=(n,),
                        desc='thrust coefficient required for steady, level flight',
                        units=None)

        ar = np.arange(n)

        self.declare_partials(of='CT', wrt='CD', rows=ar, cols=ar)
        self.declare_partials(of='CT', wrt='W_total', rows=ar, cols=ar)
        self.declare_partials(of='CT', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CT', wrt='gam', rows=ar, cols=ar)
        self.declare_partials(of='CT', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='CT', wrt='S', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        CD = inputs['CD']
        W_total = inputs['W_total']
        alpha = inputs['alpha']
        gam = inputs['gam']
        q = inputs['q']
        S = inputs['S']

        ca = np.cos(alpha)
        sgam = np.sin(gam)

        qS = q * S

        outputs['CT'] = W_total * sgam / (ca * qS) + CD / ca

    def compute_partials(self, inputs, partials):
        CD = inputs['CD']
        W_total = inputs['W_total']
        alpha = inputs['alpha']
        gam = inputs['gam']
        q = inputs['q']
        S = inputs['S']

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        sgam = np.sin(gam)
        cgam = np.cos(gam)

        qS = q * S
        dqS_dq = S
        dqS_dS = q

        dCT_dsgam = W_total / (ca * qS)
        dsgam_dgam = cgam

        dCT_dca = -W_total * sgam / (ca**2 * qS) - CD / ca**2
        dca_dalpha = -sa

        dCT_dqS = -W_total * sgam / (ca * qS**2)

        partials['CT', 'CD'] = 1.0 / ca
        partials['CT', 'W_total'] = sgam / (ca * qS)
        partials['CT', 'alpha'] = dCT_dca * dca_dalpha
        partials['CT', 'gam'] = dCT_dsgam * dsgam_dgam
        partials['CT', 'q'] = dCT_dqS * dqS_dq
        partials['CT', 'S'] = dCT_dqS * dqS_dS
