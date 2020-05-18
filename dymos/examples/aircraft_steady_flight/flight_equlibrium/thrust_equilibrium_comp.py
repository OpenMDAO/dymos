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

        self.add_output(name='thrust', shape=(n,),
                        desc='thrust required for steady flight',
                        units='N')

        ar = np.arange(n)

        self.declare_partials(of='thrust', wrt='CD', rows=ar, cols=ar)
        self.declare_partials(of='thrust', wrt='W_total', rows=ar, cols=ar)
        self.declare_partials(of='thrust', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='thrust', wrt='gam', rows=ar, cols=ar)
        self.declare_partials(of='thrust', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='thrust', wrt='S', rows=ar, cols=ar)

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

        outputs['thrust'] = W_total * sgam / ca + CD * qS / ca

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

        dT_dsgam = W_total / ca

        dsgam_dgam = cgam

        dT_dca = -W_total * sgam / ca**2 - CD * qS / ca**2
        dca_dalpha = -sa

        dT_dqS = CD / ca

        partials['thrust', 'CD'] = qS / ca
        partials['thrust', 'W_total'] = sgam / ca
        partials['thrust', 'alpha'] = dT_dca * dca_dalpha
        partials['thrust', 'gam'] = dT_dsgam * dsgam_dgam
        partials['thrust', 'q'] = dT_dqS * dqS_dq
        partials['thrust', 'S'] = dT_dqS * dqS_dS
