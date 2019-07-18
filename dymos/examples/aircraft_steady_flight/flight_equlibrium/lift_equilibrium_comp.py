import numpy as np

import openmdao.api as om


class LiftEquilibriumComp(om.ExplicitComponent):
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
        self.add_input(name='CT', shape=(n,), desc='thrust coefficient', units=None)
        self.add_input(name='W_total', shape=(n,), desc='total aircraft weight', units='N')
        self.add_input(name='alpha', shape=(n,), desc='angle of attack', units='rad')
        self.add_input(name='gam', shape=(n,), desc='flight path angle', units='rad')
        self.add_input(name='q', shape=(n,), desc='dynamic pressure', units='Pa')
        self.add_input(name='S', shape=(n,), desc='reference area', units='m**2')

        self.add_output(name='CL_eq', shape=(n,),
                        desc='lift coefficient required for steady, level flight',
                        units=None)

        ar = np.arange(n)

        self.declare_partials(of='CL_eq', wrt='CT', rows=ar, cols=ar)
        self.declare_partials(of='CL_eq', wrt='W_total', rows=ar, cols=ar)
        self.declare_partials(of='CL_eq', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='CL_eq', wrt='gam', rows=ar, cols=ar)
        self.declare_partials(of='CL_eq', wrt='q', rows=ar, cols=ar)
        self.declare_partials(of='CL_eq', wrt='S', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        CT = inputs['CT']
        W_total = inputs['W_total']
        alpha = inputs['alpha']
        gam = inputs['gam']
        q = inputs['q']
        S = inputs['S']

        sa = np.sin(alpha)
        cgam = np.cos(gam)

        qS = q * S

        outputs['CL_eq'] = W_total * cgam / qS - CT * sa

    def compute_partials(self, inputs, partials):
        CT = inputs['CT']
        W_total = inputs['W_total']
        alpha = inputs['alpha']
        gam = inputs['gam']
        q = inputs['q']
        S = inputs['S']

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cgam = np.cos(gam)
        sgam = np.sin(gam)

        qS = q * S

        partials['CL_eq', 'CT'] = -sa
        partials['CL_eq', 'W_total'] = cgam / qS
        partials['CL_eq', 'alpha'] = -CT * ca
        partials['CL_eq', 'gam'] = -W_total * sgam / qS
        partials['CL_eq', 'q'] = -W_total * cgam / (q**2 * S)
        partials['CL_eq', 'S'] = -W_total * cgam / (q * S**2)
