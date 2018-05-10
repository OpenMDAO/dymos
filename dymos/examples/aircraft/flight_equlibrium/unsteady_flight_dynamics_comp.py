import numpy as np

from openmdao.api import ExplicitComponent


class UnsteadyFlightDynamicsComp(ExplicitComponent):
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
        self.add_input(name='thrust', shape=(n,), desc='thrust', units='N')
        self.add_input(name='D', shape=(n,), desc='drag force', units='N')
        self.add_input(name='L', shape=(n,), desc='lift force', units='N')
        self.add_input(name='alpha', shape=(n,), desc='angle of attack', units='rad')
        self.add_input(name='gam', shape=(n,), desc='flight path angle', units='rad')
        self.add_input(name='mass', shape=(n,), desc='aircraft mass', units='kg')
        self.add_input(name='TAS', shape=(n,), desc='true airspeed', units='m/s')

        self.add_output(name='TAS_rate_computed', shape=(n,),
                        desc='rate of change in TAS required to match the given flight conditions',
                        units='m/s**2')

        self.add_output(name='gam_rate_computed', shape=(n,),
                        desc='rate of change in gam required to match the given flight conditions',
                        units='rad/s')

        ar = np.arange(n)

        self.declare_partials(of='TAS_rate_computed', wrt='gam', rows=ar, cols=ar)
        self.declare_partials(of='TAS_rate_computed', wrt='D', rows=ar, cols=ar)
        self.declare_partials(of='TAS_rate_computed', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='TAS_rate_computed', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='TAS_rate_computed', wrt='mass', rows=ar, cols=ar)

        self.declare_partials(of='gam_rate_computed', wrt='gam', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate_computed', wrt='L', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate_computed', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate_computed', wrt='alpha', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate_computed', wrt='mass', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate_computed', wrt='TAS', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        T = inputs['thrust']
        L = inputs['L']
        D = inputs['D']
        alpha = inputs['alpha']
        gam = inputs['gam']
        g = self._g
        m = inputs['mass']
        TAS = inputs['TAS']

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cgam = np.cos(gam)
        sgam = np.sin(gam)

        outputs['TAS_rate_computed'] = (T * ca - D) / m - g * sgam
        # TAS_dot_comp = outputs['TAS_rate_computed']

        outputs['gam_rate_computed'] = (T * sa + L) / (m * TAS) - g * cgam / TAS
        # gam_dot_comp = outputs['gam_rate_computed']

    def compute_partials(self, inputs, partials):
        T = inputs['thrust']
        L = inputs['L']
        D = inputs['D']
        alpha = inputs['alpha']
        gam = inputs['gam']
        g = self._g
        m = inputs['mass']
        TAS = inputs['TAS']

        ca = np.cos(alpha)
        sa = np.sin(alpha)
        cgam = np.cos(gam)
        sgam = np.sin(gam)

        mTAS = m * TAS

        partials['TAS_rate_computed', 'gam'] = -g * cgam
        partials['TAS_rate_computed', 'D'] = -1.0 / m
        partials['TAS_rate_computed', 'thrust'] = ca / m
        partials['TAS_rate_computed', 'alpha'] = -T * sa / m
        partials['TAS_rate_computed', 'mass'] = -(T * ca - D) / m**2

        partials['gam_rate_computed', 'gam'] = g * sgam / TAS
        partials['gam_rate_computed', 'L'] = 1.0 / mTAS
        partials['gam_rate_computed', 'thrust'] = sa / mTAS
        partials['gam_rate_computed', 'alpha'] = T * ca / mTAS
        partials['gam_rate_computed', 'mass'] = -(T * sa + L) / (m * mTAS)
        partials['gam_rate_computed', 'TAS'] = (g * cgam) / TAS**2 - (T * sa + L) / (TAS * mTAS)
