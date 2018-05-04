import numpy as np

from openmdao.api import ExplicitComponent


class UnsteadyFlightPathAngleComp(ExplicitComponent):
    """ Flight Path Angle based on velocity and altitude rate. """

    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        # Inputs
        self.add_input('h_rate',
                       val=0.01*np.ones(nn),
                       desc='Altitude rate',
                       units='m/s')

        self.add_input('h_rate2',
                       val=0.01*np.ones(nn),
                       desc='Altitude rate',
                       units='m/s**2')

        self.add_input('TAS',
                       val=77.16*np.ones(nn),
                       desc='Airspeed',
                       units='m/s')

        self.add_input('TAS_rate',
                       val=0.01*np.ones(nn),
                       desc='Airspeed rate',
                       units='m/s**2')

        # Outputs
        self.add_output(name='gam', val=0.05*np.ones(nn),
                        desc='Flight path angle', units='rad')

        self.add_output(name='gam_rate', val=0.05*np.ones(nn),
                        desc='Approximate flight path angle rate', units='rad/s')

        # Setup partials
        ar = np.arange(nn)
        self.declare_partials(of='gam', wrt='h_rate', rows=ar, cols=ar)
        self.declare_partials(of='gam', wrt='TAS', rows=ar, cols=ar)

        self.declare_partials(of='gam_rate', wrt='h_rate', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate', wrt='h_rate2', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate', wrt='TAS', rows=ar, cols=ar)
        self.declare_partials(of='gam_rate', wrt='TAS_rate', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        """ Flight path angle is calculated by taking the arcsin of the ratio
        of the altitude rate to velocity.
        """
        h_rate = inputs['h_rate']
        h_rate2 = inputs['h_rate2']
        v = inputs['TAS']
        v_rate = inputs['TAS_rate']

        ratio = np.clip(h_rate/v, -0.999, 0.999)
        outputs['gam'] = np.arcsin(ratio)
        outputs['gam_rate'] = (v * h_rate2 - h_rate * v_rate)/(v**2 * np.sqrt(1-ratio**2))

    def compute_partials(self, inputs, partials):
        """ The jacobian of the flight path angle component.

        """
        h_rate = inputs['h_rate']
        h_rate2 = inputs['h_rate2']
        v = inputs['TAS']
        v_rate = inputs['TAS_rate']

        ratio = np.clip(h_rate/v, -0.999, 0.999)
        denom = np.sqrt(1.0 - ratio**2)

        partials['gam', 'h_rate'] = 1.0 / (v * denom)
        partials['gam', 'TAS'] = -h_rate / (v**2 * denom)

        x = h_rate
        y = h_rate2
        z = v_rate

        x2 = x * x
        x3 = x2 * x
        v2 = v * v
        v3 = v2 * v

        partials['gam_rate', 'TAS'] = - denom * (v3 * y - 2 * v2 * x * z + x3 * z) / (v * (v2 - x2)**2)
        partials['gam_rate', 'h_rate'] = (x * y - v * z)/(denom * (v3 - v * x2))
        partials['gam_rate', 'TAS_rate'] = -h_rate / (v2 * denom)
        partials['gam_rate', 'h_rate2'] = 1.0 / (v * denom)
