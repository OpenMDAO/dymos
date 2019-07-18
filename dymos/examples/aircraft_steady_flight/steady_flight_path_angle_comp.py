import numpy as np

import openmdao.api as om


class SteadyFlightPathAngleComp(om.ExplicitComponent):
    """ Compute the flight path angle (gamma) based on true airspeed and climb rate. """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('TAS', val=np.zeros(nn), desc='true airspeed', units='m/s')
        self.add_input('climb_rate', val=np.zeros(nn), desc='climb rate', units='m/s')
        self.add_output('gam', val=np.zeros(nn), desc='flight path angle', units='rad')

        # Setup partials
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='gam', wrt='TAS', rows=ar, cols=ar)
        self.declare_partials(of='gam', wrt='climb_rate', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        ratio = np.clip(inputs['climb_rate'] / inputs['TAS'], -1.0, 1.0)
        outputs['gam'] = np.arcsin(ratio)

    def compute_partials(self, inputs, partials):
        tas = inputs['TAS']
        h_rate = inputs['climb_rate']

        k = np.sqrt(1 - (h_rate / tas)**2)

        dgam_dhdot = 1.0 / (tas * k)
        dgam_dtas = -h_rate / (k * tas**2)

        partials['gam', 'TAS'] = dgam_dtas
        partials['gam', 'climb_rate'] = dgam_dhdot
