import dymos as dm
import openmdao.api as om

import finite_burn_eom


class TwoBodySolution(om.ExplicitComponent):

    def initialize(self):
        nn = self.options['num_nodes']

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r0',
                       val=(1,),
                       desc='initial radius from center of attraction',
                       units='DU')

        self.add_input('r1',
                       val=(1,),
                       desc='final radius from center of attraction',
                       units='DU')

        self.add_input('theta0',
                       val=(1,),
                       desc='initial true anomaly',
                       units='rad')

        self.add_input('theta1',
                       val=(1,),
                       desc='final true anomaly',
                       units='rad')

        self.add_input('vr0',
                       val=(1,),
                       desc='initial radial velocity',
                       units='DU/TU')

        self.add_input('vt0',
                       val=(1,),
                       desc='initial horizontal velocity',
                       units='DU/TU')

        self.add_output('r',
                        shape=(nn,),
                        desc='radius from center of attraction',
                        units='DU')

        self.add_output('theta',
                        shape=(nn,),
                        desc='true anomaly',
                        units='rad')

        self.add_output('vr',
                        shape=(nn,),
                        desc='radial velocity',
                        units='DU/TU')

        self.add_output('vt',
                        shape=(nn,),
                        desc='horizontal velocity',
                        units='DU/TU')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        r0, r1, theta0, theta1, vr0, vt0 = inputs.values()

        # Specific angular momentum
        h = r0 * vt0

        # Semilatus rectum (assuming mu = 1 for canonical units)
        p = h ** 2

        c_dtrua = np.cos(theta1 - theta0)
        s_dtrua = np.sin(theta1 - theta0)

        f = 1 - r1 / p * (1 - c_dtrua)
        g = r0 * r1 / h * s_dtrua

        fdot = r0 * vr0 / (p * r0) * (1 - c_dtrua) - 1 / (r0 * h) * s_dtrua
        gdot = 1 - (r0 / p) * (1 - c_dtrua)




