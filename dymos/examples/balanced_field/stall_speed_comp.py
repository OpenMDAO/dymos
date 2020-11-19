import numpy as np
import openmdao.api as om


class StallSpeedComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('v', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input('W', val=np.ones(nn), desc='aircraft weight', units='N')
        self.add_input('rho', val=np.ones(nn), desc='atmospheric density', units='kg/m**3')
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CL_max', val=2.9, desc='maximum lift coefficient', units=None)

        self.add_output('v_stall', val=np.ones(nn), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', val=np.ones(nn), desc='stall speed ratio', units=None)

        self.declare_coloring(wrt='*', method='cs', tol=1.0E-12, show_sparsity=False, show_summary=False)
        self.declare_partials(of='v_stall', wrt='*', method='cs')

    def compute(self, inputs, outputs):
        v = inputs['v']
        W = inputs['W']
        rho = inputs['rho']
        S = inputs['S']
        CL_max = inputs['CL_max']
        outputs['v_stall'] = np.sqrt(2 * W / rho / S / CL_max)
        outputs['v_over_v_stall'] = v / outputs['v_stall']
