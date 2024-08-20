import openmdao.api as om
import numpy as np


class LowThrustODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('r', val=np.zeros(nn), units=None)
        self.add_input('theta', val=np.zeros(nn), units='rad')
        self.add_input('vr', val=np.zeros(nn), units='1/s')
        self.add_input('vt', val=np.zeros(nn), units='1/s')
        self.add_input('alpha', val=np.zeros(nn), units='rad')

        self.add_output('theta_dot', val=np.zeros(nn), units='rad/s')
        self.add_output('vr_dot', val=np.zeros(nn), units='1/s**2')
        self.add_output('vt_dot', val=np.zeros(nn), units='1/s**2')

        ar = np.arange(nn)

        self.declare_partials(of='theta_dot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='theta_dot', wrt='vt', rows=ar, cols=ar)

        self.declare_partials(of='vr_dot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vr_dot', wrt='vt', rows=ar, cols=ar)
        self.declare_partials(of='vr_dot', wrt='alpha', rows=ar, cols=ar)

        self.declare_partials(of='vt_dot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vt_dot', wrt='vr', rows=ar, cols=ar)
        self.declare_partials(of='vt_dot', wrt='vt', rows=ar, cols=ar)
        self.declare_partials(of='vt_dot', wrt='alpha', rows=ar, cols=ar)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        r = inputs['r']
        vr = inputs['vr']
        vt = inputs['vt']
        alpha = inputs['alpha']

        T = 1e-3

        outputs['theta_dot'] = vt/r
        outputs['vr_dot'] = vt**2/r - 1/r**2 + T*np.sin(alpha)
        outputs['vt_dot'] = -vr*vt/r + T*np.cos(alpha)

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        r = inputs['r']
        vr = inputs['vr']
        vt = inputs['vt']
        alpha = inputs['alpha']

        T = 1e-3

        partials['theta_dot', 'r'] = -vt/r**2
        partials['theta_dot', 'vt'] = 1/r

        partials['vr_dot', 'r'] = -(vt/r)**2 + 2/r**3
        partials['vr_dot', 'vt'] = 2*vt/r
        partials['vr_dot', 'alpha'] = T * np.cos(alpha)

        partials['vt_dot', 'r'] = vt*vr/r**2
        partials['vt_dot', 'vr'] = -vt/r
        partials['vt_dot', 'vt'] = -vr/r
        partials['vt_dot', 'alpha'] = -T * np.sin(alpha)
