import numpy as np
import openmdao.api as om


class MoonLandingProblemODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs
        self.add_input('h', val=np.ones(nn), units=None, desc='Altitude')
        self.add_input('v', val=np.ones(nn), units='1/s', desc='Velocity')
        self.add_input('m', val=np.ones(nn), units=None, desc='Mass')
        self.add_input('T', val=np.ones(nn), units=None, desc='Thrust')

        # outputs
        self.add_output('h_dot', val=np.ones(nn), units='1/s', desc='Rate of change of Altitude')
        self.add_output('v_dot', val=np.ones(nn), units='1/s**2', desc='Rate of change of Velocity')
        self.add_output('m_dot', val=np.ones(nn), units='1/s', desc='Rate of change of Mass')

        # partials
        ar = np.arange(nn)
        self.declare_partials(of='h_dot', wrt='v', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='v_dot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='T', rows=ar, cols=ar)
        self.declare_partials(of='m_dot', wrt='T', rows=ar, cols=ar, val=-1/2.349)
        self.declare_partials(of='m_dot', wrt='T', rows=ar, cols=ar, val=-1/2.349)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        v = inputs['v']
        m = inputs['m']
        T = inputs['T']

        outputs['h_dot'] = v
        outputs['v_dot'] = -1 + T/m
        outputs['m_dot'] = -T/2.349

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        m = inputs['m']
        T = inputs['T']

        partials['v_dot', 'T'] = 1/m
        partials['v_dot', 'm'] = -T/m**2
