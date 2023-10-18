import openmdao.api as om
import numpy as np


class RocketODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('h', val=np.zeros(nn), units='ft')
        self.add_input('v', val=np.zeros(nn), units='ft/s')
        self.add_input('m', val=np.zeros(nn), units='slug')
        self.add_input('T', val=np.zeros(nn), units='lbf')

        self.add_output('h_dot', val=np.zeros(nn), units='ft/s')
        self.add_output('v_dot', val=np.zeros(nn), units='ft/s**2')
        self.add_output('m_dot', val=np.zeros(nn), units='slug/s')

        ar = np.arange(nn)
        self.declare_partials(of='h_dot', wrt='v', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='v_dot', wrt='h', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='v_dot', wrt='T', rows=ar, cols=ar)

        self.declare_partials(of='m_dot', wrt='T', rows=ar, cols=ar, val=-1/1580.9425)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        h = inputs['h']
        v = inputs['v']
        m = inputs['m']
        T = inputs['T']

        sigma = 5.4915e-5
        g = 32.174
        c = 1580.9425
        h0 = 23800

        outputs['h_dot'] = v
        outputs['v_dot'] = (T - sigma * v**2 * np.exp(-h/h0))/m - g
        outputs['m_dot'] = -T / c

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        h = inputs['h']
        v = inputs['v']
        m = inputs['m']
        T = inputs['T']

        sigma = 5.4915e-5
        h0 = 23800

        partials['v_dot', 'h'] = (sigma * v**2 * np.exp(-h/h0))/(m * h0)
        partials['v_dot', 'v'] = -(2 * sigma * v * np.exp(-h/h0)) / m
        partials['v_dot', 'm'] = -(T - sigma * v**2 * np.exp(-h/h0)) / (m**2)
        partials['v_dot', 'T'] = 1 / m
