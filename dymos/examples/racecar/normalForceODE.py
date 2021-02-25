import openmdao.api as om
import numpy as np


class NormalForceODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # constants
        self.add_input('M', val=0.0, desc='mass', units='kg')
        self.add_input('g', val=9.8, desc='mass', units='m/s**2')
        self.add_input('a', val=1.8, desc='cg to front distance', units='m')
        self.add_input('b', val=1.6, desc='cg to rear distance', units='m')
        self.add_input('tw', val=0.73, desc='half track width', units='m')
        self.add_input('h', val=0.0, desc='cg height', units='m')  # changed from 0.4
        self.add_input('chi', val=0.0, desc='roll stiffness', units=None)
        self.add_input('CoP', val=1.8, desc='center of pressure (distance to front)', units='m')

        # adding downforce
        self.add_input('ClA', val=4.0, desc='lift coefficient', units='m**2')
        self.add_input('rho', val=1.2, desc='air density', units='kg/m**3')
        self.add_input('V', val=np.zeros(nn), desc='speed', units='m/s')

        # states
        self.add_input('ax', val=np.zeros(nn), desc='longitudinal acceleration', units='m/s**2')
        self.add_input('ay', val=np.zeros(nn), desc='lateral acceleration', units='m/s**2')

        # normal load outputs
        self.add_output('N_fl', val=np.zeros(nn), desc='normal force fl', units='N')
        self.add_output('N_fr', val=np.zeros(nn), desc='normal force fr', units='N')
        self.add_output('N_rl', val=np.zeros(nn), desc='normal force rl', units='N')
        self.add_output('N_rr', val=np.zeros(nn), desc='normal force rr', units='N')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        # partials
        self.declare_partials(of='N_fl', wrt='ax', rows=arange, cols=arange)
        self.declare_partials(of='N_fr', wrt='ax', rows=arange, cols=arange)
        self.declare_partials(of='N_rl', wrt='ax', rows=arange, cols=arange)
        self.declare_partials(of='N_rr', wrt='ax', rows=arange, cols=arange)

        self.declare_partials(of='N_fl', wrt='ay', rows=arange, cols=arange)
        self.declare_partials(of='N_fr', wrt='ay', rows=arange, cols=arange)
        self.declare_partials(of='N_rl', wrt='ay', rows=arange, cols=arange)
        self.declare_partials(of='N_rr', wrt='ay', rows=arange, cols=arange)

        self.declare_partials(of='N_fl', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='N_fr', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='N_rl', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='N_rr', wrt='V', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        M = inputs['M']
        g = inputs['g']
        a = inputs['a']
        b = inputs['b']
        ax = inputs['ax']
        ay = inputs['ay']
        h = inputs['h']
        chi = inputs['chi']
        tw = inputs['tw']
        rho = inputs['rho']
        V = inputs['V']
        CoP = inputs['CoP']
        ClA = inputs['ClA']

        downforce = 0.5 * rho * ClA * V ** 2

        downforce_rear = downforce * (CoP / (a + b))
        downforce_front = downforce * (1 - (CoP / (a + b)))

        outputs['N_fl'] = (M * g / 2) * (b / (a + b)) + (M / 4) * (
                          (-(ax * h) / (a + b)) + (ay * chi * h / tw)) + downforce_front / 2
        outputs['N_fr'] = (M * g / 2) * (b / (a + b)) + (M / 4) * (
                          (-(ax * h) / (a + b)) - (ay * chi * h / tw)) + downforce_front / 2
        outputs['N_rl'] = (M * g / 2) * (a / (a + b)) + (M / 4) * (
                          ((ax * h) / (a + b)) + (ay * (1 - chi) * h / tw)) + downforce_rear / 2
        outputs['N_rr'] = (M * g / 2) * (a / (a + b)) + (M / 4) * (
                          ((ax * h) / (a + b)) - (ay * (1 - chi) * h / tw)) + downforce_rear / 2

    def compute_partials(self, inputs, jacobian):
        M = inputs['M']
        a = inputs['a']
        b = inputs['b']
        h = inputs['h']
        chi = inputs['chi']
        tw = inputs['tw']
        rho = inputs['rho']
        V = inputs['V']
        CoP = inputs['CoP']
        ClA = inputs['ClA']

        jacobian['N_fl', 'ax'] = -(M * h) / (4 * (a + b))
        jacobian['N_fr', 'ax'] = -(M * h) / (4 * (a + b))
        jacobian['N_rl', 'ax'] = (M * h) / (4 * (a + b))
        jacobian['N_rr', 'ax'] = (M * h) / (4 * (a + b))

        jacobian['N_fl', 'ay'] = (M * chi * h) / (4 * tw)
        jacobian['N_rl', 'ay'] = (M * (1 - chi) * h) / (4 * tw)
        jacobian['N_fr', 'ay'] = -(M * chi * h) / (4 * tw)
        jacobian['N_rr', 'ay'] = -(M * (1 - chi) * h) / (4 * tw)

        ddownforce_dv = rho * ClA * V

        jacobian['N_fl', 'V'] = ddownforce_dv * (1 - (CoP / (a + b))) * 0.5
        jacobian['N_fr', 'V'] = ddownforce_dv * (1 - (CoP / (a + b))) * 0.5
        jacobian['N_rl', 'V'] = ddownforce_dv * (CoP / (a + b)) * 0.5
        jacobian['N_rr', 'V'] = ddownforce_dv * (CoP / (a + b)) * 0.5
