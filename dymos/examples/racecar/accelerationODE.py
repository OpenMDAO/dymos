import openmdao.api as om
import numpy as np


class AccelerationODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # constants
        self.add_input('tau_y', val=0.2, desc='lateral load transfer time constant', units='s')
        self.add_input('tau_x', val=0.2, desc='longitudinal load transfer time constant', units='s')

        # states
        self.add_input('V', val=np.zeros(nn), desc='speed', units='m/s')
        self.add_input('lambda', val=np.zeros(nn), desc='body slip angle', units='rad')
        self.add_input('omega', val=np.zeros(nn), desc='yaw rate', units='rad/s')

        self.add_input('Vdot', val=np.zeros(nn), desc='speed', units='m/s**2')
        self.add_input('lambdadot', val=np.zeros(nn), desc='body slip angle', units='rad/s')

        self.add_input('ax', val=np.zeros(nn), desc='longitudinal acceleration', units='m/s**2')
        self.add_input('ay', val=np.zeros(nn), desc='lateral acceleration', units='m/s**2')

        # outputs
        self.add_output('axdot', val=np.zeros(nn), desc='longitudinal jerk', units='m/s**3')
        self.add_output('aydot', val=np.zeros(nn), desc='lateral jerk', units='m/s**3')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        # partials
        self.declare_partials(of='axdot', wrt='ax', rows=arange, cols=arange)
        self.declare_partials(of='axdot', wrt='Vdot', rows=arange, cols=arange)
        self.declare_partials(of='axdot', wrt='omega', rows=arange, cols=arange)
        self.declare_partials(of='axdot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='axdot', wrt='lambda', rows=arange, cols=arange)

        self.declare_partials(of='aydot', wrt='ay', rows=arange, cols=arange)
        self.declare_partials(of='aydot', wrt='omega', rows=arange, cols=arange)
        self.declare_partials(of='aydot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='aydot', wrt='Vdot', rows=arange, cols=arange)
        self.declare_partials(of='aydot', wrt='lambda', rows=arange, cols=arange)
        self.declare_partials(of='aydot', wrt='lambdadot', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        tau_y = inputs['tau_y']
        tau_x = inputs['tau_x']
        V = inputs['V']
        lamb = inputs['lambda']
        omega = inputs['omega']
        Vdot = inputs['Vdot']
        lambdadot = inputs['lambdadot']
        ax = inputs['ax']
        ay = inputs['ay']

        outputs['axdot'] = (Vdot+omega*V*lamb-ax)/tau_x
        outputs['aydot'] = (omega*V-(V*lambdadot+Vdot*lamb)-ay)/tau_y

    def compute_partials(self, inputs, jacobian):
        tau_y = inputs['tau_y']
        tau_x = inputs['tau_x']
        V = inputs['V']
        lamb = inputs['lambda']
        omega = inputs['omega']
        Vdot = inputs['Vdot']
        lambdadot = inputs['lambdadot']

        jacobian['axdot', 'ax'] = -1/tau_x
        jacobian['axdot', 'Vdot'] = 1/tau_x
        jacobian['axdot', 'omega'] = (V*lamb)/tau_x
        jacobian['axdot', 'lambda'] = (omega*V)/tau_x
        jacobian['axdot', 'V'] = (omega*lamb)/tau_x

        jacobian['aydot', 'ay'] = -1/tau_y
        jacobian['aydot', 'omega'] = V/tau_y
        jacobian['aydot', 'V'] = (omega-lambdadot)/tau_y
        jacobian['aydot', 'lambda'] = -Vdot/tau_y
        jacobian['aydot', 'lambdadot'] = -V/tau_y
        jacobian['aydot', 'Vdot'] = -lamb/tau_y
