import numpy as np
import openmdao.api as om


class BrachistochroneODE(om.ExplicitComponent):

    #
    # The following dictionaries provide a way of 'tagging' the Brachistochrone ODE with
    # information about states and parameters that can be accessed from Phase.
    #
    # In this case these are class attributes, but the choice of whether or not to tie
    # this information to the ODE itself (and how to do so) is entirely up to the user.
    #
    # In a dynamic ODE model these might be instance attributes whose values vary depending on
    # the arguments to the instantiation.
    #
    states = {'x': {'rate_source': 'xdot',
                    'units': 'm'},
              'y': {'rate_source': 'ydot',
                    'units': 'm'},
              'v': {'rate_source': 'vdot',
                    'units': 'm/s'}}

    parameters = {'theta': {'units': 'rad'},
                  'g': {'units': 'm/s**2'}}

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('static_gravity', types=(bool,), default=False,
                             desc='If True, treat gravity as a static (scalar) input, rather than '
                                  'having different values at each node.')

    def setup(self):
        nn = self.options['num_nodes']
        g_default_val = 9.80665 if self.options['static_gravity'] else 9.80665 * np.ones(nn)

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=g_default_val, desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.ones(nn), desc='angle of wire', units='rad')

        self.add_output('xdot', val=np.zeros(nn), desc='velocity component in x', units='m/s')

        self.add_output('ydot', val=np.zeros(nn), desc='velocity component in y', units='m/s')

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)

        if self.options['static_gravity']:
            c = np.zeros(self.options['num_nodes'])
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=c)
        else:
            self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['xdot'] = v * sin_theta
        outputs['ydot'] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, partials):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        partials['vdot', 'g'] = cos_theta
        partials['vdot', 'theta'] = -g * sin_theta

        partials['xdot', 'v'] = sin_theta
        partials['xdot', 'theta'] = v * cos_theta

        partials['ydot', 'v'] = -cos_theta
        partials['ydot', 'theta'] = v * sin_theta

        partials['check', 'v'] = 1 / sin_theta
        partials['check', 'theta'] = -v * cos_theta / sin_theta ** 2
