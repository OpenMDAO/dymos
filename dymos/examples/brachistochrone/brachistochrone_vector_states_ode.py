import numpy as np
import openmdao.api as om


class BrachistochroneVectorStatesODE(om.ExplicitComponent):

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
    states = {'pos': {'rate_source': 'pos_dot',
                      'units': 'm',
                      'shape': (2,)},
              'v': {'rate_source': 'vdot',
                    'targets': 'v',
                    'units': 'm/s'}}

    parameters = {'theta': {'targets': 'theta',
                            'units': 'rad'},
                  'g': {'targets': 'g',
                        'units': 'm/s**2'}}

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('v', val=np.zeros(nn), desc='velocity', units='m/s')

        self.add_input('g', val=9.80665 * np.ones(nn), desc='grav. acceleration', units='m/s/s')

        self.add_input('theta', val=np.zeros(nn), desc='angle of wire', units='rad')

        self.add_output('pos_dot', val=np.zeros((nn, 2)), desc='velocity components', units='m/s')

        self.add_output('vdot', val=np.zeros(nn), desc='acceleration magnitude', units='m/s**2')

        self.add_output('check', val=np.zeros(nn), desc='check solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange)
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange)

        rs = np.arange(2*nn, dtype=int)
        cs = np.repeat(np.arange(nn, dtype=int), 2)
        self.declare_partials(of='pos_dot', wrt='v', rows=rs, cols=cs)
        self.declare_partials(of='pos_dot', wrt='theta', rows=rs, cols=cs)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g * cos_theta
        outputs['pos_dot'][:, 0] = v * sin_theta
        outputs['pos_dot'][:, 1] = -v * cos_theta
        outputs['check'] = v / sin_theta

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        jacobian['vdot', 'g'] = cos_theta
        jacobian['vdot', 'theta'] = -g * sin_theta

        jacobian['pos_dot', 'v'][0::2] = sin_theta
        jacobian['pos_dot', 'v'][1::2] = -cos_theta

        jacobian['pos_dot', 'theta'][0::2] = v * cos_theta
        jacobian['pos_dot', 'theta'][1::2] = v * sin_theta

        jacobian['check', 'v'] = 1 / sin_theta
        jacobian['check', 'theta'] = -v * cos_theta / sin_theta**2
