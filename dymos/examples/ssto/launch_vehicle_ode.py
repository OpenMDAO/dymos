import openmdao.api as om
import numpy as np

_g = {'earth': 9.80665,
      'moon': 1.61544}


class LaunchVehicleODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        self.options.declare('central_body', values=['earth', 'moon'], default='earth',
                             desc='The central gravitational body for the launch vehicle.')

        self.options.declare('CD', types=float, default=0.5,
                             desc='coefficient of drag')

        self.options.declare('S', types=float, default=7.069,
                             desc='aerodynamic reference area (m**2)')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('y',
                       val=np.zeros(nn),
                       desc='altitude',
                       units='m')

        self.add_input('vx',
                       val=np.zeros(nn),
                       desc='x velocity',
                       units='m/s')

        self.add_input('vy',
                       val=np.zeros(nn),
                       desc='y velocity',
                       units='m/s')

        self.add_input('m',
                       val=np.zeros(nn),
                       desc='mass',
                       units='kg')

        self.add_input('theta',
                       val=np.zeros(nn),
                       desc='pitch angle',
                       units='rad')

        self.add_input('thrust',
                       val=2100000 * np.ones(nn),
                       desc='thrust',
                       units='N')

        self.add_input('Isp',
                       val=265.2 * np.ones(nn),
                       desc='specific impulse',
                       units='s')
        # Outputs
        self.add_output('xdot',
                        val=np.zeros(nn),
                        desc='velocity component in x',
                        units='m/s')

        self.add_output('ydot',
                        val=np.zeros(nn),
                        desc='velocity component in y',
                        units='m/s')

        self.add_output('vxdot',
                        val=np.zeros(nn),
                        desc='x acceleration magnitude',
                        units='m/s**2')

        self.add_output('vydot',
                        val=np.zeros(nn),
                        desc='y acceleration magnitude',
                        units='m/s**2')

        self.add_output('mdot',
                        val=np.zeros(nn),
                        desc='mass rate of change',
                        units='kg/s')

        self.add_output('rho',
                        val=np.zeros(nn),
                        desc='density',
                        units='kg/m**3')
        # Setup partials
        ar = np.arange(self.options['num_nodes'])

        self.declare_partials(of='rho', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='xdot', wrt='vx', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='ydot', wrt='vy', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='vxdot', wrt='vx', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='theta', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vxdot', wrt='thrust', rows=ar, cols=ar)

        self.declare_partials(of='vydot', wrt='m', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='theta', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='vy', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='y', rows=ar, cols=ar)
        self.declare_partials(of='vydot', wrt='thrust', rows=ar, cols=ar)

        self.declare_partials(of='mdot', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='mdot', wrt='Isp', rows=ar, cols=ar)

        # # Complex-step derivatives
        # self.declare_partials(of='*', wrt='*', method='cs')
        # self.declare_coloring(wrt='*', method='cs', show_sparsity=True)

    def compute(self, inputs, outputs):
        cb = self.options['central_body']

        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        vx = inputs['vx']
        vy = inputs['vy']
        m = inputs['m']
        F_T = inputs['thrust']
        Isp = inputs['Isp']
        y = inputs['y']

        if cb == 'earth':
            rho_ref = 1.225
            h_scale = 8.44E3
        elif cb == 'moon':
            rho_ref = 0.0
            h_scale = 1.0
        else:
            raise RuntimeError('Unrecognized value for central_body: {0}'.format(cb))

        g = _g[self.options['central_body']]
        CDA = self.options['CD'] * self.options['S']

        outputs['rho'] = rho_ref * np.exp(-y / h_scale)
        outputs['xdot'] = vx
        outputs['ydot'] = vy
        outputs['vxdot'] = (F_T * cos_theta - 0.5 * CDA * outputs['rho'] * vx**2) / m
        outputs['vydot'] = (F_T * sin_theta - 0.5 * CDA * outputs['rho'] * vy**2) / m - g
        outputs['mdot'] = -F_T / (g * Isp)

    def compute_partials(self, inputs, jacobian):
        cb = self.options['central_body']

        y = inputs['y']
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        m = inputs['m']
        vx = inputs['vx']
        vy = inputs['vy']
        F_T = inputs['thrust']
        Isp = inputs['Isp']

        g = _g[self.options['central_body']]
        CDA = self.options['CD'] * self.options['S']

        if cb == 'earth':
            rho_ref = 1.225
            h_scale = 8.44E3
        elif cb == 'moon':
            rho_ref = 0.0
            h_scale = 1.0
        else:
            raise RuntimeError('Unrecognized value for central_body: {0}'.format(cb))

        jacobian['vxdot', 'vx'] = -CDA * rho_ref * np.exp(-y / h_scale) * vx / m
        jacobian['vxdot', 'y'] = (0.5 * CDA * vx ** 2 / m) * (rho_ref / h_scale) * np.exp(-y / h_scale)
        jacobian['vxdot', 'm'] = -(F_T * cos_theta - 0.5 * CDA * rho_ref * np.exp(-y / h_scale) * vx ** 2) / m ** 2
        jacobian['vxdot', 'theta'] = -(F_T / m) * sin_theta
        jacobian['vxdot', 'thrust'] = cos_theta / m

        jacobian['vydot', 'vy'] = -CDA * rho_ref * np.exp(-y / h_scale) * vy / m
        jacobian['vydot', 'y'] = (0.5 * CDA * vy ** 2 / m) * (rho_ref / h_scale) * np.exp(-y / h_scale)
        jacobian['vydot', 'm'] = -(F_T * sin_theta - 0.5 * CDA * rho_ref * np.exp(-y / h_scale) * vy ** 2) / m ** 2
        jacobian['vydot', 'theta'] = (F_T / m) * cos_theta
        jacobian['vydot', 'thrust'] = sin_theta / m

        jacobian['mdot', 'thrust'] = -1.0 / (g * Isp)
        jacobian['mdot', 'Isp'] = F_T / (g * Isp ** 2)

        jacobian['rho', 'y'] = -rho_ref * np.exp(-y / h_scale) / h_scale
