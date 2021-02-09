import openmdao.api as om
import numpy as np


class TimeODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # states
        self.add_input('sdot', val=np.zeros(nn), desc='distance along track', units='m/s')
        self.add_input('ndot', val=np.zeros(nn), desc='distance perpendicular to centerline',
                       units='m/s')
        self.add_input('alphadot', val=np.zeros(nn), desc='angle relative to centerline',
                       units='rad/s')
        self.add_input('Vdot', val=np.zeros(nn), desc='speed', units='m/s**2')
        self.add_input('lambdadot', val=np.zeros(nn), desc='body slip angle', units='rad/s')
        self.add_input('omegadot', val=np.zeros(nn), desc='yaw rate', units='rad/s**2')
        self.add_input('axdot', val=np.zeros(nn), desc='longitudinal jerk', units='m/s**3')
        self.add_input('aydot', val=np.zeros(nn), desc='lateral jerk', units='m/s**3')

        # outputs
        self.add_output('dn_ds', val=np.zeros(nn), desc='distance perpendicular to centerline',
                        units='m/m')
        self.add_output('dalpha_ds', val=np.zeros(nn), desc='angle relative to centerline',
                        units='rad/m')
        self.add_output('dV_ds', val=np.zeros(nn), desc='speed', units='1/s')
        self.add_output('dlambda_ds', val=np.zeros(nn), desc='body slip angle', units='rad/m')
        self.add_output('domega_ds', val=np.zeros(nn), desc='yaw rate', units='rad/(s*m)')
        self.add_output('dax_ds', val=np.zeros(nn), desc='longitudinal jerk', units='1/s**2')
        self.add_output('day_ds', val=np.zeros(nn), desc='lateral jerk', units='1/s**2')

        # Setup partials
        arange = np.arange(self.options['num_nodes'], dtype=int)

        # partials

        self.declare_partials(of='dn_ds', wrt='ndot', rows=arange, cols=arange)
        self.declare_partials(of='dn_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dalpha_ds', wrt='alphadot', rows=arange, cols=arange)
        self.declare_partials(of='dalpha_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dV_ds', wrt='Vdot', rows=arange, cols=arange)
        self.declare_partials(of='dV_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dlambda_ds', wrt='lambdadot', rows=arange, cols=arange)
        self.declare_partials(of='dlambda_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='domega_ds', wrt='omegadot', rows=arange, cols=arange)
        self.declare_partials(of='domega_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='dax_ds', wrt='axdot', rows=arange, cols=arange)
        self.declare_partials(of='dax_ds', wrt='sdot', rows=arange, cols=arange)

        self.declare_partials(of='day_ds', wrt='aydot', rows=arange, cols=arange)
        self.declare_partials(of='day_ds', wrt='sdot', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        omegadot = inputs['omegadot']
        sdot = inputs['sdot']
        Vdot = inputs['Vdot']
        ndot = inputs['ndot']
        lambdadot = inputs['lambdadot']
        alphadot = inputs['alphadot']
        axdot = inputs['axdot']
        aydot = inputs['aydot']

        outputs['domega_ds'] = omegadot/sdot
        outputs['dV_ds'] = Vdot/sdot
        outputs['dalpha_ds'] = alphadot/sdot
        outputs['dlambda_ds'] = lambdadot/sdot
        outputs['dn_ds'] = ndot/sdot
        outputs['dax_ds'] = axdot/sdot
        outputs['day_ds'] = aydot/sdot

    def compute_partials(self, inputs, jacobian):
        omegadot = inputs['omegadot']
        sdot = inputs['sdot']
        Vdot = inputs['Vdot']
        ndot = inputs['ndot']
        lambdadot = inputs['lambdadot']
        alphadot = inputs['alphadot']
        axdot = inputs['axdot']
        aydot = inputs['aydot']

        jacobian['dn_ds', 'sdot'] = -ndot/sdot**2
        jacobian['dn_ds', 'ndot'] = 1/sdot

        jacobian['dalpha_ds', 'sdot'] = -alphadot/sdot**2
        jacobian['dalpha_ds', 'alphadot'] = 1/sdot

        jacobian['domega_ds', 'sdot'] = -omegadot/sdot**2
        jacobian['domega_ds', 'omegadot'] = 1/sdot

        jacobian['dlambda_ds', 'sdot'] = -lambdadot/sdot**2
        jacobian['dlambda_ds', 'lambdadot'] = 1/sdot

        jacobian['dV_ds', 'sdot'] = -Vdot/sdot**2
        jacobian['dV_ds', 'Vdot'] = 1/sdot

        jacobian['dax_ds', 'sdot'] = -axdot/sdot**2
        jacobian['dax_ds', 'axdot'] = 1/sdot

        jacobian['day_ds', 'sdot'] = -aydot/sdot**2
        jacobian['day_ds', 'aydot'] = 1/sdot
