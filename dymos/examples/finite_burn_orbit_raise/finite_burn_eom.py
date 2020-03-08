import numpy as np
import openmdao.api as om
import openmdao.utils.units as units

# Add canonical units to OpenMDAO
MU_earth = 3.986592936294783e14
R_earth = 6378137.0

period = 2 * np.pi * np.sqrt(R_earth**3 / MU_earth)

# Add canonical time and distance units for these EOM
units.add_unit('TU', '{0}*s'.format(period))
units.add_unit('DU', '{0}*m'.format(R_earth))


class FiniteBurnODE(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('r',
                       val=np.ones(nn),
                       desc='radius from center of attraction',
                       units='DU')

        self.add_input('theta',
                       val=np.zeros(nn),
                       desc='anomaly term',
                       units='rad')

        self.add_input('vr',
                       val=np.zeros(nn),
                       desc='local vertical velocity component',
                       units='DU/TU')

        self.add_input('vt',
                       val=np.zeros(nn),
                       desc='local horizontal velocity component',
                       units='DU/TU')

        self.add_input('accel',
                       val=np.zeros(nn),
                       desc='acceleration due to thrust',
                       units='DU/TU**2')

        self.add_input('u1',
                       val=np.zeros(nn),
                       desc='thrust angle above local horizontal',
                       units='rad')

        self.add_input('c',
                       val=np.zeros(nn),
                       desc='exhaust velocity',
                       units='DU/TU')

        self.add_output('r_dot',
                        val=np.ones(nn),
                        desc='rate of change of radius from center of attraction',
                        units='DU/TU')

        self.add_output('theta_dot',
                        val=np.zeros(nn),
                        desc='rate of change of anomaly term',
                        units='rad/TU')

        self.add_output('vr_dot',
                        val=np.zeros(nn),
                        desc='rate of change of local vertical velocity component',
                        units='DU/TU**2')

        self.add_output('vt_dot',
                        val=np.zeros(nn),
                        desc='rate of change of local horizontal velocity component',
                        units='DU/TU**2')

        self.add_output('at_dot',
                        val=np.zeros(nn),
                        desc='rate of change of acceleration due to thrust',
                        units='DU/TU**3')

        self.add_output('deltav_dot',
                        val=np.zeros(nn),
                        desc='rate of change of delta-V',
                        units='DU/TU**2')

        self.add_output('pos_x',
                        val=np.zeros(nn),
                        desc='x-component of position',
                        units='DU')

        self.add_output('pos_y',
                        val=np.zeros(nn),
                        desc='x-component of position',
                        units='DU')

        # Setup partials
        ar = np.arange(self.options['num_nodes'])

        # r dot is a linear function of vr, so provide the partial value here
        self.declare_partials(of='r_dot', wrt='vr', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='theta_dot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='theta_dot', wrt='vt', rows=ar, cols=ar)

        self.declare_partials(of='vr_dot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vr_dot', wrt='vt', rows=ar, cols=ar)
        self.declare_partials(of='vr_dot', wrt='accel', rows=ar, cols=ar)
        self.declare_partials(of='vr_dot', wrt='u1', rows=ar, cols=ar)

        self.declare_partials(of='vt_dot', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='vt_dot', wrt='vr', rows=ar, cols=ar)
        self.declare_partials(of='vt_dot', wrt='vt', rows=ar, cols=ar)
        self.declare_partials(of='vt_dot', wrt='accel', rows=ar, cols=ar)
        self.declare_partials(of='vt_dot', wrt='u1', rows=ar, cols=ar)

        self.declare_partials(of='at_dot', wrt='accel', rows=ar, cols=ar)
        self.declare_partials(of='at_dot', wrt='c', rows=ar, cols=ar)

        self.declare_partials(of='deltav_dot', wrt='accel', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='pos_x', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='pos_x', wrt='theta', rows=ar, cols=ar)

        self.declare_partials(of='pos_y', wrt='r', rows=ar, cols=ar)
        self.declare_partials(of='pos_y', wrt='theta', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        r = inputs['r']
        theta = inputs['theta']
        vr = inputs['vr']
        vt = inputs['vt']
        at = inputs['accel']
        u1 = inputs['u1']
        c = inputs['c']

        outputs['r_dot'] = vr
        outputs['theta_dot'] = vt / r
        outputs['vr_dot'] = vt**2 / r - 1 / r**2 + at * np.sin(u1)
        outputs['vt_dot'] = -vr * vt / r + at * np.cos(u1)
        outputs['at_dot'] = at**2 / c
        outputs['deltav_dot'] = at

        outputs['pos_x'] = r * np.cos(theta)
        outputs['pos_y'] = r * np.sin(theta)

    def compute_partials(self, inputs, partials):
        r = inputs['r']
        theta = inputs['theta']
        vr = inputs['vr']
        vt = inputs['vt']
        at = inputs['accel']
        u1 = inputs['u1']
        c = inputs['c']

        su1 = np.sin(u1)
        cu1 = np.cos(u1)

        partials['theta_dot', 'r'] = -vt / r**2
        partials['theta_dot', 'vt'] = 1.0 / r

        partials['vr_dot', 'r'] = -vt**2 / r**2 + 2.0 / r**3
        partials['vr_dot', 'vt'] = 2 * vt / r
        partials['vr_dot', 'accel'] = su1
        partials['vr_dot', 'u1'] = at * cu1

        partials['vt_dot', 'r'] = vr * vt / r**2
        partials['vt_dot', 'vr'] = -vt / r
        partials['vt_dot', 'vt'] = -vr / r
        partials['vt_dot', 'accel'] = cu1
        partials['vt_dot', 'u1'] = -at * su1

        partials['at_dot', 'accel'] = 2 * at / c
        partials['at_dot', 'c'] = -at**2 / c**2

        partials['pos_x', 'r'] = np.cos(theta)
        partials['pos_x', 'theta'] = -r * np.sin(theta)

        partials['pos_y', 'r'] = np.sin(theta)
        partials['pos_y', 'theta'] = r * np.cos(theta)
