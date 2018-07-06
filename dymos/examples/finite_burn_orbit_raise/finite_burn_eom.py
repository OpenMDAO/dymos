from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent
import openmdao.utils.units as units

from dymos import declare_time, declare_state, declare_parameter

# Add canonical units to OpenMDAO
MU_earth = 3.986592936294783e14
R_earth = 6378137.0

period = 2 * np.pi * np.sqrt(R_earth**3 / MU_earth)

# Add canonical time and distance units for these EOM
units.add_unit('TU', '{0}*s'.format(period))
units.add_unit('DU', '{0}*m'.format(R_earth))


@declare_time(units='TU')
@declare_state('x1', rate_source='x1_dot', targets=['x1'], units='DU')
@declare_state('x2', rate_source='x2_dot', targets=['x2'], units='rad')
@declare_state('x3', rate_source='x3_dot', targets=['x3'], units='DU/TU')
@declare_state('x4', rate_source='x4_dot', targets=['x4'], units='DU/TU')
@declare_state('x5', rate_source='x5_dot', targets=['x5'], units='DU/TU**2')
@declare_state('deltav', rate_source='deltav_dot', units='DU/TU')
@declare_parameter('u1', targets=['u1'], units='rad')
@declare_parameter('c', targets=['c'], units='DU/TU')
class FiniteBurnODE(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('x1',
                       val=np.ones(nn),
                       desc='radius from center of attraction',
                       units='DU')

        self.add_input('x2',
                       val=np.zeros(nn),
                       desc='anomaly term',
                       units='rad')

        self.add_input('x3',
                       val=np.zeros(nn),
                       desc='local vertical velocity component',
                       units='DU/TU')

        self.add_input('x4',
                       val=np.zeros(nn),
                       desc='local horizontal velocity component',
                       units='DU/TU')

        self.add_input('x5',
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

        self.add_output('x1_dot',
                        val=np.ones(nn),
                        desc='rate of change of radius from center of attraction',
                        units='DU/TU')

        self.add_output('x2_dot',
                        val=np.zeros(nn),
                        desc='rate of change of anomaly term',
                        units='rad/TU')

        self.add_output('x3_dot',
                        val=np.zeros(nn),
                        desc='rate of change of local vertical velocity component',
                        units='DU/TU**2')

        self.add_output('x4_dot',
                        val=np.zeros(nn),
                        desc='rate of change of local horizontal velocity component',
                        units='DU/TU**2')

        self.add_output('x5_dot',
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

        # x1 dot is a linear function of x3, so provide the partial value here
        self.declare_partials(of='x1_dot', wrt='x3', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='x2_dot', wrt='x1', rows=ar, cols=ar)
        self.declare_partials(of='x2_dot', wrt='x4', rows=ar, cols=ar)

        self.declare_partials(of='x3_dot', wrt='x1', rows=ar, cols=ar)
        self.declare_partials(of='x3_dot', wrt='x4', rows=ar, cols=ar)
        self.declare_partials(of='x3_dot', wrt='x5', rows=ar, cols=ar)
        self.declare_partials(of='x3_dot', wrt='u1', rows=ar, cols=ar)

        self.declare_partials(of='x4_dot', wrt='x1', rows=ar, cols=ar)
        self.declare_partials(of='x4_dot', wrt='x3', rows=ar, cols=ar)
        self.declare_partials(of='x4_dot', wrt='x4', rows=ar, cols=ar)
        self.declare_partials(of='x4_dot', wrt='x5', rows=ar, cols=ar)
        self.declare_partials(of='x4_dot', wrt='u1', rows=ar, cols=ar)

        self.declare_partials(of='x5_dot', wrt='x5', rows=ar, cols=ar)
        self.declare_partials(of='x5_dot', wrt='c', rows=ar, cols=ar)

        self.declare_partials(of='deltav_dot', wrt='x5', rows=ar, cols=ar, val=1.0)

        self.declare_partials(of='pos_x', wrt='x1', rows=ar, cols=ar)
        self.declare_partials(of='pos_x', wrt='x2', rows=ar, cols=ar)

        self.declare_partials(of='pos_y', wrt='x1', rows=ar, cols=ar)
        self.declare_partials(of='pos_y', wrt='x2', rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']
        x5 = inputs['x5']
        u1 = inputs['u1']
        c = inputs['c']

        outputs['x1_dot'] = x3
        outputs['x2_dot'] = x4 / x1
        outputs['x3_dot'] = x4**2 / x1 - 1 / x1**2 + x5 * np.sin(u1)
        outputs['x4_dot'] = -x3 * x4 / x1 + x5 * np.cos(u1)
        outputs['x5_dot'] = x5**2 / c
        outputs['deltav_dot'] = x5

        outputs['pos_x'] = x1 * np.cos(x2)
        outputs['pos_y'] = x1 * np.sin(x2)

    def compute_partials(self, inputs, partials):
        x1 = inputs['x1']
        x2 = inputs['x2']
        x3 = inputs['x3']
        x4 = inputs['x4']
        x5 = inputs['x5']
        u1 = inputs['u1']
        c = inputs['c']

        su1 = np.sin(u1)
        cu1 = np.cos(u1)

        partials['x2_dot', 'x1'] = -x4 / x1**2
        partials['x2_dot', 'x4'] = 1.0 / x1

        partials['x3_dot', 'x1'] = -x4**2 / x1**2 + 2.0 / x1**3
        partials['x3_dot', 'x4'] = 2 * x4 / x1
        partials['x3_dot', 'x5'] = su1
        partials['x3_dot', 'u1'] = x5 * cu1

        partials['x4_dot', 'x1'] = x3 * x4 / x1**2
        partials['x4_dot', 'x3'] = -x4 / x1
        partials['x4_dot', 'x4'] = -x3 / x1
        partials['x4_dot', 'x5'] = cu1
        partials['x4_dot', 'u1'] = -x5 * su1

        partials['x5_dot', 'x5'] = 2 * x5 / c
        partials['x5_dot', 'c'] = -x5**2 / c**2

        partials['pos_x', 'x1'] = np.cos(x2)
        partials['pos_x', 'x2'] = -x1 * np.sin(x2)

        partials['pos_y', 'x1'] = np.sin(x2)
        partials['pos_y', 'x2'] = x1 * np.cos(x2)


if __name__ == '__main__':
    pass