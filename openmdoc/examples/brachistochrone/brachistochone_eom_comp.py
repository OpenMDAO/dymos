from __future__ import print_function, division, absolute_import

import numpy as np
from openmdao.api import ExplicitComponent


class BrachistochroneEOM(ExplicitComponent):
    """ The equations of motion for the Brachistochrone optimal control problem.

    Assume that a frictionless particle on a track or guide-wire, subject to
    a rectilinear gravity field, accelerates as a function of gravity and the
    angle of the slope of the guide-wire at the current location.  The equations
    of motion are thus

    ..math::

        \frac{dv}{dt} = g \cdot \cos \theta
        \frac{dx}{dt} = v \cdot \sin \theta
        \frac{dy}{dt} = -v \cdot \cos \theta

    The solution to the optimal control is that

    ..math::

        \frac{v}{\sin \theta} = K

    where K is some constant dependent upon the constraints of the problem.

    """
    def initialize(self):
        self.metadata.declare('num_nodes', types=int)

    def setup(self):
        nn = self.metadata['num_nodes']

        # Inputs
        self.add_input('v',
                       val=np.zeros(nn),
                       desc='velocity',
                       units='m/s')

        self.add_input('g',
                       val=9.80665*np.ones(nn),
                       desc='gravitational acceleration',
                       units='m/s/s')

        self.add_input('theta',
                       val=np.zeros(nn),
                       desc='angle of wire',
                       units='rad')

        self.add_output('xdot',
                        val=np.zeros(nn),
                        desc='velocity component in x',
                        units='m/s')

        self.add_output('ydot',
                        val=np.zeros(nn),
                        desc='velocity component in y',
                        units='m/s')

        self.add_output('vdot',
                        val=np.zeros(nn),
                        desc='acceleration magnitude',
                        units='m/s**2')

        self.add_output('check',
                        val=np.zeros(nn),
                        desc='A check on the solution: v/sin(theta) = constant',
                        units='m/s')

        # Setup partials
        arange = np.arange(self.metadata['num_nodes'])

        self.declare_partials(of='vdot', wrt='g', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='vdot', wrt='theta', rows=arange, cols=arange, val=1.0)

        self.declare_partials(of='xdot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='xdot', wrt='theta', rows=arange, cols=arange, val=1.0)

        self.declare_partials(of='ydot', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='ydot', wrt='theta', rows=arange, cols=arange, val=1.0)

        self.declare_partials(of='check', wrt='v', rows=arange, cols=arange, val=1.0)
        self.declare_partials(of='check', wrt='theta', rows=arange, cols=arange, val=1.0)

    def compute(self, inputs, outputs):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        outputs['vdot'] = g*cos_theta
        outputs['xdot'] = v*sin_theta
        outputs['ydot'] = -v*cos_theta
        outputs['check'] = v/sin_theta

    def compute_partials(self, inputs, jacobian):
        theta = inputs['theta']
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        g = inputs['g']
        v = inputs['v']

        jacobian['vdot', 'g'] = cos_theta
        jacobian['vdot', 'theta'] = -g*sin_theta

        jacobian['xdot', 'v'] = sin_theta
        jacobian['xdot', 'theta'] = v*cos_theta

        jacobian['ydot', 'v'] = -cos_theta
        jacobian['ydot', 'theta'] = v*sin_theta

        jacobian['check', 'v'] = 1/sin_theta
        jacobian['check', 'theta'] = -v*cos_theta/sin_theta**2
