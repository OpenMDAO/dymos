import numpy as np
import openmdao.api as om


class RobertsonODE(om.ExplicitComponent):
    """example for a stiff ODE from Robertson.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # input: state
        self.add_input('x', shape=nn, desc="state x", units=None)
        self.add_input('y', shape=nn, desc="state y", units=None)
        self.add_input('z', shape=nn, desc="state z", units=None)

        # output: derivative of state
        self.add_output('xdot', shape=nn, desc='derivative of x', units="1/s")
        self.add_output('ydot', shape=nn, desc='derivative of y', units="1/s")
        self.add_output('zdot', shape=nn, desc='derivative of z', units="1/s")

        r = np.arange(0, nn)
        self.declare_partials(of='*', wrt='*', method='exact', rows=r, cols=r)

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        xdot = -0.04 * x + 1e4 * y * z
        zdot = 3e7 * y ** 2
        ydot = - (xdot + zdot)

        outputs['xdot'] = xdot
        outputs['ydot'] = ydot
        outputs['zdot'] = zdot

    def compute_partials(self, inputs, jacobian):
        x = inputs['x']
        y = inputs['y']
        z = inputs['z']

        xdot_y = 1e4 * z
        xdot_z = 1e4 * y

        zdot_y = 6e7 * y

        jacobian['xdot', 'x'] = -0.04
        jacobian['xdot', 'y'] = xdot_y
        jacobian['xdot', 'z'] = xdot_z

        jacobian['ydot', 'x'] = 0.04
        jacobian['ydot', 'y'] = - (xdot_y + zdot_y)
        jacobian['ydot', 'z'] = - xdot_z

        jacobian['zdot', 'x'] = 0.0
        jacobian['zdot', 'y'] = zdot_y
