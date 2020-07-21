import openmdao.api as om


class FlyingRobotODE(om.ExplicitComponent):
    """
    The flying robot ODE mimics a free flying robot with thrusters in the vertical axis and
    horizontal axis.

    The controls on the robot are external accelerations in these two orthogonal axes.
    Gravity acts in the negative axis.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('u', shape=(nn, 2), units='m/s**2')
        self.add_output('u_mag2', shape=(nn,), units='m**2/s**4', desc='square of control magnitude')

        self.declare_coloring(wrt='u', method='cs')
        self.declare_partials(of='u_mag2', wrt='u', method='cs')

    def compute(self, inputs, outputs):
        u_x = inputs['u'][:, 0]
        u_y = inputs['u'][:, 1]

        outputs['u_mag2'] = u_x**2 + u_y**2
