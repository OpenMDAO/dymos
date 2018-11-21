import numpy as np

from openmdao.api import ExplicitComponent
from dymos import declare_time, declare_state, declare_parameter


@declare_time(units='s')
@declare_state('q', targets=['q'], rate_source='dXdt:q', units=None, shape=(4,))
@declare_state('w', targets=['w'], rate_source='dXdt:w', units='rad/s', shape=(3,))
@declare_parameter('I', targets=['I'], dynamic=False, units='kg*m**2', shape=(3,))
@declare_parameter('u', targets=['u'], dynamic=True, units='N*m', shape=(3,))
class SpacecraftReorientationODE(ExplicitComponent):
    """
    Provides the equations of motion for the reorientation of a spacecraft where the state
    variables are the attitude quaternion (q) and the angular velocity vector (w).

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 299, 2010.
    .. [2] Fleming, Sekhavat, and Ross, Minimum-Time Reorientation of an Assymmetric Rigid Body,
           AIAA 2008-7012.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(name='q',
                       val=np.zeros((nn, 4)),
                       desc='attitude quaternion',
                       units=None)

        self.add_input(name='w',
                       val=np.ones((nn, 3)),
                       desc='angular velocity vector',
                       units='rad/s')

        self.add_input(name='u',
                       val=np.ones((nn, 3)),
                       desc='control vector',
                       units='N*m')

        self.add_input(name='I',
                       val=np.zeros(3),
                       desc='principle moments of inertia',
                       units='kg*m**2')

        self.add_output(name='dXdt:q',
                        val=np.zeros((nn, 4)),
                        desc='rate of change of q',
                        units='1/s')

        self.add_output(name='dXdt:w',
                        val=np.zeros((nn, 3)),
                        desc='rate of change of angular velocity',
                        units='rad/s**2')

        tmp = np.ones((4, 4), dtype=int)
        np.fill_diagonal(tmp, 0)
        template = np.kron(np.eye(nn, dtype=int), tmp)
        rs, cs = np.nonzero(template)
        self.declare_partials('dXdt:q', 'q', rows=rs, cols=cs)

        tmp = np.ones((4, 3), dtype=int)
        template = np.kron(np.eye(nn, dtype=int), tmp)
        rs, cs = np.nonzero(template)
        self.declare_partials('dXdt:q', 'w', rows=rs, cols=cs, val=5.0)


        ar3 = np.arange(3*nn)
        self.declare_partials('dXdt:w', 'u', rows=ar3, cols=ar3)

        tmp = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=int)
        template = np.kron(np.eye(nn, dtype=int), tmp)
        rs, cs = np.nonzero(template)
        self.declare_partials('dXdt:w', 'w', rows=rs, cols=cs)

        self.declare_partials('dXdt:w', 'I')

    def compute(self, inputs, outputs):
        q0 = inputs['q'][:, 0]
        q1 = inputs['q'][:, 1]
        q2 = inputs['q'][:, 2]
        q3 = inputs['q'][:, 3]

        w0 = inputs['w'][:, 0]
        w1 = inputs['w'][:, 1]
        w2 = inputs['w'][:, 2]

        u0 = inputs['u'][:, 0]
        u1 = inputs['u'][:, 1]
        u2 = inputs['u'][:, 2]

        Ix = inputs['I'][0]
        Iy = inputs['I'][1]
        Iz = inputs['I'][2]

        outputs['dXdt:q'][:, 0] = 0.5 * (w0 * q3 - w1 * q2 + w2 * q1)
        outputs['dXdt:q'][:, 1] = 0.5 * (w0 * q2 + w1 * q3 - w2 * q0)
        outputs['dXdt:q'][:, 2] = 0.5 * (-w0 * q1 + w1 * q0 + w2 * q3)
        outputs['dXdt:q'][:, 3] = 0.5 * (-w0 * q0 - w1 * q1 - w2 * q2)

        outputs['dXdt:w'][:, 0] = u0 / Ix - (Iz - Iy)/Ix * w1 * w2
        outputs['dXdt:w'][:, 1] = u1 / Iy - (Ix - Iz)/Iy * w0 * w2
        outputs['dXdt:w'][:, 2] = u2 / Iz - (Iy - Ix)/Iz * w0 * w1

    def compute_partials(self, inputs, partials):
        q0 = inputs['q'][:, 0]
        q1 = inputs['q'][:, 1]
        q2 = inputs['q'][:, 2]
        q3 = inputs['q'][:, 3]

        w0 = inputs['w'][:, 0]
        w1 = inputs['w'][:, 1]
        w2 = inputs['w'][:, 2]

        u0 = inputs['u'][:, 0]
        u1 = inputs['u'][:, 1]
        u2 = inputs['u'][:, 2]

        Ix = inputs['I'][0]
        Iy = inputs['I'][1]
        Iz = inputs['I'][2]

        partials['dXdt:q', 'q'][0::12] = 0.5 * w2
        partials['dXdt:q', 'q'][1::12] = -0.5 * w1
        partials['dXdt:q', 'q'][2::12] = 0.5 * w0

        partials['dXdt:q', 'q'][3::12] = -0.5 * w2
        partials['dXdt:q', 'q'][4::12] = 0.5 * w0
        partials['dXdt:q', 'q'][5::12] = 0.5 * w1

        partials['dXdt:q', 'q'][6::12] = 0.5 * w1
        partials['dXdt:q', 'q'][7::12] = -0.5 * w0
        partials['dXdt:q', 'q'][8::12] = 0.5 * w2

        partials['dXdt:q', 'q'][9::12] = -0.5 * w0
        partials['dXdt:q', 'q'][10::12] = -0.5 * w1
        partials['dXdt:q', 'q'][11::12] = -0.5 * w2

        ###

        # dq0/dw[0, 1, 2]
        partials['dXdt:q', 'w'][0::12] = 0.5 * q3
        partials['dXdt:q', 'w'][1::12] = -0.5 * q2
        partials['dXdt:q', 'w'][2::12] = 0.5 * q1

        # dq1/dw[0, 1, 2]
        partials['dXdt:q', 'w'][3::12] = 0.5 * q2
        partials['dXdt:q', 'w'][4::12] = 0.5 * q3
        partials['dXdt:q', 'w'][5::12] = -0.5 * q0

        # dq2/dw[0, 1, 2]
        partials['dXdt:q', 'w'][6::12] = -0.5 * q1
        partials['dXdt:q', 'w'][7::12] = 0.5 * q0
        partials['dXdt:q', 'w'][8::12] = 0.5 * q3

        # dq3/dw[0, 1, 2]
        partials['dXdt:q', 'w'][9::12] = -0.5 * q0
        partials['dXdt:q', 'w'][10::12] = -0.5 * q1
        partials['dXdt:q', 'w'][11::12] = -0.5 * q2

        ###

        partials['dXdt:w', 'u'][0::3] = 1 / Ix
        partials['dXdt:w', 'u'][1::3] = 1 / Iy
        partials['dXdt:w', 'u'][2::3] = 1 / Iz

        ###

        partials['dXdt:w', 'I'][0::3, 0] = -u0 / Ix**2 + (Iz - Iy) / Ix**2 * w1 * w2
        partials['dXdt:w', 'I'][0::3, 1] = w1 * w2 / Ix
        partials['dXdt:w', 'I'][0::3, 2] = -w1 * w2 / Ix

        partials['dXdt:w', 'I'][1::3, 0] = -w0 * w2 / Iy
        partials['dXdt:w', 'I'][1::3, 1] = -u1 / Iy**2 + (Ix - Iz) / Iy**2 * w0 * w2
        partials['dXdt:w', 'I'][1::3, 2] = w0 * w2 / Iy

        partials['dXdt:w', 'I'][2::3, 0] = w0 * w1 / Iz
        partials['dXdt:w', 'I'][2::3, 1] = -w0 * w1 / Iz
        partials['dXdt:w', 'I'][2::3, 2] = -u2 / Iz**2 + (Iy - Ix) / Iz**2 * w0 * w1

        ###

        # Partials of dXdt:w0 wrt w1 and w2
        partials['dXdt:w', 'w'][0::6] = -w2 * (Iz - Iy) / Ix  # w1
        partials['dXdt:w', 'w'][1::6] = -w1 * (Iz - Iy) / Ix  # w2

        # Partials of dXdt:w1 wrt w0 and w2
        partials['dXdt:w', 'w'][2::6] = -w2 * (Ix - Iz) / Iy  # w0
        partials['dXdt:w', 'w'][3::6] = -w0 * (Ix - Iz) / Iy  # w2

        # Partials of dXdt:w2 wrt w0 and w1
        partials['dXdt:w', 'w'][4::6] = -w1 * (Iy - Ix) / Iz  # w0
        partials['dXdt:w', 'w'][5::6] = -w0 * (Iy - Ix) / Iz  # w1
