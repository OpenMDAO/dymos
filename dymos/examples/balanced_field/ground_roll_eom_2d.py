import numpy as np

import openmdao.api as om


class GroundRollEOM2D(om.ExplicitComponent):
    """
    Computes the acceleration of an aircraft on a runway, per Raymer Eq. 17.97 _[1]
    Computes the position and velocity equations of motion using a modification of the 2D flight
    path parameterization of states per equations 4.42 - 4.46 of _[1].  Flight path angle
    and altitude are static quantities during the ground roll and are not integrated as states.

    References
    ----------
    .. [1] Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', types=(float, int), default=9.80665, desc='gravitational acceleration in m/s**2)')

    def setup(self):
        g = 9.80665
        nn = self.options['num_nodes']

        self.add_input(name='m',
                       val=79016 * np.ones(nn),  # MTOW of 174200 lb
                       desc='aircraft mass',
                       units='kg')

        self.add_input(name='v',
                       val=np.ones(nn),
                       desc='aircraft velocity magnitude',
                       units='m/s')

        self.add_input(name='T',
                       val=120101.98,  # 27000 lbf
                       desc='thrust',
                       units='N')

        self.add_input(name='alpha',
                       val=np.ones(nn),
                       desc='angle of attack',
                       units='rad')

        self.add_input(name='L',
                       val=np.ones(nn),
                       desc='lift force',
                       units='N')

        self.add_input(name='D',
                       val=np.ones(nn),
                       desc='drag force',
                       units='N')

        self.add_input(name='mu_r',
                       val=0.05,
                       desc='runway friction coefficient',
                       units=None)

        self.add_output(name='v_dot',
                        val=np.ones(nn),
                        desc='rate of change of velocity magnitude',
                        units='m/s**2')

        self.add_output(name='r_dot',
                        val=np.ones(nn),
                        desc='rate of change of range',
                        units='m/s')

        self.add_output(name='F_r',
                        val=np.ones(nn),
                        desc='runway normal force',
                        units='N')

        # self.add_output(name='W',
        #                 val=np.ones(nn),
        #                 desc='aircraft weight',
        #                 units='N')

        ar = np.arange(nn)

        # self.declare_partials('v_dot', 'T', rows=ar, cols=np.zeros(nn))
        # self.declare_partials('v_dot', 'D', rows=ar, cols=ar)
        # self.declare_partials('v_dot', 'alpha', rows=ar, cols=ar)
        # self.declare_partials('v_dot', 'mu_r', rows=ar, cols=np.zeros(nn))
        # self.declare_partials('v_dot', 'm', rows=ar, cols=ar)

        # self.declare_partials(of='v_dot', wrt='*', method='cs')
        # self.declare_partials(of='F_r', wrt='*', method='cs')
        # self.declare_partials(of='r_dot', wrt='v', method='cs')
        # self.declare_partials(of='W', wrt='m', method='cs')

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs')

        # self.declare_partials('r_dot', 'v', rows=ar, cols=ar, val=1.0)

        # self.declare_partials('F_r', 'm', rows=ar, cols=ar, val=g)
        # self.declare_partials('F_r', 'L', rows=ar, cols=ar)
        # self.declare_partials('F_r', 'T', rows=ar, cols=np.zeros(nn))
        # self.declare_partials('F_r', 'alpha', rows=ar, cols=ar)

        # self.declare_partials('W', 'm', rows=ar, cols=ar, val=g)

    def compute(self, inputs, outputs):
        g = 9.80665
        m = inputs['m']
        v = inputs['v']
        T = inputs['T']
        L = inputs['L']
        D = inputs['D']
        mu_r = inputs['mu_r']
        alpha = inputs['alpha']

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        outputs['F_r'] = m * g - L * calpha - T * salpha
        outputs['v_dot'] = (T * calpha - D - outputs['F_r'] * mu_r) / m

        outputs['r_dot'] = v

        # outputs['W'] = m * g

    # def compute_partials(self, inputs, partials):
    #     g = 9.80665
    #     m = inputs['m']
    #     T = inputs['T']
    #     L = inputs['L']
    #     D = inputs['D']
    #     alpha = inputs['alpha']
    #     mu_r = inputs['mu_r']
    #
    #     calpha = np.cos(alpha)
    #     salpha = np.sin(alpha)
    #
    #     # F_r = m * g - L * calpha - T * salpha
    #
    #     # partials['v_dot', 'T'] = calpha / m
    #     # partials['v_dot', 'D'] = -1.0 / m
    #     # partials['v_dot', 'mu_r'] = -1.0 / m
    #     # # partials['v_dot', 'm'] = (mu_r * F_r + D - T * calpha) / (m**2)
    #     # partials['v_dot', 'alpha'] = -T * salpha / m
    #
    #     # partials['F_r', 'L'] = -calpha
    #     # partials['F_r', 'T'] = -salpha
    #     # partials['F_r', 'alpha'] = L * salpha - T * calpha

if __name__ == "__main__":

    import openmdao.api as om
    p = om.Problem()
    p.model = GroundRollEOM2D(num_nodes=20)

    p.setup(force_alloc_complex=True)
    p.run_model()

    with np.printoptions(linewidth=1024):
        p.check_partials(method='cs', compact_print=True)
