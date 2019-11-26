import numpy as np
import openmdao.api as om


class AerodynamicHeating(om.ExplicitComponent):
    """
    Defines the Aerodynamic heating equations for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('rho', val=np.ones(nn), desc='local density', units='slug/ft**3')
        self.add_input('v', val=np.ones(nn), desc='velocity of shuttle', units='ft/s')
        self.add_input('alpha', val=np.ones(nn), desc='angle of attack of shuttle',
                       units='deg')

        self.add_output('q', val=np.ones(nn),
                        desc='aerodynamic heating on leading edge of shuttle',
                        units='Btu/ft**2/s')

        partial_range = np.arange(nn)

        self.declare_partials('q', 'rho', rows=partial_range, cols=partial_range)
        self.declare_partials('q', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('q', 'alpha', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        rho = inputs['rho']
        v = inputs['v']
        alpha = inputs['alpha']
        c_0 = 1.0672181
        c_1 = -0.19213774e-1
        c_2 = 0.21286289e-3
        c_3 = -0.10117249e-5

        if np.any(v < 0):
            raise om.AnalysisError('Negative velocity magnitude encountered')

        q_r = 17700.0 * np.sqrt(rho) * (0.0001 * v) ** 3.07
        q_a = c_0 + c_1 * alpha + c_2 * alpha ** 2 + c_3 * alpha ** 3

        outputs['q'] = q_r * q_a

    def compute_partials(self, inputs, partials):
        rho = inputs['rho']
        v = inputs['v']
        alpha = inputs['alpha']
        c_0 = 1.0672181
        c_1 = -.19213774e-1
        c_2 = .21286289e-3
        c_3 = -.10117249e-5

        if np.any(v < 0):
            raise om.AnalysisError('Negative velocity magnitude encountered')

        sqrt_rho = np.sqrt(rho)

        q_r = 17700 * sqrt_rho * (.0001 * v) ** 3.07
        q_a = c_0 + c_1 * alpha + c_2 * alpha ** 2 + c_3 * alpha ** 3

        dqr_drho = 0.5 * q_r / rho
        dqr_dv = 17700 * sqrt_rho * 0.0001 * 3.07 * (0.0001 * v) ** 2.07

        dqa_dalpha = c_1 + 2 * c_2 * alpha + 3 * c_3 * alpha ** 2

        partials['q', 'rho'] = dqr_drho * q_a
        partials['q', 'v'] = dqr_dv * q_a
        partials['q', 'alpha'] = dqa_dalpha * q_r
