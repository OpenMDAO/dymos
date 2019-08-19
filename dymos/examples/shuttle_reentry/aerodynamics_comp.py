import numpy as np
import openmdao.api as om


class Aerodynamics(om.ExplicitComponent):
    """
    Defines the aerodynamics for the shuttle reentry problem.

    References
    ----------
    .. [1] Betts, John T., Practical Methods for Optimal Control and Estimation Using Nonlinear
           Programming, p. 248, 2010.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('alpha', val=np.ones(nn), desc='angle of attack', units='deg')
        self.add_input('v', val=np.ones(nn), desc='velocity of shuttle', units='ft/s')
        self.add_input('rho', val=np.ones(nn), desc='local atmospheric density',
                       units='slug/ft**3')

        self.add_output('drag', val=np.ones(nn), desc='drag on shuttle', units='lb')
        self.add_output('lift', val=np.ones(nn), desc='lift on shuttle', units='lb')

        partial_range = np.arange(nn, dtype=int)

        self.declare_partials('drag', 'alpha', rows=partial_range, cols=partial_range)
        self.declare_partials('drag', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('drag', 'rho', rows=partial_range, cols=partial_range)
        self.declare_partials('lift', 'alpha', rows=partial_range, cols=partial_range)
        self.declare_partials('lift', 'v', rows=partial_range, cols=partial_range)
        self.declare_partials('lift', 'rho', rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):
        a_0 = -.20704
        a_1 = .029244
        b_0 = .07854
        b_1 = -.61592e-2
        b_2 = .621408e-3
        S = 2690
        alpha = inputs['alpha']
        v = inputs['v']
        rho = inputs['rho']
        c_L = a_0 + a_1 * alpha
        c_D = b_0 + b_1 * alpha + b_2 * alpha ** 2

        outputs['drag'] = .5 * c_D * S * rho * v ** 2
        outputs['lift'] = .5 * c_L * S * rho * v ** 2

    def compute_partials(self, inputs, J):
        alpha = inputs['alpha']
        v = inputs['v']
        rho = inputs['rho']
        a_0 = -.20704
        a_1 = .029244
        b_0 = .07854
        b_1 = -.61592e-2
        b_2 = .621408e-3
        S = 2690
        c_L = a_0 + a_1 * alpha
        c_D = b_0 + b_1 * alpha + b_2 * alpha ** 2

        dD_dCD = .5 * S * rho * v ** 2
        dCD_dalpha = b_1 + 2 * alpha * b_2
        dL_dCL = dD_dCD
        dCL_dalpha = a_1

        J['drag', 'alpha'] = dD_dCD * dCD_dalpha
        J['drag', 'v'] = c_D * S * rho * v
        J['drag', 'rho'] = .5 * c_D * S * v ** 2
        J['lift', 'alpha'] = dL_dCL * dCL_dalpha
        J['lift', 'v'] = c_L * S * rho * v
        J['lift', 'rho'] = .5 * c_L * S * v ** 2
