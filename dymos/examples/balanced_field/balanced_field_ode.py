import numpy as np
import jax.numpy as jnp
import openmdao.api as om
from openmdao.jax.smooth import act_tanh


class BalancedFieldODEComp(om.ExplicitComponent):
    """
    The ODE System for an aircraft takeoff climb.

    Computes the rates for states v (true airspeed) gam (flight path angle) r (range) and h (altitude).

    References
    ----------
    .. [1] Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', types=(float, int), default=9.80665, desc='gravitational acceleration (m/s**2)')
        self.options.declare('mode', values=('runway', 'climb'), desc='mode of operation (ground roll or flight)')
        self.options.declare('attitude_input', values=('pitch', 'alpha'), default='alpha', desc='attitude input type')
        self.options.declare('control', values=('attitude', 'gam_rate'), default='gam_rate',
                             desc='Whether pitch or alpha serves as control when climbing.')

    def setup(self):
        nn = self.options['num_nodes']

        if self.options['attitude_input'] == 'alpha':
            self.add_input('alpha', shape=(nn,), desc='angle of attack', units='rad')
            self.add_output('pitch', shape=(nn,), desc='vehicle +x angle above horizon', units='rad')
        else:
            self.add_input('pitch', shape=(nn,), desc='vehicle +x angle above horizon', units='rad')
            self.add_output('alpha', shape=(nn,), desc='angle of attack', units='rad')

        if self.options['control'] == 'gam_rate':
            self.add_input('gam_rate', shape=(nn,), val=0.0, desc='controlled rate of change of flight path angle', units='rad/s')

        # Scalar (constant) inputs
        self.add_input('rho', val=1.225, desc='atmospheric density at runway', units='kg/m**3')
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)
        self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
        self.add_input('CL_max', val=2.0, desc='maximum lift coefficient for linear fit', units=None)
        self.add_input('alpha_max', val=np.radians(10), desc='angle of attack at CL_max', units='rad')
        self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
        self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
        self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
        self.add_input('span', val=35.7, desc='Wingspan', units='m')
        self.add_input('T', val=1.0, desc='thrust', units='N')
        self.add_input('mu_r', val=0.05, desc='runway friction coefficient', units=None)

        # Dynamic inputs (can assume a different value at every node)
        self.add_input('m', shape=(nn,), desc='aircraft mass', units='kg')
        self.add_input('v', shape=(nn,), desc='aircraft true airspeed', units='m/s')
        self.add_input('h', shape=(nn,), desc='altitude', units='m')
        self.add_input('gam', shape=(nn,), val=0.0, desc='flight path angle', units='rad')

        # Outputs
        self.add_output('CL', shape=(nn,), desc='lift coefficient', units=None)
        self.add_output('q', shape=(nn,), desc='dynamic pressure', units='Pa')
        self.add_output('L', shape=(nn,), desc='lift force', units='N')
        self.add_output('D', shape=(nn,), desc='drag force', units='N')
        self.add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)
        self.add_output('F_r', shape=(nn,), desc='runway normal force', units='N')
        self.add_output('v_dot', shape=(nn,), desc='rate of change of speed', units='m/s**2',
                        tags=['dymos.state_rate_source:v'])
        self.add_output('r_dot', shape=(nn,), desc='rate of change of range', units='m/s',
                        tags=['dymos.state_rate_source:r'])
        self.add_output('W', shape=(nn,), desc='aircraft weight', units='N')
        self.add_output('v_stall', shape=(nn,), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', shape=(nn,), desc='stall speed ratio', units=None)
        self.add_output('climb_gradient', shape=(nn,),
                        desc='altitude rate divided by range rate',
                        units='unitless')
        self.add_output('gam_dot', shape=(nn,), desc='rate of change of flight path angle', units='rad/s')
        self.add_output('h_dot', shape=(nn,), desc='rate of change of altitude', units='m/s')
        self.add_output('thrust', shape=(nn,), units='N')

        # self.declare_coloring(wrt='*', method='cs')
        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        g = self.options['g']

        # Compute factor k to include ground effect on lift
        rho = inputs['rho']
        gam = inputs['gam']
        v = inputs['v']
        S = inputs['S']
        CD0 = inputs['CD0']
        m = inputs['m']
        T = inputs['T']
        h = inputs['h']
        h_w = inputs['h_w']
        span = inputs['span']
        AR = inputs['AR']
        CL0 = inputs['CL0']
        alpha_max = inputs['alpha_max']
        CL_max = inputs['CL_max']
        e = inputs['e']
        mu_r = inputs['mu_r']

        # Cross-compute alpha and pitch depeding on which one is input.
        if self.options['attitude_input'] == 'alpha':
            alpha = inputs['alpha']
            outputs['pitch'] = pitch = gam + alpha
        elif self.options['attitude_input'] == 'pitch':
            pitch = inputs['pitch']
            outputs['alpha'] = alpha = pitch - gam

        if self.options['control'] == 'gam_rate':
            gam_rate = inputs['gam_rate']

        outputs['W'] = W = m * g
        outputs['v_stall'] = v_stall = np.sqrt(2 * W / rho / S / CL_max)
        outputs['v_over_v_stall'] = v / v_stall

        outputs['CL'] = CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
        K_nom = 1.0 / (np.pi * AR * e)
        b = span / 2.0

        # Note the use of clip here.  If altitude drops below zero while the solver is iterating,
        # the non-clipped equation will result in NaN and ruin the analysis.
        # Since we're using a gradient-free nonlinear block GS to converge thedo we n
        fact = (np.clip(h + h_w, 0.0, 1000.0) / b) ** 1.5
        outputs['K'] = K = K_nom * 33 * fact / (1.0 + 33 * fact)

        outputs['q'] = q = 0.5 * rho * v ** 2
        outputs['L'] = L = q * S * CL
        outputs['D'] = D = q * S * (CD0 + K * CL ** 2)

        # Compute the downward force on the landing gear
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        # Runway normal force
        if self.options['mode'] == 'runway':
            outputs['F_r'] = F_r = m * g - L * calpha - T * salpha
        else:
            outputs['F_r'] = F_r = 0.0

        # # Compute the dynamics
        # if self.options['mode'] == 'climb':
        cgam = np.cos(gam)
        sgam = np.sin(gam)
        outputs['v_dot'] = (T * calpha - D - F_r * mu_r) / m - g * sgam
        outputs['h_dot'] = v * sgam
        outputs['r_dot'] = v * cgam
        outputs['climb_gradient'] = sgam
        outputs['thrust'][...] = T

        if self.options['control'] == 'gam_rate':
            outputs['gam_dot'] = gam_rate
        else:
            outputs['gam_dot'] = (T * salpha + L) / (m * v) - (g / v) * cgam


def ground_effect(h, pitch, CD_min, CL, CD, AR, wing_height, span):
    """
    Compute lift and drag coefficients in ground effect using John DeYoung ground effect equations.

    A smooth activation function is used to transition CL and CD from the ground effect values
    to the nominal values centered around hf = 1.05.

    Parameters
    ----------
    h : array-like
        altitude (ft)
    pitch : array-like
        pitch angle (flight path angle + alpha) (rad)
    CD_min : array-like
        Minimum drag coefficient
    CL : array-like
        Nominal lift coefficient
    CD : array-like
        Nominal drag coefficient
    AR : array-like
        Wing aspect ratio
    wing_height : array-like
        Height of wing above ground on runway at zero altitude (ft)

    Returns
    -------
    CL_out : array-like
        The lift coefficient accounting for ground effect, if the aircraft is within ground effect.
    CD_out : array-like
        The drag coefficient accounting for ground effect, if the aircraft is within ground effect.
    """
    hf = jnp.clip(jnp.real(h + wing_height)) / span

    A = (6.0 + AR) ** 2 / (36.0 + AR)
    B = 32.0 * (hf * A) ** 2 + 1.0
    denom = B - 0.5 + 4.0 * hf * A * jnp.sqrt(B)
    cl_fact = 1.0 + 1.0 / denom
    B = 1.0 + 32.0 * hf ** 2
    denom = 4.0 * hf * jnp.sqrt(B) + B
    PHII = 1.0 - 1.0 / denom
    CD_ge = CD_min + cl_fact * PHII * (CD - CD_min) + pitch * CL * (cl_fact - 1.0)
    CL_ge = CL * cl_fact

    CL_out = act_tanh(hf, mu=0.01, z=1.05, a=CL_ge, b=CL)
    CD_out = act_tanh(hf, mu=0.01, z=1.05, a=CD_ge, b=CD)

    return CL_out, CD_out


class BalancedFieldJaxODEComp(om.JaxExplicitComponent):
    """
    The ODE System for an aircraft takeoff climb.

    Computes the rates for states v (true airspeed) gam (flight path angle) r (range) and h (altitude).

    References
    ----------
    .. [1] Raymer, Daniel. Aircraft design: a conceptual approach. American Institute of
    Aeronautics and Astronautics, Inc., 2012.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', types=(float, int), default=9.80665, desc='gravitational acceleration (m/s**2)')
        self.options.declare('mode', values=('runway', 'climb'), desc='mode of operation (ground roll or flight)')
        self.options.declare('attitude_input', values=('pitch', 'alpha'), default='alpha', desc='attitude input type')
        self.options.declare('control', values=('attitude', 'gam_rate'), default='gam_rate',
                             desc='Whether pitch or alpha serves as control when climbing.')

    def get_self_statics(self):
        return (self.options['num_nodes'], self.options['g'], self.options['mode'],
                self.options['attitude_input'], self.options['control'])

    def setup(self):
        nn = self.options['num_nodes']

        # Scalar (constant) inputs
        self.add_input('rho', val=1.225, desc='atmospheric density at runway', units='kg/m**3')
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)
        self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
        self.add_input('CL_max', val=2.0, desc='maximum lift coefficient for linear fit', units=None)
        self.add_input('alpha_max', val=jnp.radians(10), desc='angle of attack at CL_max', units='rad')
        self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')
        self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
        self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
        self.add_input('span', val=35.7, desc='Wingspan', units='m')
        self.add_input('T', val=1.0, desc='thrust', units='N')
        self.add_input('mu_r', val=0.05, desc='runway friction coefficient', units=None)

        # Dynamic inputs (can assume a different value at every node)
        self.add_input('m', shape=(nn,), desc='aircraft mass', units='kg')
        self.add_input('v', shape=(nn,), desc='aircraft true airspeed', units='m/s')
        self.add_input('h', shape=(nn,), desc='altitude', units='m')
        self.add_input('gam', shape=(nn,), val=0.0, desc='flight path angle', units='rad')
        self.add_input('pitch', shape=(nn,), desc='vehicle +x angle above horizon', units='rad')

        self.add_input('gam_rate', shape=(nn,), val=0.0, desc='controlled rate of change of flight path angle', units='rad/s')

        # Outputs
        self.add_output('alpha', shape=(nn,), desc='angle of attack', units='rad')
        self.add_output('CL', shape=(nn,), desc='lift coefficient', units=None)
        self.add_output('q', shape=(nn,), desc='dynamic pressure', units='Pa')
        self.add_output('L', shape=(nn,), desc='lift force', units='N')
        self.add_output('D', shape=(nn,), desc='drag force', units='N')
        self.add_output('K', val=jnp.ones(nn), desc='drag-due-to-lift factor', units=None)
        self.add_output('F_r', shape=(nn,), desc='runway normal force', units='N')
        self.add_output('v_dot', shape=(nn,), desc='rate of change of speed', units='m/s**2',
                        tags=['dymos.state_rate_source:v'])
        self.add_output('r_dot', shape=(nn,), desc='rate of change of range', units='m/s',
                        tags=['dymos.state_rate_source:r'])
        self.add_output('W', shape=(nn,), desc='aircraft weight', units='N')
        self.add_output('v_stall', shape=(nn,), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', shape=(nn,), desc='stall speed ratio', units=None)
        self.add_output('climb_gradient', shape=(nn,),
                        desc='altitude rate divided by range rate',
                        units='unitless')
        self.add_output('gam_dot', shape=(nn,), desc='rate of change of flight path angle', units='rad/s')
        self.add_output('h_dot', shape=(nn,), desc='rate of change of altitude', units='m/s')

        self.declare_coloring(wrt='*', method='jax', show_summary=False)

    def compute_primal(self, rho, S, CD0, CL0, CL_max, alpha_max, h_w, AR, e, span, T, mu_r, m, v, h, gam, pitch, gam_rate):
        g = self.options['g']

        alpha = alpha = pitch - gam

        W = m * g
        v_stall = jnp.sqrt(2 * W / rho / S / CL_max)
        v_over_v_stall = v / v_stall

        CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
        K_nom = 1.0 / (jnp.pi * AR * e)
        b = span / 2.0

        # Note the use of clip here.  If altitude drops below zero while the solver is iterating,
        # the non-clipped equation will result in NaN and ruin the analysis.
        # Since we're using a gradient-free nonlinear block GS to converge thedo we n
        fact = (jnp.clip(jnp.real(h + h_w), 0.0, 1000.0) / b) ** 1.5
        K = K_nom * 33 * fact / (1.0 + 33 * fact)

        CD = (CD0 + K * CL ** 2)

        # CL, CD = ground_effect(h, pitch, CD0, CL, CD, AR, h_w, span)

        q = 0.5 * rho * v ** 2
        L = q * S * CL
        D = q * S * CD

        # Compute the downward force on the landing gear
        calpha = jnp.cos(alpha)
        salpha = jnp.sin(alpha)

        # Runway normal force
        if self.options['mode'] == 'runway':
            F_r = m * g - L * calpha - T * salpha
        else:
            F_r = 0.0

        #  Compute the dynamics
        cgam = jnp.cos(gam)
        sgam = jnp.sin(gam)
        v_dot = (T * calpha - D - F_r * mu_r) / m - g * sgam
        h_dot = v * sgam
        r_dot = v * cgam
        climb_gradient = sgam

        if self.options['control'] == 'gam_rate':
            gam_dot = gam_rate
        else:
            gam_dot = (T * salpha + L) / (m * v) - (g / v) * cgam

        return alpha, CL, q, L, D, K, F_r, v_dot, r_dot, W, v_stall, v_over_v_stall, climb_gradient, gam_dot, h_dot
