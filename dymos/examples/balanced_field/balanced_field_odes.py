import numpy as np
import openmdao.api as om


def compute_stall_speed(m, g, rho, S, CL_max, v):
    """
    Compute the weight, stall speed, and ratio of the current airspeed to the stall speed.

    Notes
    -----
    This function is complex-step-differentiable.
    User is expected to provide inputs in a units-consistent manner.

    Parameters
    ----------
    m : float or np.array
        Aircraft mass
    g : float
        Gravitational acceleration
    rho : float or np.array
        Atmospheric density
    S : float or np.array
        Aerodynamic reference area
    CL_max : float or np.array
        Maximum lift coefficient of the aircraft
    v : float or np.array
        True airspeed

    Returns
    -------
    W : float or np.array
        The aircraft weight.
    v_stall : float or np.array
        The stall speed of the aircraft.
    v_over_v_stall : float or np.array
        The ratio of the current aircraft speed to its stall speed.
    """
    W = m * g
    v_stall = np.sqrt(2 * W / rho / S / CL_max)
    v_over_v_stall = v / v_stall
    return W, v_stall, v_over_v_stall


def compute_aero(CL0, CD0, S, alpha, alpha_max, CL_max, AR, e, span, h, h_w, rho, v):
    """
    Compute the aerodynamic forces (lift and drag) as well as some other auxiliary aerodynamic output variables.

    Notes
    -----
    This function is complex-step-differentiable.
    User is expected to provide inputs in a units-consistent manner.

    Parameters
    ----------
    CL0 : float or np.array
        Zero-alpha lift coefficient
    CD0 : float or np.array
        Zero-lift drag coefficient
    S : float or np.array
        Aerodynamic reference area
    alpha : float or np.array
        Angle of attack
    alpha_max : float or np.array
        Maximum angle of attack
    CL_max : float or np.array
        Maximum lift coefficient
    AR : float or np.array
        Wing aspect ratio
    e : float or np.array
        Oswald's efficiency factor
    span : float or np.array
        Wingspan
    h : float or np.array
        Altitude
    h_w : float or np.array
        Height of wing above aircraft center of mass
    rho : float or np.array
        Atmospheric density
    v : float or np.array
        True airspeed

    Returns
    -------
    CL : float or np.array
        Lift coefficient
    K : float or np.array
        Drag-due-to-lift factor
    q : float or np.array
        Dynamic pressure
    L : float or np.array
        Lift
    D : float or np.array
        Drag
    """
    CL = CL0 + (alpha / alpha_max) * (CL_max - CL0)
    K_nom = 1.0 / (np.pi * AR * e)
    b = span / 2.0
    K = K_nom * 33 * ((h + h_w) / b) ** 1.5 / (1.0 + 33 * ((h + h_w) / b) ** 1.5)

    q = 0.5 * rho * v ** 2
    L = q * S * CL
    D = q * S * (CD0 + K * CL ** 2)

    return CL, K, q, L, D


class GroundRollODEComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', types=(float, int), default=9.80665, desc='gravitational acceleration (m/s**2)')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('rho', val=1.225, desc='atmospheric density', units='kg/m**3')
        self.add_input('S', val=124.7, desc='aerodynamic reference area', units='m**2')
        self.add_input('CD0', val=0.03, desc='zero-lift drag coefficient', units=None)

        self.add_input('AR', val=9.45, desc='wing aspect ratio', units=None)
        self.add_input('e', val=0.801, desc='Oswald span efficiency factor', units=None)
        self.add_input('span', val=35.7, desc='Wingspan', units='m')
        self.add_input('h', val=0.0, desc='altitude', units='m')
        self.add_input('h_w', val=1.0, desc='height of the wing above the CG', units='m')

        self.add_input('CL0', val=0.5, desc='zero-alpha lift coefficient', units=None)
        self.add_input('CL_max', val=2.0, desc='maximum lift coefficient', units=None)
        self.add_input('alpha', val=np.ones(nn), desc='angle of attack', units='rad')
        self.add_input('alpha_max', val=np.radians(10), desc='angle of attack at CL_max', units='rad')
        self.add_input('T', val=120101.98, desc='thrust', units='N')
        self.add_input('mu_r', val=0.05, desc='runway friction coefficient', units=None)

        self.add_input('v', val=np.ones(nn), desc='true airspeed', units='m/s')
        self.add_input('m', val=np.ones(nn), desc='aircraft mass', units='kg')

        self.add_output('CL', val=np.ones(nn), desc='lift coefficient', units=None)
        self.add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)
        self.add_output('q', val=np.ones(nn), desc='dynamic pressure', units='Pa')
        self.add_output('L', val=np.ones(nn), desc='lift', units='N')
        self.add_output('D', val=np.ones(nn), desc='drag', units='N')
        self.add_output('W', val=np.ones(nn), desc='aircraft weight', units='N')
        self.add_output('v_stall', val=np.ones(nn), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', val=np.ones(nn), desc='stall speed ratio', units=None)

        self.add_output(name='v_dot', val=np.ones(nn), desc='rate of change of velocity magnitude', units='m/s**2', tags=['state_rate_source:v'])
        self.add_output(name='r_dot', val=np.ones(nn), desc='rate of change of range', units='m/s', tags=['state_rate_source:r'])
        self.add_output(name='F_r', val=np.ones(nn), desc='runway normal force', units='N')

        self.declare_coloring(wrt='*', method='cs')
        self.declare_partials(of='*', wrt='*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        g = self.options['g']
        rho = inputs['rho']
        v = inputs['v']
        S = inputs['S']
        CD0 = inputs['CD0']

        h = inputs['h']
        h_w = inputs['h_w']
        span = inputs['span']
        AR = inputs['AR']
        e = inputs['e']

        CL0 = inputs['CL0']
        alpha = inputs['alpha']
        alpha_max = inputs['alpha_max']
        CL_max = inputs['CL_max']
        m = inputs['m']

        T = inputs['T']
        mu_r = inputs['mu_r']

        outputs['W'], outputs['v_stall'], outputs['v_over_v_stall'] = \
            compute_stall_speed(m, g, rho, S, CL_max, v)

        outputs['CL'], outputs['K'], outputs['q'], outputs['L'], outputs['D'] = \
            compute_aero(CL0, CD0, S, alpha, alpha_max, CL_max, AR, e, span, h, h_w, rho, v)

        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        outputs['F_r'] = m * g - outputs['L'] * calpha - T * salpha
        outputs['v_dot'] = (T * calpha - outputs['D'] - outputs['F_r'] * mu_r) / m
        outputs['r_dot'] = v


class TakeoffClimbODEComp(om.ExplicitComponent):
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

    def setup(self):
        nn = self.options['num_nodes']

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

        # Dynamic inputs (can assume a different value at every node)
        self.add_input('h', shape=(nn,), desc='altitude', units='m')
        self.add_input('m', shape=(nn,), desc='aircraft mass', units='kg')
        self.add_input('v', shape=(nn,), desc='aircraft true airspeed', units='m/s')
        self.add_input('gam', shape=(nn,), desc='flight path angle', units='rad')
        self.add_input('alpha', shape=(nn,), desc='angle of attack', units='rad')

        # Outputs
        self.add_output('CL', shape=(nn,), desc='lift coefficient', units=None)
        self.add_output('q', shape=(nn,), desc='dynamic pressure', units='Pa')
        self.add_output('L', shape=(nn,), desc='lift force', units='N')
        self.add_output('D', shape=(nn,), desc='drag force', units='N')
        self.add_output('K', val=np.ones(nn), desc='drag-due-to-lift factor', units=None)
        self.add_output('F_r', shape=(nn,), desc='runway normal force', units='N')
        self.add_output('v_dot', shape=(nn,), desc='rate of change of speed', units='m/s**2', tags=['state_rate_source:v'])
        self.add_output('gam_dot', shape=(nn,), desc='rate of change of flight path angle', units='rad/s', tags=['state_rate_source:gam'])
        self.add_output('h_dot', shape=(nn,), desc='rate of change of altitude', units='m/s', tags=['state_rate_source:h'])
        self.add_output('r_dot', shape=(nn,), desc='rate of change of range', units='m/s', tags=['state_rate_source:r'])
        self.add_output('W', shape=(nn,), desc='aircraft weight', units='N')
        self.add_output('v_stall', shape=(nn,), desc='stall speed', units='m/s')
        self.add_output('v_over_v_stall', shape=(nn,), desc='stall speed ratio', units=None)

        self.declare_partials(of='*', wrt='*', method='cs')
        self.declare_coloring(wrt='*', method='cs')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        g = self.options['g']

        # Compute factor k to include ground effect on lift
        rho = inputs['rho']
        v = inputs['v']
        S = inputs['S']
        CD0 = inputs['CD0']
        m = inputs['m']
        T = inputs['T']
        gam = inputs['gam']
        h = inputs['h']
        h_w = inputs['h_w']
        span = inputs['span']
        AR = inputs['AR']
        CL0 = inputs['CL0']
        alpha = inputs['alpha']
        alpha_max = inputs['alpha_max']
        CL_max = inputs['CL_max']
        e = inputs['e']

        outputs['W'], outputs['v_stall'], outputs['v_over_v_stall'] = \
            compute_stall_speed(m, g, rho, S, CL_max, v)

        outputs['CL'], outputs['K'], outputs['q'], outputs['L'], outputs['D'] = \
            compute_aero(CL0, CD0, S, alpha, alpha_max, CL_max, AR, e, span, h, h_w, rho, v)

        # Compute the downward force on the landing gear
        calpha = np.cos(alpha)
        salpha = np.sin(alpha)

        # Compute the dynamics
        cgam = np.cos(gam)
        sgam = np.sin(gam)

        outputs['F_r'] = m * g - outputs['L'] * calpha - T * salpha
        outputs['v_dot'] = (T * calpha - outputs['D']) / m - g * sgam
        outputs['gam_dot'] = (T * salpha + outputs['L']) / (m * v) - (g / v) * cgam
        outputs['h_dot'] = v * sgam
        outputs['r_dot'] = v * cgam
