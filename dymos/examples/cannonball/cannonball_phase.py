import dymos as dm
from .cannonball_ode import CannonballODE


class CannonballPhase(dm.Phase):
    """
    CannonballPhase serves as a demonstration of how to subclass Phase in order to associate
    certain metadata of a set of states with a given ODE.
    """

    def initialize(self):

        # First perform the standard phase initialization.
        # After this step no more options may be declared, but the options
        # will be available for assignment.
        super(CannonballPhase, self).initialize()

        # Here we set the ODE class to be used.
        # Note if this phase is instantiated with an ode_class argument it will be overridden here!
        self.options['ode_class'] = CannonballODE

        # Here we only set default units, rate_sources, and targets.
        # Other options are generally more problem-specific.
        self.add_state('r', units='m', rate_source='eom.r_dot')
        self.add_state('h', units='m', rate_source='eom.h_dot', targets=['atmos.h'])
        self.add_state('gam', units='rad', rate_source='eom.gam_dot', targets=['eom.gam'])
        self.add_state('v', units='m/s', rate_source='eom.v_dot',
                       targets=['dynamic_pressure.v', 'eom.v', 'kinetic_energy.v'])
