import openmdao.api as om

from .thrust_comp import ThrustComp
from .max_thrust_comp import MaxThrustComp
from .throttle_comp import ThrottleComp
from .tsfc_comp import SFCComp
from .fuel_burn_rate_comp import FuelBurnRateComp


class PropulsionGroup(om.Group):
    """
    The PropulsionGroup computes propulsive forces (thrust), the specific fuel consumption and
    fuel expenditure rate, and the aircraft throttle setting.
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        assumptions = self.add_subsystem('assumptions', subsys=om.IndepVarComp())

        assumptions.add_output('tsfc_sl', val=2 * 8.951e-6 * 9.80665, units='1/s',
                               desc='thrust specific fuel consumption at sea-level')

        assumptions.add_output('max_thrust_sl', val=1.02E6, units='N',
                               desc='maximum thrust at sea-level')

        self.add_subsystem(name='thrust_comp',
                           subsys=ThrustComp(num_nodes=n),
                           promotes_inputs=['CT', 'q', 'S'],
                           promotes_outputs=['thrust'])

        self.add_subsystem(name='max_thrust_comp',
                           subsys=MaxThrustComp(num_nodes=n),
                           promotes_inputs=['pres', 'max_thrust_sl'],
                           promotes_outputs=['max_thrust'])

        self.add_subsystem(name='throttle_comp',
                           subsys=ThrottleComp(num_nodes=n),
                           promotes_inputs=['thrust', 'max_thrust'],
                           promotes_outputs=['tau'])

        self.add_subsystem(name='tsfc_comp',
                           subsys=SFCComp(num_nodes=n),
                           promotes_inputs=['alt'],
                           promotes_outputs=['tsfc'])

        self.add_subsystem(name='fuel_burn_rate_comp',
                           subsys=FuelBurnRateComp(num_nodes=n),
                           promotes_inputs=['tsfc', 'thrust'],
                           promotes_outputs=['dXdt:mass_fuel'])

        self.connect('assumptions.tsfc_sl', 'tsfc_comp.tsfc_sl')
        self.connect('assumptions.max_thrust_sl', 'max_thrust_sl')
