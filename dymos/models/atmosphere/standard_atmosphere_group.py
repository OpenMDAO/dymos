from openmdao.api import Group

from .pressure_comp import PressureComp
from .temperature_comp import TemperatureComp
from .sos_comp import SpeedOfSoundComp
from .density_comp import DensityComp


class StandardAtmosphereGroup(Group):
    """
    Model of the 1976 Standard Atmosphere.

    Inputs
    ------
    h : m
        Altitude above sea-level

    Outputs
    -------
    pres : Pa
        Atmospheric Pressure
    temp : K
        Atmospheric Temperature
    rho : kg/m**3
        Atmospheric Density

    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        n = self.options['num_nodes']

        self.add_subsystem('pres_comp',
                           PressureComp(num_nodes=n),
                           promotes_inputs=['h'],
                           promotes_outputs=['pres'])

        self.add_subsystem('temp_comp',
                           TemperatureComp(num_nodes=n),
                           promotes_inputs=['h'],
                           promotes_outputs=['temp'])

        self.add_subsystem('sos_comp',
                           SpeedOfSoundComp(num_nodes=n),
                           promotes_inputs=['temp'],
                           promotes_outputs=['sos'])

        self.add_subsystem('rho_comp',
                           DensityComp(num_nodes=n),
                           promotes_inputs=['pres', 'temp'],
                           promotes_outputs=['rho'])
