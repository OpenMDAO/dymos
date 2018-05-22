from __future__ import print_function, division, absolute_import
import numpy as np
from openmdao.api import MetaModelStructuredComp

_FT2M = 0.3048

_LBF2N = 4.4482216

# Note in the data that Mach varies fastest (the first 10 datapoints correspond to Alt=0)
# Altitude is given in ft and thrust is given in lbf
THR_DATA = {'mach': np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8], dtype=np.float64),
            'h': np.array([0.0, 5.0E3, 10.0E3, 15.0E3, 20.0E3,
                           25.0E3, 30.0E3, 40.0E3, 50.0E3, 70.0E3]),
            'thrust': np.array([24200.0, 28000.0, 28300.0, 30800.0, 34500.0,
                                37900.0, 36100.0, 34300.0, 32500.0, 30700.0,
                                #  alt=5000
                                24000.0, 24600.0, 25200.0, 27200.0, 30300.0,
                                34300.0, 38000.0, 36600.0, 35200.0, 33800.0,
                                # ! alt=10000
                                20300.0, 21100.0, 21900.0, 23800.0, 26600.0,
                                30400.0, 34900.0, 38500.0, 42100.0, 45700.0,
                                #  alt=15000
                                17300.0, 18100.0, 18700.0, 20500.0, 23200.0,
                                26800.0, 31300.0, 36100.0, 38700.0, 41300.0,
                                #  alt=20000
                                14500.0, 15200.0, 15900.0, 17300.0, 19800.0,
                                23300.0, 27300.0, 31600.0, 35700.0, 39800.0,
                                #  alt=25000
                                12200.0, 12800.0, 13400.0, 14700.0, 16800.0,
                                19800.0, 23600.0, 28100.0, 32000.0, 34600.0,
                                #  alt=30000
                                10200.0, 10700.0, 11200.0, 12300.0, 14100.0,
                                16800.0, 20100.0, 24200.0, 28100.0, 31100.0,
                                #  alt=40000
                                5700.0,   6500.0,  7300.0,  8100.0,  9400.0,
                                11200.0, 13400.0, 16200.0, 19300.0, 21700.0,
                                #  alt=50000
                                3400.0,   3900.0,  4400.0,  4900.0,  5600.0,
                                6800.0,   8300.0, 10000.0, 11900.0, 13300.0,
                                #  alt=70000
                                100.0,     200.0,   400.0,   800.0,  1100.0,
                                1400.0,   1700.0,  2200.0,  2900.0,  3100.0]).reshape((10, 10))}


class MaxThrustComp(MetaModelStructuredComp):
    """ Interpolates max thrust for 2 J79 jet engines. """

    def setup(self):
        nn = self.options['vec_size']
        self.add_input(name='h', val=0.0 * np.ones(nn), units='ft', training_data=THR_DATA['h'])
        self.add_input(name='mach', val=0.2 * np.ones(nn), training_data=THR_DATA['mach'])
        self.add_output(name='max_thrust', val=np.zeros(nn), units='lbf',
                        training_data=THR_DATA['thrust'])
