from __future__ import print_function, division, absolute_import

from pyxdsm.XDSM import XDSM

from openmdao.api import Problem
from dymos.examples.ssto.launch_vehicle_ode import LaunchVehicleODE

def main():
    p = Problem()
    p.model = LaunchVehicleODE(num_nodes=1)
    p.setup()

    p.run_model()

    xdsm = XDSM()
    xdsm.from_openmdao_group(p.model)
    xdsm.write('ssto_xdsm', build=True, cleanup=True)




if __name__ == '__main__':
    main()