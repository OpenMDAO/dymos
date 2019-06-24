from __future__ import print_function, division, absolute_import

from pyxdsm.XDSM import XDSM

import openmdao.api as om
from dymos.examples.ssto.launch_vehicle_linear_tangent_ode import LaunchVehicleLinearTangentODE


def main():  # pragma: no cover
    p = om.Problem()
    p.model = LaunchVehicleLinearTangentODE(num_nodes=1)
    p.setup()

    p.run_model()

    var_map = {'rho': r'$\rho$',
               'x': r'$x$',
               'y': r'$y$',
               'vx': r'$v_x$',
               'vy': r'$v_y$',
               'm': r'$m$',
               'mdot': r'$\dot{m}$',
               'Isp': r'$I_{sp}$',
               'thrust': r'$F_{T}$',
               'ydot': r'$\dot{y}$',
               'xdot': r'$\dot{x}$',
               'vxdot': r'$\dot{v}_x$',
               'vydot': r'$\dot{v}_y$',
               'theta': r'$\theta$',
               'a_ctrl': r'$a$',
               'b_ctrl': r'$b$',
               'time': r'$time$'}

    xdsm = XDSM()
    xdsm.from_openmdao_group(p.model, var_map=var_map)
    xdsm.write('ssto_linear_tangent_xdsm', build=True, cleanup=True)


if __name__ == '__main__':  # pragma: no cover
    main()
