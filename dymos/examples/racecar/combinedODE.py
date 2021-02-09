import openmdao.api as om

from .carODE import CarODE
from .tireODE import TireODE
from .normalForceODE import NormalForceODE
from .accelerationODE import AccelerationODE
from .tireConstraintODE import TireConstraintODE
from .timeODE import TimeODE
from .timeAdder import TimeAdder
from .curvature import Curvature


class CombinedODE(om.Group):
    def initialize(self):
        self.options.declare("num_nodes", types=int,
                             desc="Number of nodes to be evaluated in the RHS")

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_subsystem(name="normal", subsys=NormalForceODE(num_nodes=nn),
                           promotes_inputs=["ax", "ay", "V"],
                           promotes_outputs=["N_fr", "N_fl", "N_rr", "N_rl"], )

        self.add_subsystem(name="tire", subsys=TireODE(num_nodes=nn),
                           promotes_inputs=["lambda", "omega", "V", "N_fr", "N_fl", "N_rr", "N_rl",
                                            "thrust", "delta", ],
                           promotes_outputs=["S_fr", "S_fl", "S_rr", "S_rl", "F_fr", "F_fl", "F_rr",
                                             "F_rl", ], )

        self.add_subsystem(name="curv", subsys=Curvature(num_nodes=nn))

        self.connect("curv.kappa", "car.kappa")

        self.add_subsystem(name="car", subsys=CarODE(num_nodes=nn),
                           promotes_inputs=["n", "alpha", "V", "lambda", "omega", "S_fr", "S_fl",
                                            "S_rr", "S_rl", "F_fr", "F_fl", "F_rr", "F_rl",
                                            "delta", ],
                           promotes_outputs=["sdot", "ndot", "alphadot", "omegadot", "Vdot",
                                             "lambdadot", "power", ], )

        self.add_subsystem(name="accel", subsys=AccelerationODE(num_nodes=nn),
                           promotes_inputs=["V", "lambda", "omega", "Vdot", "lambdadot", "ax",
                                            "ay"], promotes_outputs=["axdot", "aydot"], )

        self.add_subsystem(name="tireconstraint", subsys=TireConstraintODE(num_nodes=nn),
                           promotes_inputs=["S_fr", "S_fl", "S_rr", "S_rl", "F_fr", "F_fl", "F_rr",
                                            "F_rl", "N_fr", "N_fl", "N_rr", "N_rl", ],
                           promotes_outputs=["c_rr", "c_rl", "c_fr", "c_fl"], )

        self.add_subsystem(name="time", subsys=TimeODE(num_nodes=nn),
                           promotes_inputs=["ndot", "sdot", "omegadot", "lambdadot", "Vdot",
                                            "axdot", "aydot", "alphadot", ],
                           promotes_outputs=["dn_ds", "dV_ds", "domega_ds", "dlambda_ds",
                                             "dalpha_ds", "dax_ds", "day_ds", ], )

        self.add_subsystem(name="timeAdder", subsys=TimeAdder(num_nodes=nn),
                           promotes_inputs=["sdot"], promotes_outputs=["dt_ds"], )
