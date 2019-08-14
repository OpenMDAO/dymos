from openmdao.api import Problem, Group
from heating_comp import AerodynamicHeating
from flight_dynamics_comp import FlightDynamics
from aerodynamics_comp import Aerodynamics
from atmosphere_comp import Atmosphere
import numpy as np

class ShuttleODE(Group):

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self.add_subsystem("atmosphere", subsys=Atmosphere(num_nodes=nn), promotes_inputs=["h"], promotes_outputs=["rho"])
        self.add_subsystem("aerodynamics", subsys=Aerodynamics(num_nodes=nn), promotes_inputs=["alpha", "v", "rho"], promotes_outputs=["lift", "drag"])
        self.add_subsystem("heating", subsys=AerodynamicHeating(num_nodes=nn), promotes_inputs=["rho", "v", "alpha"], promotes_outputs=["q"])
        self.add_subsystem("eom", subsys=FlightDynamics(num_nodes=nn), promotes_inputs=["beta", "gamma", "h", "psi", "theta", "v", "lift", "drag"], promotes_outputs=["hdot", "gammadot", "phidot", "psidot", "thetadot", "vdot"])

def test_shuttle_ode():
    prob = Problem()
    prob.model = ShuttleODE(num_nodes=5)

    prob.setup(check=False, force_alloc_complex=True)

    prob.run_model()

    return(prob)

if __name__ == "__main__":

    prob = test_shuttle_ode()
    prob.check_partials(compact_print=True, method="cs")
    print(prob["psidot"])