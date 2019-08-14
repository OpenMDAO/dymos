import numpy as np 
import matplotlib.pyplot as plt 
from openmdao.api import ExplicitComponent, Problem

class AerodynamicHeating(ExplicitComponent):

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):

        nn = self.options["num_nodes"]

        self.add_input("rho", val=np.ones(nn), desc="local density", units="slug/ft**3")
        self.add_input("v", val=np.ones(nn), desc="velocity of shuttle", units="ft/s")
        self.add_input("alpha", val=np.ones(nn), desc="angle of attack of shuttle", units="deg")

        self.add_output("q", val=np.ones(nn), desc="aerodynamic heating on leading edge of shuttle", units="Btu/ft**2/s")

        partial_range = np.arange(nn) 

        self.declare_partials("q", "rho", rows=partial_range, cols=partial_range)
        self.declare_partials("q", "v", rows=partial_range, cols=partial_range)
        self.declare_partials("q", "alpha", rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):

        rho = inputs["rho"]
        v = inputs["v"]
        alpha = inputs["alpha"]
        c_0 = 1.0672181
        c_1 = -.19213774e-1
        c_2 = .21286289e-3
        c_3 = -.10117249e-5
        q_r = 17700*rho**.5*(.0001*v)**3.07
        q_a = c_0 + c_1*alpha + c_2*alpha**2 + c_3*alpha**3

        outputs["q"] = q_r*q_a

    def compute_partials(self, inputs, J):
        
        rho = inputs["rho"]
        v = inputs["v"]
        alpha = inputs["alpha"]
        c_0 = 1.0672181
        c_1 = -.19213774e-1
        c_2 = .21286289e-3
        c_3 = -.10117249e-5

        q_r = 17700*rho**.5*(.0001*v)**3.07
        q_a = c_0 + c_1*alpha + c_2*alpha**2 + c_3*alpha**3

        J["q", "rho"] = q_a*.5*17700*rho**(-.5)*(.0001*v)**3.07
        J["q", "v"] = q_a*3.07*17700*rho**.5*(.0001)**3.07*v**2.07
        J["q", "alpha"] = q_r*(c_1 + 2*c_2*alpha + 3*c_3*alpha**2)

def test_heating():
    prob = Problem()
    prob.model = AerodynamicHeating(num_nodes=5)

    prob.setup(check=False, force_alloc_complex=True)

    prob.run_model()

    return(prob)

if __name__ == "__main__":

    prob = test_heating()
    prob.check_partials(compact_print=True, method="cs")