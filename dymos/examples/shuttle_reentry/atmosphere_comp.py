import numpy as np 
import matplotlib.pyplot as plt 
from openmdao.api import ExplicitComponent, Problem

class Atmosphere(ExplicitComponent):

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):

        nn = self.options["num_nodes"]

        self.add_input("h", val=np.ones(nn), desc="altitude", units="ft")

        self.add_output("rho", val=np.ones(nn), desc="local density", units="slug/ft**3")

        partial_range = np.arange(nn, dtype=int)

        self.declare_partials("rho", "h", rows=partial_range, cols=partial_range)

    def compute(self, inputs, outputs):

        h = inputs["h"]
        h_r = 23800
        rho_0 = .002378

        outputs["rho"] = rho_0*np.exp(-h/h_r)

    def compute_partials(self, inputs, J):

        h = inputs["h"]
        h_r = 23800
        rho_0 = .002378

        J["rho", "h"] = -1/(h_r)*rho_0*np.exp(-h/h_r)


def test_atmosphere():
    prob = Problem()
    prob.model = Atmosphere(num_nodes=5)

    prob.setup(check=False, force_alloc_complex=True)

    prob.run_model()

    return(prob)

if __name__ == "__main__":

    prob = test_atmosphere()
    prob.check_partials(compact_print=True, method="cs")

    