import openmdao.api as om
import numpy as np


class TireConstraintODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # constants
        self.add_input("mu0_y", val=1.68, desc="lateral friction coefficient", units=None)
        self.add_input("mu0_x", val=1.68, desc="longitudinal friction coefficient", units=None)
        self.add_input("k_mu", val=-0.0, desc="tire load sensitivity", units=None)
        self.add_input("a", val=1.8, desc="cg to front distance", units="m")
        self.add_input("b", val=1.6, desc="cg to rear distance", units="m")
        self.add_input("M", val=0.0, desc="mass", units="kg")
        self.add_input("g", val=9.8, desc="mass", units="m/s**2")

        # states
        self.add_input("S_fl", val=np.zeros(nn), desc="longitudinal force fl", units="N")
        self.add_input("S_fr", val=np.zeros(nn), desc="longitudinal force fr", units="N")
        self.add_input("S_rl", val=np.zeros(nn), desc="longitudinal force rl", units="N")
        self.add_input("S_rr", val=np.zeros(nn), desc="longitudinal force rr", units="N")

        self.add_input("F_fl", val=np.zeros(nn), desc="lateral force fl", units="N")
        self.add_input("F_fr", val=np.zeros(nn), desc="lateral force fr", units="N")
        self.add_input("F_rl", val=np.zeros(nn), desc="lateral force rl", units="N")
        self.add_input("F_rr", val=np.zeros(nn), desc="lateral force rr", units="N")

        self.add_input("N_rr", val=np.zeros(nn), desc="normal load rr", units="N")
        self.add_input("N_fr", val=np.zeros(nn), desc="normal load fr", units="N")
        self.add_input("N_rl", val=np.zeros(nn), desc="normal load rl", units="N")
        self.add_input("N_fl", val=np.zeros(nn), desc="normal load fl", units="N")

        # outputs
        self.add_output("c_rr", val=np.zeros(nn), desc="tire load constraint rr", units=None)
        self.add_output("c_rl", val=np.zeros(nn), desc="tire load constraint rr", units=None)
        self.add_output("c_fr", val=np.zeros(nn), desc="tire load constraint rr", units=None)
        self.add_output("c_fl", val=np.zeros(nn), desc="tire load constraint rr", units=None)

        # # Setup partials
        arange = np.arange(self.options["num_nodes"], dtype=int)

        # #partials
        self.declare_partials(of="c_rr", wrt="S_rr", rows=arange, cols=arange)
        self.declare_partials(of="c_rr", wrt="F_rr", rows=arange, cols=arange)
        self.declare_partials(of="c_rr", wrt="N_rr", rows=arange, cols=arange)

        self.declare_partials(of="c_rl", wrt="S_rl", rows=arange, cols=arange)
        self.declare_partials(of="c_rl", wrt="F_rl", rows=arange, cols=arange)
        self.declare_partials(of="c_rl", wrt="N_rl", rows=arange, cols=arange)

        self.declare_partials(of="c_fr", wrt="S_fr", rows=arange, cols=arange)
        self.declare_partials(of="c_fr", wrt="F_fr", rows=arange, cols=arange)
        self.declare_partials(of="c_fr", wrt="N_fr", rows=arange, cols=arange)

        self.declare_partials(of="c_fl", wrt="S_fl", rows=arange, cols=arange)
        self.declare_partials(of="c_fl", wrt="F_fl", rows=arange, cols=arange)
        self.declare_partials(of="c_fl", wrt="N_fl", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        mu0_y = inputs["mu0_y"]
        mu0_x = inputs["mu0_x"]
        k_mu = inputs["k_mu"]
        a = inputs["a"]
        b = inputs["b"]
        M = inputs["M"]
        g = inputs["g"]

        S_fl = inputs["S_fl"]
        S_fr = inputs["S_fr"]
        S_rl = inputs["S_rl"]
        S_rr = inputs["S_rr"]

        N_fl = inputs["N_fl"]
        N_fr = inputs["N_fr"]
        N_rl = inputs["N_rl"]
        N_rr = inputs["N_rr"]

        F_fl = inputs["F_fl"]
        F_fr = inputs["F_fr"]
        F_rl = inputs["F_rl"]
        F_rr = inputs["F_rr"]

        N0_rr = (M * g / 2) * (a / (a + b))
        N0_rl = (M * g / 2) * (a / (a + b))
        N0_fr = (M * g / 2) * (b / (a + b))
        N0_fl = (M * g / 2) * (b / (a + b))

        mu_x_rr = mu0_x + k_mu * N_rr / N0_rr
        mu_x_rl = mu0_x + k_mu * N_rl / N0_rl
        mu_x_fr = mu0_x + k_mu * N_fr / N0_fr
        mu_x_fl = mu0_x + k_mu * N_fl / N0_fl

        mu_y_rr = mu0_y + k_mu * N_rr / N0_rr
        mu_y_rl = mu0_y + k_mu * N_rl / N0_rl
        mu_y_fr = mu0_y + k_mu * N_fr / N0_fr
        mu_y_fl = mu0_y + k_mu * N_fl / N0_fl

        outputs["c_rr"] = (S_rr / (N_rr * mu_x_rr)) ** 2 + (F_rr / (N_rr * mu_y_rr)) ** 2
        outputs["c_rl"] = (S_rl / (N_rl * mu_x_rl)) ** 2 + (F_rl / (N_rl * mu_y_rl)) ** 2
        outputs["c_fr"] = (S_fr / (N_fr * mu_x_fr)) ** 2 + (F_fr / (N_fr * mu_y_fr)) ** 2
        outputs["c_fl"] = (S_fl / (N_fl * mu_x_fl)) ** 2 + (F_fl / (N_fl * mu_y_fl)) ** 2

    def compute_partials(self, inputs, jacobian):
        mu0_y = inputs["mu0_y"]
        mu0_x = inputs["mu0_x"]
        k_mu = inputs["k_mu"]
        a = inputs["a"]
        b = inputs["b"]
        M = inputs["M"]
        g = inputs["g"]

        S_fl = inputs["S_fl"]
        S_fr = inputs["S_fr"]
        S_rl = inputs["S_rl"]
        S_rr = inputs["S_rr"]

        N_fl = inputs["N_fl"]
        N_fr = inputs["N_fr"]
        N_rl = inputs["N_rl"]
        N_rr = inputs["N_rr"]

        F_fl = inputs["F_fl"]
        F_fr = inputs["F_fr"]
        F_rl = inputs["F_rl"]
        F_rr = inputs["F_rr"]

        N0_rr = (M * g / 2) * (a / (a + b))
        N0_rl = (M * g / 2) * (a / (a + b))
        N0_fr = (M * g / 2) * (b / (a + b))
        N0_fl = (M * g / 2) * (b / (a + b))

        jacobian["c_rr", "S_rr"] = (2 * S_rr) / (N_rr ** 2 * (mu0_x + (N_rr * k_mu) / N0_rr) ** 2)
        jacobian["c_fr", "S_fr"] = (2 * S_fr) / (N_fr ** 2 * (mu0_x + (N_fr * k_mu) / N0_fr) ** 2)
        jacobian["c_fl", "S_fl"] = (2 * S_fl) / (N_fl ** 2 * (mu0_x + (N_fl * k_mu) / N0_fl) ** 2)
        jacobian["c_rl", "S_rl"] = (2 * S_rl) / (N_rl ** 2 * (mu0_x + (N_rl * k_mu) / N0_rl) ** 2)

        jacobian["c_rr", "F_rr"] = (2 * F_rr) / (N_rr ** 2 * (mu0_y + (N_rr * k_mu) / N0_rr) ** 2)
        jacobian["c_fr", "F_fr"] = (2 * F_fr) / (N_fr ** 2 * (mu0_y + (N_fr * k_mu) / N0_fr) ** 2)
        jacobian["c_fl", "F_fl"] = (2 * F_fl) / (N_fl ** 2 * (mu0_y + (N_fl * k_mu) / N0_fl) ** 2)
        jacobian["c_rl", "F_rl"] = (2 * F_rl) / (N_rl ** 2 * (mu0_y + (N_rl * k_mu) / N0_rl) ** 2)

        jacobian["c_rr", "N_rr"] = (
            -(2 * F_rr ** 2) / (N_rr ** 3 * (mu0_y + (k_mu * N_rr) / N0_rr) ** 2) -
            (2 * S_rr ** 2) / (N_rr ** 3 * (mu0_x + (k_mu * N_rr) / N0_rr) ** 2) -
            (2 * F_rr ** 2 * k_mu) /
            (N_rr ** 2 * N0_rr * (mu0_y + (k_mu * N_rr) / N0_rr) ** 3) -
            (2 * k_mu * S_rr ** 2) /
            (N_rr ** 2 * N0_rr * (mu0_x + (k_mu * N_rr) / N0_rr) ** 3)
        )
        jacobian["c_rl", "N_rl"] = (
            -(2 * F_rl ** 2) / (N_rl ** 3 * (mu0_y + (k_mu * N_rl) / N0_rl) ** 2) -
            (2 * S_rl ** 2) / (N_rl ** 3 * (mu0_x + (k_mu * N_rl) / N0_rl) ** 2) -
            (2 * F_rl ** 2 * k_mu) /
            (N_rl ** 2 * N0_rl * (mu0_y + (k_mu * N_rl) / N0_rl) ** 3) -
            (2 * k_mu * S_rl ** 2) /
            (N_rl ** 2 * N0_rl * (mu0_x + (k_mu * N_rl) / N0_rl) ** 3)
        )
        jacobian["c_fr", "N_fr"] = (
            -(2 * F_fr ** 2) / (N_fr ** 3 * (mu0_y + (k_mu * N_fr) / N0_fr) ** 2) -
            (2 * S_fr ** 2) / (N_fr ** 3 * (mu0_x + (k_mu * N_fr) / N0_fr) ** 2) -
            (2 * F_fr ** 2 * k_mu) /
            (N_fr ** 2 * N0_fr * (mu0_y + (k_mu * N_fr) / N0_fr) ** 3) -
            (2 * k_mu * S_fr ** 2) /
            (N_fr ** 2 * N0_fr * (mu0_x + (k_mu * N_fr) / N0_fr) ** 3)
        )
        jacobian["c_fl", "N_fl"] = (
            -(2 * F_fl ** 2) / (N_fl ** 3 * (mu0_y + (k_mu * N_fl) / N0_fl) ** 2) -
            (2 * S_fl ** 2) / (N_fl ** 3 * (mu0_x + (k_mu * N_fl) / N0_fl) ** 2) -
            (2 * F_fl ** 2 * k_mu) /
            (N_fl ** 2 * N0_fl * (mu0_y + (k_mu * N_fl) / N0_fl) ** 3) -
            (2 * k_mu * S_fl ** 2) /
            (N_fl ** 2 * N0_fl * (mu0_x + (k_mu * N_fl) / N0_fl) ** 3)
        )
