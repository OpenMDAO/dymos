import openmdao.api as om
import numpy as np


class CarODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # constants
        self.add_input("M", val=0.0, desc="mass", units="kg")
        self.add_input("a", val=1.8, desc="cg to front distance", units="m")
        self.add_input("b", val=1.6, desc="cg to rear distance", units="m")
        self.add_input("tw", val=0.73, desc="half track width", units="m")
        self.add_input("Iz", val=450.0, desc="yaw inertia", units="kg*m**2")
        self.add_input("rho", val=1.2, desc="air density", units="kg/m**3")
        self.add_input("kappa", val=np.zeros(nn), desc="track curvature", units="1/m")
        self.add_input("CdA", val=1.35, desc="drag coefficient", units="m**2")

        # states
        self.add_input("n", val=np.zeros(nn), desc="distance perpendicular to centerline",
                       units="m", )
        self.add_input("alpha", val=np.zeros(nn), desc="angle relative to centerline", units="rad")
        self.add_input("V", val=np.zeros(nn), desc="speed", units="m/s")
        self.add_input("lambda", val=np.zeros(nn), desc="body slip angle", units="rad")
        self.add_input("omega", val=np.zeros(nn), desc="yaw rate", units="rad/s")

        # tire loads
        self.add_input("S_fl", val=np.zeros(nn), desc="longitudinal force fl", units="N")
        self.add_input("S_fr", val=np.zeros(nn), desc="longitudinal force fr", units="N")
        self.add_input("S_rl", val=np.zeros(nn), desc="longitudinal force rl", units="N")
        self.add_input("S_rr", val=np.zeros(nn), desc="longitudinal force rr", units="N")

        self.add_input("F_fl", val=np.zeros(nn), desc="lateral force fl", units="N")
        self.add_input("F_fr", val=np.zeros(nn), desc="lateral force fr", units="N")
        self.add_input("F_rl", val=np.zeros(nn), desc="lateral force rl", units="N")
        self.add_input("F_rr", val=np.zeros(nn), desc="lateral force rr", units="N")

        # controls
        self.add_input("delta", val=np.zeros(nn), desc="steering angle", units="rad")

        # outputs
        self.add_output("sdot", val=np.zeros(nn), desc="distance along track", units="m/s")
        self.add_output("ndot", val=np.zeros(nn), desc="distance perpendicular to centerline",
                        units="m/s", )
        self.add_output("alphadot", val=np.zeros(nn), desc="angle relative to centerline",
                        units="rad/s", )
        self.add_output("Vdot", val=np.zeros(nn), desc="speed", units="m/s**2")
        self.add_output("lambdadot", val=np.zeros(nn), desc="body slip angle", units="rad/s")
        self.add_output("omegadot", val=np.zeros(nn), desc="yaw rate", units="rad/s**2")

        # constraint (power)
        self.add_output("power", val=np.zeros(nn), desc="tractive power", units="W")

        # Setup partials
        arange = np.arange(self.options["num_nodes"], dtype=int)

        # sdot
        self.declare_partials(of="sdot", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="sdot", wrt="alpha", rows=arange, cols=arange)
        self.declare_partials(of="sdot", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="sdot", wrt="n", rows=arange, cols=arange)
        self.declare_partials(of="sdot", wrt="kappa", rows=arange, cols=arange)

        # ndot
        self.declare_partials(of="ndot", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="ndot", wrt="alpha", rows=arange, cols=arange)
        self.declare_partials(of="ndot", wrt="lambda", rows=arange, cols=arange)

        # alphadot
        self.declare_partials(of="alphadot", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="alphadot", wrt="alpha", rows=arange, cols=arange)
        self.declare_partials(of="alphadot", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="alphadot", wrt="n", rows=arange, cols=arange)
        self.declare_partials(of="alphadot", wrt="omega", rows=arange, cols=arange)
        self.declare_partials(of="alphadot", wrt="kappa", rows=arange, cols=arange)

        # vdot
        self.declare_partials(of="Vdot", wrt="S_fl", rows=arange, cols=arange)
        self.declare_partials(of="Vdot", wrt="S_fr", rows=arange, cols=arange)
        self.declare_partials(of="Vdot", wrt="S_rl", rows=arange, cols=arange)
        self.declare_partials(of="Vdot", wrt="S_rr", rows=arange, cols=arange)

        self.declare_partials(of="Vdot", wrt="F_fr", rows=arange, cols=arange)
        self.declare_partials(of="Vdot", wrt="F_fl", rows=arange, cols=arange)

        self.declare_partials(of="Vdot", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="Vdot", wrt="omega", rows=arange, cols=arange)
        self.declare_partials(of="Vdot", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="Vdot", wrt="delta", rows=arange, cols=arange)

        # lambdadot
        self.declare_partials(of="lambdadot", wrt="F_fl", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="F_fr", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="F_rl", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="F_rr", rows=arange, cols=arange)

        self.declare_partials(of="lambdadot", wrt="S_fr", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="S_fl", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="S_rl", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="S_rr", rows=arange, cols=arange)

        self.declare_partials(of="lambdadot", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="omega", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="lambdadot", wrt="delta", rows=arange, cols=arange)

        # omegadot
        self.declare_partials(of="omegadot", wrt="S_fl", rows=arange, cols=arange)
        self.declare_partials(of="omegadot", wrt="S_fr", rows=arange, cols=arange)
        self.declare_partials(of="omegadot", wrt="S_rl", rows=arange, cols=arange)
        self.declare_partials(of="omegadot", wrt="S_rr", rows=arange, cols=arange)

        self.declare_partials(of="omegadot", wrt="F_fr", rows=arange, cols=arange)
        self.declare_partials(of="omegadot", wrt="F_fl", rows=arange, cols=arange)
        self.declare_partials(of="omegadot", wrt="F_rl", rows=arange, cols=arange)
        self.declare_partials(of="omegadot", wrt="F_rr", rows=arange, cols=arange)

        self.declare_partials(of="power", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="power", wrt="S_rr", rows=arange, cols=arange)
        self.declare_partials(of="power", wrt="S_rl", rows=arange, cols=arange)
        self.declare_partials(of="power", wrt="S_fl", rows=arange, cols=arange)
        self.declare_partials(of="power", wrt="S_fr", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        omega = inputs["omega"]
        V = inputs["V"]
        n = inputs["n"]
        lamb = inputs["lambda"]
        delta = inputs["delta"]
        alpha = inputs["alpha"]
        S_fl = inputs["S_fl"]
        S_fr = inputs["S_fr"]
        S_rl = inputs["S_rl"]
        S_rr = inputs["S_rr"]
        F_fl = inputs["F_fl"]
        F_fr = inputs["F_fr"]
        F_rl = inputs["F_rl"]
        F_rr = inputs["F_rr"]
        kappa = inputs["kappa"]
        M = inputs["M"]
        rho = inputs["rho"]
        Iz = inputs["Iz"]
        a = inputs["a"]
        b = inputs["b"]
        tw = inputs["tw"]
        CdA = inputs["CdA"]

        F_all = F_fl + F_fr + F_rl + F_rr
        S_all = S_fl + S_fr + S_rl + S_rr
        F_front = F_fl + F_fr
        S_front = S_fl + S_fr

        # drag equation
        drag = 0.5 * rho * CdA * V ** 2

        sin_alpha_minus_lamb = np.sin(alpha - lamb)
        cos_alpha_minus_lamb = np.cos(alpha - lamb)

        outputs["sdot"] = (V * cos_alpha_minus_lamb) / (1 - n * kappa)
        outputs["ndot"] = V * sin_alpha_minus_lamb
        outputs["alphadot"] = omega - (kappa * V * cos_alpha_minus_lamb) / (1 - n * kappa)
        Vdot = S_all / M - delta * F_front / M - drag / M - omega * V * lamb
        outputs["Vdot"] = Vdot
        outputs["lambdadot"] = (omega - Vdot * lamb / V - delta * S_front /
                                (M * V) - F_all / (M * V))
        outputs["omegadot"] = ((a * (F_fr + F_fl)) / Iz - (b * (F_rr + F_rl)) / Iz + (
                               tw * (-S_rr + S_rl - S_fr + S_fl)) / Iz)
        outputs["power"] = V * S_all

    def compute_partials(self, inputs, jacobian):
        omega = inputs["omega"]
        V = inputs["V"]
        n = inputs["n"]
        lamb = inputs["lambda"]
        delta = inputs["delta"]
        alpha = inputs["alpha"]
        S_fl = inputs["S_fl"]
        S_fr = inputs["S_fr"]
        S_rl = inputs["S_rl"]
        S_rr = inputs["S_rr"]
        F_fl = inputs["F_fl"]
        F_fr = inputs["F_fr"]
        F_rl = inputs["F_rl"]
        F_rr = inputs["F_rr"]
        kappa = inputs["kappa"]
        M = inputs["M"]
        rho = inputs["rho"]
        Iz = inputs["Iz"]
        a = inputs["a"]
        b = inputs["b"]
        tw = inputs["tw"]
        CdA = inputs["CdA"]

        F_all = F_fl + F_fr + F_rl + F_rr
        S_all = S_fl + S_fr + S_rl + S_rr
        F_front = F_fl + F_fr
        S_front = S_fl + S_fr

        ddrag_dv = 2 * CdA * V

        sin_alpha_minus_lamb = np.sin(alpha - lamb)
        cos_alpha_minus_lamb = np.cos(alpha - lamb)
        jacobian["sdot", "V"] = cos_alpha_minus_lamb / (1 - n * kappa)
        jacobian["sdot", "alpha"] = (V * sin_alpha_minus_lamb) / (kappa * n - 1)
        jacobian["sdot", "lambda"] = -(V * sin_alpha_minus_lamb) / (kappa * n - 1)
        jacobian["sdot", "n"] = (kappa * V * cos_alpha_minus_lamb) / (1 - kappa * n) ** 2
        jacobian["sdot", "kappa"] = n * V * cos_alpha_minus_lamb / (1 - kappa * n) ** 2

        jacobian["ndot", "V"] = sin_alpha_minus_lamb
        jacobian["ndot", "alpha"] = V * cos_alpha_minus_lamb
        jacobian["ndot", "lambda"] = -V * cos_alpha_minus_lamb

        jacobian["alphadot", "omega"] = 1.0
        jacobian["alphadot", "V"] = -(kappa * cos_alpha_minus_lamb) / (1 - n * kappa)
        jacobian["alphadot", "alpha"] = (kappa * V * sin_alpha_minus_lamb) / (1 - kappa * n)
        jacobian["alphadot", "lambda"] = (kappa * V * sin_alpha_minus_lamb) / (kappa * n - 1)
        jacobian["alphadot", "n"] = -(kappa ** 2 * V * cos_alpha_minus_lamb) / (
                                     (1 - kappa * n) ** 2)
        jacobian["alphadot", "kappa"] = -(V * cos_alpha_minus_lamb) / ((1 - kappa * n) ** 2)

        jacobian["Vdot", "S_rr"] = 1 / M
        jacobian["Vdot", "S_rl"] = 1 / M
        jacobian["Vdot", "S_fr"] = 1 / M
        jacobian["Vdot", "S_fl"] = 1 / M

        jacobian["Vdot", "delta"] = -(F_fr + F_fl) / M
        jacobian["Vdot", "F_fr"] = -delta / M
        jacobian["Vdot", "F_fl"] = -delta / M

        jacobian["Vdot", "V"] = -(0.5 * rho * ddrag_dv) / M - omega * lamb
        jacobian["Vdot", "omega"] = -V * lamb
        jacobian["Vdot", "lambda"] = -omega * V

        jacobian["omegadot", "F_fr"] = a / Iz
        jacobian["omegadot", "F_fl"] = a / Iz
        jacobian["omegadot", "F_rr"] = -b / Iz
        jacobian["omegadot", "F_rl"] = -b / Iz
        jacobian["omegadot", "S_rr"] = -tw / Iz
        jacobian["omegadot", "S_rl"] = tw / Iz
        jacobian["omegadot", "S_fr"] = -tw / Iz
        jacobian["omegadot", "S_fl"] = tw / Iz

        jacobian["lambdadot", "omega"] = 1 + lamb ** 2
        jacobian["lambdadot", "lambda"] = (2.0 * lamb * omega + (V * (0.5 * CdA * rho)) / M - (
                                           1.0 * (S_all - 1.0 * F_front * delta)) / (M * V))
        jacobian["lambdadot", "delta"] = (lamb * (F_fr + F_fl) / (M * V)) - (S_fr + S_fl) / (M * V)
        jacobian["lambdadot", "F_fr"] = (lamb * delta / (M * V)) - 1 / (M * V)
        jacobian["lambdadot", "F_fl"] = (lamb * delta / (M * V)) - 1 / (M * V)
        jacobian["lambdadot", "F_rr"] = -1 / (M * V)
        jacobian["lambdadot", "F_rl"] = -1 / (M * V)
        jacobian["lambdadot", "S_fr"] = -(lamb + delta) / (M * V)
        jacobian["lambdadot", "S_fl"] = -(lamb + delta) / (M * V)
        jacobian["lambdadot", "S_rr"] = -lamb / (M * V)
        jacobian["lambdadot", "S_rl"] = -lamb / (M * V)

        jacobian["lambdadot", "V"] = (0.5 * CdA * lamb * rho) / M +\
                                     (F_all + S_front * delta + S_all * lamb -
                                      F_front * delta * lamb) / (M * V ** 2)
        jacobian["power", "V"] = S_all
        jacobian["power", "S_fl"] = V
        jacobian["power", "S_fr"] = V
        jacobian["power", "S_rr"] = V
        jacobian["power", "S_rl"] = V
