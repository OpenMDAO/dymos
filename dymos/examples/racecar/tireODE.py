import openmdao.api as om
import numpy as np


class TireODE(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        # constants
        self.add_input("M", val=0.0, desc="mass", units="kg")
        self.add_input("g", val=9.8, desc="mass", units="m/s**2")  # N
        self.add_input("a", val=1.8, desc="cg to front distance", units="m")
        self.add_input("b", val=1.6, desc="cg to rear distance", units="m")
        self.add_input("tw", val=0.73, desc="half track width", units="m")
        self.add_input("beta", val=0.0, desc="braking bias", units=None)  # val = 0.62
        self.add_input("k_lambda", val=44.0, desc="tire lateral stiffness", units=None)

        # states
        self.add_input("V", val=np.zeros(nn), desc="speed", units="m/s")
        self.add_input("lambda", val=np.zeros(nn), desc="body slip angle", units="rad")
        self.add_input("omega", val=np.zeros(nn), desc="yaw rate", units="rad/s")

        # normal load inputs
        self.add_input("N_rr", val=np.zeros(nn), desc="normal load rr", units="N")
        self.add_input("N_fr", val=np.zeros(nn), desc="normal load fr", units="N")
        self.add_input("N_rl", val=np.zeros(nn), desc="normal load rl", units="N")
        self.add_input("N_fl", val=np.zeros(nn), desc="normal load fl", units="N")

        # tire load outputs
        self.add_output("S_fl", val=np.zeros(nn), desc="longitudinal force fl", units="N")
        self.add_output("S_fr", val=np.zeros(nn), desc="longitudinal force fr", units="N")
        self.add_output("S_rl", val=np.zeros(nn), desc="longitudinal force rl", units="N")
        self.add_output("S_rr", val=np.zeros(nn), desc="longitudinal force rr", units="N")

        self.add_output("F_fl", val=np.zeros(nn), desc="lateral force fl", units="N")
        self.add_output("F_fr", val=np.zeros(nn), desc="lateral force fr", units="N")
        self.add_output("F_rl", val=np.zeros(nn), desc="lateral force rl", units="N")
        self.add_output("F_rr", val=np.zeros(nn), desc="lateral force rr", units="N")

        # controls
        self.add_input("thrust", val=np.zeros(nn), desc="thrust", units=None)
        self.add_input("delta", val=np.zeros(nn), desc="steering angle", units="rad")

        # Setup partials
        arange = np.arange(self.options["num_nodes"], dtype=int)

        self.declare_partials(of="S_fl", wrt="thrust", rows=arange, cols=arange)
        self.declare_partials(of="S_fr", wrt="thrust", rows=arange, cols=arange)
        self.declare_partials(of="S_rl", wrt="thrust", rows=arange, cols=arange)
        self.declare_partials(of="S_rr", wrt="thrust", rows=arange, cols=arange)

        self.declare_partials(of="F_rr", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="F_rr", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="F_rr", wrt="omega", rows=arange, cols=arange)
        self.declare_partials(of="F_rr", wrt="N_rr", rows=arange, cols=arange)

        self.declare_partials(of="F_fr", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="F_fr", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="F_fr", wrt="omega", rows=arange, cols=arange)
        self.declare_partials(of="F_fr", wrt="N_fr", rows=arange, cols=arange)

        self.declare_partials(of="F_rl", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="F_rl", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="F_rl", wrt="omega", rows=arange, cols=arange)
        self.declare_partials(of="F_rl", wrt="N_rl", rows=arange, cols=arange)

        self.declare_partials(of="F_fl", wrt="V", rows=arange, cols=arange)
        self.declare_partials(of="F_fl", wrt="lambda", rows=arange, cols=arange)
        self.declare_partials(of="F_fl", wrt="omega", rows=arange, cols=arange)
        self.declare_partials(of="F_fl", wrt="N_fl", rows=arange, cols=arange)

        self.declare_partials(of="F_fr", wrt="delta", rows=arange, cols=arange)
        self.declare_partials(of="F_fl", wrt="delta", rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        omega = inputs["omega"]
        V = inputs["V"]
        lamb = inputs["lambda"]
        M = inputs["M"]
        g = inputs["g"]
        a = inputs["a"]
        b = inputs["b"]
        tw = inputs["tw"]
        N_rr = inputs["N_rr"]
        N_rl = inputs["N_rl"]
        N_fr = inputs["N_fr"]
        N_fl = inputs["N_fl"]
        delta = inputs["delta"]
        beta = inputs["beta"]
        k_lambda = inputs["k_lambda"]
        thrust = inputs["thrust"]

        # split thrust signal into a throttle and brake signal
        signs = np.sign(thrust)
        signs2 = np.where(signs < 1, 0, 1)

        brake = (signs2 - 1) * thrust  # positive
        throttle = signs2 * thrust

        outputs["S_fl"] = -(M * g / 2) * brake * beta
        outputs["S_fr"] = -(M * g / 2) * brake * beta
        outputs["S_rl"] = (M * g / 2) * (throttle - brake * (1 - beta))
        outputs["S_rr"] = (M * g / 2) * (throttle - brake * (1 - beta))

        outputs["F_rr"] = N_rr * k_lambda * (lamb + (omega * (b + lamb * tw)) / V)
        outputs["F_rl"] = N_rl * k_lambda * (lamb + (omega * (b - lamb * tw)) / V)
        outputs["F_fr"] = (N_fr * k_lambda * (lamb + delta - (omega * (a - lamb * tw)) / V))
        outputs["F_fl"] = (N_fl * k_lambda * (lamb + delta - (omega * (a + lamb * tw)) / V))

    def compute_partials(self, inputs, jacobian):
        omega = inputs["omega"]
        V = inputs["V"]
        lamb = inputs["lambda"]
        M = inputs["M"]
        g = inputs["g"]
        a = inputs["a"]
        b = inputs["b"]
        tw = inputs["tw"]
        N_rr = inputs["N_rr"]
        N_rl = inputs["N_rl"]
        N_fr = inputs["N_fr"]
        N_fl = inputs["N_fl"]
        delta = inputs["delta"]
        beta = inputs["beta"]
        k_lambda = inputs["k_lambda"]
        thrust = inputs["thrust"]

        signs = np.sign(thrust)
        signs2 = np.where(signs < 1, 0, 1)  # -1 where brake is active

        brake = signs2 - 1  # positive
        throttle = signs2

        jacobian["S_fl", "thrust"] = -(M * g / 2) * beta * brake
        jacobian["S_fr", "thrust"] = -(M * g / 2) * beta * brake
        jacobian["S_rl", "thrust"] = (-(M * g / 2) * (1 - beta) * brake + (M * g / 2) * throttle)
        jacobian["S_rr", "thrust"] = (-(M * g / 2) * (1 - beta) * brake + (M * g / 2) * throttle)

        jacobian["F_rr", "N_rr"] = k_lambda * (lamb + (omega * (b + lamb * tw)) / V)
        jacobian["F_rl", "N_rl"] = k_lambda * (lamb + (omega * (b - lamb * tw)) / V)
        jacobian["F_fr", "N_fr"] = k_lambda * (lamb + delta - (omega * (a - lamb * tw)) / V)
        jacobian["F_fl", "N_fl"] = k_lambda * (lamb + delta - (omega * (a + lamb * tw)) / V)

        jacobian["F_rr", "lambda"] = N_rr * k_lambda * (1 + (omega * tw / V))
        jacobian["F_fr", "lambda"] = N_fr * k_lambda * (1 + (omega * tw / V))
        jacobian["F_rl", "lambda"] = N_rl * k_lambda * (1 - (omega * tw / V))
        jacobian["F_fl", "lambda"] = N_fl * k_lambda * (1 - (omega * tw / V))

        jacobian["F_rr", "omega"] = N_rr * k_lambda * ((b + lamb * tw) / V)
        jacobian["F_fr", "omega"] = N_fr * k_lambda * (-(a - lamb * tw) / V)
        jacobian["F_rl", "omega"] = N_rl * k_lambda * ((b - lamb * tw) / V)
        jacobian["F_fl", "omega"] = N_fl * k_lambda * (-(a + lamb * tw) / V)

        jacobian["F_fr", "delta"] = N_fr * k_lambda
        jacobian["F_fl", "delta"] = N_fl * k_lambda

        jacobian["F_rr", "V"] = N_rr * k_lambda * -(omega * (b + lamb * tw)) / V ** 2
        jacobian["F_fr", "V"] = N_fr * k_lambda * (omega * (a - lamb * tw)) / V ** 2
        jacobian["F_rl", "V"] = N_rl * k_lambda * -(omega * (b - lamb * tw)) / V ** 2
        jacobian["F_fl", "V"] = N_fl * k_lambda * (omega * (a + lamb * tw)) / V ** 2
