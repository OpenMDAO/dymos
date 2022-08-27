"""
Cart-pole dynamics (ODE)
"""

import numpy as np
import openmdao.api as om


class CartPoleDynamics(om.ExplicitComponent):
    """
    Computes the time derivatives of states given state variables and control inputs.

    Parameters
    ----------
    m_cart : float
        Mass of cart.
    m_pole : float
        Mass of pole.
    l_pole : float
        Length of pole.
    theta : 1d array
        Angle of pole, 0 for vertical downward, positive counter clockwise.
    theta_dot : 1d array
        Angluar velocity of pole.
    f : 1d array
        x-wise force applied to the cart.

    Returns
    -------
    x_dotdot : 1d array
        Acceleration of cart in x direction.
    theta_dotdot : 1d array
        Angular acceleration of pole.
    e_dot : 1d array
        Rate of "energy" state.
    """

    def initialize(self):
        self.options.declare(
            "num_nodes", default=1, desc="number of nodes to be evaluated (i.e., length of vectors x, theta, etc)"
        )
        self.options.declare("g", default=9.81, desc="gravity constant")

    def setup(self):
        nn = self.options["num_nodes"]

        # --- inputs ---
        # cart-pole parameters
        self.add_input("m_cart", shape=(1,), units="kg", desc="cart mass")
        self.add_input("m_pole", shape=(1,), units="kg", desc="pole mass")
        self.add_input("l_pole", shape=(1,), units="m", desc="pole length")
        # state variables. States x, x_dot, energy have no influence on the outputs, so we don't need them as inputs.
        self.add_input("theta", shape=(nn,), units="rad", desc="pole angle")
        self.add_input("theta_dot", shape=(nn,), units="rad/s", desc="pole angle velocity")
        # control input
        self.add_input("f", shape=(nn,), units="N", desc="force applied to cart in x direction")

        # --- outputs ---
        # rate of states (accelerations)
        self.add_output("x_dotdot", shape=(nn,), units="m/s**2", desc="x acceleration of cart")
        self.add_output("theta_dotdot", shape=(nn,), units="rad/s**2", desc="angular acceleration of pole")
        # also computes force**2, which will be integrated to compute the objective
        self.add_output("e_dot", shape=(nn,), units="N**2", desc="square of force to be integrated")

        # ---  partials ---.
        # Jacobian of outputs w.r.t. state/control inputs is diagonal
        # because each node (corresponds to time discretization) is independent
        self.declare_partials(of=["*"], wrt=["theta", "theta_dot", "f"], method="exact", rows=np.arange(nn), cols=np.arange(nn))

        # partials of outputs w.r.t. cart-pole parameters. We will use complex-step, but still declare the sparsity structure.
        # NOTE: since the cart-pole parameters are fixed during optimization, these partials are not necessary to declare.
        self.declare_partials(of=["*"], wrt=["m_cart", "m_pole", "l_pole"], method="cs", rows=np.arange(nn), cols=np.zeros(nn))
        self.set_check_partial_options(wrt=["m_cart", "m_pole", "l_pole"], method="fd", step=1e-7)

    def compute(self, inputs, outputs):
        g = self.options["g"]
        mc = inputs["m_cart"]
        mp = inputs["m_pole"]
        lpole = inputs["l_pole"]
        theta = inputs["theta"]
        omega = inputs["theta_dot"]
        f = inputs["f"]

        sint = np.sin(theta)
        cost = np.cos(theta)
        det = mp * lpole * cost**2 - lpole * (mc + mp)
        outputs["x_dotdot"] = (-mp * lpole * g * sint * cost - lpole * (f + mp * lpole * omega**2 * sint)) / det
        outputs["theta_dotdot"] = ((mc + mp) * g * sint + cost * (f + mp * lpole * omega**2 * sint)) / det
        outputs["e_dot"] = f**2

    def compute_partials(self, inputs, jacobian):
        g = self.options["g"]
        mc = inputs["m_cart"]
        mp = inputs["m_pole"]
        lpole = inputs["l_pole"]
        theta = inputs["theta"]
        theta_dot = inputs["theta_dot"]
        f = inputs["f"]

        # --- derivatives of x_dotdot ---
        # Collecting Theta Derivative
        low = mp * lpole * np.cos(theta) ** 2 - lpole * mc - lpole * mp
        dhigh = (
            mp * g * lpole * np.sin(theta) ** 2 -
            mp * g * lpole * np.cos(theta) ** 2 -
            lpole**2 * mp * theta_dot**2 * np.cos(theta)
        )
        high = -mp * g * lpole * np.cos(theta) * np.sin(theta) - lpole * f - lpole**2 * mp * theta_dot**2 * np.sin(theta)
        dlow = 2.0 * mp * lpole * np.cos(theta) * (-np.sin(theta))

        jacobian["x_dotdot", "theta"] = (low * dhigh - high * dlow) / low**2
        jacobian["x_dotdot", "theta_dot"] = (
            -2.0 * theta_dot * lpole**2 * mp * np.sin(theta) / (mp * lpole * np.cos(theta) ** 2 - lpole * mc - lpole * mp)
        )
        jacobian["x_dotdot", "f"] = -lpole / (mp * lpole * np.cos(theta) ** 2 - lpole * mc - lpole * mp)

        # --- derivatives of theta_dotdot ---
        # Collecting Theta Derivative
        low = mp * lpole * np.cos(theta) ** 2 - lpole * mc - lpole * mp
        dlow = 2.0 * mp * lpole * np.cos(theta) * (-np.sin(theta))
        high = (mc + mp) * g * np.sin(theta) + f * np.cos(theta) + mp * lpole * theta_dot**2 * np.sin(theta) * np.cos(theta)
        dhigh = (
            (mc + mp) * g * np.cos(theta) -
            f * np.sin(theta) +
            mp * lpole * theta_dot**2 * (np.cos(theta) ** 2 - np.sin(theta) ** 2)
        )

        jacobian["theta_dotdot", "theta"] = (low * dhigh - high * dlow) / low**2
        jacobian["theta_dotdot", "theta_dot"] = (
            2.0 *
            theta_dot *
            mp *
            lpole *
            np.sin(theta) *
            np.cos(theta) /
            (mp * lpole * np.cos(theta) ** 2 - lpole * mc - lpole * mp)
        )
        jacobian["theta_dotdot", "f"] = np.cos(theta) / (mp * lpole * np.cos(theta) ** 2 - lpole * mc - lpole * mp)

        # --- derivatives of e_dot ---
        jacobian["e_dot", "theta"] = 0.0
        jacobian["e_dot", "theta_dot"] = 0.0
        jacobian["e_dot", "f"] = 2.0 * f
