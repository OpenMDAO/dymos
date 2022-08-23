import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.utils.testing_utils import use_tempdirs

from dymos.examples.cart_pole.cartpole_dynamics import CartPoleDynamics


@use_tempdirs
class TestCartPoleDynamics(unittest.TestCase):
    def test_cartpole_ode(self):

        p = om.Problem()
        p.model.add_subsystem("dynamics", CartPoleDynamics(num_nodes=2), promotes=["*"])
        # set input values
        p.model.set_input_defaults("m_cart", 1.0, units="kg")
        p.model.set_input_defaults("m_pole", 0.3, units="kg")
        p.model.set_input_defaults("l_pole", 0.5, units="m")
        p.model.set_input_defaults("theta", [0.0, 1.0], units="rad")
        p.model.set_input_defaults("theta_dot", [0.0, 0.1], units="rad/s")
        p.model.set_input_defaults("f", [10, -10], units="N")

        # run model
        p.setup(check=False)
        p.run_model()
        # get outputs
        x_dotdot = p.get_val("x_dotdot", units="m/s**2")
        theta_dotdot = p.get_val("theta_dotdot", units="rad/s**2")
        assert_near_equal(x_dotdot, [10.0, -7.14331021], tolerance=1e-6)
        assert_near_equal(theta_dotdot, [-20.0, -8.79056677], tolerance=1e-6)

        # check partials
        partials = p.check_partials(compact_print=True, method="cs")
        assert_check_partials(partials, atol=1e-5, rtol=1e-5)


if __name__ == "___main__":
    unittest.main()
