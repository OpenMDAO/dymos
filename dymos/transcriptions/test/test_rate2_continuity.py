import unittest
from openmdao.utils.testing_utils import use_tempdirs
import dymos as dm
from dymos.examples.cart_pole.cartpole_dynamics import CartPoleDynamics


@use_tempdirs
class TestRate2Continuity(unittest.TestCase):
    def test_rate2_continuity(self):
        p = dm.Phase(transcription=dm.GaussLobatto(num_segments=3,
                                                   order=[5, 3, 3],
                                                   segment_ends=[0.0, 3.0, 10.0, 20.0],
                                                   compressed=True),
                     ode_class=CartPoleDynamics)

        p.add_control('test', rate2_continuity=True)

        p.setup()

        self.assertEqual(p.control_options['test']['rate2_continuity'], False)


if __name__ == '__main__':
    unittest.main()
