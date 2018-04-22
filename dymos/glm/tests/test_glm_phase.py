import unittest

from parameterized import parameterized
from itertools import product

from dymos.glm.ozone.methods_list import method_classes

from dymos import Phase
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestGLMPhase(unittest.TestCase):

    @parameterized.expand(product(
        ['optimizer-based', 'solver-based', 'time-marching'],
        method_classes.keys(),
    ))
    def test_compressed_transcription_error_message(self, glm_formulation, glm_integrator):

        with self.assertRaises(ValueError) as e:
            phase = Phase('glm',
                          ode_class=BrachistochroneODE,
                          num_segments=10,
                          formulation=glm_formulation,
                          method_name=glm_integrator)

        expected = 'GLMPhase does not currently support compressed transcription. ' \
                   'Specify `compressed=False` when initializing the phase.'
        self.assertEqual(str(e.exception), expected)
