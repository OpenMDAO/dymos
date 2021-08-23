import unittest

import numpy as np

import openmdao.api as om
import dymos as dm
from dymos.examples.brachistochrone.brachistochrone_ode import BrachistochroneODE


class TestExplicitShooting(unittest.TestCase):

    def test_brachistochrone_explicit_shooting(self):
        prob = om.Problem()

        tx = dm.transcriptions.ExplicitShooting(num_segments=10, grid='gauss-lobatto',
                                                order=3, num_steps_per_segment=10, compressed=True)

        phase = dm.Phase(ode_class=BrachistochroneODE, transcription=tx)

        phase.set_time_options(units='s')

        # automatically discover states

        phase.add_parameter('g', val=9.80665, units='m/s**2', opt=False)
        phase.add_control('theta', val=45.0, units='deg', opt=True)

        prob.model.add_subsystem('phase0', phase)

        prob.setup()

        prob.set_val('phase0.t_initial', 0.0)
        prob.set_val('phase0.t_duration', 1.8016)
        prob.set_val('phase0.states:x', 0.0)
        prob.set_val('phase0.states:y', 10.0)
        prob.set_val('phase0.states:v', 1.0E-6)
        prob.set_val('phase0.parameters:g', 9.80665, units='m/s**2')
        prob.set_val('phase0.controls:theta', phase.interp('theta', ys=[0.01, 100]), units='deg')

        prob.run_model()

        with np.printoptions(linewidth=1024):
            prob.model.phase0._get_subsystem('integrator')._prob.check_partials(compact_print=False)

        # print(prob.get_val('phase0.controls:theta', units='deg'))

        prob.model.list_outputs(print_arrays=True)
