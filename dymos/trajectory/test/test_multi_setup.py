import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import require_pyoptsparse

import dymos as dm
from dymos.utils.misc import om_version
from dymos.examples.balanced_field.balanced_field_ode import BalancedFieldODEComp


class TestMultiSetup(unittest.TestCase):

    def test_call_setup_twice(self):

        ode_class = BalancedFieldODEComp
        tx = dm.Radau(num_segments=2, order=3, compressed=True)

        p = om.Problem()

        # First Phase: Brake release to V1 - both engines operable
        br_to_v1 = dm.Phase(ode_class=ode_class, transcription=tx,
                            ode_init_kwargs={'mode': 'runway'})

        # Instantiate the trajectory and add phases
        traj = dm.Trajectory()
        p.model.add_subsystem('traj', traj)
        traj.add_phase('br_to_v1', br_to_v1)
        all_phases = ['br_to_v1']

        # Add parameters common to multiple phases to the trajectory
        traj.add_parameter('m', val=174200., opt=False, units='lbm',
                           desc='aircraft mass',
                           targets={phase: ['m'] for phase in all_phases})

        p.setup()
        p.setup()  # This fails in dymos 1.13.2-dev
        p.final_setup()
