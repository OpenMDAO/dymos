import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
import dymos as dm

from dymos.transcriptions.timestepping.test.test_euler_integration_comp import SimpleODE
from dymos.transcriptions.timestepping.tau_comp import TauComp


class TestTauComp(unittest.TestCase):

    def test_eval(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=3, transcription='gauss-lobatto',
                                                         transcription_order=3)

        p = om.Problem()
        p.model.add_subsystem('tau_comp', TauComp(grid_data=grid_data, time_units='s'))
        p.setup(force_alloc_complex=True)

        p.set_val('tau_comp.t_initial', 0)
        p.set_val('tau_comp.t_duration', 10)

        t = np.linspace(0, 10, 500)
        ptau = np.zeros_like(t)
        stau = np.zeros_like(t)

        for i, time in enumerate(t):
            p.set_val('tau_comp.time', time)
            p.run_model()
            ptau[i] = p.get_val('tau_comp.ptau')
            stau[i] = p.get_val('tau_comp.stau')

        p.set_val('tau_comp.time', 1.5)
        p.run_model()
        cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)
