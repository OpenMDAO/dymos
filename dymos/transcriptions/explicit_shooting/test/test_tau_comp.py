import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
import dymos as dm

from dymos.transcriptions.explicit_shooting.tau_comp import TauComp


class TestTauComp(unittest.TestCase):

    def test_eval(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=3, transcription='gauss-lobatto',
                                                         transcription_order=3)

        p = om.Problem()
        tau_comp = p.model.add_subsystem('tau_comp', TauComp(grid_data=grid_data, time_units='s'))
        p.setup(force_alloc_complex=True)

        tau_comp.options['segment_index'] = 0

        p.set_val('tau_comp.t_initial', 0)
        p.set_val('tau_comp.t_duration', 15)

        p.set_val('tau_comp.time', 1.5)
        p.run_model()

        frac = 0.1
        ptau_expected = -1 + 2 * frac
        assert_near_equal(p.get_val('tau_comp.ptau'), ptau_expected)

        frac = 1.5/5.0
        stau_expected = -1 + 2 * frac
        assert_near_equal(p.get_val('tau_comp.stau'), stau_expected)

        cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)

    def test_eval_vectorized(self):
        grid_data = dm.transcriptions.grid_data.GridData(num_segments=3, transcription='gauss-lobatto',
                                                         transcription_order=3)

        p = om.Problem()
        tau_comp = p.model.add_subsystem('tau_comp', TauComp(vec_size=4, grid_data=grid_data, time_units='s'))
        p.setup(force_alloc_complex=True)

        tau_comp.options['segment_index'] = 0

        p.set_val('tau_comp.t_initial', 0)
        p.set_val('tau_comp.t_duration', 15)

        p.set_val('tau_comp.time', np.linspace(0, 5, 4))
        p.run_model()

        print(p.get_val('tau_comp.ptau'))

        cpd = p.check_partials(method='cs')
        assert_check_partials(cpd)
