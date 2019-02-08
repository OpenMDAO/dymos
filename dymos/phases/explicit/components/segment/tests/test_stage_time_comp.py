from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.phases.options import TimeOptionsDictionary, StateOptionsDictionary
from dymos.phases.explicit.components.segment.stage_time_comp import StageTimeComp


class TestStageStateComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_steps = 4

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('seg_t0_tf', val=np.array([0.0, 2.0]), units='s')

        stage_time_comp = StageTimeComp(num_steps=4, method='rk4', time_options=time_options)

        cls.p.model.add_subsystem('stage_time_comp', stage_time_comp)

        cls.p.model.connect('seg_t0_tf', 'stage_time_comp.seg_t0_tf')

        cls.p.setup(force_alloc_complex=True)

        cls.p.run_model()

    def test_results(self):

        assert_rel_error(self, self.p['stage_time_comp.h'], [0.5, 0.5, 0.5, 0.5])

        t_stage_expected = np.array([[0.0, 0.25, 0.25, 0.5],
                                     [0.5, 0.75, 0.75, 1.0],
                                     [1.0, 1.25, 1.25, 1.5],
                                     [1.5, 1.75, 1.75, 2.0]])

        assert_rel_error(self, self.p['stage_time_comp.t_stage'], t_stage_expected)

    def test_partials(self):
        cpd = self.p.check_partials()
        assert_check_partials(cpd)

if __name__ == '__main__':
    unittest.main()
