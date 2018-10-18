from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.phases.options import TimeOptionsDictionary, StateOptionsDictionary
from dymos.phases.explicit.components.segment.stage_k_comp import StageKComp


def _f(y, t):
    return y - t**2 + 1


class TestKComp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        num_steps = 4

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        state_options = {}
        state_options['y'] = StateOptionsDictionary()
        state_options['y']['units'] = 'm'

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('h', val=0.5 * np.ones(num_steps), units='s')
        ivc.add_output('state_rates:y', val=0.5 * np.ones((num_steps, 4, 1)), units='m/s')

        k_comp = StageKComp(num_steps=4,
                            method='rk4',
                            time_options=time_options,
                            state_options=state_options)

        cls.p.model.add_subsystem('k_comp', k_comp)

        cls.p.model.connect('state_rates:y', 'k_comp.state_rates:y')
        cls.p.model.connect('h', 'k_comp.h')

        cls.p.setup(force_alloc_complex=True)

        cls.p['h'] = np.array([0.5, 0.5, 0.5, 0.5])

        t = np.array([[0.00, 0.25, 0.25, 0.50],
                      [0.50, 0.75, 0.75, 1.00],
                      [1.00, 1.25, 1.25, 1.50],
                      [1.50, 1.75, 1.75, 2.00]])

        y = np.array([[0.50, 0.875, 0.953125, 1.4453125],
                     [1.425130208333, 1.968912760416667, 2.0267333984375, 2.657246907552083],
                     [2.639602661132812, 3.299503326416016, 3.323853492736816, 4.020279407501221],
                     [4.006818970044454, 4.696023712555567, 4.665199898183346, 5.308168919136127]])

        cls.p['state_rates:y'] = np.reshape(_f(y, t), (4, 4, 1))

        cls.p.run_model()

    def test_results(self):

        k_expected = \
            np.array([[0.75, 0.90625, 0.9453125, 1.09765625],
                      [1.087565104166667, 1.203206380208333, 1.23211669921875, 1.328623453776042],
                      [1.319801330566406, 1.368501663208008, 1.380676746368408, 1.385139703750610],
                      [1.378409485022227, 1.316761856277783, 1.301349949091673, 1.154084459568063]])

        assert_rel_error(self,
                         self.p['k_comp.k:y'],
                         np.reshape(k_expected, (4, 4, 1)),
                         tolerance=1.0E-12)

    def test_partials(self):
        cpd = self.p.check_partials(method='cs')
        assert_check_partials(cpd)

#
#
#
# class TestYCompStage1(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#
#         time_options = TimeOptionsDictionary()
#         time_options['units'] = 's'
#
#         state_options = {}
#         state_options['y'] = StateOptionsDictionary()
#         state_options['y']['units'] = 'm'
#
#         cls.p = Problem(model=Group())
#
#         ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
#
#         ivc.add_output('states:y_0', val=0.5, units='m')
#
#         y_comp = StageYComp(stage=1, method='RK4', state_options=state_options)
#         cls.p.model.add_subsystem('y_comp', y_comp)
#
#         cls.p.model.connect('states:y_0', 'y_comp.states:y_0')
#
#         cls.p.setup(force_alloc_complex=True)
#
#         cls.p.run_model()
#
#     def test_compute(self):
#         assert_rel_error(self, self.p.get_val('y_comp.states:y_1'), 0.5)
#
#     def test_partials(self):
#         cpd = self.p.check_partials(method='cs', out_stream=None)
#         assert_check_partials(cpd)
#
# class TestYCompStage2(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#
#         time_options = TimeOptionsDictionary()
#         time_options['units'] = 's'
#
#         state_options = {}
#         state_options['y'] = StateOptionsDictionary()
#         state_options['y']['units'] = 'm'
#
#         cls.p = Problem(model=Group())
#
#         ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
#
#         ivc.add_output('states:y_1', val=0.5, units='m')
#         ivc.add_output('k:y_1', val=0.75, units='m')
#
#         y_comp = StageYComp(stage=2, method='RK4', state_options=state_options)
#         cls.p.model.add_subsystem('y_comp', y_comp)
#
#         cls.p.model.connect('states:y_1', 'y_comp.states:y_1')
#         cls.p.model.connect('k:y_1', 'y_comp.k:y_1')
#
#         cls.p.setup(force_alloc_complex=True)
#
#         cls.p.run_model()
#
#     def test_compute(self):
#         assert_rel_error(self, self.p.get_val('y_comp.states:y_2'), 0.875)
#
#     def test_partials(self):
#         cpd = self.p.check_partials(method='cs', out_stream=None)
#         assert_check_partials(cpd)
#
# class TestYCompStage3(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#
#         time_options = TimeOptionsDictionary()
#         time_options['units'] = 's'
#
#         state_options = {}
#         state_options['y'] = StateOptionsDictionary()
#         state_options['y']['units'] = 'm'
#
#         cls.p = Problem(model=Group())
#
#         ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
#
#         ivc.add_output('states:y_2', val=0.5, units='m')
#         ivc.add_output('k:y_2', val=0.90625, units='m')
#
#         y_comp = StageYComp(stage=3, method='RK4', state_options=state_options)
#         cls.p.model.add_subsystem('y_comp', y_comp)
#
#         cls.p.model.connect('states:y_2', 'y_comp.states:y_2')
#         cls.p.model.connect('k:y_2', 'y_comp.k:y_2')
#
#         cls.p.setup(force_alloc_complex=True)
#
#         cls.p.run_model()
#
#     def test_compute(self):
#         assert_rel_error(self, self.p.get_val('y_comp.states:y_3'), 0.953125)
#
#     def test_partials(self):
#         cpd = self.p.check_partials(method='cs', out_stream=None)
#         assert_check_partials(cpd)
#
# class TestYCompStage4(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#
#         time_options = TimeOptionsDictionary()
#         time_options['units'] = 's'
#
#         state_options = {}
#         state_options['y'] = StateOptionsDictionary()
#         state_options['y']['units'] = 'm'
#
#         cls.p = Problem(model=Group())
#
#         ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])
#
#         ivc.add_output('states:y_3', val=0.5, units='m')
#         ivc.add_output('k:y_3', val=0.9453125, units='m')
#
#         y_comp = StageYComp(stage=4, method='RK4', state_options=state_options)
#         cls.p.model.add_subsystem('y_comp', y_comp)
#
#         cls.p.model.connect('states:y_3', 'y_comp.states:y_3')
#         cls.p.model.connect('k:y_3', 'y_comp.k:y_3')
#
#         cls.p.setup(force_alloc_complex=True)
#
#         cls.p.run_model()
#
#     def test_compute(self):
#         assert_rel_error(self, self.p.get_val('y_comp.states:y_4'), 1.4453125)
#
#     def test_partials(self):
#         cpd = self.p.check_partials(method='cs', out_stream=None)
#         assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
