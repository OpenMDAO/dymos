from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.phases.options import TimeOptionsDictionary, StateOptionsDictionary
from dymos.phases.explicit.components.segment.stage_state_comp import StageStateComp


def _f(y, t):
    return y - t**2 + 1

class TestYComp(unittest.TestCase):

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

        ivc.add_output('step_states:y', val=0.5 * np.ones(num_steps + 1), units='m')

        y_comp = StageStateComp(num_steps=4, method='rk4', state_options=state_options)

        cls.p.model.add_subsystem('y_comp', y_comp)

        cls.p.model.connect('step_states:y', 'y_comp.step_states:y')

        cls.p.setup(force_alloc_complex=True)

        cls.p.run_model()

    def test_results(self):

        print(self.p['y_comp.step_states:y'])
        print('stage_states')
        print(self.p['y_comp.stage_states:y'])






class TestYCompStage1(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        state_options = {}
        state_options['y'] = StateOptionsDictionary()
        state_options['y']['units'] = 'm'

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('states:y_0', val=0.5, units='m')

        y_comp = StageYComp(stage=1, method='RK4', state_options=state_options)
        cls.p.model.add_subsystem('y_comp', y_comp)

        cls.p.model.connect('states:y_0', 'y_comp.states:y_0')

        cls.p.setup(force_alloc_complex=True)

        cls.p.run_model()

    def test_compute(self):
        assert_rel_error(self, self.p.get_val('y_comp.states:y_1'), 0.5)

    def test_partials(self):
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

class TestYCompStage2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        state_options = {}
        state_options['y'] = StateOptionsDictionary()
        state_options['y']['units'] = 'm'

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('states:y_1', val=0.5, units='m')
        ivc.add_output('k:y_1', val=0.75, units='m')

        y_comp = StageYComp(stage=2, method='RK4', state_options=state_options)
        cls.p.model.add_subsystem('y_comp', y_comp)

        cls.p.model.connect('states:y_1', 'y_comp.states:y_1')
        cls.p.model.connect('k:y_1', 'y_comp.k:y_1')

        cls.p.setup(force_alloc_complex=True)

        cls.p.run_model()

    def test_compute(self):
        assert_rel_error(self, self.p.get_val('y_comp.states:y_2'), 0.875)

    def test_partials(self):
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

class TestYCompStage3(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        state_options = {}
        state_options['y'] = StateOptionsDictionary()
        state_options['y']['units'] = 'm'

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('states:y_2', val=0.5, units='m')
        ivc.add_output('k:y_2', val=0.90625, units='m')

        y_comp = StageYComp(stage=3, method='RK4', state_options=state_options)
        cls.p.model.add_subsystem('y_comp', y_comp)

        cls.p.model.connect('states:y_2', 'y_comp.states:y_2')
        cls.p.model.connect('k:y_2', 'y_comp.k:y_2')

        cls.p.setup(force_alloc_complex=True)

        cls.p.run_model()

    def test_compute(self):
        assert_rel_error(self, self.p.get_val('y_comp.states:y_3'), 0.953125)

    def test_partials(self):
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

class TestYCompStage4(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        time_options = TimeOptionsDictionary()
        time_options['units'] = 's'

        state_options = {}
        state_options['y'] = StateOptionsDictionary()
        state_options['y']['units'] = 'm'

        cls.p = Problem(model=Group())

        ivc = cls.p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('states:y_3', val=0.5, units='m')
        ivc.add_output('k:y_3', val=0.9453125, units='m')

        y_comp = StageYComp(stage=4, method='RK4', state_options=state_options)
        cls.p.model.add_subsystem('y_comp', y_comp)

        cls.p.model.connect('states:y_3', 'y_comp.states:y_3')
        cls.p.model.connect('k:y_3', 'y_comp.k:y_3')

        cls.p.setup(force_alloc_complex=True)

        cls.p.run_model()

    def test_compute(self):
        assert_rel_error(self, self.p.get_val('y_comp.states:y_4'), 1.4453125)

    def test_partials(self):
        cpd = self.p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)


if __name__ == '__main__':
    unittest.main()
