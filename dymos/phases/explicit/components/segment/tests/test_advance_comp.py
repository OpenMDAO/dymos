# from __future__ import print_function, division, absolute_import
#
# import unittest
#
# import numpy as np
#
# from openmdao.api import Problem, Group, IndepVarComp
# from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials
#
# from dymos.phases.options import TimeOptionsDictionary, StateOptionsDictionary
# from dymos.phases.explicit.components.segment.advance_comp import AdvanceComp
#
#
# def _f(y, t):
#     return y - t**2 + 1
#
#
# class TestKComp(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         num_steps = 4
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
#         ivc.add_output('states:y_0', val=np.ones((1,)), units='m')
#         ivc.add_output('k:y', val=np.ones((num_steps, 4, 1)), units='m')
#
#         advance_comp = AdvanceComp(num_steps=4,
#                                    method='rk4',
#                                    state_options=state_options)
#
#         cls.p.model.add_subsystem('advance_comp', advance_comp)
#
#         cls.p.model.connect('states:y_0', 'advance_comp.initial_states:y')
#         cls.p.model.connect('k:y', 'advance_comp.k:y')
#
#         cls.p.setup(force_alloc_complex=True)
#
#         k = np.array([[0.75, 0.90625, 0.9453125, 1.09765625],
#                       [1.087565104166667, 1.203206380208333, 1.23211669921875, 1.328623453776042],
#                       [1.319801330566406, 1.368501663208008, 1.380676746368408, 1.385139703750610],
#                       [1.378409485022227, 1.316761856277783, 1.301349949091673, 1.154084459568063]])
#
#         cls.p['k:y'] = np.reshape(k, (4, 4, 1))
#
#         cls.p['states:y_0'] = np.array([0.5])
#
#         cls.p.run_model()
#
#     def test_results(self):
#
#         y_expected = np.array([[0.5],
#                                [1.425130208333333],
#                                [2.639602661132812],
#                                [4.006818970044454],
#                                [5.301605229265987]])
#
#         assert_rel_error(self,
#                          self.p['advance_comp.step_states:y'],
#                          y_expected,
#                          tolerance=1.0E-12)
#
#     def test_partials(self):
#         np.set_printoptions(linewidth=1024)
#         cpd = self.p.check_partials(out_stream=None)
#         assert_check_partials(cpd)
#
#
# if __name__ == '__main__':
#     unittest.main()
