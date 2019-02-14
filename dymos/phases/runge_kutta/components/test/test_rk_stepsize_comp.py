from __future__ import print_function, division, absolute_import

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error, assert_check_partials

from dymos.phases.runge_kutta.components.runge_kutta_stepsize_comp import RungeKuttaStepsizeComp


class TestRKStepsizeComp(unittest.TestCase):

    def test_rk_stepsize_comp(self):
        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('t_duration', shape=(1,), units='s')

        p.model.add_subsystem('c', RungeKuttaStepsizeComp(num_segments=4,
                                                          seg_rel_lengths=[1, 1, 1, 1],
                                                          time_units='s'))

        p.model.connect('t_duration', 'c.t_duration')

        p.setup(check=True, force_alloc_complex=True)

        p['t_duration'] = 2.0

        np.set_printoptions(linewidth=1024)

        p.run_model()

        expected = np.array([0.5, 0.5, 0.5, 0.5])

        assert_rel_error(self, p.get_val('c.h'), expected, tolerance=1.0E-9)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)

    def test_rk_stepsize_comp_nonuniform(self):
        p = Problem(model=Group())

        ivc = p.model.add_subsystem('ivc', IndepVarComp(), promotes_outputs=['*'])

        ivc.add_output('t_duration', shape=(1,), units='s')

        p.model.add_subsystem('c', RungeKuttaStepsizeComp(num_segments=4,
                                                          seg_rel_lengths=[1, 1, 0.5, 0.25],
                                                          time_units='s'))

        p.model.connect('t_duration', 'c.t_duration')

        p.setup(check=True, force_alloc_complex=True)

        p['t_duration'] = 5.5

        np.set_printoptions(linewidth=1024)

        p.run_model()

        expected = np.array([2.0, 2.0, 1.0, 0.5])

        assert_rel_error(self, p.get_val('c.h'), expected, tolerance=1.0E-9)

        cpd = p.check_partials(method='cs', out_stream=None)
        assert_check_partials(cpd)