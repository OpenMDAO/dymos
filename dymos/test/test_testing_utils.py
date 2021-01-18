import os
import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs
import dymos as dm

from dymos.utils.testing_utils import assert_cases_equal, assert_timeseries_near_equal


@use_tempdirs
class TestAssertCasesEqual(unittest.TestCase):

    def tearDown(self):
        for file in ('p1.db', 'p2.db'):
            try:
                os.remove(file)
                print('removed', file)
            except:
                print(f'no file named {file}')

    def test_different_variables(self):

        p1 = om.Problem()
        ivc = p1.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('a', val=0.0)
        ivc.add_output('b', val=1.0)
        ivc.add_output('c', val=2.0)
        ivc.add_output('x', val=3.0)
        p1.add_recorder(om.SqliteRecorder('p1.db'))
        p1.setup()

        p2 = om.Problem()
        ivc = p2.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('x', val=3.0)
        ivc.add_output('y', val=4.0)
        ivc.add_output('z', val=5.0)
        p2.add_recorder(om.SqliteRecorder('p2.db'))
        p2.setup()

        p1.run_model()
        p2.run_model()

        p1.record('final')
        p1.cleanup()

        p2.record('final')
        p2.cleanup()

        c1 = om.CaseReader('p1.db').get_case('final')
        c2 = om.CaseReader('p2.db').get_case('final')

        with self.assertRaises(AssertionError) as e:
            assert_cases_equal(c1, c2)

        expected = "\nrequire_same_vars=True but cases contain different variables.\nVariables in " \
                   "case1 but not in case2: ['a', 'b', 'c']\nVariables in case2 but not in " \
                   "case1: ['y', 'z']"

        self.assertEqual(str(e.exception), expected)

    def test_allow_different_variables(self):
        p1 = om.Problem()
        ivc = p1.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('a', val=0.0)
        ivc.add_output('b', val=1.0)
        ivc.add_output('c', val=2.0)
        ivc.add_output('x', val=3.0)
        p1.add_recorder(om.SqliteRecorder('p1.db'))
        p1.setup()

        p2 = om.Problem()
        ivc = p2.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('x', val=3.0)
        ivc.add_output('y', val=4.0)
        ivc.add_output('z', val=5.0)
        p2.add_recorder(om.SqliteRecorder('p2.db'))
        p2.setup()

        p1.run_model()
        p2.run_model()

        p1.record('final')
        p1.cleanup()

        p2.record('final')
        p2.cleanup()

        c1 = om.CaseReader('p1.db').get_case('final')
        c2 = om.CaseReader('p2.db').get_case('final')

        assert_cases_equal(c1, c2, require_same_vars=False)

    def test_different_shapes(self):

        p1 = om.Problem()
        ivc = p1.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('a', val=np.array([0, 1, 2]))
        ivc.add_output('b', val=np.array([[0, 1, 2], [3, 4, 5]]))
        ivc.add_output('c', val=np.eye(3, dtype=float))
        p1.add_recorder(om.SqliteRecorder('p1.db'))
        p1.setup()

        p2 = om.Problem()
        ivc = p2.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('a', val=3.0)
        ivc.add_output('b', val=np.array([0, 1, 2]))
        ivc.add_output('c', val=np.ones(3, dtype=float))
        p2.add_recorder(om.SqliteRecorder('p2.db'))
        p2.setup()

        p1.run_model()
        p2.run_model()

        p1.record('final')
        p1.cleanup()

        p2.record('final')
        p2.cleanup()

        c1 = om.CaseReader('p1.db').get_case('final')
        c2 = om.CaseReader('p2.db').get_case('final')

        with self.assertRaises(AssertionError) as e:
            assert_cases_equal(c1, c2)

        expected = "\nThe following variables have different shapes/sizes:\na has shape (3,) in " \
                   "case1 but shape (1,) in case2\nb has shape (2, 3) in case1 but shape (3,) in " \
                   "case2\nc has shape (3, 3) in case1 but shape (3,) in case2"

        self.assertEqual(str(e.exception), expected)

    def test_different_values(self):

        p1 = om.Problem()
        ivc = p1.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('a', val=np.array([0, 1, 2]))
        ivc.add_output('b', val=np.array([[0, 1, 2], [3, 4, 5]]))
        ivc.add_output('c', val=np.eye(3, dtype=float))
        p1.add_recorder(om.SqliteRecorder('p1.db'))
        p1.setup()

        p2 = om.Problem()
        ivc = p2.model.add_subsystem('ivc', om.IndepVarComp(), promotes_outputs=['*'])
        ivc.add_output('a', val=3 * np.array([0, 1, 2]))
        ivc.add_output('b', val=2 * np.array([[0, 1, 2], [3, 4, 5]]))
        ivc.add_output('c', val=5 * np.eye(3, dtype=float))
        p2.add_recorder(om.SqliteRecorder('p2.db'))
        p2.setup()

        p1.run_model()
        p2.run_model()

        p1.record('final')
        p1.cleanup()

        p2.record('final')
        p2.cleanup()

        c1 = om.CaseReader('p1.db').get_case('final')
        c2 = om.CaseReader('p2.db').get_case('final')

        expected = "\nThe following variables contain different values:\nvar: " \
                   "error\na: [0. 2. 4.]\nb: [[0. 1. 2.]\n [3. 4. 5.]]\n" \
                   "c: [[4. 0. 0.]\n [0. 4. 0.]\n [0. 0. 4.]]"

        with self.assertRaises(AssertionError) as e:
            assert_cases_equal(c1, c2)

        self.assertEqual(str(e.exception), expected)


@use_tempdirs
class TestAssertTimeseriesNearEqual(unittest.TestCase):

    def test_assert_different_shape(self):

        t1 = np.linspace(0, 100, 50)
        t2 = np.linspace(0, 100, 60)

        x1 = np.atleast_2d(np.sin(t1)).T
        x2 = np.atleast_3d(np.sin(t2)).T

        with self.assertRaises(ValueError) as e:
            assert_timeseries_near_equal(t1, x1, t2, x2)

        expected = "The shape of the variable in the two timeseries is not equal x1 is (1,)  x2 is (60, 1)"

        self.assertEqual(expected, str(e.exception))

    def test_assert_different_initial_time(self):

        t1 = np.linspace(0, 100, 50)
        t2 = np.linspace(5, 100, 50)

        x1 = np.atleast_2d(np.sin(t1)).T
        x2 = np.atleast_2d(np.sin(t2)).T

        with self.assertRaises(ValueError) as e:
            assert_timeseries_near_equal(t1, x1, t2, x2)

        expected = "The initial time of the two timeseries is not the same. t1[0]=0.0  " \
                   "t2[0]=5.0  difference: 5.0"

        self.assertEqual(str(e.exception), expected)

    def test_assert_different_final_time(self):

        t1 = np.linspace(0, 100, 50)
        t2 = np.linspace(0, 102, 50)

        x1 = np.atleast_2d(np.sin(t1)).T
        x2 = np.atleast_2d(np.sin(t2)).T

        with self.assertRaises(ValueError) as e:
            assert_timeseries_near_equal(t1, x1, t2, x2)

        expected = "The final time of the two timeseries is not the same. t1[0]=100.0  " \
                   "t2[0]=102.0  difference: 2.0"

        self.assertEqual(str(e.exception), expected)

    def test_assert_different_values(self):

        t1 = np.linspace(0, 100, 50)
        t2 = np.linspace(0, 100, 50)

        x1 = np.atleast_2d(np.sin(t1)).T
        x2 = np.atleast_2d(np.cos(t2)).T

        with self.assertRaises(ValueError) as e:
            assert_timeseries_near_equal(t1, x1, t2, x2)
