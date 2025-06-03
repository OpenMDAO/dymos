import os
import unittest
import numpy as np

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from dymos.utils.testing_utils import assert_cases_equal, assert_timeseries_near_equal


def create_linear_time_series(n, t_begin, t_end, x_begin, x_end):
    """
    Simple little function to generate a time series to be used for the tests in this file
    """
    slope = (x_end - x_begin) / (t_end - t_begin)

    t = np.linspace(t_begin, t_end, n).reshape(n, 1)
    x = slope * (t - t_begin) + x_begin
    return t, x


class TestAssertTimeseriesNearEqual(unittest.TestCase):

    def test_assert_different_shape(self):
        t_ref = np.linspace(0, 100, 50)
        t_check = np.linspace(0, 100, 60)

        x_ref = np.atleast_2d(np.sin(t_ref)).T
        x_check = np.atleast_3d(np.sin(t_check)).T

        with self.assertRaises(ValueError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-03)
        expected = "The shape of the variable in the two timeseries is not equal x_ref is (1," \
                   ")  x_check is (60, 1)"
        self.assertEqual(expected, str(e.exception))

    def test_equal_time_series(self):
        # use the same time series and see if the assert correctly says they are the same
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)

        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3,
                                     abs_tolerance=10.0)
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=10.0)

    def test_unequal_time_series_rel_only(self):
        # slightly modify the "to be checked" time series and check that the assert is working
        # Try both when the mod is slightly less than the tolerance and slightly more
        # Only use relative tolerance
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        rel_tolerance = 0.1

        x_check_5_orig = float(x_check[5])
        tolerance = x_check_5_orig * rel_tolerance

        x_check[5] = x_check_5_orig + tolerance * 0.9  # should not cause an error since rel error
        # will be less than tolerance
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=rel_tolerance)

        x_check[5] = x_check_5_orig + tolerance * 1.1  # should cause an error since rel error will
        # be less than tolerance

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         rel_tolerance=rel_tolerance)

        start_of_expected_errmsg = f"The following timeseries data are out of tolerance due to " \
                                   f"absolute (None) or relative ({rel_tolerance}) tolerance " \
                                   f"violations"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with {start_of_expected_errmsg} but "
                        f"instead was {actual_errmsg}")
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 2)
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 0)

    def test_unequal_time_series_abs_only(self):
        # slightly modify the "to be checked" time series and check that the assert is working
        # Try both when the mod is slightly less than the tolerance and slightly more
        # Only use absolute tolerance
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_check_5_orig = float(x_check[5])

        abs_tolerance = 10.0

        # should not cause an error since rel error will be less than tolerance
        x_check[5] = x_check_5_orig + abs_tolerance * 0.9
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=abs_tolerance)

        # should cause an error since rel error will be less than tolerance
        x_check[5] = x_check_5_orig + abs_tolerance * 1.1
        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         abs_tolerance=abs_tolerance)
        start_of_expected_errmsg = "The following timeseries data are out of tolerance due to " \
                                   f"absolute ({abs_tolerance}) or relative (None) tolerance " \
                                   "violations"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with {start_of_expected_errmsg} but "
                        f"instead was {actual_errmsg}")
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 2)
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 0)

    def test_unequal_time_series_abs_and_rel(self):
        # slightly modify the "to be checked" time series and check that the assert is working
        # Try both when the mod is slightly less than the tolerance and slightly more
        # Use both and absolute and relative tolerance.
        # Do the mods both in the range where the absolute tolerance is used (small values) and
        #   also relative tolerance (large values)
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_check_5_orig = float(x_check[5])

        abs_tolerance = 10.0
        rel_tolerance = 0.1

        # for < 100, uses the abs, x_check[5] is ~ 50
        # should not cause an error since rel error will be less than tolerance
        x_check[5] = x_check_5_orig + abs_tolerance * 0.9
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                     abs_tolerance=abs_tolerance,
                                     rel_tolerance=rel_tolerance
                                     )

        # should cause an error since rel error will be less than tolerance
        x_check[5] = x_check_5_orig + abs_tolerance * 1.1
        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         abs_tolerance=abs_tolerance,
                                         rel_tolerance=rel_tolerance
                                         )
        start_of_expected_errmsg = "The following timeseries data are out of tolerance due to " \
                                   "absolute"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with '{start_of_expected_errmsg}' but "
                        f"instead was '{actual_errmsg}'")
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 2)
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 2)

        # for > 100, uses the rel, x_check[15] is ~ 150
        x_check[5] = x_check_5_orig
        x_check_15_orig = float(x_check[15])
        tolerance = x_check_15_orig * rel_tolerance
        # should not cause an error since rel error will be less than tolerance
        x_check[15] = x_check_15_orig + tolerance * 0.9
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                     abs_tolerance=abs_tolerance,
                                     rel_tolerance=rel_tolerance
                                     )

        # should cause an error since rel error will be greater than tolerance
        x_check[15] = x_check_15_orig + tolerance * 1.1
        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         abs_tolerance=abs_tolerance,
                                         rel_tolerance=rel_tolerance
                                         )
        start_of_expected_errmsg = "The following timeseries data are out of tolerance due to " \
                                   "absolute"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with '{start_of_expected_errmsg}' but "
                        f"instead was '{actual_errmsg}'")
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 2)
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 2)

        # Combine the two cases where one data paint fails because of abs error and one because
        #   of rel error
        # should cause an error since rel error will be less than tolerance
        x_check[5] = x_check_5_orig + abs_tolerance * 1.1
        tolerance = x_check_15_orig * rel_tolerance
        # should cause an error since rel error will be less than tolerance
        x_check[15] = x_check_15_orig + tolerance * 1.1
        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         abs_tolerance=abs_tolerance,
                                         rel_tolerance=rel_tolerance
                                         )
        start_of_expected_errmsg = "The following timeseries data are out of tolerance due to " \
                                   "absolute"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with '{start_of_expected_errmsg}' but "
                        f"instead was '{actual_errmsg}'")
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 3)
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 3)

    def test_no_overlapping_time(self):
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 510.0, 1500.0, 0.0, 1000.0)
        with self.assertRaises(ValueError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=1)
        expected_msg = "There is no overlapping time between the two time series"
        actual_errmsg = str(e.exception)
        self.assertEqual(actual_errmsg, expected_msg)

    def test_with_overlapping_times(self):
        # checked time series shifted to the right
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 250.0, 750.0, 500.0, 1500.0)
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)

        # checked time series shifted to the left
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, -250.0, 250.0, -500.0, 500.0)
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)

        # checked time series is subset of reference
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 100.0, 400.0, 200.0, 800.0)
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)

        # checked time series is superset of reference
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, -500.0, 1000.0, -1000.0, 2000.0)
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)

    def test_multi_dimensional_equal(self):
        t_ref, x_ref_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_ref, x_ref_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_ref = np.stack((x_ref_1, x_ref_2), axis=1)

        t_check, x_check_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_check = np.stack((x_check_1, x_check_2), axis=1)

        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)

    def test_multi_dimensional_unequal(self):
        t_ref, x_ref_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_ref, x_ref_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_ref = np.stack((x_ref_1, x_ref_2), axis=1)

        # change the timeseries data values to be checked a little
        t_check, x_check_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 990.0)
        t_check, x_check_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 990.0)
        x_check = np.stack((x_check_1, x_check_2), axis=1)

        rel_tolerance = 1.0E-3

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         rel_tolerance=rel_tolerance)
        start_of_expected_errmsg = "The following timeseries data are out of tolerance due to " \
                                   f"absolute (None) or relative ({rel_tolerance}) tolerance " \
                                   "violations"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with {start_of_expected_errmsg} but "
                        f"instead was {actual_errmsg}")
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 0)
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 199)

    def test_multi_dimensional_unequal_abs_and_rel(self):
        t_ref, x_ref_1 = create_linear_time_series(10, 0.0, 500.0, 0.0, 1000.0)
        t_ref, x_ref_2 = create_linear_time_series(10, 0.0, 500.0, 0.0, 1000.0)
        x_ref = np.stack((x_ref_1, x_ref_2), axis=1)

        # change the timeseries data values to be checked a little
        t_check, x_check_1 = create_linear_time_series(10, 0.0, 500.0, 0.0, 990.0)
        t_check, x_check_2 = create_linear_time_series(10, 0.0, 500.0, 0.0, 990.0)
        x_check = np.stack((x_check_1, x_check_2), axis=1)

        abs_tolerance = 1.0E-3
        rel_tolerance = 0.5E-5

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)
        start_of_expected_errmsg = "The following timeseries data are out of tolerance due to " \
                                   f"absolute ({abs_tolerance}) or relative ({rel_tolerance}) " \
                                   "tolerance violations"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with {start_of_expected_errmsg} but "
                        f"instead was {actual_errmsg}")
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 19)
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 19)

    def test_multi_dimensional_with_overlapping_times(self):
        t_ref, x_ref_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_ref, x_ref_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_ref = np.stack((x_ref_1, x_ref_2), axis=1)

        t_check, x_check_1 = create_linear_time_series(100, 250.0, 750.0, 500.0, 1500.0)
        t_check, x_check_2 = create_linear_time_series(100, 250.0, 750.0, 500.0, 1500.0)
        x_check = np.stack((x_check_1, x_check_2), axis=1)

        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)

    def test_multi_dimensional_unequal_with_overlapping_times(self):
        t_ref, x_ref_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_ref, x_ref_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_ref = np.stack((x_ref_1, x_ref_2), axis=1)

        t_check, x_check_1 = create_linear_time_series(100, 250.0, 750.0, 500.0, 1000.0)
        t_check, x_check_2 = create_linear_time_series(100, 250.0, 750.0, 500.0, 1000.0)
        x_check = np.stack((x_check_1, x_check_2), axis=1)

        rel_tolerance = 1.0E-3

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         rel_tolerance=rel_tolerance)
        start_of_expected_errmsg = "The following timeseries data are out of tolerance due to " \
                                   f"absolute (None) or relative ({rel_tolerance}) tolerance " \
                                   "violations"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with {start_of_expected_errmsg} but "
                        f"instead was {actual_errmsg}")
        self.assertEqual(actual_errmsg.count('>ABS_TOL'), 0)
        self.assertEqual(actual_errmsg.count('>REL_TOL'), 99)


@use_tempdirs
class TestAssertCasesEqual(unittest.TestCase):

    def tearDown(self):
        for file in ('p1.db', 'p2.db'):
            try:
                os.remove(file)
            except Exception:
                pass

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

        p1_db = p1.get_outputs_dir() / 'p1.db'
        p2_db = p2.get_outputs_dir() / 'p2.db'

        c1 = om.CaseReader(p1_db).get_case('final')
        c2 = om.CaseReader(p2_db).get_case('final')

        with self.assertRaises(AssertionError) as e:
            assert_cases_equal(c1, c2)

        expected = "\nrequire_same_vars=True but cases contain different variables.\nVariables in "\
                   "" \
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

        p1_db = p1.get_outputs_dir() / 'p1.db'
        p2_db = p2.get_outputs_dir() / 'p2.db'

        c1 = om.CaseReader(p1_db).get_case('final')
        c2 = om.CaseReader(p2_db).get_case('final')

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

        p1_db = p1.get_outputs_dir() / 'p1.db'
        p2_db = p2.get_outputs_dir() / 'p2.db'

        c1 = om.CaseReader(p1_db).get_case('final')
        c2 = om.CaseReader(p2_db).get_case('final')

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

        p1_db = p1.get_outputs_dir() / 'p1.db'
        p2_db = p2.get_outputs_dir() / 'p2.db'

        c1 = om.CaseReader(p1_db).get_case('final')
        c2 = om.CaseReader(p2_db).get_case('final')

        expected = "\nThe following variables contain different values:\n" \
                   "var        max error       mean error\n" \
                   "--- ---------------- ----------------\n" \
                   "  a  4.000000000e+00  2.000000000e+00\n" \
                   "  b  5.000000000e+00  2.500000000e+00\n" \
                   "  c  4.000000000e+00  1.333333333e+00\n"

        with self.assertRaises(AssertionError) as e:
            assert_cases_equal(c1, c2)

        self.assertEqual(str(e.exception), expected)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
