import unittest
import numpy as np
from dymos.utils.misc import get_rate_units
from dymos.utils.testing_utils import assert_timeseries_near_equal

def create_linear_time_series(n, t_begin, t_end, x_begin, x_end):
    slope = (x_end - x_begin) / (t_end - t_begin)
    line = lambda t: slope * (t - t_begin) + x_begin

    t = np.linspace(t_begin, t_end, n).reshape(n, 1)
    x = line(t)
    return t, x

class TestAssertTimeseriesNearEqual(unittest.TestCase):

    def test_equal_time_series(self):
        # use the same time series and see if the assert correctly says they are the same
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)

        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3, abs_tolerance=10.0)
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

        x_check[5] = x_check_5_orig + tolerance * 0.9  # should not cause an error since rel error will be less than tolerance
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=rel_tolerance)

        x_check[5] = x_check_5_orig + tolerance * 1.1  # should cause an error since rel error will be less than tolerance

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         rel_tolerance=rel_tolerance)

        start_of_expected_errmsg = f"timeseries not equal within relative tolerance of {rel_tolerance}"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with f{start_of_expected_errmsg} but instead was f{actual_errmsg}")

    def test_unequal_time_series_abs_only(self):
        # slightly modify the "to be checked" time series and check that the assert is working
        # Try both when the mod is slightly less than the tolerance and slightly more
        # Only use absolute tolerance
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        abs_tolerance = 10.0

        x_check_5_orig = float(x_check[5])

        x_check[5] = x_check_5_orig + abs_tolerance * 0.9  # should not cause an error since rel error will be less than tolerance
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=abs_tolerance)

        x_check[5] = x_check_5_orig + abs_tolerance * 1.1  # should cause an error since rel error will be less than tolerance

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         abs_tolerance=abs_tolerance)

        start_of_expected_errmsg = f"timeseries not equal within absolute tolerance of {abs_tolerance}"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with f{start_of_expected_errmsg} but instead was f{actual_errmsg}")

    def test_unequal_time_series_abs_and_rel(self):
        # slightly modify the "to be checked" time series and check that the assert is working
        # Try both when the mod is slightly less than the tolerance and slightly more
        # Use both and absolute and relative tolerance.
        # Do the mods both in the range where the absolute tolerance is used (small values) and
        #   also relative tolerance (large values)
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        abs_tolerance = 10.0
        rel_tolerance = 0.1

        transition_tolerance = abs_tolerance / rel_tolerance

        # for < 100, uses the abs, x_check[5] is ~ 50
        x_check_5_orig = float(x_check[5])

        x_check[5] = x_check_5_orig + abs_tolerance * 0.9  # should not cause an error since rel error will be less than tolerance
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                     abs_tolerance=abs_tolerance,
                                     rel_tolerance=rel_tolerance
                                     )

        x_check[5] = x_check_5_orig + abs_tolerance * 1.1  # should cause an error since rel error will be less than tolerance

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         abs_tolerance=abs_tolerance,
                                         rel_tolerance=rel_tolerance
                                         )

        start_of_expected_errmsg = f"timeseries not equal within absolute tolerance of {abs_tolerance}"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with f{start_of_expected_errmsg} but instead was f{actual_errmsg}")

        # for > 100, uses the rel, x_check[15] is ~ 150
        x_check[5] = x_check_5_orig
        x_check_15_orig = float(x_check[15])
        tolerance = x_check_15_orig * rel_tolerance

        x_check[15] = x_check_15_orig + tolerance * 0.9  # should not cause an error since rel error will be less than tolerance
        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                     abs_tolerance=abs_tolerance,
                                     rel_tolerance=rel_tolerance
                                     )

        x_check[15] = x_check_15_orig + tolerance * 1.1  # should cause an error since rel error will be less than tolerance

        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check,
                                         abs_tolerance=abs_tolerance,
                                         rel_tolerance=rel_tolerance
                                         )


        start_of_expected_errmsg = f"timeseries not equal within relative tolerance of {rel_tolerance}"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with f{start_of_expected_errmsg} but instead was f{actual_errmsg}")

    def test_no_overlapping_time(self):
        t_ref, x_ref = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_check, x_check = create_linear_time_series(100, 510.0, 1500.0, 0.0, 1000.0)
        with self.assertRaises(ValueError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=1)
        expected_msg = f"There is no overlapping time between the two time series"
        self.assertEqual(str(e.exception), expected_msg)

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
        t_ref, x_ref_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0) # change this array
        x_ref = np.stack((x_ref_1, x_ref_2), axis=1)

        t_check, x_check_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 999.0)
        t_check, x_check_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 999.0)
        x_check = np.stack((x_check_1, x_check_2), axis=1)

        rel_tolerance = 1.0E-3
        with self.assertRaises(AssertionError) as e:
            assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=rel_tolerance)

        start_of_expected_errmsg = f"timeseries not equal within relative tolerance of {rel_tolerance}"
        actual_errmsg = str(e.exception)
        self.assertTrue(actual_errmsg.startswith(start_of_expected_errmsg),
                        f"Error message expected to start with f{start_of_expected_errmsg} but instead was f{actual_errmsg}")

    def test_multi_dimensional_with_overlapping_times(self):
        t_ref, x_ref_1 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        t_ref, x_ref_2 = create_linear_time_series(100, 0.0, 500.0, 0.0, 1000.0)
        x_ref = np.stack((x_ref_1, x_ref_2), axis=1)

        t_check, x_check_1 = create_linear_time_series(100, 250.0, 750.0, 500.0, 1500.0)
        t_check, x_check_2 = create_linear_time_series(100, 250.0, 750.0, 500.0, 1500.0)
        x_check = np.stack((x_check_1, x_check_2), axis=1)

        assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, rel_tolerance=1.0E-3)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
