import unittest

from dymos.utils.misc import get_rate_units


class TestMisc(unittest.TestCase):

    def test_get_rate_units(self):
        rate_units = get_rate_units('m', 's', deriv=1)

        self.assertEqual(rate_units, 'm/s')

        rate2_units = get_rate_units('m', 's', deriv=2)

        self.assertEqual(rate2_units, 'm/s**2')

    def test_rate_units_for_unitless(self):
        rate_units = get_rate_units('unitless', 's', deriv=1)
        self.assertEqual(rate_units, '1/s')
        rate2_units = get_rate_units('unitless', 's', deriv=2)
        self.assertEqual(rate2_units, '1/s**2')

    def test_rate_units_for_unitless_time(self):
        for time_units in (None, 'unitless'):
            with self.subTest(f'{time_units=}'):
                rate_units = get_rate_units('unitless', time_units, deriv=1)
                self.assertEqual(rate_units, None)
                rate2_units = get_rate_units('unitless', time_units, deriv=2)
                self.assertEqual(rate2_units, None)

    def test_get_rate_units_invalid_deriv(self):

        with self.assertRaises(ValueError) as e:
            get_rate_units('m', 's', deriv=0)
        self.assertEqual(str(e.exception), 'deriv argument must be 1 or 2.')


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
