import openmdao.utils.assert_utils as _om_assert_utils


def assert_check_partials(data, atol=1.0E-6, rtol=1.0E-6):
    """
    Calls OpenMDAO's assert_check_partials but verifies that the dictionary of assertion data is
    not empty due to dymos.options['include_check_partials'] being False.
    """
    assert len(data) >= 1, "No check partials data found.  Is " \
                           "dymos.options['include_check_partials'] set to True?"
    _om_assert_utils.assert_check_partials(data, atol, rtol)
