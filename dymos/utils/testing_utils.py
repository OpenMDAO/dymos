import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.interpolate import interp1d
import openmdao.utils.assert_utils as _om_assert_utils


def assert_check_partials(data, atol=1.0E-6, rtol=1.0E-6):
    """
    Calls OpenMDAO's assert_check_partials but verifies that the dictionary of assertion data is
    not empty due to dymos.options['include_check_partials'] being False.
    """
    assert len(data) >= 1, "No check partials data found.  Is " \
                           "dymos.options['include_check_partials'] set to True?"
    _om_assert_utils.assert_check_partials(data, atol, rtol)


def assert_cases_equal(case1, case2, tol=1.0E-12, require_same_vars=True):
    """
    Raise AssertionError if the data in two OpenMDAO Cases is different.

    Parameters
    ----------
    case1 : om.Case
        The first OpenMDAO Case for comparison.
    case2 : om.Case
        The second OpenMDAO Case for comparison.
    tol : float
        The absolute value of the allowable difference in values between two variables.
    require_same_vars : bool
        If True, require that the two files contain the same set of variables.

    Raises
    ------
    AssertionError
        Raised in the following cases:  If require_same_vars is True, then AssertionError is raised
        if the two cases contain different variables.  Otherwise, this error is raised if case1
        and case2 contain the same variable but the variable has a different size/shape in the two
        cases, or if the variables have the same shape but different values (as given by tol).
    """
    case1_vars = {t[1]['prom_name']: t[1] for t in
                  case1.list_inputs(values=True, units=True, prom_name=True, out_stream=None)}
    case1_vars.update({t[1]['prom_name']: t[1] for t in
                       case1.list_outputs(values=True, units=True, prom_name=True, out_stream=None)})

    case2_vars = {t[1]['prom_name']: t[1] for t in
                  case2.list_inputs(values=True, units=True, prom_name=True, out_stream=None)}
    case2_vars.update({t[1]['prom_name']: t[1] for t in
                       case2.list_outputs(values=True, units=True, prom_name=True, out_stream=None)})

    # Warn if a and b don't contain the same sets of variables
    diff_err_msg = ''
    if require_same_vars:
        case1_minus_case2 = set(case1_vars.keys()) - set(case2_vars.keys())
        case2_minus_case1 = set(case2_vars.keys()) - set(case1_vars.keys())
        diff_err_msg = ''
        if case1_minus_case2:
            diff_err_msg += f'\nVariables in case1 but not in case2: {case1_minus_case2}'
        if case2_minus_case1:
            diff_err_msg += f'\nVariables in case2 but not in case1: {case2_minus_case1}'

    shape_errors = set()
    val_errors = set()
    shape_err_msg = '\nThe following variables have different shapes/sizes:'
    val_err_msg = '\nThe following variables contain different values:'

    for var in set(case1_vars.keys()).intersection(case2_vars.keys()):
        a = case1_vars[var]['value']
        b = case2_vars[var]['value']
        if a.shape != b.shape:
            shape_errors.add(var)
            shape_err_msg += f'\n{var}'
            continue
        err = np.abs(a - b)
        if np.any(err > tol):
            val_errors.add(var)
            val_err_msg += f'\n{var}: {err}'

    err_msg = ''
    if diff_err_msg:
        err_msg += diff_err_msg
    if shape_errors:
        err_msg += shape_err_msg
    if val_errors:
        err_msg += val_err_msg

    if err_msg:
        raise AssertionError(err_msg)


def assert_timeseries_near_equal(t1, x1, t2, x2, tolerance=1.0E-6, num_points=20):
    """
    Assert that two timeseries of data are approximately equal.
    This is done by fitting a 1D interpolant to each index of each timeseries, and then comparing
    the values of the two interpolants at some equally spaced number of points.

    Parameters
    ----------
    t1 : np.array
        Time values for the first timeseries.
    x1 : np.array
        Data values for the first timeseries.
    t2 : np.array
        Time values for the second timeseries.
    x2 : np.array
        Data values for the second timeseries.
    tolerance : float
        The tolerance for any errors along at each point checked.
    num_points : int
        The number of points along the timeseries to be compared.

    Raises
    ------
    AssertionError
        When one or more elements of the interpolated timeseries are not within the
        desired tolerance.
    """
    shape1 = x1.shape[1:]
    shape2 = x2.shape[1:]

    if shape1 != shape2:
        raise ValueError('The shape variable in the two timeseries is not equal')

    if abs(t1[0] - t2[0]) > 1.0E-12:
        raise ValueError('The initial time of the two timeseries is not the same.')

    if abs(t1[-1] - t2[-1]) > 1.0E-12:
        raise ValueError('The final time of the two timeseries is not the same.')

    size = np.prod(shape1)

    nn1 = x1.shape[0]
    a1 = np.reshape(x1, newshape=(nn1, size))
    t1_unique, idxs1 = np.unique(t1.ravel(), return_index=True)

    nn2 = x2.shape[0]
    a2 = np.reshape(x2, newshape=(nn2, size))
    t2_unique, idxs2 = np.unique(t2.ravel(), return_index=True)

    interp1 = interp1d(x=t1_unique, y=a1[idxs1, ...], kind='slinear', axis=0)
    interp2 = interp1d(x=t2_unique, y=a2[idxs2, ...], kind='slinear', axis=0)

    t_interp = np.linspace(t1[0], t1[-1], num_points)

    y1 = np.reshape(interp1(t_interp), newshape=(num_points,) + shape1)
    y2 = np.reshape(interp2(t_interp), newshape=(num_points,) + shape2)

    _om_assert_utils.assert_near_equal(y1, y2, tolerance=tolerance)
