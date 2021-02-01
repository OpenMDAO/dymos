import numpy as np
from scipy.interpolate import interp1d

import openmdao.utils.assert_utils as _om_assert_utils
from openmdao.utils.general_utils import warn_deprecation


def assert_check_partials(data, atol=1.0E-6, rtol=1.0E-6):
    """
    Wrapper around OpenMDAO's assert_check_partials with a dymos-specific message.

    Calls OpenMDAO's assert_check_partials but verifies that the dictionary of assertion data is
    not empty due to dymos.options['include_check_partials'] being False.

    Parameters
    ----------
    data : dict of dicts of dicts
            First key:
                is the component name;
            Second key:
                is the (output, input) tuple of strings;
            Third key:
                is one of ['rel error', 'abs error', 'magnitude', 'J_fd', 'J_fwd', 'J_rev'];

            For 'rel error', 'abs error', 'magnitude' the value is: A tuple containing norms for
                forward - fd, adjoint - fd, forward - adjoint.
            For 'J_fd', 'J_fwd', 'J_rev' the value is: A numpy array representing the computed
                Jacobian for the three different methods of computation.
    atol : float
        Absolute error. Default is 1e-6.
    rtol : float
        Relative error. Default is 1e-6.
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
        if case1_minus_case2 or case2_minus_case1:
            diff_err_msg = '\nrequire_same_vars=True but cases contain different variables.'
        if case1_minus_case2:
            diff_err_msg += f'\nVariables in case1 but not in case2: {sorted(case1_minus_case2)}'
        if case2_minus_case1:
            diff_err_msg += f'\nVariables in case2 but not in case1: {sorted(case2_minus_case1)}'

    shape_errors = set()
    val_errors = set()
    shape_err_msg = '\nThe following variables have different shapes/sizes:'
    val_err_msg = '\nThe following variables contain different values:\nvar: error'

    for var in sorted(set(case1_vars.keys()).intersection(case2_vars.keys())):
        a = case1_vars[var]['value']
        b = case2_vars[var]['value']
        if a.shape != b.shape:
            shape_errors.add(var)
            shape_err_msg += f'\n{var} has shape {a.shape} in case1 but shape {b.shape} in case2'
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


def assert_timeseries_near_equal(t1, x1, t2, x2, tolerance=1.0E-6, num_points=None):
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
    if num_points is not None:
        warn_deprecation('Argument num_points is deprecated and will be removed in dymos 1.0.0')

    shape1 = x1.shape[1:]
    shape2 = x2.shape[1:]

    if shape1 != shape2:
        raise ValueError('The shape of the variable in the two timeseries is not equal '
                         f'x1 is {shape1}  x2 is {shape2}')

    if abs(t1[0] - t2[0]) > 1.0E-12:
        raise ValueError('The initial time of the two timeseries is not the same. '
                         f't1[0]={t1[0]}  t2[0]={t2[0]}  difference: {t2[0] - t1[0]}')

    if abs(t1[-1] - t2[-1]) > 1.0E-12:
        raise ValueError('The final time of the two timeseries is not the same. '
                         f't1[0]={t1[-1]}  t2[0]={t2[-1]}  difference: {t2[-1] - t1[-1]}')

    size = np.prod(shape1)

    nn1 = x1.shape[0]
    a1 = np.reshape(x1, newshape=(nn1, size))
    t1_unique, idxs1 = np.unique(t1.ravel(), return_index=True)

    nn2 = x2.shape[0]
    a2 = np.reshape(x2, newshape=(nn2, size))
    t2_unique, idxs2 = np.unique(t2.ravel(), return_index=True)

    if nn1 > nn2:
        # The first timeseries is more dense
        t_unique = t1_unique
        x_to_interp = a1[idxs1, ...]
        t_check = t2.ravel()
        x_check = x2
    else:
        # The second timeseries is more dense
        t_unique = t2_unique
        x_to_interp = a2[idxs2, ...]
        t_check = t1.ravel()
        x_check = x1

    interp = interp1d(x=t_unique, y=x_to_interp, kind='slinear', axis=0)
    num_points = np.prod(t_check.shape)

    y_interp = np.reshape(interp(t_check), newshape=(num_points,) + shape1)

    _om_assert_utils.assert_near_equal(y_interp, x_check, tolerance=tolerance)
