import numpy as np

from scipy.interpolate import interp1d

from numpy.testing import assert_array_less

import openmdao.api as om
from openmdao.utils.array_utils import shape_to_len
import openmdao.utils.assert_utils as _om_assert_utils


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
        a = case1_vars[var]['val']
        b = case2_vars[var]['val']
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


def assert_timeseries_near_equal(t1, x1, t2, x2, tolerance=None, atol=1.0E-2, rtol=1.0E-2, check_time=True):
    """
    Assert that two timeseries of data are approximately equal.

    The more temporally-dense timeseries is always interpolated onto the times of the less dense series that
    fall into the region where the times of the two series overlap.

    When testing time, the first timeseries is considered the "true" value.

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
        Deprecated. This input is replaced by atol and rtol.
    atol : float
        Absolute tolerance for error in the timeseries value at each point.
    rtol : float
        Relative tolerance for error in the timeseries value at each point.
    check_time : bool
        If True, assert that the start and end times of the two timeseries are within the specified tolerances.

    Raises
    ------
    AssertionError
        When one or more elements of the interpolated timeseries are not within the
        desired tolerance.
    """
    shape1 = x1.shape[1:]
    shape2 = x2.shape[1:]

    if shape1 != shape2:
        raise ValueError('The shape of the variable in the two timeseries is not equal '
                         f'x1 is {shape1}  x2 is {shape2}')

    if tolerance is not None:
        om.issue_warning('The tolerance argument in assert_timeseries_near_equal is deprecated. Please specify'
                         'atol and rtol.', om.OMDeprecationWarning)
        rtol = tolerance
        atol = tolerance

    if check_time:
        assert np.all(np.abs(t1[0] - t2[0]) < rtol * np.abs(t1[0]) + atol), \
            'The initial value of time in the two timeseries differ by more than the allowable tolerance.\n'\
            f't1_initial: {t1[0]}  t2_initial: {t2[0]}\n'\
            'Pass argument `check_time=False` to ignore this error and only compare the values in the two timeseries ' \
            'in the overlapping region of time.'

        assert np.all(np.abs(t1[-1] - t2[-1]) < rtol * np.abs(t1[-1]) + atol), \
            'The final value of time in the two timeseries differ by more than the allowable tolerance.\n'\
            f't1_final: {t1[-1]}  t2_final: {t2[-1]}\n'\
            'Pass argument `check_time=False` to ignore this error and only compare the values in the two timeseries ' \
            'in the overlapping region of time.'

    size = np.prod(shape1)

    nn1 = x1.shape[0]
    a1 = np.reshape(x1, newshape=(nn1, size))
    t1_unique, idxs1 = np.unique(t1.ravel(), return_index=True)

    nn2 = x2.shape[0]
    a2 = np.reshape(x2, newshape=(nn2, size))
    t2_unique, idxs2 = np.unique(t2.ravel(), return_index=True)

    # The interval in which the two timeseries overlap.
    t_overlap = (max(t1[0], t2[0]), min(t1[-1], t2[-1]))

    t1_overlap_idxs = np.where(np.logical_and(t_overlap[0] < t1_unique, t1_unique < t_overlap[1]))[0]
    t2_overlap_idxs =  np.where(np.logical_and(t_overlap[0] < t2_unique, t2_unique < t_overlap[1]))[0]

    if len(t1_overlap_idxs) == 0:
        raise ValueError(f'There are no values for the first timeseries in the region of overlapping time {t_overlap}')
    if len(t2_overlap_idxs) == 0:
        raise ValueError(f'There are no values for the first timeseries in the region of overlapping time {t_overlap}')

    if nn1 > nn2:
        # The first timeseries is more dense
        t_to_interp = t1_unique[t1_overlap_idxs]
        x_to_interp = a1[idxs1[t1_overlap_idxs], ...]
        t_check = t2_unique[t2_overlap_idxs]
        x_check = x2[t2_overlap_idxs, ...]
    else:
        # The second timeseries is more dense
        t_to_interp = t2_unique[t2_overlap_idxs]
        x_to_interp = a2[idxs2[t2_overlap_idxs], ...]
        t_check = t1_unique[t1_overlap_idxs]
        x_check = x1[t1_overlap_idxs, ...]

    interp = interp1d(x=t_to_interp, y=x_to_interp, kind='slinear', axis=0, bounds_error=False, fill_value='extrapolate')
    num_points = shape_to_len(t_check.shape)
    x_interp = np.reshape(interp(t_check), newshape=(num_points,) + shape1)

    error_calc = np.abs(x_check - x_interp) < rtol * np.abs(x_check) + atol

    if not error_calc.all():
        max_err_idx = np.argmax(np.abs(x_check - x_interp) - rtol * np.abs(x_check) + atol)
        x1_at_err = x_interp[max_err_idx] if nn1 > nn2 else x_check[max_err_idx]
        x2_at_err = x_check[max_err_idx] if nn1 > nn2 else x_interp[max_err_idx]
        abs_err = np.abs(x_check[max_err_idx] - x_interp[max_err_idx])
        rel_err = abs_err / np.abs(x_check[max_err_idx])

        msg = f'The two timeseries do not agree to the specified tolerance (atol: {atol} rtol: {rtol}).\n' \
              'The largest discrepancy is:\n' \
              f'time: {np.round(t_check[max_err_idx], 6)}\n' \
              f'x1: {np.round(x1_at_err, 6)}\n' \
              f'x2: {np.round(x2_at_err, 6)}\n' \
              f'rel err: {np.round(rel_err, 6)}\n' \
              f'abs err: {np.round(abs_err, 6)}'

        assert np.all(np.abs(x_check - x_interp) < rtol * np.abs(x_check) + atol), msg
