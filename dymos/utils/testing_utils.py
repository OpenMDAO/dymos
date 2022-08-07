import io

import numpy as np

from scipy.interpolate import interp1d

import openmdao.api as om
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
    _case1 = case1.model if isinstance(case1, om.Problem) else case1
    _case2 = case2.model if isinstance(case2, om.Problem) else case2

    case1_vars = {t[1]['prom_name']: t[1] for t in
                  _case1.list_inputs(val=True, units=True, prom_name=True, out_stream=None)}
    case1_vars.update({t[1]['prom_name']: t[1] for t in
                       _case1.list_outputs(val=True, units=True, prom_name=True, out_stream=None)})

    case2_vars = {t[1]['prom_name']: t[1] for t in
                  _case2.list_inputs(val=True, units=True, prom_name=True, out_stream=None)}
    case2_vars.update({t[1]['prom_name']: t[1] for t in
                       _case2.list_outputs(val=True, units=True, prom_name=True, out_stream=None)})

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
    val_errors = {}
    shape_err_msg = '\nThe following variables have different shapes/sizes:'
    val_err_msg = io.StringIO()

    for var in sorted(set(case1_vars.keys()).intersection(case2_vars.keys())):
        a = case1_vars[var]['val']
        b = case2_vars[var]['val']
        if a.shape != b.shape:
            shape_errors.add(var)
            shape_err_msg += f'\n{var} has shape {a.shape} in case1 but shape {b.shape} in case2'
            continue
        err = np.abs(a - b)
        max_err = np.max(err)
        mean_err = np.mean(err)
        if np.any(max_err > tol):
            val_errors[var] = (max_err, mean_err)

    err_msg = ''
    if diff_err_msg:
        err_msg += diff_err_msg
    if shape_errors:
        err_msg += shape_err_msg
    if val_errors:
        val_err_msg.write('\nThe following variables contain different values:\n')
        max_var_len = max(3, max([len(s) for s in val_errors.keys()]))
        val_err_msg.write(f"{'var'.rjust(max_var_len)} {'max error'.rjust(16)} {'mean error'.rjust(16)}\n")
        val_err_msg.write(max_var_len * '-' + ' ' + 16 * '-' + ' ' + 16 * '-' + '\n')
        for varname, (max_err, mean_err) in val_errors.items():
            val_err_msg.write(f"{varname.rjust(max_var_len)} {max_err:16.9e} {mean_err:16.9e}\n")
        err_msg += val_err_msg.getvalue()

    if err_msg:
        raise AssertionError(err_msg)

#
#
# def assert_timeseries_near_equal_old(t1, x1, t2, x2, tolerance=1.0E-6):
#     """
#     Assert that two timeseries of data are approximately equal.
#
#     This is done by fitting a 1D interpolant to each index of each timeseries, and then comparing
#     the values of the two interpolants at some equally spaced number of points.
#
#     Parameters
#     ----------
#     t1 : np.array
#         Time values for the first timeseries.
#     x1 : np.array
#         Data values for the first timeseries.
#     t2 : np.array
#         Time values for the second timeseries.
#     x2 : np.array
#         Data values for the second timeseries.
#     tolerance : float
#         The tolerance for any errors along at each point checked.
#
#     Raises
#     ------
#     AssertionError
#         When one or more elements of the interpolated timeseries are not within the
#         desired tolerance.
#     """
#     shape1 = x1.shape[1:]
#     shape2 = x2.shape[1:]
#
#     if shape1 != shape2:
#         raise ValueError('The shape of the variable in the two timeseries is not equal '
#                          f'x1 is {shape1}  x2 is {shape2}')
#
#     if abs(t1[0] - t2[0]) > 1.0E-12:
#         raise ValueError('The initial time of the two timeseries is not the same. '
#                          f't1[0]={t1[0]}  t2[0]={t2[0]}  difference: {t2[0] - t1[0]}')
#
#     if abs(t1[-1] - t2[-1]) > 1.0E-12:
#         raise ValueError('The final time of the two timeseries is not the same. '
#                          f't1[0]={t1[-1]}  t2[0]={t2[-1]}  difference: {t2[-1] - t1[-1]}')
#
#     size = np.prod(shape1)
#
#     nn1 = x1.shape[0]
#     a1 = np.reshape(x1, newshape=(nn1, size))
#     t1_unique, idxs1 = np.unique(t1.ravel(), return_index=True)
#
#     nn2 = x2.shape[0]
#     a2 = np.reshape(x2, newshape=(nn2, size))
#     t2_unique, idxs2 = np.unique(t2.ravel(), return_index=True)
#
#     if nn1 > nn2:
#         # The first timeseries is more dense
#         t_unique = t1_unique
#         x_to_interp = a1[idxs1, ...]
#         t_check = t2.ravel()
#         x_check = x2
#     else:
#         # The second timeseries is more dense
#         t_unique = t2_unique
#         x_to_interp = a2[idxs2, ...]
#         t_check = t1.ravel()
#         x_check = x1
#
#     interp = interp1d(x=t_unique, y=x_to_interp, kind='slinear', axis=0)
#     num_points = np.prod(t_check.shape)
#
#     x_ref_interp = np.reshape(interp(t_check), newshape=(num_points,) + shape1)
#
#     _om_assert_utils.assert_near_equal(x_ref_interp, x_check, tolerance=tolerance)
#

def assert_timeseries_near_equal_v1(t1, x1, t2, x2, tolerance=1.0E-6):
    """
    Assert that two timeseries of data are approximately equal.

    This is done by fitting a 1D interpolant to each index of each timeseries, and then comparing
    the values of the two interpolants at some equally spaced number of points.

    The first timeseries, defined by t1, x1, serves as the reference.

    Parameters
    ----------
    t1 : np.array
        Time values for the reference timeseries.
    x1 : np.array
        Data values for the reference timeseries.
    t2 : np.array
        Time values for the timeseries that is compared to the reference.
    x2 : np.array
        Data values for the timeseries that is compared to the reference.
    tolerance : float
        The tolerance for any errors along at each point checked.

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
    # a2 = np.reshape(x2, newshape=(nn2, size))
    # t2_unique, idxs2 = np.unique(t2.ravel(), return_index=True)

    # Assuming that The first timeseries is the reference and therefore more dense
    t_unique = t1_unique
    x_to_interp = a1[idxs1, ...]
    t_check = t2.ravel()
    x_check = x2

    interp = interp1d(x=t_unique, y=x_to_interp, kind='slinear', axis=0)
    num_points = np.prod(t_check.shape)

    x_ref_interp = np.reshape(interp(t_check), newshape=(num_points,) + shape1)

    _om_assert_utils.assert_near_equal(x_ref_interp, x_check, tolerance=tolerance)


def _write_out_timeseries_values_out_of_tolerance(isclose, x_check, x_ref):
    err_msg = ''
    for idx, item_close in np.ndenumerate(isclose):
        if not item_close:
            err_msg += f"{idx}: {x_check[idx]} != {x_ref[idx]}\n"
    return err_msg

def assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=None, rel_tolerance=None):
    """
    Assert that two timeseries of data are approximately equal.

    The first timeseries, defined by t_ref, x_ref, serves as the reference.

    This is done by fitting a 1D interpolant to the reference, and then comparing
    the values of the two interpolants at some equally spaced number of points.


    Parameters
    ----------
    t_ref : np.array
        Time values for the reference timeseries.
    x_ref : np.array
        Data values for the reference timeseries.
    t_check : np.array
        Time values for the timeseries that is compared to the reference.
    x_check : np.array
        Data values for the timeseries that is compared to the reference.
    abs_tolerance : float
        The absolute tolerance for any errors along at each point checked.
    rel_tolerance : float
        The relative tolerance for any errors along at each point checked.

    Raises
    ------
    AssertionError
        When one or more elements of the interpolated timeseries are not within the
        desired tolerance.
    """
    shape_ref = x_ref.shape[1:]
    shape_check = x_check.shape[1:]

    if abs_tolerance is None and rel_tolerance is None:
        raise ValueError('abs_tolerance and rel_tolerance cannot be both None')

    if shape_ref != shape_check:
        raise ValueError('The shape of the variable in the two timeseries is not equal '
                         f'x_ref is {shape_ref}  x_check is {shape_check}')

    # get the overlapping time period between t_ref and t_check
    t_begin = max(t_ref[0], t_check[0])
    t_end = min(t_ref[-1], t_check[-1])

    if t_begin > t_end:
        raise ValueError("There is no overlapping time between the two time series")

    # if abs(t_ref[0] - t_check[0]) > 1.0E-12:
    #     raise ValueError('The initial time of the two timeseries is not the same. '
    #                      f't_ref[0]={t_ref[0]}  t_check[0]={t_check[0]}  difference: {t_check[0] - t_ref[0]}')
    #
    # if abs(t_ref[-1] - t_check[-1]) > 1.0E-12:
    #     raise ValueError('The final time of the two timeseries is not the same. '
    #                      f't_ref[0]={t_ref[-1]}  t_check[0]={t_check[-1]}  difference: {t_check[-1] - t_ref[-1]}')

    size = np.prod(shape_ref)

    nn1 = x_ref.shape[0]
    a_ref = np.reshape(x_ref, newshape=(nn1, size))
    t_ref_unique, idxs_ref = np.unique(t_ref.ravel(), return_index=True)

    t_unique = t_ref_unique
    x_to_interp = a_ref[idxs_ref, ...]
    t_check = t_check.ravel()
    x_check = x_check

    interp = interp1d(x=t_unique, y=x_to_interp, kind='slinear', axis=0)
    num_points = np.prod(t_check.shape)

    # only want t_check in the range of t_begin and t_end





    t_check_in_range_condition = np.logical_and(t_check >= t_begin, t_check <= t_end)
    # t_check = np.extract(t_check_in_range_condition, t_check)
    # x_check = np.extract(t_check_in_range_condition, x_check)
    t_check = np.compress(t_check_in_range_condition, t_check)
    x_check = np.compress(t_check_in_range_condition, x_check, axis=0)
    num_points = np.prod(t_check.shape)
    #



    x_ref_interp = np.reshape(interp(t_check), newshape=(num_points,) + shape_ref)

    # Need to only use values from between t_begin and t_end
    # So filter out x_check and x_ref_interp based on that
    # below_t_begin_condition = np.logical_and(t_unique >= t_begin, t_unique <= t_end)
    # x_check = np.extract(below_t_begin_condition, x_check)
    # x_ref_interp = np.extract(below_t_begin_condition, x_ref_interp)

    if abs_tolerance is None:
        # need to use rel_tolerance
        # _om_assert_utils.assert_near_equal(x_ref_interp, x_check, tolerance=rel_tolerance)
        isclose = np.isclose(x_check, x_ref_interp, rtol=rel_tolerance, atol=0.0)

        all_close = np.all(isclose)

        # all_close = np.allclose(x_check, x_ref_interp, rtol=rel_tolerance, atol=0.0)
        if not all_close:
            err_msg = f"timeseries not equal within relative tolerance of {rel_tolerance}\n" + \
                      "The following values are out of tolerance:\n"
            out_of_tol_msg = _write_out_timeseries_values_out_of_tolerance(isclose, x_check, x_ref_interp)
            err_msg + out_of_tol_msg
            raise AssertionError(err_msg)
        # assert all_close, f"timeseries not equal within relative tolerance of {rel_tolerance}"
    elif rel_tolerance is None:
        # need to use abs_tolerance
        # error = x_check - x_ref_interp
        # For assert_near_equal, actual followed by desired. If `desired` is zero, then use absolute error
        # _om_assert_utils.assert_near_equal(error, 0.0, tolerance=abs_tolerance)
        # all_close = np.allclose(x_check, x_ref_interp, rtol=0.0, atol=abs_tolerance)
        isclose = np.isclose(x_check, x_ref_interp, rtol=0.0, atol=abs_tolerance)

        all_close = np.all(isclose)
        if not all_close:
            err_msg = f"timeseries not equal within absolute tolerance of {abs_tolerance}\n" + \
                      "The following values are out of tolerance:\n"
            out_of_tol_msg = _write_out_timeseries_values_out_of_tolerance(isclose, x_check, x_ref_interp)
            err_msg + out_of_tol_msg
            raise AssertionError(err_msg)

    else:
        # need to use a hybrid of abs and rel
        transition_tolerance = abs_tolerance / rel_tolerance

        # for values > transition_tolerance, use rel_tolerance
        transition_condition = abs(x_ref_interp) >= transition_tolerance
        above_transition_x_ref_interp = np.extract(transition_condition, x_ref_interp)
        above_transition_x_check = np.extract(transition_condition, x_check)
        # _om_assert_utils.assert_near_equal(above_transition_x_ref_interp, above_transition_x_check,
        #                                    tolerance=rel_tolerance)


        # all_close = np.allclose(above_transition_x_check, above_transition_x_ref_interp, rtol=rel_tolerance, atol=0.0)

        # TODO ???? Need to handle the fact that these arrays were extracted!
        isclose = np.isclose(above_transition_x_check, above_transition_x_ref_interp, rtol=rel_tolerance, atol=0.0)
        all_close = np.all(isclose)
        if not all_close:
            err_msg = f"timeseries not equal within relative tolerance of {rel_tolerance}\n" + \
                      "The following values are out of tolerance:\n"
            out_of_tol_msg = _write_out_timeseries_values_out_of_tolerance(isclose, x_check, x_ref_interp)
            err_msg + out_of_tol_msg
            raise AssertionError(err_msg)

        # for values < transition_tolerance, use abs_tolerance
        transition_condition = abs(x_ref_interp) < transition_tolerance
        below_transition_x_ref_interp = np.extract(transition_condition, x_ref_interp)
        below_transition_x_check = np.extract(transition_condition, x_check)

        error = below_transition_x_check - below_transition_x_ref_interp
        # _om_assert_utils.assert_near_equal(error, np.zeros(error.shape), tolerance=abs_tolerance)
        # all_close = np.allclose(below_transition_x_check, below_transition_x_ref_interp, rtol=0.0, atol=abs_tolerance)
        isclose = np.isclose(below_transition_x_check, below_transition_x_ref_interp, rtol=0.0, atol=abs_tolerance)
        all_close = np.all(isclose)
        if not all_close:
            err_msg = f"timeseries not equal within absolute tolerance of {abs_tolerance}\n" + \
                      "The following values are out of tolerance:\n"
            out_of_tol_msg = _write_out_timeseries_values_out_of_tolerance(isclose, x_check, x_ref_interp)
            err_msg + out_of_tol_msg
            raise AssertionError(err_msg)

        # assert all_close, f"timeseries not equal within absolute tolerance of {abs_tolerance}"
