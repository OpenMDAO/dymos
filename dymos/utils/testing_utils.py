import io
import pathlib
from packaging.version import Version

import numpy as np

from scipy.interpolate import interp1d

import openmdao.api as om
import openmdao.utils.assert_utils as _om_assert_utils
from openmdao import __version__ as openmdao_version


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

def _write_out_timeseries_values_out_of_tolerance(isclose, tolerance_type, tolerance,
                                                  t_check, x_check, x_ref):
    err_msg = f"\nTimeseries data not equal within {tolerance_type} tolerance of {tolerance}\n" + \
              "The following timeseries data are out of tolerance.\n" + \
              "  The data is shown as:\n"
    header = f"{'time_index':10s} | " + \
             f"{'data_indices':12s} | " + \
             f"{'time':13s} | " + \
             f"{'ref_data':13s} | " + \
             f"{'checked_data':13s} | " + \
             f"{'abs_error':14s} | " + \
             f"{'rel_error':14s}\n"
    err_msg += header
             # f"time_index | data_indices |   time       | ref_data       | checked_data       | abs     |        err\n"
    for idx, item_close in np.ndenumerate(isclose):
        if not item_close:
            if tolerance_type == 'absolute':
                abs_error_indicator = '*'
                rel_error_indicator = ' '
            else:
                abs_error_indicator = ' '
                rel_error_indicator = '*'
            abs_error = abs(x_check[idx] - x_ref[idx])
            rel_error = abs(x_check[idx] - x_ref[idx]) / abs(x_ref[idx])
            err_msg += f"{idx[0]:10,d} | {str(idx[1:]):>12s} | {t_check[idx[0]]:13.6e} | {x_ref[idx]:13.6e} | {x_check[idx]:13.6e} | {abs_error:13.6e}{abs_error_indicator} | {rel_error:13.6e}{rel_error_indicator}\n"
    return err_msg

def assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=None,
                                 rel_tolerance=None):
    """
    Assert that two timeseries of data are approximately equal.

    The first timeseries, defined by t_ref, x_ref, serves as the reference.

    The second timeseries, defined by t_check, x_check is what is checked for near equality.

    The check is done by fitting a 1D interpolant to the reference, and then comparing
    the values of the interpolant at the times in t_check.

    Only where the time series overlap is used for the check.

    When both abs_tolerance and rel_tolerance are given, only one is actually used for any given
    data point. When the absolute values of the data values are small, the abs_tolerance is used,
    otherwise the rel_tolerance is used. The transition point is given by

        abs_tolerance / rel_tolerance

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
    # get shapes for the time series values
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

    # Flatten the timeseries data arrays
    num_elements = np.prod(shape_ref)
    time_series_len = x_ref.shape[0]
    x_ref_data_flattened = np.reshape(x_ref, newshape=(time_series_len, num_elements))
    t_ref_unique, idxs_unique_ref = np.unique(t_ref.ravel(), return_index=True)
    x_to_interp = x_ref_data_flattened[idxs_unique_ref, ...]
    t_check = t_check.ravel()

    interp = interp1d(x=t_ref_unique, y=x_to_interp, kind='slinear', axis=0)
    # num_points = np.prod(t_check.shape)

    # only want t_check in the overlapping range of t_begin and t_end
    t_check_in_range_condition = np.logical_and(t_check >= t_begin, t_check <= t_end)
    t_check = np.compress(t_check_in_range_condition, t_check)
    x_check = np.compress(t_check_in_range_condition, x_check, axis=0)

    # get the interpolated values of the reference at the values of t_check
    # Reshape back to unflattened data values
    x_ref_interp = np.reshape(interp(t_check), newshape=(t_check.size,) + shape_ref)

    if abs_tolerance is None:  # so only have rel_tolerance
        isclose = np.isclose(x_check, x_ref_interp, rtol=rel_tolerance, atol=0.0)
        all_close = np.all(isclose)
        if not all_close:
            err_msg = _write_out_timeseries_values_out_of_tolerance(isclose,
                                                                           'relative',
                                                                           rel_tolerance,
                                                                           t_check,
                                                                           x_check,
                                                                           x_ref_interp,
                                                                           )
            raise AssertionError(err_msg)
    elif rel_tolerance is None:  # so only have abs_tolerance
        isclose = np.isclose(x_check, x_ref_interp, rtol=0.0, atol=abs_tolerance)
        all_close = np.all(isclose)
        if not all_close:
            err_msg = _write_out_timeseries_values_out_of_tolerance(isclose,
                                                                           'absolute',
                                                                           abs_tolerance,
                                                                           t_check,
                                                                           x_check,
                                                                           x_ref_interp,
                                                                           )
            raise AssertionError(err_msg)
    else:  # need to use a hybrid of abs and rel

        err_msg = ''

        # At what value of x does the check switch between using the absolute vs relative tolerance
        transition_tolerance = abs_tolerance / rel_tolerance

        # for values > transition_tolerance, use rel_tolerance
        transition_condition = abs(x_ref_interp) >= transition_tolerance

        above_transition_x_ref_interp = np.full(x_ref_interp.shape, np.nan)
        np.copyto(above_transition_x_ref_interp, x_ref_interp, where=transition_condition)
        above_transition_x_check = np.full(x_ref_interp.shape, np.nan)
        np.copyto(above_transition_x_check, x_check, where=transition_condition)


        # old way
        # above_transition_x_ref_interp = np.extract(transition_condition, x_ref_interp)
        # above_transition_x_check = np.extract(transition_condition, x_check)

        # TODO ???? Need to handle the fact that these arrays were extracted!
        isclose = np.isclose(above_transition_x_check, above_transition_x_ref_interp,
                             rtol=rel_tolerance, atol=0.0, equal_nan=True)




        all_close = np.all(isclose)
        if not all_close:
            err_msg += _write_out_timeseries_values_out_of_tolerance(isclose,
                                                                    'relative',
                                                                    rel_tolerance,
                                                                    t_check,
                                                                    x_check,
                                                                    x_ref_interp,
                                                                    )
        # for values < transition_tolerance, use abs_tolerance
        transition_condition = abs(x_ref_interp) < transition_tolerance

        below_transition_x_ref_interp = np.full(x_ref_interp.shape, np.nan)
        np.copyto(below_transition_x_ref_interp, x_ref_interp, where=transition_condition)
        below_transition_x_check = np.full(x_ref_interp.shape, np.nan)
        np.copyto(below_transition_x_check, x_check, where=transition_condition)




        # below_transition_x_ref_interp = np.extract(transition_condition, x_ref_interp)
        # below_transition_x_check = np.extract(transition_condition, x_check)

        # error = below_transition_x_check - below_transition_x_ref_interp
        isclose = np.isclose(below_transition_x_check, below_transition_x_ref_interp, rtol=0.0,
                             atol=abs_tolerance, equal_nan=True)
        all_close = np.all(isclose)
        if not all_close:
            err_msg += _write_out_timeseries_values_out_of_tolerance(isclose,
                                                                           'absolute',
                                                                           abs_tolerance,
                                                                           t_check,
                                                                           x_check,
                                                                           x_ref_interp,
                                                                           )
        if err_msg:
            raise AssertionError(err_msg)

def _get_reports_dir(prob):
    # need this to work with older OM versions with old reports system API
    # reports API changed between 3.18 and 3.19, so handle it here in order to be able to test against older
    # versions of openmdao
    if Version(openmdao_version) > Version("3.18"):
        return prob.get_reports_dir()

    from openmdao.utils.reports_system import get_reports_dir
    return get_reports_dir(prob)
