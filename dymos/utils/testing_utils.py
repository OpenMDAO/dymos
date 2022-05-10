import math
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


def assert_timeseries_near_equal(t_nom, x_nom, t_check, x_check, tolerance=None, atol=1.0E-2, rtol=1.0E-2,
                                 atol_cutoff=1.0E-6, assert_time=True):
    """
    Assert that two timeseries of data are approximately equal.

    Parameters
    ----------
    t_nom : np.array
        Time values for the nominal timeseries.
    x_nom : np.array
        Data values for the nominal timeseries.
    t_check : np.array
        Time values for the timeseries to test.
    x_check : np.array
        Data values for the timeseries to test.
    tolerance : float
        The tolerance for any errors along at each point checked.
        Deprecated. This input is replaced by atol and rtol.
    atol : float
        Absolute tolerance for error in the timeseries value at each point.  atol is only used when the absolute
        value of x_nom is below the threshold value givenby atol_cutoff.
    rtol : float
        Relative tolerance for error in the timeseries value at each point.
    atol_cutoff : float
        If abs(x_nom) is above this value, check the relative error.  Otherwise, check the absolute error.
    assert_time : bool
        If True, assert that the start and end times of the two timeseries are within the specified tolerances.

    Raises
    ------
    AssertionError
        When one or more elements of the interpolated timeseries are not within the desired tolerance.

    Warns
    -----
    UserWarning
        UserWarning is raised if the second timeseries is less temporally-dense than the first timeseries. Since it is
        interpolated onto the times defining the first timeseries, a sparse timeseries defined by (t, x) can lead to
        errors due to interpolation.
    """
    shape1 = x_nom.shape[1:]
    shape2 = x_check.shape[1:]

    if shape1 != shape2:
        raise ValueError('The shape of the variable in the two timeseries is not equal '
                         f'x1 is {shape1}  x2 is {shape2}')

    size = shape_to_len(shape1)

    if tolerance is not None:
        om.issue_warning('The tolerance argument in assert_timeseries_near_equal is deprecated. Please specify '
                         'atol and rtol.', om.OMDeprecationWarning)
        rtol = tolerance
        atol = tolerance

    sigfigs = int(max(-math.log10(atol), -math.log10(rtol)))
    format_str = f'{{:0.{sigfigs}f}}'

    with np.printoptions():#formatter={'float': format_str.format}):

        if assert_time:
            assert np.all(np.abs(t_check[0] - t_nom[0]) < rtol * np.abs(t_nom[0]) + atol), \
                'The initial value of time in the two timeseries differ by more than the allowable tolerance.\n'\
                f't_nom_initial: {t_nom[0]}  t_check_initial: {t_check[0]}\n'\
                'Pass argument `assert_time=False` to ignore this error and only compare the values in the two ' \
                'timeseries in the overlapping region of time.'

            assert np.all(np.abs(t_check[-1] - t_nom[-1]) < rtol * np.abs(t_nom[-1]) + atol), \
                'The final value of time in the two timeseries differ by more than the allowable tolerance.\n'\
                f't_nom_final: {t_nom[-1]}  t_check_final: {t_check[-1]}\n'\
                'Pass argument `assert_time=False` to ignore this error and only compare the values in the two ' \
                'timeseries in the overlapping region of time.'

        nn_nom = x_nom.shape[0]
        a1 = np.reshape(x_nom, newshape=(nn_nom, size))
        t_nom_unique, idxs_nom_unique = np.unique(t_nom.ravel(), return_index=True)

        nn_check = x_check.shape[0]
        a2 = np.reshape(x_check, newshape=(nn_check, size))
        t_check_unique, idxs_check_unique = np.unique(t_check.ravel(), return_index=True)

        # The interval in which the two timeseries overlap.
        t_overlap = (max(t_nom[0], t_check[0]), min(t_nom[-1], t_check[-1]))

        t_nom_overlap_idxs = np.where(np.logical_and(t_overlap[0] <= t_nom_unique, t_nom_unique <= t_overlap[1]))[0]
        t_check_overlap_idxs = np.where(np.logical_and(t_overlap[0] <= t_check_unique, t_check_unique <= t_overlap[1]))[0]

        if len(t_nom_overlap_idxs) == 0:
            raise ValueError(f'There are no values for the nominsl timeseries in the region of overlapping '
                             f'time {t_overlap}')
        elif len(t_check_overlap_idxs) == 0:
            raise ValueError(f'There are no values for the check timeseries in the region of overlapping '
                             f'time {t_overlap}')
        elif t_check_overlap_idxs.size < t_nom_overlap_idxs.size:
            om.issue_warning(f't_check has fewer values on the overlapping interval {t_overlap} than t_nom. '
                             'Interpolation errors may result.')

        # The second timeseries is more dense
        t_to_interp = t_check_unique
        x_to_interp = a2[idxs_check_unique, ...]
        t_to_test = t_nom_unique[t_nom_overlap_idxs]
        x_to_test = x_nom[idxs_nom_unique[t_nom_overlap_idxs], ...]

        interp = interp1d(x=t_to_interp, y=x_to_interp, kind='slinear', axis=0, bounds_error=False, fill_value='extrapolate')
        num_points = shape_to_len(t_to_test.shape)
        x_interp = np.reshape(interp(t_to_test), newshape=(num_points,) + shape1)

        abs_idxs = np.where(np.abs(x_to_test) <= atol_cutoff)[0]
        rel_idxs = np.where(np.abs(x_to_test) > atol_cutoff)[0]

        abs_err = np.abs(x_to_test[abs_idxs] - x_interp[abs_idxs])
        rel_err = np.abs(x_to_test[rel_idxs] - x_interp[rel_idxs]) / np.abs(x_to_test[rel_idxs])

        if np.any(rel_err > rtol):
            idx_max_rel_err = np.argmax(rel_err)
            msg = f'Relative error between timeseries exceeded tolerance ({rtol:0.{sigfigs}f})\n' \
                  f'time: {t_to_test[idx_max_rel_err]:0.{sigfigs}f}\n' \
                  f'rel_err: {rel_err[idx_max_rel_err]}\n' \
                  f'x_nom: {x_to_test[rel_idxs][idx_max_rel_err]}\n' \
                  f'x_check (interpolated): {x_interp[rel_idxs][idx_max_rel_err]}'

            assert np.all(rel_err <= rtol), msg

        if np.any(abs_err > atol):
            idx_max_abs_err = np.argmax(abs_err)
            msg = f'Absolute error between timeseries exceeded tolerance ({atol:0.{sigfigs}f})\n' \
                  f'time: {t_to_test[idx_max_abs_err]:0.{sigfigs}f}\n' \
                  f'abs_err: {abs_err[idx_max_abs_err]}\n' \
                  f'x_nom: {x_to_test[abs_idxs][idx_max_abs_err]}\n' \
                  f'x_check (interpolated): {x_interp[abs_idxs][idx_max_abs_err]}'

            assert np.all(abs_err <= atol), msg
