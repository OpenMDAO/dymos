import io

from packaging.version import Version

import numpy as np

from scipy.interpolate import Akima1DInterpolator

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
        val_err_msg.write(
            f"{'var'.rjust(max_var_len)} {'max error'.rjust(16)} {'mean error'.rjust(16)}\n")
        val_err_msg.write(max_var_len * '-' + ' ' + 16 * '-' + ' ' + 16 * '-' + '\n')
        for varname, (max_err, mean_err) in val_errors.items():
            val_err_msg.write(f"{varname.rjust(max_var_len)} {max_err:16.9e} {mean_err:16.9e}\n")
        err_msg += val_err_msg.getvalue()

    if err_msg:
        raise AssertionError(err_msg)


def _write_out_timeseries_values_out_of_tolerance(isclose, rel_tolerance, abs_tolerance,
                                                  t_check, x_check, x_ref):
    """
    Helper function used to write out a table of values indicating which timeseries values
    were out of tolerance.

    Parameters
    ----------
    isclose : array of bool
        Boolean array indicating where data value is in tolerance. Has same shape as the
        time series array
    rel_tolerance : float
        Allowed relative tolerance error
    abs_tolerance : float
        Allowed absolute tolerance error
    t_check : np.array
        Array of time values for the timeseries
    x_check : np.array
        Array of data values for the timeseries to be check/compared to the reference value, x_ref
    x_ref : np.array
        Array of data values for the timeseries to be used as the reference
    """
    err_msg = f"The following timeseries data are out of tolerance due to absolute (" \
              f"{abs_tolerance}) or relative ({rel_tolerance}) tolerance violations\n"
    header = f"{'time_index':10s} | " + \
             f"{'data_indices':12s} | " + \
             f"{'time':13s} | " + \
             f"{'ref_data':13s} | " + \
             f"{'checked_data':13s} | " + \
             f"{'abs_error':13s} | " + \
             f"{'rel_error':13} | " + \
             " ABS or REL error "
    err_msg += f"{header}\n"
    err_msg += len(header) * '-' + '\n'

    rel_error_max = 0.0
    err_line_max = 0
    for idx, item_close in np.ndenumerate(isclose):
        if not item_close:
            error_string = ''
            abs_error = abs(x_check[idx] - x_ref[idx])
            if x_ref[idx] != 0.0:
                rel_error = abs(x_check[idx] - x_ref[idx]) / abs(x_ref[idx])
            else:
                rel_error = float('nan')
            if abs_tolerance is not None:
                if abs_error > abs_tolerance:
                    error_string += ' >ABS_TOL'

            if rel_tolerance is not None:
                if rel_error > rel_tolerance:
                    error_string += ' >REL_TOL'

            err_line = f"{idx[0]:10,d} | {str(idx[1:]):>12s} | {t_check[idx[0]]:13.6e} |" \
                       f"{x_ref[idx]:13.6e} | {x_check[idx]:13.6e} | {abs_error:13.6e} | " \
                       f"{rel_error:13.6e} | {error_string}\n"
            err_msg += err_line

            if rel_error > rel_error_max:
                rel_error_max = rel_error
                err_line_max = err_line

    # show the item with the max rel error
    max_rel_error_header_txt = 'Time series data value with the largest relative error'
    max_rel_error_msg = f"\n{len(max_rel_error_header_txt) * '#'}\n{max_rel_error_header_txt}\n" \
                        f"{len(max_rel_error_header_txt) * '#'}\n"

    max_rel_error_msg += f"{header}\n"
    max_rel_error_msg += len(header) * '-' + '\n'
    max_rel_error_msg += err_line_max

    err_msg += max_rel_error_msg

    return err_msg


def assert_timeseries_near_equal(t_ref, x_ref, t_check, x_check, abs_tolerance=None,
                                 rel_tolerance=None):
    """
    Assert that two timeseries of data are approximately equal.

    The first timeseries, defined by t_ref, x_ref, serves as the reference.

    The second timeseries, defined by t_check, x_check is what is checked for near equality.

    The check is done by fitting a 1D interpolant to the reference, and then comparing
    the values of the interpolant at the times in t_check. The check for errors within
    tolerance are done on a point-by-point basis. If any point is out of tolerance, throw
    an AssertionError.

    Only the times where the two timeseries overlap are used for the check.

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
        When one or more elements of the timeseries to be checked are not with in the desired
        tolerance of the interpolated reference timeseries, an AssertionError is raised.
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
    num_elements = np.prod(shape_ref, dtype=int)
    time_series_len = x_ref.shape[0]
    x_ref_data_flattened = np.reshape(x_ref, (time_series_len, num_elements))
    t_ref_unique, idxs_unique_ref = np.unique(t_ref.ravel(), return_index=True)
    x_to_interp = x_ref_data_flattened[idxs_unique_ref, ...]
    t_check = t_check.ravel()

    interp = Akima1DInterpolator(t_ref_unique, x_to_interp)

    # only want t_check in the overlapping range of t_begin and t_end
    t_check_in_range_condition = np.logical_and(t_check >= t_begin, t_check <= t_end)
    t_check = np.compress(t_check_in_range_condition, t_check)
    x_check = np.compress(t_check_in_range_condition, x_check, axis=0)

    # get the interpolated values of the reference at the values of t_check
    # Reshape back to unflattened data values
    x_ref_interp = np.reshape(interp(t_check), (t_check.size,) + shape_ref)

    if abs_tolerance is None:  # so only have rel_tolerance
        isclose = np.isclose(x_check, x_ref_interp, rtol=rel_tolerance, atol=0.0)
        all_close = np.all(isclose)
        if not all_close:
            err_msg = _write_out_timeseries_values_out_of_tolerance(isclose,
                                                                    rel_tolerance,
                                                                    abs_tolerance,
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
                                                                    rel_tolerance,
                                                                    abs_tolerance,
                                                                    t_check,
                                                                    x_check,
                                                                    x_ref_interp,
                                                                    )
            raise AssertionError(err_msg)
    else:  # need to use a hybrid of abs and rel tolerances
        err_msg = ''

        # At what value of absolute value of the data does the tolerance check switch between
        #    using the absolute vs relative tolerance
        transition_tolerance = abs_tolerance / rel_tolerance

        # for values > transition_tolerance, use rel_tolerance
        transition_condition = abs(x_ref_interp) >= transition_tolerance
        above_transition_x_ref_interp = np.full(x_ref_interp.shape, np.nan)
        np.copyto(above_transition_x_ref_interp, x_ref_interp, where=transition_condition)
        above_transition_x_check = np.full(x_ref_interp.shape, np.nan)
        np.copyto(above_transition_x_check, x_check, where=transition_condition)
        isclose_using_rel_tolerance = np.isclose(above_transition_x_check,
                                                 above_transition_x_ref_interp,
                                                 rtol=rel_tolerance, atol=0.0, equal_nan=True)

        # for values < transition_tolerance, use abs_tolerance
        transition_condition = abs(x_ref_interp) < transition_tolerance
        below_transition_x_ref_interp = np.full(x_ref_interp.shape, np.nan)
        np.copyto(below_transition_x_ref_interp, x_ref_interp, where=transition_condition)
        below_transition_x_check = np.full(x_ref_interp.shape, np.nan)
        np.copyto(below_transition_x_check, x_check, where=transition_condition)
        isclose_using_abs_tolerance = np.isclose(below_transition_x_check,
                                                 below_transition_x_ref_interp, rtol=0.0,
                                                 atol=abs_tolerance, equal_nan=True)

        # combine the two
        isclose_using_both_tolerance = isclose_using_rel_tolerance & isclose_using_abs_tolerance
        all_close = np.all(isclose_using_both_tolerance)
        if not all_close:
            err_msg += _write_out_timeseries_values_out_of_tolerance(isclose_using_both_tolerance,
                                                                     rel_tolerance,
                                                                     abs_tolerance,
                                                                     t_check,
                                                                     x_check,
                                                                     x_ref_interp,
                                                                     )
        if err_msg:
            raise AssertionError(err_msg)


def _get_reports_dir(prob):
    # need this to work with older OM versions with old reports system API
    # reports API changed between 3.18 and 3.19, so handle it here in order to be able to
    #  test against older versions of openmdao
    if Version(openmdao_version) > Version("3.18"):
        return prob.get_reports_dir()

    from openmdao.utils.reports_system import get_reports_dir
    return get_reports_dir(prob)


class PhaseStub():
    """
    A stand-in for the Phase during config_io for testing.

    It just supports the classify_var method and returns "ode", the only value needed for unittests.
    """
    def __init__(self):
        self.nonlinear_solver = None
        self.linear_solver = None

    def classify_var(self, name):
        """
        A stand-in for classify_var that always sets the variable type to name.

        Parameters
        ----------
        name : str
            The name of the variable to classify.

        Returns
        -------
        str
            The variable classification.
        """
        return 'ode'


class SimpleODE(om.ExplicitComponent):
    """
    A simple ODE for testing purposes.

    Source: https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Component options.
    """
    def initialize(self):
        """
        Declare options for SimpleODE.
        """
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        """
        Add inputs and outputs to SimpleODE.
        """
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='s**2')
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('p', shape=(nn,), units='s**2')

        self.add_output('x_dot', shape=(nn,), units='s')

        ar = np.arange(nn, dtype=int)
        self.declare_partials(of='x_dot', wrt='x', rows=ar, cols=ar, val=1.0)
        self.declare_partials(of='x_dot', wrt='t', rows=ar, cols=ar)
        self.declare_partials(of='x_dot', wrt='p', rows=ar, cols=ar, val=1.0)

    def compute(self, inputs, outputs):
        """
        Compute the outputs of SimpleVectorizedODE.

        Parameters
        ----------
        inputs : Vector
            Vector of inputs.
        outputs : Vector
            Vector of outputs.
        """
        x = inputs['x']
        t = inputs['t']
        p = inputs['p']
        outputs['x_dot'] = x - t**2 + p

    def compute_partials(self, inputs, partials):
        """
        Compute the partials of SimpleVectorizedODE.

        Parameters
        ----------
        inputs : Vector
            Vector of inputs.
        partials : Dictionary
            Vector of partials.
        """
        t = inputs['t']
        partials['x_dot', 't'] = -2*t


class SimpleVectorizedODE(om.ExplicitComponent):
    """
    A simple vector-valued ODE.

    Source: https://math.okstate.edu/people/yqwang/teaching/math4513_fall11/Notes/rungekutta.pdf

    Parameters
    ----------
    **kwargs : dict of keyword arguments
        Keyword arguments that will be mapped into the Component options.
    """
    def initialize(self):
        """
        Declare options for SimpleVectorizedODE.
        """
        self.options.declare('num_nodes', types=(int,))

    def setup(self):
        """
        Add inputs and outputs to SimpleVectorizedODE.
        """
        nn = self.options['num_nodes']
        self.add_input('z', shape=(nn, 2), units='s**2')
        self.add_input('t', shape=(nn,), units='s')
        self.add_input('p', shape=(nn,), units='s**2')

        self.add_output('z_dot', shape=(nn, 2), units='s')

        cs = np.repeat(np.arange(nn, dtype=int), 2)
        ar2 = np.arange(2 * nn, dtype=int)
        dzdot_dz_pattern = np.arange(2 * nn, step=2, dtype=int)
        self.declare_partials(of='z_dot', wrt='z', rows=dzdot_dz_pattern, cols=dzdot_dz_pattern, val=1.0)
        self.declare_partials(of='z_dot', wrt='t', rows=ar2, cols=cs)
        dzdot_dp_rows = np.arange(2 * nn, step=2, dtype=int)
        dzdot_dp_cols = np.arange(nn, dtype=int)
        self.declare_partials(of='z_dot', wrt='p', rows=dzdot_dp_rows, cols=dzdot_dp_cols, val=1.0)

    def compute(self, inputs, outputs):
        """
        Compute the outputs of SimpleVectorizedODE.

        Parameters
        ----------
        inputs : Vector
            Vector of inputs.
        outputs : Vector
            Vector of outputs.
        """
        z = inputs['z']
        t = inputs['t']
        p = inputs['p']
        outputs['z_dot'][:, 0] = z[:, 0] - t**2 + p
        outputs['z_dot'][:, 1] = 10 * t

    def compute_partials(self, inputs, partials):
        """
        Compute the partials of SimpleVectorizedODE.

        Parameters
        ----------
        inputs : Vector
            Vector of inputs.
        partials : Dictionary
            Vector of partials.
        """
        t = inputs['t']
        partials['z_dot', 't'][0::2] = -2*t
        partials['z_dot', 't'][1::2] = 10
