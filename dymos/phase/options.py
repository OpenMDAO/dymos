from collections.abc import Iterable
from numbers import Number
import numpy as np

import openmdao.api as om
from ..utils.misc import _unspecified


class ControlOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary specific to controls.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(ControlOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str,
                     desc='The name of ODE system parameter to be controlled.')

        self.declare(name='units', default=_unspecified,
                     allow_none=True, desc='The units in which the control variable is defined.')

        self.declare(name='desc', types=str, default='',
                     desc='The description of the control variable.')

        self.declare(name='opt', default=True, types=bool,
                     desc='If True, the control value will be a design variable '
                          'for the optimization problem.  If False, allow the '
                          'control to be connected externally.')

        self.declare(name='fix_initial', types=bool, default=False,
                     desc='If True, the initial value of this control is fixed and not a '
                          'design variable. This option is invalid if opt=False.')

        self.declare(name='fix_final', types=bool, default=False,
                     desc='If True, the final value of this control is fixed and not a '
                          'design variable. This option is invalid if opt=False.')

        self.declare(name='targets', allow_none=True, default=_unspecified,
                     desc='Targets in the ODE to which the state is connected')

        self.declare(name='rate_targets', allow_none=True, default=_unspecified,
                     desc='The targets in the ODE to which the control rate is connected')

        self.declare(name='rate2_targets', allow_none=True, default=_unspecified,
                     desc='The targets in the ODE to which the control 2nd derivative '
                          'is connected')

        self.declare(name='val', types=(Iterable, np.ndarray, Number), default=np.zeros(1),
                     desc='The default value of the control variable at the '
                          'control discretization nodes.')

        self.declare(name='shape', types=Iterable, allow_none=True, default=None,
                     desc='The shape of the control variable at each point in time.')

        self.declare(name='lower', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The lower bound of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='upper', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The upper bound of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='scaler', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The scaler of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='adder', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The adder of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='ref0', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The zero-reference value of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='ref', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The unit-reference value of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='continuity', types=(bool, dict), default=True,
                     desc='Enforce continuity of control values at segment boundaries. This '
                          'option is invalid if opt=False.')

        self.declare(name='rate_continuity', types=(bool, dict), default=True,
                     desc='Enforce continuity of control first derivatives in dimensionless time '
                          'at segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate_continuity_scaler', types=(Number,), default=1.0,
                     desc='Scaler of the dimensionless rate continuity constraint at '
                          'segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate2_continuity', types=(bool, dict), default=False,
                     desc='Enforce continuity of control second derivatives at segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate2_continuity_scaler', types=(Number,), default=1.0,
                     desc='Scaler of the dimensionless rate continuity constraint at '
                          'segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare('dynamic', default=True, types=bool,
                     desc='If True, the value of the shape of the parameter will '
                          'be (num_nodes, ...), allowing the variable to be used as either a '
                          'static or dynamic control.  This impacts the shape of the partial '
                          'derivatives matrix.  Unless a parameter is large and broadcasting a '
                          'value to each individual node would be inefficient, users should stick '
                          'to the default value of True.')


class PolynomialControlOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary specific to controls.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(PolynomialControlOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str,
                     desc='The name of ODE system parameter to be controlled.')

        self.declare(name='units', default=_unspecified,
                     allow_none=True, desc='The units in which the control variable is defined.')

        self.declare(name='desc', types=str, default='',
                     desc='The description of the control variable.')

        self.declare(name='opt', default=True, types=bool,
                     desc='If True, the control value will be a design variable '
                          'for the optimization problem.  If False, allow the '
                          'control to be connected externally.')

        self.declare(name='fix_initial', types=bool, default=False,
                     desc='If True, the initial value of this control is fixed and not a '
                          'design variable. This option is invalid if opt=False.')

        self.declare(name='fix_final', types=bool, default=False,
                     desc='If True, the final value of this control is fixed and not a '
                          'design variable. This option is invalid if opt=False.')

        self.declare(name='targets', allow_none=True, default=_unspecified,
                     desc='Targets in the ODE to which the state is connected')

        self.declare(name='rate_targets', allow_none=True, default=_unspecified,
                     desc='The targets in the ODE to which the polynomial control rate is connected')

        self.declare(name='rate2_targets', allow_none=True, default=_unspecified,
                     desc='The targets in the ODE to which the polynomial control 2nd derivative '
                          'is connected')

        self.declare(name='val', types=(Iterable, np.ndarray, Number), default=np.zeros(1),
                     desc='The default value of the control variable at the '
                          'control discretization nodes.')

        self.declare(name='shape', types=Iterable, allow_none=True, default=None,
                     desc='The shape of the control variable at each point in time.')

        self.declare(name='lower', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The lower bound of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='upper', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The upper bound of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='scaler', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The scaler of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='adder', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The adder of the control variable at the nodes. This'
                          'option is invalid if opt=False.')

        self.declare(name='ref0', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The zero-reference value of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='ref', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The unit-reference value of the control variable at the nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='order', types=(int,), default=None, allow_none=True,
                     desc='A integer that provides the interpolation order when the control is '
                          'to assume a single polynomial basis across the entire phase, or None '
                          'to use the default control behavior.')

        self.declare('dynamic', default=True, types=bool,
                     desc='If True, the value of the shape of the parameter will '
                          'be (num_nodes, ...), allowing the variable to be used as either a '
                          'static or dynamic control.  This impacts the shape of the partial '
                          'derivatives matrix.  Unless a parameter is large and broadcasting a '
                          'value to each individual node would be inefficient, users should stick '
                          'to the default value of True.')


def check_valid_shape(name, value):
    """
    Raise an exception if the value specified for a shape is invalid.

    Parameters
    ----------
    name : str
        Name of the option.
    value : object
        Shape to check, should be a Iterable, Number, list, or tuple.
    """
    if name == 'shape':
        if value is not _unspecified and not isinstance(value, (Iterable, Number, list, tuple)):
            raise ValueError(f"Option '{name}' with value {value} is not valid.")


class ParameterOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary specific to parameters.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(ParameterOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str,
                     desc='The name of ODE system parameter to be set via parameter.')

        self.declare(name='units', default=_unspecified,
                     allow_none=True, desc='The units in which the parameter is defined.')

        self.declare(name='desc', types=str, default='',
                     desc='The description of the parameter.')

        self.declare(name='opt', default=True, types=bool,
                     desc='If True, the control value will be a design variable '
                          'for the optimization problem.  If False, allow the '
                          'control to be connected externally.')

        self.declare(name='dynamic', types=bool, default=True,
                     desc='True if this parameter can be used as a dynamic control, else False')

        self.declare(name='targets', allow_none=True, default=_unspecified,
                     desc='Targets in the ODE to which the state is connected')

        self.declare(name='val', types=(Iterable, np.ndarray, Number), default=np.zeros(1),
                     desc='The default value of the parameter in the phase.')

        self.declare(name='shape',  check_valid=check_valid_shape, default=_unspecified,
                     desc='The shape of the parameter.')

        self.declare(name='lower', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The lower bound of the parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='upper', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The upper bound of the parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='scaler', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The scaler of the parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='adder', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The adder of the parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='ref0', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The zero-reference value of the parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='ref', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The unit-reference value of the parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='include_timeseries', types=bool, default=True,
                     desc='True if the static parameters should be included in output timeseries, else False')


class TrajParameterOptionsDictionary(ParameterOptionsDictionary):
    """
    An OptionsDictionary specific to trajectory design parameters.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(TrajParameterOptionsDictionary, self).__init__(read_only)

        self.declare(name='custom_targets', types=dict, default=None, allow_none=True,
                     desc='Used to override the default targets of the trajectory input parameter'
                          ' in each phase.  By default its target will be the same as its name')

        self._dict.pop('targets')

        self.declare(name='targets', types=dict, default=None, allow_none=True,
                     desc='Used to specify the targets for the parameter in each phase. '
                          'If None, Dymos will attempt to connect it to a parameter of '
                          'the same name in each phase.  Otherwise, targets should be a given '
                          'as a dictionary.  For each phase name given as a key in the dictionary,'
                          'if the associated value is a string, connect the parameter to the phase'
                          ' input parameter given by the string. If the associated value is a'
                          ' sequence, treat it as a list of ODE-relative targets for the parameter'
                          ' in that phase')


class StateOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary specific to states.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(StateOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str,
                     desc='name of ODE state variable')

        self.declare(name='units', default=_unspecified,
                     allow_none=True, desc='units in which the state variable is defined')

        self.declare(name='opt', types=bool, default=True,
                     desc='If true, the values of this state are a design variable '
                          'for the optimizer.  Otherwise it exists as an unconnected '
                          'input.')

        self.declare(name='fix_initial', types=(bool, Iterable), default=False,
                     desc='If True, the initial value of this state is fixed and not a '
                          'design variable. If the state variable has a non-scalar shape, '
                          'this may be an iterable of bool for each index. '
                          'This option is invalid if opt=False.')

        self.declare(name='fix_final', types=(bool, Iterable), default=False,
                     desc='If True, the final value of this state is fixed and not a '
                          'design variable. If the state variable has a non-scalar shape, '
                          'this may be an iterable of bool for each index. This option is '
                          'invalid if opt=False.')

        self.declare(name='initial_bounds', types=Iterable, default=None, allow_none=True,
                     desc='Bounds on the value of the state at the start of the phase. This '
                          'option is invalid if opt=False.')

        self.declare(name='final_bounds', types=Iterable, default=None, allow_none=True,
                     desc='Bounds on the value of the state at the end of the phase. This '
                          'option is invalid if opt=False.')

        self.declare(name='val', types=(Iterable, Number), default=1.0,
                     desc='Default value of the state variable at the discretization nodes')

        self.declare(name='desc', types=str, default='',
                     desc='description of the state variable')

        self.declare(name='shape', types=Iterable, allow_none=True, default=None,
                     desc='shape of the state variable, as determined by introspection')

        self.declare(name='rate_source', types=str, allow_none=True, default=None,
                     desc='ODE-path to the derivative of the state variable')

        self.declare(name='targets', allow_none=True, default=_unspecified,
                     desc='Targets in the ODE to which the state is connected')

        self.declare(name='lower', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Lower bound of the state variable at the discretization nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='upper',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Upper bound of the state variable at the discretization nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='scaler',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Scaler of the state variable at the discretization nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='adder',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Adder of the state variable at the discretization nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='ref0',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Zero-reference value of the state variable at the discretization nodes. '
                          'This option is invalid if opt=False.')

        self.declare(name='ref',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Unit-reference value of the state variable at the discretization nodes. '
                          'This option is invalid if opt=False.')

        self.declare(name='defect_scaler',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Scaler of the state variable defects at the collocation nodes. '
                          'If defect_scaler and defect_ref are both None but the state scaler '
                          'or ref are provided, use those values as the defect scaler or ref. '
                          'This option is invalid if opt=False.')

        self.declare(name='defect_ref',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Unit-reference value of the state defects at the collocation nodes. This'
                          ' option is invalid if opt=False.  If provided, this option overrides'
                          ' defect_scaler. If defect_scaler and defect_ref are both None but the '
                          'state scaler or ref are provided, use those values as the defect '
                          'scaler or ref.')

        self.declare(name='continuity', types=(bool, dict), default=True,
                     desc='Enforce continuity of state values at segment boundaries. This '
                          'option is invalid if opt=False.')

        self.declare(name='solve_segments', default=None, allow_none=True,
                     values=(True, False, 'forward', 'backward'),
                     desc='If True (deprecated) or \'forward\', collocation defects within each'
                          'segment are solved with a Newton solver by fixing the initial value in the'
                          'phase (if using compressed transcription) or segment (if not using '
                          'compressed transcription). This provides a forward shooting (or multiple shooting)'
                          'method.  If \'backward\', the final value in the phase or segment is fixed'
                          'and a solver finds the other ones to mimic reverse propagation. If None, '
                          '(the default) use the value of solve_segments in the transcription. Set '
                          'to False to explicitly disable the use of a solver to converge the state'
                          'time history.')

        self.declare(name='connected_initial', default=False, types=bool,
                     desc='Whether an input is created to pass in the initial state. This may be '
                          'set by a trajectory that links phases.')


class TimeOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary for time options.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(TimeOptionsDictionary, self).__init__(read_only)

        self.declare('units', types=str, allow_none=True,
                     default='s', desc='Units for the integration variable')

        self.declare(name='fix_initial', types=bool, default=False,
                     desc='If True, the initial value of time is not a design variable.')

        self.declare(name='fix_duration', types=bool, default=False,
                     desc='If True, the  phase duration is not a design variable.')

        self.declare(name='input_initial', types=bool, default=False,
                     desc='If True, the initial value of time (t_initial) is expected to be '
                          'connected to an external output source.')

        self.declare(name='input_duration', types=bool, default=False,
                     desc='If True, the phase duration (t_duration) is expected to be '
                          'connected to an external output source.')

        self.declare('initial_val', types=Number, default=0.0,
                     desc='Value of the integration variable at the start of the phase.')

        self.declare('initial_bounds', types=Iterable, default=(None, None),
                     desc='Tuple of (lower, upper) bounds for the integration variable at '
                          'the start of the phase.')

        self.declare('initial_scaler', types=Number, allow_none=True, default=None,
                     desc='Scalar for the initial value of the integration variable.')

        self.declare('initial_adder', types=Number, allow_none=True, default=None,
                     desc='Adder for the initial value of the integration variable.')

        self.declare('initial_ref0', types=Number, allow_none=True, default=None,
                     desc='Zero-reference value for the initial value of the integration variable.')

        self.declare('initial_ref', types=Number, allow_none=True, default=None,
                     desc='Unit-reference value for the initial value of the integration variable.')

        self.declare('duration_val', types=Number, default=1.0,
                     desc='Value of the duration of the integration variable across the phase.')

        self.declare('duration_bounds', types=Iterable, default=(None, None),
                     desc='Tuple of (lower, upper) bounds for the duration '
                          'of the integration variable across the phase.')

        self.declare('duration_scaler', types=Number, allow_none=True, default=None,
                     desc='Scalar for the duration of the integration variable across the phase.')

        self.declare('duration_adder', types=Number, allow_none=True, default=None,
                     desc='Adder for the duration of the integration variable across the phase.')

        self.declare('duration_ref0', types=Number, allow_none=True, default=None,
                     desc='Zero-reference value for the duration of the integration variable '
                          'across the phase.')

        self.declare('duration_ref', types=Number, allow_none=True, default=None,
                     desc='Unit-reference value for the duration of the integration variable '
                          'across the phase.')

        self.declare(name='targets', types=Iterable, allow_none=True, default=[],
                     desc='targets in the ODE to which the integration variable is connected')

        self.declare(name='time_phase_targets', types=Iterable, allow_none=True, default=[],
                     desc='targets in the ODE to which the elapsed duration of the phase is '
                          'connected')

        self.declare(name='t_initial_targets', types=Iterable, allow_none=True, default=[],
                     desc='targets in the ODE to which the initial time of the phase is '
                          'connected')

        self.declare(name='t_duration_targets', types=Iterable, allow_none=True, default=[],
                     desc='targets in the ODE to which the total duration of the phase is '
                          'connected')


class GridRefinementOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary for grid refinement options in a Phase.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(GridRefinementOptionsDictionary, self).__init__(read_only)

        self.declare('refine', types=bool, default=True,
                     desc='If True, this Phase may be refined during the grid refinement procedure.')

        self.declare(name='tolerance', types=float, default=1.0E-4,
                     desc='Default tolerance for grid refinement in this phase.')

        self.declare(name='min_order', types=int, default=3,
                     desc='Minimum transcription order for segments in this phase.')

        self.declare(name='max_order', types=int, default=14,
                     desc='Maximum transcription order for segments in this phase.')
        self.declare(name='smoothness_factor', types=float, default=1.2,
                     desc='Maximum allowed ratio of state second derivatives across refinement iterations')


class _ForDocs(object):  # pragma: no cover
    """
    This class is provided as a way to automatically display options dictionaries in the docs,
    since these option dictionaries typically don't exist in instantiated form in the code base.
    """

    def __init__(self):

        self.time_options = TimeOptionsDictionary()
        self.state_options = StateOptionsDictionary()
        self.control_options = ControlOptionsDictionary()
        self.parameter_options = ParameterOptionsDictionary()
