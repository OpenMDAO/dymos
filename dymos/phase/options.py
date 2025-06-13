from collections.abc import Iterable
from numbers import Number
import numpy as np

import openmdao.api as om
from dymos.utils.misc import is_none_or_unspecified, _unspecified


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

        self.declare(name='control_type', types=str, default='full',
                     desc='The type of control variable.  Options are `full` or `polynomial`.')

        self.declare(name='order', types=(int,), default=None, allow_none=True,
                     desc='A integer that provides the interpolation order when the control is '
                          'to assume a single polynomial basis across the entire phase, or None '
                          'to use the default control behavior.')

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

        self.declare(name='continuity_scaler', types=(Number,), default=None, allow_none=True,
                     desc='Scaler for continuity at segment boundaries. This '
                          'option is invalid if opt=False.')

        self.declare(name='continuity_ref', types=(Number,), default=None, allow_none=True,
                     desc='Reference unit value for continuity at segment boundaries instead of scaler.'
                          'This option is invalid if opt=False.')

        self.declare(name='rate_continuity', types=(bool, dict), default=True,
                     desc='Enforce continuity of control first derivatives in dimensionless time '
                          'at segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate_continuity_scaler', types=(Number,), default=None, allow_none=True,
                     desc='Scaler of the rate continuity constraint at segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate_continuity_ref', types=(Number,), default=None, allow_none=True,
                     desc='Reference unit value for rate continuity at segment boundaries instead of scaler.'
                          'This option is invalid if opt=False.')

        self.declare(name='rate2_continuity', types=(bool, dict), default=False,
                     desc='Enforce continuity of control second derivatives at segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate2_continuity_scaler', types=(Number,), default=None, allow_none=True,
                     desc='Scaler of the rate2 continuity constraint at segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate2_continuity_ref', types=(Number,), default=None, allow_none=True,
                     desc='Reference unit value for rate2 continuity at segment boundaries instead of scaler.'
                          'This option is invalid if opt=False.')


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
        if not is_none_or_unspecified(value) and not isinstance(value, (Iterable, Number, list, tuple)):
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

        self.declare(name='dynamic', values=[True, False, _unspecified], default=_unspecified,
                     desc='True if this parameter can be used as a dynamic control, else False.'
                          'If _unspecified, attempt to determine through introspection.',
                     deprecation="Option dynamic has been replaced by option 'static_target' and "
                                 "will be removed in Dymos 2.0.0.\nNote that 'static_target' has "
                                 "the opposite meaning of option 'dynamic', so parameters with "
                                 "option 'dynamic' set to False should now use 'static_target' set "
                                 "to True.")

        self.declare(name='static_target', default=_unspecified,
                     desc='True if the target of this parameter does NOT have a unique value at '
                          'each node in the ODE.'
                          'If _unspecified, attempt to determine through introspection.',
                     deprecation='Use option `static_targets` to specify whether all targets\n'
                                 'are static (static_targets=True), none are static (static_targets=False),\n'
                                 'static_targets are determined via introspection (static_targets=_unspecified),\n'
                                 'or give an explicit sequence of the static targets.')

        self.declare(name='static_targets', default=_unspecified,
                     desc='If a boolean, specifies whether all targets are static (True), or no\n'
                          'targets are static (False). Otherwise, provide a list of the static\n'
                          'targets within the ODE. If left unspecified, static targets will be\n'
                          'determined by finding inptus tagged with \'dymos.static_target\'.')

        self.declare(name='targets', allow_none=True, default=_unspecified,
                     desc='Targets in the ODE to which the state is connected')

        self.declare(name='val', types=(Iterable, np.ndarray, Number), default=0.0,
                     desc='The default value of the parameter in the phase.')

        self.declare(name='shape', check_valid=check_valid_shape, default=_unspecified, allow_none=True,
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

        self.declare(name='include_timeseries', types=bool, default=None,
                     allow_none=True,
                     desc='True if the static parameters should be included in output timeseries, else False.'
                          'If None (default) set the value based on Phase.timeseries_options["include_parameters"]')


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
                     desc='ODE-path or phase variable providing the derivative of the state variable')

        self.declare(name='source', types=str, allow_none=True, default=None,
                     desc='RHS-path or phase variable providing value of the state variable, for Analytic '
                          'transcription only!')

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

        self.declare(name='continuity_scaler', types=(Number,), default=None, allow_none=True,
                     desc='Scaler for continuity at segment boundaries. This '
                          'option is invalid if opt=False.')

        self.declare(name='continuity_ref', types=(Number,), default=None, allow_none=True,
                     desc='Reference unit value for continuity at segment boundaries instead of scaler.'
                          'This option is invalid if opt=False.')

        self.declare(name='solve_segments', default=None, allow_none=True,
                     values=(False, 'forward', 'backward'),
                     desc='If \'forward\', collocation defects within each'
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
                          'set by a trajectory that links phases.',
                     deprecation="State option 'connected_initial' is deprecated. Please use 'input_initial' instead.")

        self.declare(name='input_initial', default=False, types=bool,
                     desc='Whether the initial value of the state is expected to be connected to an exterior value.')

        self.declare(name='input_final', default=False, types=bool,
                     desc='Whether the final value of the state is expected to be connected to an exterior value.')


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

        self.declare('name', types=str, default='time', desc='Name of the integraiton variable in the phase.')

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

        self.declare(name='targets', allow_none=True, default=_unspecified,
                     desc='targets in the ODE to which the integration variable is connected')

        self.declare(name='time_phase_targets', allow_none=True, default=_unspecified,
                     desc='targets in the ODE to which the elapsed duration of the phase is connected')

        self.declare(name='t_initial_targets', allow_none=True, default=[],
                     desc='targets in the ODE to which the initial time of the phase is connected')

        self.declare(name='t_duration_targets', allow_none=True, default=[],
                     desc='targets in the ODE to which the total duration of the phase is connected')

        self.declare(name='t_final_targets', allow_none=True, default=[],
                     desc='targets in the ODE to which the final time of the phase is connected')

        self.declare(name='dt_dstau_targets', allow_none=True, default=[],
                     desc='targets in the ODE to which the ratio of segment time duration to nondim duration is connected')

        self.declare(name='t_duration_balance_options', default={},
                     desc='options dictionary for the duration residual')


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


class SimulateOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary for simulate options in a Phase.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(SimulateOptionsDictionary, self).__init__(read_only)

        self.declare('method', values=('RK23', 'RK45', 'DOP853', 'BDF', 'Radau', 'LSODA'), default='DOP853',
                     desc='The method used by simulate to propagate the ODE.')

        self.declare(name='atol', types=(float, np.array), default=1.0E-6,
                     desc='Absolute error tolerance for variable step integration.')

        self.declare(name='rtol', types=(float, np.array), default=1.0E-3,
                     desc='Relative error tolerance for variable step integration.')

        self.declare(name='first_step', types=float, allow_none=True, default=None,
                     desc='Initial step size, or None if the algorithm should choose.')

        self.declare(name='max_step', types=float, default=np.inf,
                     desc='Maximum allowable step size')

        self.declare(name='times_per_seg', types=int, allow_none=True, default=10,
                     desc='The default number of output times per segment for Phase.simulate.'
                          'If None, and not provided as an argument to simulate, use the'
                          'same grid as the Phase transcription.')


class ConstraintOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary for path and boundary constraints.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(ConstraintOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str, default=None, allow_none=True,
                     desc='Name or ODE-relative path of the variable to be constrained.')

        self.declare(name='constraint_name', types=str, default=None, allow_none=True,
                     desc='Name of the variable when used as a constraint, to avoid name collisions.')

        self.declare(name='constraint_path', types=str, default=None, allow_none=True,
                     desc='Path in the phase to the constrained output. Determined automatically by dymos.')

        self.declare(name='lower', types=(Iterable, Number), default=None, allow_none=True,
                     desc='Lower bound of the constraint.')

        self.declare(name='upper', types=(Iterable, Number), default=None, allow_none=True,
                     desc='Upper bound of the constraint.')

        self.declare(name='equals', types=(Iterable, Number), default=None, allow_none=True,
                     desc='Desired vlue for an equality constraint.')

        self.declare(name='scaler', types=(Iterable, Number), default=None, allow_none=True,
                     desc='Scaler of the variable.')

        self.declare(name='adder', types=(Iterable, Number), default=None, allow_none=True,
                     desc='Adder of the state variable.')

        self.declare(name='ref0', types=(Iterable, Number), default=None, allow_none=True,
                     desc='Zero-reference value of the variable.')

        self.declare(name='ref', types=(Iterable, Number), default=None, allow_none=True,
                     desc='Unit-reference value of the variable.')

        self.declare(name='indices', types=(Iterable,), default=None, allow_none=True,
                     desc='Indices value of the variable, format is controlled by the `flat_indices` option.')

        self.declare(name='shape', types=(Iterable,), default=None, allow_none=True,
                     desc='The shape of the constrained variable. This is generally determined automatically by dymos.')

        self.declare(name='linear', types=(bool,), default=False,
                     desc='If True, tell the optimizer to treat this as a linear constraint. Setting this to True '
                          'when the constraint is not actually linear will result in a failure of the optimization.')

        self.declare(name='units', types=str, default=None, allow_none=True,
                     desc='Units to be used for the constraint bounds, or None to use the units of the constrained '
                          'variable.')

        self.declare(name='flat_indices', types=bool, default=True,
                     desc='If True, the given indices will be treated as indices into a C-order flattened array based '
                          'on the shaped of the constrained variable at a point in time.')


class TimeseriesOutputOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary for timeseries outputs.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super(TimeseriesOutputOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=str, default=None, allow_none=True,
                     desc='Name or ODE-relative path of the variable to be output in the timeseries.')

        self.declare(name='output_name', types=str, default=None, allow_none=True,
                     desc='Name of the variable used as the output from the timeseries, to avoid name collisions.')

        self.declare(name='wildcard_units', types=dict, default=None, allow_none=True,
                     desc='Variable name, unit mapping that can be provided if timeseries are specified '
                          'with wildcards.')

        self.declare(name='shape', default=None, allow_none=True,
                     desc='The shape of the timeseries output variable.'
                          ' This is generally determined automatically by dymos.')

        self.declare(name='units', default=None, allow_none=True,
                     desc='Units to be used for the timeseries output, or None to leave the units unchanged.')

        self.declare(name='src', types=str, default=None, allow_none=True,
                     desc='The phase-relative path of the source of this timeseries output,'
                          ' used when issuing connections.')

        self.declare(name='src_idxs', types=(Iterable,), default=None, allow_none=True,
                     desc='The indices of the source of this timeseries output to be used when issuing connections.')

        self.declare(name='val', types=bool, default=True,
                     desc='If True, include the value of this variable as a timeseries output.')

        self.declare(name='is_rate', default=False, allow_none=False,
                     desc='If True this is a rate.')


class PhaseTimeseriesOptionsDictionary(om.OptionsDictionary):
    """
    An OptionsDictionary for phase options related to timeseries.

    Parameters
    ----------
    read_only : bool
        If True, setting (via __setitem__ or update) is not permitted.
    """
    def __init__(self, read_only=False):
        super().__init__(read_only)

        self.declare(name='use_prefix', types=bool, default=False,
                     desc='If True, prefix the timeseries variable output with the type '
                          'of variable (this is legacy behavior that changed in Dymos 1.8.0)')

        self.declare(name='include_state_rates', types=bool, default=False,
                     desc='If True, include state rates in the timeseries outputs by default.')

        self.declare(name='include_control_rates', types=bool, default=False,
                     desc='If True, include control rates in the timeseries outputs by default.')

        self.declare(name='include_t_phase', types=bool, default=False,
                     desc='If True, include the elapsed phase time in the timeseries outputs by default.')

        self.declare(name='include_parameters', types=bool, default=False,
                     desc='If True, include the parameters in the timeseries outputs by default.')
