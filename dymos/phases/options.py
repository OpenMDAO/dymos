from collections import Iterable
from numbers import Number
from six import string_types

import numpy as np

from openmdao.api import OptionsDictionary


class ControlOptionsDictionary(OptionsDictionary):
    """
    An OptionsDictionary specific to controls.
    """

    def __init__(self, read_only=False):
        super(ControlOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=string_types,
                     desc='The name of ODE system parameter to be controlled.')

        self.declare(name='units', types=string_types, default=None,
                     allow_none=True, desc='The units in which the control variable is defined.')

        self.declare(name='desc', types=string_types, default='',
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

        self.declare(name='targets', types=Iterable, default=[],
                     desc='Used to store target information for ShootingPhase.  Should not be'
                          'set by the user in add_control.')

        self.declare(name='rate_param', types=Iterable, allow_none=True,
                     default=None,
                     desc='The parameter in the ODE to which the control rate is connected')

        self.declare(name='rate2_param', types=Iterable, allow_none=True,
                     default=None,
                     desc='The parameter in the ODE to which the control 2nd derivative '
                          'is connected.')

        self.declare(name='val', types=(Iterable, np.ndarray, Number), default=np.zeros(1),
                     desc='The default value of the control variable at the '
                          'control discretization nodes.')

        self.declare(name='shape', types=Iterable, default=(1,),
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

        self.declare(name='rate2_continuity', types=(bool, dict), default=True,
                     desc='Enforce continuity of control second derivatives at segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='rate2_continuity_scaler', types=(Number,), default=1.0,
                     desc='Scaler of the dimensionless rate continuity constraint at '
                          'segment boundaries. '
                          'This option is invalid if opt=False.')

        self.declare(name='interp_order', types=(int,), default=None, allow_none=True,
                     desc='A integer that provides the interpolation order when the control is'
                          'to assume a single polynomial basis across the entire phase, or None'
                          'to use the default control behavior.')


class DesignParameterOptionsDictionary(OptionsDictionary):
    """
    An OptionsDictionary specific to design parameters.
    """

    def __init__(self, read_only=False):
        super(DesignParameterOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=string_types,
                     desc='The name of ODE system parameter to be set via design parameter.')

        self.declare(name='units', types=string_types, default=None,
                     allow_none=True, desc='The units in which the design parameter is defined.')

        self.declare(name='desc', types=string_types, default='',
                     desc='The description of the design parameter.')

        self.declare(name='opt', default=True, types=bool,
                     desc='If True, the control value will be a design variable '
                          'for the optimization problem.  If False, allow the '
                          'control to be connected externally.')

        self.declare(name='dynamic', types=bool, default=True,
                     desc='True if this parameter can be used as a dynamic control, else False')

        self.declare(name='target_params', types=dict, default=None, allow_none=True,
                     desc='Used to store target information on a per-phase basis for trajectories.')

        self.declare(name='val', types=(Iterable, np.ndarray, Number), default=np.zeros(1),
                     desc='The default value of the design parameter in the phase.')

        self.declare(name='shape', types=Iterable, default=(1,),
                     desc='The shape of the design parameter.')

        self.declare(name='lower', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The lower bound of the design parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='upper', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The upper bound of the design parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='scaler', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The scaler of the design parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='adder', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The adder of the design parameter. This'
                          'option is invalid if opt=False.')

        self.declare(name='ref0', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The zero-reference value of the design parameter. This '
                          'option is invalid if opt=False.')

        self.declare(name='ref', types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='The unit-reference value of the design parameter. This '
                          'option is invalid if opt=False.')


class InputParameterOptionsDictionary(OptionsDictionary):
    """
    An OptionsDictionary specific to input parameters.
    """

    def __init__(self, read_only=False):
        super(InputParameterOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=string_types,
                     desc='The name of ODE system parameter to be set via input parameter, or'
                          'an alias.  If an alias is provided, then "target_param" should provide'
                          'the ODE system parameter name.')

        self.declare(name='units', types=string_types, default=None,
                     allow_none=True, desc='The units in which the design parameter is defined.')

        self.declare(name='desc', types=string_types, default='',
                     desc='The description of the design parameter.')

        self.declare(name='dynamic', types=bool, default=True,
                     desc='True if this parameter can be used as a dynamic control, else False')

        self.declare(name='target_params', types=dict, default=None, allow_none=True,
                     desc='Used to store target information on a per-phase basis for trajectories.')

        self.declare(name='val', types=(Iterable, np.ndarray, Number), default=np.zeros(1),
                     desc='The default value of the design parameter in the phase.')

        self.declare(name='shape', types=Iterable, default=(1,),
                     desc='The shape of the design parameter.')


class StateOptionsDictionary(OptionsDictionary):
    """
    An OptionsDictionary specific to controls.
    """

    def __init__(self, read_only=False):
        super(StateOptionsDictionary, self).__init__(read_only)

        self.declare(name='name', types=string_types,
                     desc='name of ODE state variable')

        self.declare(name='units', types=string_types, default=None,
                     allow_none=True, desc='units in which the state variable is defined')

        self.declare(name='opt', types=bool, default=True,
                     desc='If true, the values of this state are a design variable '
                          'for the optimizer.  Otherwise it exists as an unconnected'
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

        self.declare(name='val', types=(Iterable, Number), default=0.0,
                     desc='Default value of the state variable at the discretization nodes')

        self.declare(name='desc', types=string_types, default='',
                     desc='description of the state variable')

        self.declare(name='shape', types=Iterable, default=(1,),
                     desc='shape of the state variable')

        self.declare(name='rate_source', types=string_types,
                     desc='ODE-path to the derivative of the state variable')

        self.declare(name='targets', types=Iterable, allow_none=True, default=None,
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
                     types=(Iterable, Number), default=1.0,
                     allow_none=True,
                     desc='Scaler of the state variable defects at the collocation nodes. This '
                          'option is invalid if opt=False.')

        self.declare(name='defect_ref',
                     types=(Iterable, Number), default=None,
                     allow_none=True,
                     desc='Unit-reference value of the state defects at the collocation nodes. This'
                          ' option is invalid if opt=False.  If provide, this option overrides'
                          ' defect_scaler.')

        self.declare(name='continuity', types=(bool, dict), default=True,
                     desc='Enforce continuity of state values at segment boundaries. This '
                          'option is invalid if opt=False.')

        self.declare('solve_segments', default=False, types=bool,
                     desc='setting to control if segment collocation defects are '
                          'solved with a newton solver')

        self.declare('time_direction', default='forward', values=('forward', 'backward'),
                     desc='Whether the numerical propagation occurs forward or backward '
                          'in time for this state.  This poses restrictions on whether '
                          'states can have fixed or connected initial and final values.')


class TimeOptionsDictionary(OptionsDictionary):
    """
    An OptionsDictionary for time options
    """

    def __init__(self, read_only=False):
        super(TimeOptionsDictionary, self).__init__(read_only)

        self.declare('units', types=string_types, allow_none=True,
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

        self.declare(name='targets', types=Iterable, allow_none=True, default=None,
                     desc='targets in the ODE to which the integration variable is connected')

        self.declare(name='time_phase_targets', types=Iterable, allow_none=True, default=None,
                     desc='targets in the ODE to which the elapsed duration of the phase is '
                          'connected')

        self.declare(name='t_initial_targets', types=Iterable, allow_none=True, default=None,
                     desc='targets in the ODE to which the initial time of the phase is '
                          'connected')

        self.declare(name='t_duration_targets', types=Iterable, allow_none=True, default=None,
                     desc='targets in the ODE to which the total duration of the phase is '
                          'connected')


class _ForDocs(object):  # pragma: no cover
    """
    This class is provided as a way to automatically display options dictionaries in the docs,
    since these option dictionaries typically don't exist in instantiated form in the code base.
    """

    def __init__(self):

        self.time_options = TimeOptionsDictionary()
        self.state_options = StateOptionsDictionary()
        self.control_options = ControlOptionsDictionary()
        self.design_parameter_options = DesignParameterOptionsDictionary()
