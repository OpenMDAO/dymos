from __future__ import division, print_function, absolute_import

from collections import Iterable
import inspect
from six import iteritems
import warnings

import numpy as np

from scipy import interpolate

from openmdao.api import Problem, Group, IndepVarComp, SqliteRecorder
from openmdao.core.system import System

from dymos.phases.components import BoundaryConstraintComp
from dymos.phases.components import InputParameterComp
from dymos.phases.options import ControlOptionsDictionary, DesignParameterOptionsDictionary, \
    InputParameterOptionsDictionary, StateOptionsDictionary, TimeOptionsDictionary, \
    PolynomialControlOptionsDictionary
from dymos.phases.components import PolynomialControlGroup, ControlGroup

from dymos.utils.constants import INF_BOUND


_unspecified = object()


class PhaseBase(Group):
    def __init__(self, **kwargs):

        super(PhaseBase, self).__init__(**kwargs)

        # Dictioanries of variable options that are set by the user via the API
        # These will be applied over any defaults specified by decorators on the ODE
        self.user_time_options = {}
        self.user_state_options = {}
        self.user_control_options = {}
        self.user_polynomial_control_options = {}
        self.user_design_parameter_options = {}
        self.user_input_parameter_options = {}
        self.user_traj_parameter_options = {}

        self._initial_boundary_constraints = {}
        self._final_boundary_constraints = {}
        self._path_constraints = {}
        self._timeseries_outputs = {}
        self._objectives = {}
        self._ode_controls = {}
        self.grid_data = None
        self._time_extents = []

    def initialize(self):
        self.options.declare('num_segments', types=int, desc='Number of segments')
        self.options.declare('ode_class',
                             desc='System defining the ODE')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('transcription', values=['gauss-lobatto', 'radau-ps', 'explicit'],
                             desc='Transcription technique of the optimal control problem.')
        self.options.declare('segment_ends', default=None, types=Iterable, allow_none=True,
                             desc='Iterable of locations of segment ends or None for equally'
                                  'spaced segments')
        self.options.declare('transcription_order', default=3, types=(int, Iterable),
                             desc='Order of the transcription')
        self.options.declare('compressed', default=True, types=bool,
                             desc='Use compressed transcription')

    def set_state_options(self, name, **kwargs):
        """
        Set options that apply the EOM state variable of the given name.

        Parameters
        ----------
        name : str
            Name of the state variable in the RHS.
        units : str or None
            Units in which the state variable is defined.  Internally components may use different
            units for the state variable, but the IndepVarComp which provides its value will provide
            it in these units, and collocation defects will use these units.  If units is not
            specified here then the value as defined in the ODEOptions (@declare_state) will be
            used.
        val :  ndarray
            The default value of the state at the state discretization nodes of the phase.
        fix_initial : bool(False)
            If True, omit the first value of the state from the design variables (prevent the
            optimizer from changing it).
        fix_final : bool(False)
            If True, omit the final value of the state from the design variables (prevent the
            optimizer from changing it).
        lower : float or ndarray or None (None)
            The lower bound of the state at the nodes of the phase.
        upper : float or ndarray or None (None)
            The upper bound of the state at the nodes of the phase.
        scaler : float or ndarray or None (None)
            The scaler of the state value at the nodes of the phase.
        adder : float or ndarray or None (None)
            The adder of the state value at the nodes of the phase.
        ref0 : float or ndarray or None (None)
            The zero-reference value of the state at the nodes of the phase.
        ref : float or ndarray or None (None)
            The unit-reference value of the state at the nodes of the phase
        defect_scaler : float or ndarray (1.0)
            The scaler of the state defect at the collocation nodes of the phase.
        defect_ref : float or ndarray (1.0)
            The unit-reference value of the state defect at the collocation nodes of the phase. If
            provided, this value overrides defect_scaler.
        solve_segments : bool(False)
            If True, a solver will be used to converge the collocation defects within a segment.
            Note that the state continuity defects between segements will still be
            handled by the optimizer.
        connected_initial : bool
            If True, then the initial value for this state comes from an externally connected
            source.
        rate_source : str
            The path to the ODE output which provides the rate of this state variable.
        targets : Sequence of str
            The path to the targets of the state variable in the ODE system.
        """
        if name not in self.user_state_options:
            self.user_state_options[name] = {}

        kwargs['name'] = name

        for kw in kwargs:
            if kw not in StateOptionsDictionary():
                raise KeyError('Invalid argument to set_state_options: {0}'.format(kw))

        self.user_state_options[name].update(kwargs)

    def _check_parameter(self, name):
        """
        Checks that the parameter of the given name is valid.

        First name is checked against all existing controls, input parameters, and design
        parameters.  If it has already been assigned to one of those, ValueError is raised.
        Finally, if *dynamic* is True, the control is not a dynamic parameter in the ODE,
        ValueError is raised.

        Parameters
        ----------
        name : str
            The name of the controllable parameter.

        Raises
        ------
        ValueError
            Raised if the parameter of the given name is previously assigned or
            incompatible with the type of control to which it is assigned.


        """
        # ode_params = None if self.ode_options is None else self.ode_options._parameters
        if name in self.user_control_options:
            raise ValueError('{0} has already been added as a control.'.format(name))
        if name in self.user_design_parameter_options:
            raise ValueError('{0} has already been added as a design parameter.'.format(name))
        if name in self.user_input_parameter_options:
            raise ValueError('{0} has already been added as an input parameter.'.format(name))
        if name in self.user_polynomial_control_options:
            raise ValueError('{0} has already been added as an interpolated control.'.format(name))
        if name in self.user_traj_parameter_options:
            raise ValueError('{0} has already been added as a trajectory-level '
                             'parameter.'.format(name))

    def add_control(self, name, **kwargs):
        """
        Adds a dynamic control variable to be tied to a parameter in the ODE.

        Parameters
        ----------
        name : str
            The name assigned to the control variable.  If the ODE has been decorated with
            parameters, this should be the name of a control in the system.
        units : str or None
            The units with which the control parameter in this phase will be defined.  It must be
            compatible with the units of the targets to which the control is connected.
        desc : str
            A description of the control variable.
        opt : bool
            If True, the control value will be a design variable for the optimization problem.
            If False, allow the control to be connected externally.
        fix_initial : bool
            If True, the initial value of this control is fixed and not a design variable.
            This option is invalid if opt=False.
        fix_final : bool
            If True, the final value of this control is fixed and not a design variable.
            This option is invalid if opt=False.
        targets : Sequence of str or None
            Targets in the ODE to which this control is connected.
        rate_targets : Sequence of str or None
            The targets in the ODE to which the control rate is connected.
        rate2_targets : Sequence of str or None
            The parameter in the ODE to which the control 2nd derivative is connected.
        val : float
            The default value of the control variable at the control input nodes.
        shape : Sequence of int
            The shape of the control variable at each point in time.
        lower : Sequence of Number or None
            The lower bound of the control variable at the nodes.
            This option is invalid if opt=False.
        upper : Sequence or Number or None
            The upper bound of the control variable at the nodes.
            This option is invalid if opt=False.
        scaler : float or None
            The scaler of the control variable at the nodes.
            This option is invalid if opt=False.
        adder : float or None
            The adder of the control variable at the nodes.
            This option is invalid if opt=False.
        ref0 : float or None
            The zero-reference value of the control variable at the nodes.
            This option is invalid if opt=False.
        ref : float or None
            The unit-reference value of the control variable at the nodes.
            This option is invalid if opt=False.
        continuity : bool
            Enforce continuity of control values at segment boundaries.
            This option is invalid if opt=False.
        rate_continuity : bool
            Enforce continuity of control first derivatives  (in dimensionless time) at
            segment boundaries.
            This option is invalid if opt=False.
        rate_continuity_scaler : float
            Scaler of the rate continuity constraint at segment boundaries.
            This option is invalid if opt=False.
        rate2_continuity : bool
            Enforce continuity of control second derivatives at segment boundaries.
            This option is invalid if opt=False.
        rate2_continuity_scaler : float
            Scaler of the dimensionless rate continuity constraint at segment boundaries.
            This option is invalid if opt=False.
        dynamic : bool
            If True, the value of the shape of the parameter will be (num_nodes, ...),
            allowing the variable to be used as either a static or dynamic control.
            This impacts the shape of the partial derivatives matrix.  Unless a parameter is
            large and broadcasting a value to each individual node would be inefficient,
            users should stick to the default value of True.)

        Notes
        -----
        rate and rate2 continuity are not enforced for input controls.

        """
        self._check_parameter(name)

        if name not in self.user_control_options:
            self.user_control_options[name] = {}

        kwargs['name'] = name

        for kw in kwargs:
            if kw not in ControlOptionsDictionary():
                raise KeyError('Invalid argument to add_control: {0}'.format(kw))

        self.user_control_options[name].update(kwargs)

    def add_polynomial_control(self, name, **kwargs):
        """
        Adds an polynomial control variable to be tied to a parameter in the ODE.

        Polynomial controls are defined by values at the Legendre-Gauss-Lobatto nodes of a
        single polynomial, defined on [-1, 1] in phase tau space.

        For a polynomial control of a given order, the number of nodes used to define the
        polynomial is (order + 1).

        Parameters
        ----------
        name : str
            Name of the controllable parameter in the ODE.
        order : int
            The order of the interpolating polynomial used to represent the control valeu in
            phase tau space.
        val : float or ndarray
            Default value of the control at all nodes.  If val scalar and the control
            is dynamic it will be broadcast.
        units : str or None or 0
            Units in which the control variable is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True (default) the value(s) of this control will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False, the values of this control will exist as aainput controls:{name}
        lower : float or ndarray
            The lower bound of the control at the nodes of the phase.
        upper : float or ndarray
            The upper bound of the control at the nodes of the phase.
        scaler : float or ndarray
            The scaler of the control value at the nodes of the phase.
        adder : float or ndarray
            The adder of the control value at the nodes of the phase.
        ref0 : float or ndarray
            The zero-reference value of the control at the nodes of the phase.
        ref : float or ndarray
            The unit-reference value of the control at the nodes of the phase
        rate_param : None or str
            The name of the parameter in the ODE to which the first time-derivative
            of the control value is connected.
        rate2_param : None or str
            The name of the parameter in the ODE to which the second time-derivative
            of the control value is connected.
        targets : Sequence of str or None
            Targets in the ODE to which this polynomial control is connected.
        """
        self._check_parameter(name)

        if name not in self.user_polynomial_control_options:
            self.user_polynomial_control_options[name] = {}

        if 'order' not in kwargs:
            raise RuntimeError('Keyword argument \'order\' must be specified for polynomial '
                               'control \'{0}\''.format(name))

        kwargs['name'] = name

        for kw in kwargs:
            if kw not in PolynomialControlOptionsDictionary():
                raise KeyError('Invalid argument to add_polynomial_control: {0}'.format(kw))

        self.user_polynomial_control_options[name].update(kwargs)

    def add_design_parameter(self, name, **kwargs):
        """
        Add a design parameter (static control variable) to the phase.

        Parameters
        ----------
        name : str
            Name of the design parameter.
        val : float or ndarray
            Default value of the design parameter at all nodes.
        units : str or None or 0
            Units in which the design parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True (default) the value(s) of this design parameter will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False, the this design parameter will still be owned by an IndepVarComp in the phase,
            but it will not be a design variable in the optimization.
        lower : float or ndarray
            The lower bound of the design parameter value.
        upper : float or ndarray
            The upper bound of the design parameter value.
        scaler : float or ndarray
            The scaler of the design parameter value for the optimizer.
        adder : float or ndarray
            The adder of the design parameter value for the optimizer.
        ref0 : float or ndarray
            The zero-reference value of the design parameter for the optimizer.
        ref : float or ndarray
            The unit-reference value of the design parameter for the optimizer.
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.

        """
        self._check_parameter(name)

        if name not in self.user_design_parameter_options:
            self.user_design_parameter_options[name] = {}

        kwargs['name'] = name

        for kw in kwargs:
            if kw not in DesignParameterOptionsDictionary():
                raise KeyError('Invalid argument to add_design_parameter: {0}'.format(kw))

        self.user_design_parameter_options[name].update(kwargs)

    def add_input_parameter(self, name, **kwargs):
        """
        Add an input parameter (static control variable) to the phase.

        Parameters
        ----------
        name : str
            Name of the ODE parameter to be controlled via this input parameter.
        val : float or ndarray
            Default value of the design parameter at all nodes.
        units : str or None or 0
            Units in which the design parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        """
        self._check_parameter(name)

        if name not in self.user_input_parameter_options:
            self.user_input_parameter_options[name] = {}

        kwargs['name'] = name

        for kw in kwargs:
            if kw not in InputParameterOptionsDictionary():
                raise KeyError('Invalid argument to add_input_parameter: {0}'.format(kw))

        self.user_input_parameter_options[name].update(kwargs)

    def add_traj_parameter(self, name, **kwargs):
        """
        Add an input parameter to the phase that is connected to an input or design parameter
        in the parent trajectory.

        Parameters
        ----------
        name : str
            Name of the ODE parameter to be controlled via this input parameter.
        val : float or ndarray
            Default value of the design parameter at all nodes.
        units : str or None or 0
            Units in which the design parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        """
        self._check_parameter(name)

        if name not in self.user_traj_parameter_options:
            self.user_traj_parameter_options[name] = {}

        kwargs['name'] = name

        for kw in kwargs:
            if kw not in InputParameterOptionsDictionary():
                raise KeyError('Invalid argument to add_traj_parameter: {0}'.format(kw))

        self.user_traj_parameter_options[name].update(kwargs)

    def add_boundary_constraint(self, name, loc, constraint_name=None, units=None,
                                shape=None, indices=None, lower=None, upper=None, equals=None,
                                scaler=None, adder=None, ref=None, ref0=None, linear=False):
        r"""
        Add a boundary constraint to a variable in the phase.

        Parameters
        ----------
        name : string
            Name of the variable to constrain.  If name is not a state, control, or 'time',
            then this is assumed to be the path of the variable to be constrained in the ODE.
        loc : string
            The location of the boundary constraint ('initial' or 'final')
        constraint_name : string or None
            The name of the variable as provided to the boundary constraint comp.  By
            default this is the last element in `name` when split by dots.  The user may
            override the constraint name if splitting the path causes name collisions.
        units : str or None
            The units in which the boundary constraint is to be applied.  If None, use the
            units associated with the constrained output.  If provided, must be compatible with
            the variables units.
        shape : tuple, list, ndarray, or None
            The shape of the variable being boundary-constrained.  This can be inferred
            automatically for time, states, controls, and input/design parameters, but is required
            if the constrained variable is an output of the ODE system.
        indices : tuple, list, ndarray, or None
            The indices of the output variable to be boundary constrained.  Indices assumes C-order
            flattening.  For instance, when constraining element [0, 1] of a variable of shape
            [2, 2], indices would be [3].
        lower : float or ndarray, optional
            Lower boundary for the variable
        upper : float or ndarray, optional
            Upper boundary for the variable
        equals : float or ndarray, optional
            Equality constraint value for the variable
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        linear : bool
            Set to True if constraint is linear. Default is False.
        """
        if loc not in ['initial', 'final']:
            raise ValueError('Invalid boundary constraint location "{0}". Must be '
                             '"initial" or "final".'.format(loc))

        if constraint_name is None:
            constraint_name = name.split('.')[-1]

        bc_dict = self._initial_boundary_constraints \
            if loc == 'initial' else self._final_boundary_constraints

        bc_dict[name] = {}
        bc_dict[name]['constraint_name'] = constraint_name

        bc_dict[name]['shape'] = shape
        bc_dict[name]['indices'] = indices
        bc_dict[name]['lower'] = lower
        bc_dict[name]['upper'] = upper
        bc_dict[name]['equals'] = equals
        bc_dict[name]['scaler'] = scaler
        bc_dict[name]['adder'] = adder
        bc_dict[name]['ref0'] = ref0
        bc_dict[name]['ref'] = ref
        bc_dict[name]['linear'] = linear
        bc_dict[name]['units'] = units

    def add_path_constraint(self, name, constraint_name=None, units=None, shape=None, indices=None,
                            lower=None, upper=None, equals=None, scaler=None, adder=None, ref=None,
                            ref0=None, linear=False):
        r"""
        Add a path constraint to a variable in the phase.

        Parameters
        ----------
        name : string
            Name of the response variable in the system.
        constraint_name : string or None
            The name of the variable as provided to the boundary constraint comp.  By
            default this is the last element in `name` when split by dots.  The user may
            override the constraint name if splitting the path causes name collisions.
        units : str or None
            The units in which the boundary constraint is to be applied.  If None, use the
            units associated with the constrained output.  If provided, must be compatible with
            the variables units.
        shape : tuple, list, ndarray, or None
            The shape of the variable being boundary-constrained.  This can be inferred
            automatically for time, states, controls, and input/design parameters, but is required
            if the constrained variable is an output of the ODE system.
        indices : tuple, list, ndarray, or None
            The indices of the output variable to be path constrained.  Indices assumes C-order
            flattening.  For instance, when constraining element [0, 1] of a variable of shape
            [2, 2], indices would be [3].
        lower : float or ndarray, optional
            Lower boundary for the variable
        upper : float or ndarray, optional
            Upper boundary for the variable
        equals : float or ndarray, optional
            Equality constraint value for the variable
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        linear : bool
            Set to True if constraint is linear. Default is False.

        """
        if constraint_name is None:
            constraint_name = name.split('.')[-1]

        if name not in self._path_constraints:
            self._path_constraints[name] = {}
            self._path_constraints[name]['constraint_name'] = constraint_name

        self._path_constraints[name]['lower'] = lower
        self._path_constraints[name]['upper'] = upper
        self._path_constraints[name]['equals'] = equals
        self._path_constraints[name]['scaler'] = scaler
        self._path_constraints[name]['adder'] = adder
        self._path_constraints[name]['ref0'] = ref0
        self._path_constraints[name]['ref'] = ref
        self._path_constraints[name]['indices'] = indices
        self._path_constraints[name]['shape'] = shape
        self._path_constraints[name]['linear'] = linear
        self._path_constraints[name]['units'] = units

    def add_timeseries_output(self, name, output_name=None, units=None, shape=(1,)):
        r"""
        Add a variable to the timeseries outputs of the phase.

        Parameters
        ----------
        name : string
            The name of the variable to be used as a timeseries output.  Must be one of
            'time', 'time_phase', 't_initial', 't_duration', or one of the states, controls,
            control rates, or parameters in the phase.
        output_name : string or None
            The name of the variable as listed in the phase timeseries outputs.  By
            default this is the last element in `name` when split by dots.  The user may
            override the constraint name if splitting the path causes name collisions.
        units : str or None
            The units in which the boundary constraint is to be applied.  If None, use the
            units associated with the constrained output.  If provided, must be compatible with
            the variables units.
        shape : tuple
            The shape of the timeseries output variable.  This must be provided (if not scalar)
            since Dymos doesn't necessarily know the shape of ODE outputs until setup time.
        """
        if output_name is None:
            output_name = name.split('.')[-1]

        if name not in self._timeseries_outputs:
            self._timeseries_outputs[name] = {}
            self._timeseries_outputs[name]['output_name'] = output_name

        self._timeseries_outputs[name]['units'] = units
        self._timeseries_outputs[name]['shape'] = shape

    def add_objective(self, name, loc='final', index=None, shape=(1,), ref=None, ref0=None,
                      adder=None, scaler=None, parallel_deriv_color=None,
                      vectorize_derivs=False):
        """
        Allows the user to add an objective in the phase.  If name is not a state,
        control, control rate, or 'time', then this is assumed to be the path of the variable
        to be constrained in the RHS.

        Parameters
        ----------
        name : str
            Name of the objective variable.  This should be one of 'time', a state or control
            variable, or the path to an output from the top level of the RHS.
        loc : str
            Where in the phase the objective is to be evaluated.  Valid
            options are 'initial' and 'final'.  The default is 'final'.
        index : int, optional
            If variable is an array at each point in time, this indicates which index is to be
            used as the objective, assuming C-ordered flattening.
        shape : int, optional
            The shape of the objective variable, at a point in time
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        parallel_deriv_color : string
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        vectorize_derivs : bool
            If True, vectorize derivative calculations.
        """
        obj_dict = {'loc': loc,
                    'index': index,
                    'shape': shape,
                    'ref': ref,
                    'ref0': ref0,
                    'adder': adder,
                    'scaler': scaler,
                    'parallel_deriv_color': parallel_deriv_color,
                    'vectorize_derivs': vectorize_derivs}
        self._objectives[name] = obj_dict

    def _setup_objective(self):
        """
        Find the path of the objective(s) and add the objective using the standard OpenMDAO method.
        """
        for name, options in iteritems(self._objectives):
            index = options['index']
            loc = options['loc']

            obj_path, shape, units, _ = self._get_boundary_constraint_src(name, loc)

            shape = options['shape'] if shape is None else shape

            size = int(np.prod(shape))

            if size > 1 and index is None:
                raise ValueError('Objective variable is non-scaler {0} but no index specified '
                                 'for objective'.format(shape))

            idx = 0 if index is None else index
            if idx < 0:
                idx = size + idx

            if idx >= size or idx < -size:
                raise ValueError('Objective index={0}, but the shape of the objective '
                                 'variable is {1}'.format(index, shape))

            if loc == 'final':
                obj_index = -size + idx
            elif loc == 'initial':
                obj_index = idx
            else:
                raise ValueError('Invalid value for objective loc: {0}. Must be '
                                 'one of \'initial\' or \'final\'.'.format(loc))

            super(PhaseBase, self).add_objective(obj_path, ref=options['ref'], ref0=options['ref0'],
                                                 index=obj_index, adder=options['adder'],
                                                 scaler=options['scaler'],
                                                 parallel_deriv_color=options['parallel_deriv_color'],
                                                 vectorize_derivs=options['vectorize_derivs'])

    def set_time_options(self, **kwargs):
        """
        Set options for the time (or the integration variable) in the Phase.

        Parameters
        ----------
        opt_initial : bool, deprecated
            If True, the initial time of the phase is a design variable
            for optimization, otherwise False. This option is deprecated in favor of fix_initial.
        opt_duration : bool, deprecated
            If True, the duration of the phase is a design variable
            for optimization, otherwise False. This option is deprecated in favor of fix_duration.
        fix_initial : bool
            If True, the initial time of the phase is not a design variable.
        fix_duration : bool
            If True, the duration of the phase is not a design variable
        input_initial : bool
            If True, the user is expected to link phase.t_initial to an external output source.
            Providing input_initial=True makes all initial time optimization settings irrelevant.
        input_duration : bool
            If True, the user is expected to link phase.t_duration to an external output source.
            Providing input_duration=True makes all time duration optimization settings irrelevant.
        initial_val : float
            Default value of the time at the start of the phase.
        initial_bounds : Iterable of size 2
            Tuple of (lower, upper) bounds for time at the start of the phase.
        initial_scaler : float
            Scalar for the initial value of time.
        initial_adder : float
            Adder for the initial value of time.
        initial_ref0 : float
            Zero-reference value for the initial value of time.
        initial_ref : float
            Unit-reference value for the initial value of time.
        duration_val : float
            Value of the duration of time across the phase.
        duration_bounds : Iterable of size 2
            Tuple of (lower, upper) bounds for the duration of time
            across the phase.
        duration_scaler : float
            Scalar for the duration of time across the phase.
        duration_adder : float
            Adder for the duration of time across the phase.
        duration_ref0 : float
            Zero-reference value for the duration of time across the phase.
        duration_ref : float
            Unit-reference value for the duration of time across the phase.
        """
        self.user_time_options.update(kwargs)

    def _classify_var(self, var):
        """
        Classifies a variable of the given name or path.

        This method searches for it as a time variable, state variable,
        control variable, or parameter.  If it is not found to be one
        of those variables, it is assumed to be the path to a variable
        relative to the top of the ODE system for the phase.

        Parameters
        ----------
        var : str
            The name of the variable to be classified.

        Returns
        -------
        str
            The classification of the given variable, which is one of
            'time', 'state', 'input_control', 'indep_control', 'control_rate',
            'control_rate2', 'input_polynomial_control', 'indep_polynomial_control',
            'polynomial_control_rate', 'polynomial_control_rate2', 'design_parameter',
            'input_parameter', or 'ode'.

        """
        if var == 'time':
            return 'time'
        elif var == 'time_phase':
            return 'time_phase'
        elif var in self.state_options:
            return 'state'
        elif var in self.control_options:
            if self.control_options[var]['opt']:
                return 'indep_control'
            else:
                return 'input_control'
        elif var in self.polynomial_control_options:
            if self.polynomial_control_options[var]['opt']:
                return 'indep_polynomial_control'
            else:
                return 'input_polynomial_control'
        elif var in self.design_parameter_options:
            return 'design_parameter'
        elif var in self.input_parameter_options:
            return 'input_parameter'
        elif var in self.traj_parameter_options:
            return 'traj_parameter'
        elif var.endswith('_rate') and var[:-5] in self.control_options:
                return 'control_rate'
        elif var.endswith('_rate2') and var[:-6] in self.control_options:
                return 'control_rate2'
        elif var.endswith('_rate') and var[:-5] in self.polynomial_control_options:
                return 'polynomial_control_rate'
        elif var.endswith('_rate2') and var[:-6] in self.polynomial_control_options:
                return 'polynomial_control_rate2'
        else:
            return 'ode'

    def finalize_variables(self):
        """ Finalize the variable options by combining the user-defined options and the ODE options.

        First apply any variable options that may be defined via ODEOptions properties on the ODE
        class.  Then apply any user-specified options over those.
        """
        self.time_options = TimeOptionsDictionary()
        self.state_options = {}
        self.control_options = {}
        self.polynomial_control_options = {}
        self.design_parameter_options = {}
        self.input_parameter_options = {}
        self.traj_parameter_options = {}

        # First apply any defaults set in the ode options
        if self.options['ode_class'] is not None and hasattr(self.options['ode_class'], 'ode_options'):
            ode_options = self.options['ode_class'].ode_options
        else:
            ode_options = None

        # Now update with any user-supplied options
        if ode_options:
            self.time_options.update(ode_options._time_options)
        self.time_options.update(self.user_time_options)

        if ode_options:
            for state in ode_options._states:
                self.state_options[state] = StateOptionsDictionary()
                self.state_options[state].update(ode_options._states[state])
        for state in list(self.user_state_options.keys()):
            if state not in self.state_options:
                self.state_options[state] = StateOptionsDictionary()
            self.state_options[state].update(self.user_state_options[state])

        for control in list(self.user_control_options.keys()):
            self.control_options[control] = ControlOptionsDictionary()
            if ode_options and control in ode_options._parameters:
                self.control_options[control].update(ode_options._parameters[control])
            self.control_options[control].update(self.user_control_options[control])

        for pc in list(self.user_polynomial_control_options.keys()):
            self.polynomial_control_options[pc] = PolynomialControlOptionsDictionary()
            if ode_options and pc in ode_options._parameters:
                self.polynomial_control_options[pc].update(ode_options._parameters[pc])
            self.polynomial_control_options[pc].update(self.user_polynomial_control_options[pc])

        for dp in list(self.user_design_parameter_options.keys()):
            self.design_parameter_options[dp] = DesignParameterOptionsDictionary()
            if ode_options and dp in ode_options._parameters:
                self.design_parameter_options[dp].update(ode_options._parameters[dp])
            self.design_parameter_options[dp].update(self.user_design_parameter_options[dp])

        for ip in list(self.user_input_parameter_options.keys()):
            self.input_parameter_options[ip] = InputParameterOptionsDictionary()
            if ode_options and ip in ode_options._parameters:
                self.input_parameter_options[ip].update(ode_options._parameters[ip])
            self.input_parameter_options[ip].update(self.user_input_parameter_options[ip])

        for tp in list(self.user_traj_parameter_options.keys()):
            self.traj_parameter_options[tp] = InputParameterOptionsDictionary()
            if ode_options and tp in ode_options._parameters:
                self.traj_parameter_options[tp].update(ode_options._parameters[tp])
            self.traj_parameter_options[tp].update(self.user_traj_parameter_options[tp])

    def setup(self):
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        self.finalize_variables()

        self._time_extents = self._setup_time()

        # The control interpolation comp to which we'll connect controls
        if self.control_options:
            self._setup_controls()

        if self.polynomial_control_options:
            self._setup_polynomial_controls()

        if self.input_parameter_options:
            self._setup_input_parameters()

        if self.design_parameter_options:
            self._setup_design_parameters()

        if self.traj_parameter_options:
            self._setup_traj_input_parameters()

        self._setup_rhs()
        self._setup_defects()
        self._setup_states()

        self._setup_endpoint_conditions()
        self._setup_boundary_constraints('initial')
        self._setup_boundary_constraints('final')
        self._setup_path_constraints()
        self._setup_objective()

        self._setup_timeseries_outputs()

        self.is_setup = True

    def _check_time_options(self):
        """
        Check that time options are valid and issue warnings if invalid options are provided.

        Warnings
        --------
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        if self.time_options['fix_initial'] or self.time_options['input_initial']:
            invalid_options = []
            init_bounds = self.time_options['initial_bounds']
            if init_bounds is not None and init_bounds != (None, None):
                invalid_options.append('initial_bounds')
            for opt in 'initial_scaler', 'initial_adder', 'initial_ref', 'initial_ref0':
                if self.time_options[opt] is not None:
                    invalid_options.append(opt)
            if invalid_options:
                warnings.warn('Phase time options have no effect because fix_initial=True for '
                              'phase \'{0}\': {1}'.format(self.name, ', '.join(invalid_options)),
                              RuntimeWarning)

        if self.time_options['fix_initial'] and self.time_options['input_initial']:
            warnings.warn('Phase \'{0}\' initial time is an externally-connected input, '
                          'therefore fix_initial has no effect.'.format(self.name),
                          RuntimeWarning)

        if self.time_options['fix_duration'] or self.time_options['input_duration']:
            invalid_options = []
            duration_bounds = self.time_options['duration_bounds']
            if duration_bounds is not None and duration_bounds != (None, None):
                invalid_options.append('duration_bounds')
            for opt in 'duration_scaler', 'duration_adder', 'duration_ref', 'duration_ref0':
                if self.time_options[opt] is not None:
                    invalid_options.append(opt)
            if invalid_options:
                warnings.warn('Phase time options have no effect because fix_duration=True for '
                              'phase \'{0}\': {1}'.format(self.name, ', '.join(invalid_options)))

        if self.time_options['fix_duration'] and self.time_options['input_duration']:
            warnings.warn('Phase \'{0}\' time duration is an externally-connected input, '
                          'therefore fix_duration has no effect.'.format(self.name),
                          RuntimeWarning)

    def _check_control_options(self):
        """
        Check that time options are valid and issue warnings if invalid options are provided.

        Warnings
        --------
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        for name, options in iteritems(self.control_options):
            if not options['opt']:
                invalid_options = []
                for opt in 'lower', 'upper', 'scaler', 'adder', 'ref', 'ref0':
                    if options[opt] is not None:
                        invalid_options.append(opt)
                if invalid_options:
                    warnings.warn('Invalid options for non-optimal control \'{0}\' in phase \'{1}\': '
                                  '{2}'.format(name, self.name, ', '.join(invalid_options)),
                                  RuntimeWarning)

                # Do not enforce rate continuity/rate continuity for non-optimal controls
                self.control_options[name]['continuity'] = False
                self.control_options[name]['rate_continuity'] = False
                self.control_options[name]['rate2_continuity'] = False

    def _check_design_parameter_options(self):
        """
        Check that time options are valid and issue warnings if invalid options are provided.

        Warnings
        --------
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        for name, options in iteritems(self.design_parameter_options):
            if not options['opt']:
                invalid_options = []
                for opt in 'lower', 'upper', 'scaler', 'adder', 'ref', 'ref0':
                    if options[opt] is not None:
                        invalid_options.append(opt)
                if invalid_options:
                    warnings.warn('Invalid options for non-optimal design_parameter \'{0}\' in '
                                  'phase \'{1}\': {2}'.format(name, self.name, ', '.join(invalid_options)),
                                  RuntimeWarning)

    def _setup_time(self):
        """
        Setup up the time component and time extents for the phase.

        Returns
        -------
        comps
            A list of the component names needed for time extents.
        """
        time_units = self.time_options['units']

        indeps = []
        default_vals = {'t_initial': self.time_options['initial_val'],
                        't_duration': self.time_options['duration_val']}
        externals = []
        comps = []

        # Warn about invalid options
        self._check_time_options()

        if self.time_options['input_initial']:
            externals.append('t_initial')
        else:
            indeps.append('t_initial')
            self.connect('t_initial', 'time.t_initial')

        if self.time_options['input_duration']:
            externals.append('t_duration')
        else:
            indeps.append('t_duration')
            self.connect('t_duration', 'time.t_duration')

        if indeps:
            indep = IndepVarComp()

            for var in indeps:
                indep.add_output(var, val=default_vals[var], units=time_units)

            self.add_subsystem('time_extents', indep, promotes_outputs=['*'])
            comps += ['time_extents']

        if not (self.time_options['input_initial'] or self.time_options['fix_initial']):
            lb, ub = self.time_options['initial_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            self.add_design_var('t_initial',
                                lower=lb,
                                upper=ub,
                                scaler=self.time_options['initial_scaler'],
                                adder=self.time_options['initial_adder'],
                                ref0=self.time_options['initial_ref0'],
                                ref=self.time_options['initial_ref'])

        if not (self.time_options['input_duration'] or self.time_options['fix_duration']):
            lb, ub = self.time_options['duration_bounds']
            lb = -INF_BOUND if lb is None else lb
            ub = INF_BOUND if ub is None else ub

            self.add_design_var('t_duration',
                                lower=lb,
                                upper=ub,
                                scaler=self.time_options['duration_scaler'],
                                adder=self.time_options['duration_adder'],
                                ref0=self.time_options['duration_ref0'],
                                ref=self.time_options['duration_ref'])

        return indeps, externals, comps

    def _setup_controls(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        self._check_control_options()

        if self.control_options:
            control_group = ControlGroup(control_options=self.control_options,
                                         time_units=self.time_options['units'],
                                         grid_data=self.grid_data)

            self.add_subsystem('control_group',
                               subsys=control_group,
                               promotes=['controls:*', 'control_values:*', 'control_rates:*'])
            self.connect('time.dt_dstau', 'control_group.dt_dstau')

    def _setup_polynomial_controls(self):
        """
        Adds the polynomial control group to the model if any polynomial controls are present.
        """
        if self.polynomial_control_options:
            sys = PolynomialControlGroup(grid_data=self.grid_data,
                                         polynomial_control_options=self.polynomial_control_options,
                                         time_units=self.time_options['units'])
            self.add_subsystem('polynomial_control_group', subsys=sys,
                               promotes_inputs=['*'], promotes_outputs=['*'])

    def _setup_design_parameters(self):
        """
        Adds an IndepVarComp if necessary and issues appropriate connections based
        on transcription.
        """
        self._check_design_parameter_options()

        if self.design_parameter_options:
            indep = self.add_subsystem('design_params', subsys=IndepVarComp(),
                                       promotes_outputs=['*'])

        for name, options in iteritems(self.design_parameter_options):
            src_name = 'design_parameters:{0}'.format(name)

            if options['opt']:
                lb = -INF_BOUND if options['lower'] is None else options['lower']
                ub = INF_BOUND if options['upper'] is None else options['upper']

                self.add_design_var(name=src_name,
                                    lower=lb,
                                    upper=ub,
                                    scaler=options['scaler'],
                                    adder=options['adder'],
                                    ref0=options['ref0'],
                                    ref=options['ref'])

            _shape = (1,) + options['shape']

            indep.add_output(name=src_name,
                             val=options['val'],
                             shape=_shape,
                             units=options['units'])

            for tgts, src_idxs in self._get_parameter_connections(name):
                self.connect(src_name, [t for t in tgts],
                             src_indices=src_idxs, flat_src_indices=True)

    def _setup_input_parameters(self):
        """
        Adds a InputParameterComp to allow input parameters to be connected from sources
        external to the phase.
        """
        if self.input_parameter_options:
            passthru = \
                InputParameterComp(input_parameter_options=self.input_parameter_options)

            self.add_subsystem('input_params', subsys=passthru, promotes_inputs=['*'],
                               promotes_outputs=['*'])

        for name in self.input_parameter_options:
            src_name = 'input_parameters:{0}_out'.format(name)

            for tgts, src_idxs in self._get_parameter_connections(name):
                self.connect(src_name, [t for t in tgts],
                             src_indices=src_idxs, flat_src_indices=True)

    def _setup_traj_input_parameters(self):
        """
        Adds a InputParameterComp to allow input parameters to be connected from sources
        external to the phase.
        """
        if self.traj_parameter_options:
            passthru = \
                InputParameterComp(input_parameter_options=self.traj_parameter_options,
                                   traj_params=True)

            self.add_subsystem('traj_params', subsys=passthru, promotes_inputs=['*'],
                               promotes_outputs=['*'])

        for name, options in iteritems(self.traj_parameter_options):
            src_name = 'traj_parameters:{0}_out'.format(name)

            for tgts, src_idxs in self._get_parameter_connections(name):
                self.connect(src_name, [t for t in tgts], src_indices=src_idxs)

    def _get_parameter_connections(self, name):
        """
        Returns a list containing tuples of each path and related indices to which the
        given parameter name is to be connected.

        Returns
        -------
        connection_info : list of (paths, indices)
            A list containing a tuple of target paths and corresponding src_indices to which the
            given design variable is to be connected.
        """
        raise NotImplementedError('Phase class {0} does not implement '
                                  '_get_parameter_connections'.format(self.__class__.__name__))

    def _get_rate_source_path(self, state_name, nodes=None, **kwargs):
        """
        Given the name of a variable to be used as a rate source, provide the source connection
        path for that variable in the Phase.

        Parameters
        ----------
        state_name : str
            The name of the state variable whose source path and indices is desired.
        nodes : str
            The name of the node subset from which the rate source is desired.

        Returns
        -------
        path : str
            The full path to the rate source in the system.
        src_idxs : np.ndarray
            The source indices in the resulting src that provide the values at the given nodes.
        """
        raise NotImplementedError('Phase class {0} does not implement '
                                  '_get_rate_source_path'.format(self.__class__.__name__))

    def _setup_rhs(self):
        if not inspect.isclass(self.options['ode_class']):
            raise ValueError('ode_class must be a class, not an instance.')
        if not issubclass(self.options['ode_class'], System):
            raise ValueError('ode_class must be derived from openmdao.core.System.')

    def _setup_defects(self):
        raise NotImplementedError()

    def _setup_states(self):
        raise NotImplementedError()

    def _setup_endpoint_conditions(self):
        raise NotImplementedError()

    def _get_boundary_constraint_src(self, var, loc):
        """
        Get the path of the boundary constraint source within the phase.

        Parameters
        ----------
        var : str
            The variable within the phase whose value is to be constrained at a boundary.
        loc : str
            The location of the boundary constraint in the phase.  Must be one of 'initial' or
            'final'.

        Returns
        -------
        src : str
            The phase-relative path of the boundary constraint source.
        src_idxs : np.array of int
            The source indices that of src that provide the boundary constraint values.
        shape : tuple of int
            The shape of the variable being boundary-constrained.
        units : str
            The units of the output to be boundary-constrained.
        linear : bool
            True if this boundary constraint is constrained linearly, otherwise False.

        """
        raise NotImplementedError('This phase class does not implement '
                                  '_get_boundary_constraint_src.')

    def _setup_boundary_constraints(self, loc):
        """
        Adds BoundaryConstraintComp for initial and/or final boundary constraints if necessary
        and issues appropriate connections.

        Parameters
        ----------
        loc : str
            The kind of boundary constraints being setup.  Must be one of 'initial' or 'final'.

        """
        if loc not in ('initial', 'final'):
            raise ValueError('loc must be one of \'initial\' or \'final\'.')
        bc_comp = None

        bc_dict = self._initial_boundary_constraints \
            if loc == 'initial' else self._final_boundary_constraints

        if bc_dict:
            bc_comp = self.add_subsystem('{0}_boundary_constraints'.format(loc),
                                         subsys=BoundaryConstraintComp(loc=loc))

        for var, options in iteritems(bc_dict):
            con_name = options['constraint_name']

            # Constraint options are a copy of options with constraint_name key removed.
            con_options = options.copy()
            con_options.pop('constraint_name')

            src, shape, units, linear = self._get_boundary_constraint_src(var, loc)

            con_units = options.get('units', None)

            shape = options['shape'] if shape is None else shape
            if shape is None:
                warnings.warn('\nUnable to infer shape of boundary constraint {0}. Assuming scalar. '
                              '\nIf variable is not scalar, provide shape in '
                              'add_boundary_constraint. \nIn Dymos 1.0 an error will be raised if '
                              'a constrained ODE output shape is not specified in '
                              'add_boundary_constraint.'.format(var), DeprecationWarning)
                shape = (1,)

            if options['indices'] is not None:
                # Indices are provided, make sure lower/upper/equals are compatible.
                con_shape = (len(options['indices']),)
                # Indices provided, make sure lower/upper/equals have shape of the indices.
                if options['lower'] and not np.isscalar(options['lower']) and \
                        np.asarray(options['lower']).shape != con_shape:
                    raise ValueError('The lower bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['upper'] and not np.isscalar(options['upper']) and \
                        np.asarray(options['upper']).shape != con_shape:
                    raise ValueError('The upper bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['equals'] and not np.isscalar(options['equals']) and \
                        np.asarray(options['equals']).shape != con_shape:
                    raise ValueError('The equality boundary constraint value on {0} is not '
                                     'compatible the provided indices. Provide them as a '
                                     'flat array with the same size as indices.'.format(var))

            elif options['lower'] or options['upper'] or options['equals']:
                # Indices not provided, make sure lower/upper/equals have shape of source.
                if options['lower'] and not np.isscalar(options['lower']) and \
                        np.asarray(options['lower']).shape != shape:
                    raise ValueError('The lower bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['upper'] and not np.isscalar(options['upper']) and \
                        np.asarray(options['upper']).shape != shape:
                    raise ValueError('The upper bounds of boundary constraint on {0} are not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))

                if options['equals'] and not np.isscalar(options['equals']) \
                        and np.asarray(options['equals']).shape != shape:
                    raise ValueError('The equality boundary constraint value on {0} is not '
                                     'compatible with its shape, and no indices were '
                                     'provided.'.format(var))
                con_shape = (np.prod(shape),)

            size = np.prod(shape)
            con_options['shape'] = shape if shape is not None else con_shape
            con_options['units'] = units if con_units is None else con_units
            con_options['linear'] = linear

            # Build the correct src_indices regardless of shape
            if loc == 'initial':
                src_idxs = np.arange(size, dtype=int).reshape(shape)
            else:
                src_idxs = np.arange(-size, 0, dtype=int).reshape(shape)

            bc_comp._add_constraint(con_name, **con_options)

            self.connect(src,
                         '{0}_boundary_constraints.{0}_value_in:{1}'.format(loc, con_name),
                         src_indices=src_idxs,
                         flat_src_indices=True)

    def _setup_path_constraints(self):
        raise NotImplementedError('_setup_path_constraints has not been implemented '
                                  'for phase type {0}'.format(self.__class__))

    def _setup_timeseries_outputs(self):
        raise NotImplementedError('_setup_timeseries_outputs has not been implemented '
                                  'for phase type {0}'.format(self.__class__))

    def set_values(self, var, value, nodes=None, kind='linear', axis=0):
        """
        Retrieve the values of the given variable at the given
        subset of nodes.

        Parameters
        ----------
        var : str
            The variable whose values are to be returned.  This may be
            the name 'time', the name of a state, control, or parameter,
            or the path to a variable in the ODE system of the phase.
        value : ndarray
            Array of time/control/state/parameter values.
        nodes : str
            The name of the node subset or None (default).
        kind : str
            Specifies the kind of interpolation, as per the scipy.interpolate package.
            One of ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.
        axis : int
            Specifies the axis along which interpolation should be performed.  Default is
            the first axis (0).
        """
        raise NotImplementedError('set_values has not been implemented for this class.')

    def interpolate(self, xs=None, ys=None, nodes='all', kind='linear', axis=0):
        """
        Return an array of values on interpolated to the given node subset of the phase.

        Parameters
        ----------
        xs :  ndarray
            Array of integration variable values.
        ys :  ndarray
            Array of control/state/parameter values.
        nodes : str or None
            The name of the node subset or None (default).
        kind : str
            Specifies the kind of interpolation, as per the scipy.interpolate package.
            One of ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
            where 'zero', 'slinear', 'quadratic' and 'cubic' refer to a spline
            interpolation of zeroth, first, second or third order) or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.
        axis : int
            Specifies the axis along which interpolation should be performed.  Default is
            the first axis (0).

        Returns
        -------
        np.array
            The values of y interpolated at nodes of the specified type.
        """
        if not isinstance(ys, Iterable):
            raise ValueError('ys must be provided as an Iterable of length at least 2.')
        if nodes not in ('col', 'disc', 'all', 'state_disc', 'state_input', 'control_disc',
                         'control_input', 'segment_ends'):
            raise ValueError("nodes must be one of 'col', 'all', 'state_disc', "
                             "'state_input', 'control_disc', 'control_input', or 'segment_ends'")
        if xs is None:
            if len(ys) != 2:
                raise ValueError('xs may only be unspecified when len(ys)=2')
            if kind != 'linear':
                raise ValueError('kind must be linear when xs is unspecified.')
            xs = [-1, 1]
        elif len(xs) != np.prod(np.asarray(xs).shape):
            raise ValueError('xs must be viewable as a 1D array')

        node_locations = self.grid_data.node_ptau[self.grid_data.subset_node_indices[nodes]]
        # if self.options['compressed']:
        #     node_locations = np.array(sorted(list(set(node_locations))))
        # Affine transform xs into tau space [-1, 1]
        _xs = np.asarray(xs).ravel()
        m = 2.0 / (_xs[-1] - _xs[0])
        b = 1.0 - (m * _xs[-1])
        taus = m * _xs + b
        interpfunc = interpolate.interp1d(taus, ys, axis=axis, kind=kind,
                                          bounds_error=False, fill_value='extrapolate')
        res = np.atleast_2d(interpfunc(node_locations))
        if res.shape[0] == 1:
            res = res.T
        return res

    def get_simulation_phase(self, times_per_seg=None, method='RK45', atol=1.0E-9, rtol=1.0E-9):
        """
        Return a SolveIVPPhase initialized based on data from this Phase instance and
        the given simulation times.

        Parameters
        ----------
        times_per_seg : int or None
            Number of equally distributed output times per segment in the phase simulation.  If
            None, output to all nodes provided by this phases GridData.
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for the scipy.integrate.solve_ivp method.
        rtol : float
            Relative convergence tolerance for the scipy.integrate.solve_ivp method.

        Returns
        -------
        SimulationPhase
            An instance of SimulationPhase initialized based on data from this Phase and the given
            times.  This instance has not yet been setup.
        """

        from .solve_ivp.solve_ivp_phase import SolveIVPPhase

        sim_phase = SolveIVPPhase(from_phase=self,
                                  method=method,
                                  atol=atol,
                                  rtol=rtol,
                                  output_nodes_per_seg=times_per_seg)

        return sim_phase

    def simulate(self, times_per_seg=10, method='RK45', atol=1.0E-9, rtol=1.0E-9,
                 record_file=None):
        """
        Simulate the Phase using scipy.integrate.solve_ivp.

        Parameters
        ----------
        times_per_seg : int or None
            Number of equally spaced times per segment at which output is requested.  If None,
            output will be provided at all Nodes.
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for scipy.integrate.solve_ivp.
        rtol : float
            Relative convergence tolerance for scipy.integrate.solve_ivp.
        record_file : str or None
            If a string, the file to which the result of the simulation will be saved.
            If None, no record of the simulation will be saved.

        Returns
        -------
        problem
            An OpenMDAO Problem in which the simulation is implemented.  This Problem interface
            can be interrogated to obtain timeseries outputs in the same manner as other Phases
            to obtain results at the requested times.

        """

        sim_prob = Problem(model=Group())

        sim_phase = self.get_simulation_phase(times_per_seg, method=method, atol=atol, rtol=rtol)

        sim_prob.model.add_subsystem(self.name, sim_phase)

        if record_file is not None:
            rec = SqliteRecorder(record_file)
            sim_prob.model.recording_options['includes'] = ['*.timeseries.*']
            sim_prob.model.add_recorder(rec)

        sim_prob.setup(check=True)

        sim_phase.initialize_values_from_phase(sim_prob)

        print('\nSimulating phase {0}'.format(self.pathname))
        sim_prob.run_model()
        print('Done simulating phase {0}'.format(self.pathname))

        sim_prob.cleanup()

        return sim_prob
