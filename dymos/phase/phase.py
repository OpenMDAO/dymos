from __future__ import division, print_function, absolute_import

from collections import Iterable, Sequence
import inspect
import warnings

from six import iteritems, string_types

import numpy as np

from scipy import interpolate

import openmdao.api as om
from openmdao.core.system import System
import dymos as dm

from .options import ControlOptionsDictionary, DesignParameterOptionsDictionary, \
    InputParameterOptionsDictionary, StateOptionsDictionary, TimeOptionsDictionary, \
    PolynomialControlOptionsDictionary

from ..transcriptions.transcription_base import TranscriptionBase


_unspecified = object()


class Phase(om.Group):
    """
    The Phase object in Dymos.

    The Phase object is an OpenMDAO Group which contains the options for the variables in the
    optimal control problem (states, times, controls, parameters), the transcription, and
    the ODE class.

    The role of the Phase is to unite the problem formulation with the transcription and the ODE
    in order to transcribe a single portion of a trajectory into a nonlinear programming problem
    to be solved by the optimizer.

    On setup, the Phase runs through its setup stack which will add the appropriate OpenMDAO
    systems as prescribed by its associated Transcription.

    Parameters
    ----------
    from_phase: Phase or None
        A phase instance from which the initialized phase should copy its data.
    transcription: TranscriptionBase
        The transcription to be utilized within the Phase.
    ode_class
        An OpenMDAO system class serving as the ODE for the phase.
    ode_init_kwargs: dict
        Keyword arguments used to initialize ode_class.

    """

    def __init__(self, from_phase=None, **kwargs):

        _kwargs = kwargs.copy()

        # Dictionaries of variable options that are set by the user via the API
        # These will be applied over any defaults specified by decorators on the ODE
        if from_phase is None:
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
            self._timeseries = {}
            self._timeseries['timeseries'] = {'transcription': None,
                                              'subset': 'all',
                                              'outputs': {}}
            self._objectives = {}
        else:
            self.user_time_options = TimeOptionsDictionary()
            self.user_time_options.update(from_phase.time_options)
            self.user_state_options = from_phase.state_options.copy()
            self.user_control_options = from_phase.control_options.copy()
            self.user_polynomial_control_options = from_phase.polynomial_control_options.copy()
            self.user_design_parameter_options = from_phase.design_parameter_options.copy()
            self.user_input_parameter_options = from_phase.input_parameter_options.copy()
            # Don't copy over the trajectory parameters.  The owning trajectory object will
            # handle that.
            self.user_traj_parameter_options = {}

            self._initial_boundary_constraints = from_phase._initial_boundary_constraints.copy()
            self._final_boundary_constraints = from_phase._final_boundary_constraints.copy()
            self._path_constraints = from_phase._path_constraints.copy()
            self._timeseries = from_phase._timeseries.copy()
            self._objectives = from_phase._objectives.copy()

            _kwargs['ode_class'] = from_phase.options['ode_class']
            _kwargs['ode_init_kwargs'] = from_phase.options['ode_init_kwargs']

        super(Phase, self).__init__(**_kwargs)

    def initialize(self):
        self.options.declare('ode_class', default=None,
                             desc='System defining the ODE')
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('transcription', types=TranscriptionBase,
                             desc='Transcription technique of the optimal control problem.')
        self.options.declare('timeseries', types=(dict,),
                             desc='Alternative timeseries.')

    def add_state(self, name, units=_unspecified, shape=_unspecified,
                  rate_source=_unspecified, targets=_unspecified,
                  val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                  lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                  ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                  defect_ref=_unspecified, solve_segments=_unspecified, connected_initial=_unspecified):
        """
        Add a state variable to be integrated by the phase.

        Parameters
        ----------
        name : str
            Name of the state variable in the RHS.
        units : str or None
            Units in which the state variable is defined.  Internally components may use different
            units for the state variable, but the IndepVarComp which provides its value will provide
            it in these units, and collocation defects will use these units.  If units is not
            specified here then the value as defined in the ODEOptions (@dm.declare_state) will be
            used.
        shape : tuple of int
            The shape of the state variable.  For instance, a 3D cartesian position vector would have
            a shape of (3,).
        rate_source : str
            The path to the ODE output which provides the rate of this state variable.
        targets : str or Sequence of str
            The path to the targets of the state variable in the ODE system.  If given
            this will override the value given by the @declare_state decorator on the ODE.
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
        defect_scaler : float or ndarray
            The scaler of the state defect at the collocation nodes of the phase.
        defect_ref : float or ndarray
            The unit-reference value of the state defect at the collocation nodes of the phase. If
            provided, this value overrides defect_scaler.
        solve_segments : bool(False)
            If True, a solver will be used to converge the collocation defects within a segment.
            Note that the state continuity defects between segements will still be
            handled by the optimizer.
        connected_initial : bool
            If True, then the initial value for this state comes from an externally connected
            source.
        """
        if name not in self.user_state_options:
            self.user_state_options[name] = {'name': name}

        if units is not _unspecified:
            self.user_state_options[name]['units'] = units

        if shape is not _unspecified:
            self.user_state_options[name]['shape'] = shape

        if rate_source is not _unspecified:
            self.user_state_options[name]['rate_source'] = rate_source

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.user_state_options[name]['targets'] = (targets,)
            else:
                self.user_state_options[name]['targets'] = targets

        if val is not _unspecified:
            self.user_state_options[name]['val'] = val

        if fix_initial is not _unspecified:
            self.user_state_options[name]['fix_initial'] = fix_initial

        if fix_final is not _unspecified:
            self.user_state_options[name]['fix_final'] = fix_final

        if lower is not _unspecified:
            self.user_state_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.user_state_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.user_state_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.user_state_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.user_state_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.user_state_options[name]['ref'] = ref

        if defect_scaler is not _unspecified:
            self.user_state_options[name]['defect_scaler'] = defect_scaler

        if defect_ref is not _unspecified:
            self.user_state_options[name]['defect_ref'] = defect_ref

        if solve_segments is not _unspecified:
            self.user_state_options[name]['solve_segments'] = solve_segments

        if connected_initial is not _unspecified:
            self.user_state_options[name]['connected_initial'] = connected_initial

    def set_state_options(self, name, units=_unspecified, shape=_unspecified,
                          rate_source=_unspecified, targets=_unspecified,
                          val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                          lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                          ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                          defect_ref=_unspecified, solve_segments=_unspecified, connected_initial=_unspecified):
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
            specified here then the value as defined in the ODEOptions (@dm.declare_state) will be
            used.
        shape : tuple of int
            The shape of the state variable.  For instance, a 3D cartesian position vector would have
            a shape of (3,).
        rate_source : str
            The path to the ODE output which provides the rate of this state variable.
        targets : str or Sequence of str
            The path to the targets of the state variable in the ODE system.  If given
            this will override the value given by the @declare_state decorator on the ODE.
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
        defect_scaler : float or ndarray
            The scaler of the state defect at the collocation nodes of the phase.
        defect_ref : float or ndarray
            The unit-reference value of the state defect at the collocation nodes of the phase. If
            provided, this value overrides defect_scaler.
        solve_segments : bool(False)
            If True, a solver will be used to converge the collocation defects within a segment.
            Note that the state continuity defects between segements will still be
            handled by the optimizer.
        connected_initial : bool
            If True, then the initial value for this state comes from an externally connected
            source.
        """
        warnings.warn('Phase method set_state_options for Dymos has been deprecated. '
                      'Use the add_state method on phase to provide units, targets and '
                      'other options for the state variable.', DeprecationWarning)

        self.add_state(name=name, units=units, shape=shape, rate_source=rate_source, targets=targets,
                       val=val, fix_initial=fix_initial, fix_final=fix_final, lower=lower,
                       upper=upper, scaler=scaler, adder=adder, ref0=ref0, ref=ref,
                       defect_scaler=defect_scaler, defect_ref=defect_ref,
                       solve_segments=solve_segments, connected_initial=connected_initial)

    def check_parameter(self, name):
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
            raise ValueError('{0} has already been added as a polynomial control.'.format(name))
        if name in self.user_traj_parameter_options:
            raise ValueError('{0} has already been added as a trajectory-level '
                             'parameter.'.format(name))

    def add_control(self, name, units=_unspecified, desc=_unspecified, opt=_unspecified,
                    fix_initial=_unspecified, fix_final=_unspecified, targets=_unspecified,
                    rate_targets=_unspecified, rate2_targets=_unspecified, val=_unspecified,
                    shape=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                    adder=_unspecified, ref0=_unspecified, ref=_unspecified, continuity=_unspecified,
                    continuity_scaler=_unspecified, rate_continuity=_unspecified,
                    rate_continuity_scaler=_unspecified, rate2_continuity=_unspecified,
                    rate2_continuity_scaler=_unspecified):
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
        continuity_scaler : bool
            Scaler of the continuity constraint. This option is invalid if opt=False.  This
            option is only relevant in the Radau pseudospectral transcription where the continuity
            constraint is nonlinear.  For Gauss-Lobatto the continuity constraint is linear.
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

        Notes
        -----
        rate and rate2 continuity are not enforced for input controls.

        """
        self.check_parameter(name)

        if name not in self.user_control_options:
            self.user_control_options[name] = {'name': name}

        if units is not _unspecified:
            self.user_control_options[name]['units'] = units

        if opt is not _unspecified:
            self.user_control_options[name]['opt'] = opt

        if desc is not _unspecified:
            self.user_control_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.user_control_options[name]['targets'] = (targets,)
            else:
                self.user_control_options[name]['targets'] = targets

        if rate_targets is not _unspecified:
            if isinstance(rate_targets, string_types):
                self.user_control_options[name]['rate_targets'] = (rate_targets,)
            else:
                self.user_control_options[name]['rate_targets'] = rate_targets

        if rate2_targets is not _unspecified:
            if isinstance(rate2_targets, string_types):
                self.user_control_options[name]['rate2_targets'] = (rate2_targets,)
            else:
                self.user_control_options[name]['rate2_targets'] = rate2_targets

        if val is not _unspecified:
            self.user_control_options[name]['val'] = val

        if shape is not _unspecified:
            self.user_control_options[name]['shape'] = shape

        if fix_initial is not _unspecified:
            self.user_control_options[name]['fix_initial'] = fix_initial

        if fix_final is not _unspecified:
            self.user_control_options[name]['fix_final'] = fix_final

        if lower is not _unspecified:
            self.user_control_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.user_control_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.user_control_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.user_control_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.user_control_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.user_control_options[name]['ref'] = ref

        if continuity is not _unspecified:
            self.user_control_options[name]['continuity'] = continuity

        if continuity_scaler is not _unspecified:
            self.user_control_options[name]['continuity_scaler'] = continuity_scaler

        if rate_continuity is not _unspecified:
            self.user_control_options[name]['rate_continuity'] = rate_continuity

        if rate_continuity_scaler is not _unspecified:
            self.user_control_options[name]['rate_continuity_scaler'] = rate_continuity_scaler

        if rate2_continuity is not _unspecified:
            self.user_control_options[name]['rate2_continuity'] = rate2_continuity

        if rate2_continuity_scaler is not _unspecified:
            self.user_control_options[name]['rate2_continuity_scaler'] = rate2_continuity_scaler

    def add_polynomial_control(self, name, order, desc=_unspecified, val=_unspecified, units=_unspecified,
                               opt=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                               lower=_unspecified, upper=_unspecified,
                               scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                               ref=_unspecified, targets=_unspecified, rate_targets=_unspecified,
                               rate2_targets=_unspecified, shape=_unspecified):
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
        desc : str
            A description of the polynomial control.
        units : str or None or 0
            Units in which the control variable is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True (default) the value(s) of this control will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False, the values of this control will exist as input controls:{name}
        fix_initial : bool
            If True, the given initial value of the polynomial control is not a design variable and
            will not be changed during the optimization.
        fix_final : bool
            If True, the given final value of the polynomial control is not a design variable and
            will not be changed during the optimization.
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
        targets : Sequence of str or None
            Targets in the ODE to which this polynomial control is connected.
        rate_targets : None or str
            The name of the parameter in the ODE to which the first time-derivative
            of the control value is connected.
        rate2_targets : None or str
            The name of the parameter in the ODE to which the second time-derivative
            of the control value is connected.
        shape : Sequence of int
            The shape of the control variable at each point in time.
        """
        self.check_parameter(name)

        if name not in self.user_polynomial_control_options:
            self.user_polynomial_control_options[name] = {'name': name,
                                                          'order': order}

        if units is not _unspecified:
            self.user_polynomial_control_options[name]['units'] = units

        if opt is not _unspecified:
            self.user_polynomial_control_options[name]['opt'] = opt

        if desc is not _unspecified:
            self.user_polynomial_control_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.user_polynomial_control_options[name]['targets'] = (targets,)
            else:
                self.user_polynomial_control_options[name]['targets'] = targets

        if rate_targets is not _unspecified:
            if isinstance(rate_targets, string_types):
                self.user_polynomial_control_options[name]['rate_targets'] = (rate_targets,)
            else:
                self.user_polynomial_control_options[name]['rate_targets'] = rate_targets

        if rate2_targets is not _unspecified:
            if isinstance(rate2_targets, string_types):
                self.user_polynomial_control_options[name]['rate2_targets'] = (rate2_targets,)
            else:
                self.user_polynomial_control_options[name]['rate2_targets'] = rate2_targets

        if val is not _unspecified:
            self.user_polynomial_control_options[name]['val'] = val

        if shape is not _unspecified:
            self.user_polynomial_control_options[name]['shape'] = shape

        if fix_initial is not _unspecified:
            self.user_polynomial_control_options[name]['fix_initial'] = fix_initial

        if fix_final is not _unspecified:
            self.user_polynomial_control_options[name]['fix_final'] = fix_final

        if lower is not _unspecified:
            self.user_polynomial_control_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.user_polynomial_control_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.user_polynomial_control_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.user_polynomial_control_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.user_polynomial_control_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.user_polynomial_control_options[name]['ref'] = ref

    def add_design_parameter(self, name, val=_unspecified, units=_unspecified, opt=_unspecified,
                             desc=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                             adder=_unspecified, ref0=_unspecified, ref=_unspecified, targets=_unspecified,
                             shape=_unspecified, dynamic=_unspecified):
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
        desc : str
            A description of the design parameter.
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
        shape : Sequence of int
            The shape of the design parameter.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.
        """
        self.check_parameter(name)

        if name not in self.user_design_parameter_options:
            self.user_design_parameter_options[name] = {'name': name}

        if units is not _unspecified:
            self.user_design_parameter_options[name]['units'] = units

        if opt is not _unspecified:
            self.user_design_parameter_options[name]['opt'] = opt

        if desc is not _unspecified:
            self.user_design_parameter_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.user_design_parameter_options[name]['targets'] = (targets,)
            else:
                self.user_design_parameter_options[name]['targets'] = targets

        if val is not _unspecified:
            self.user_design_parameter_options[name]['val'] = val

        if shape is not _unspecified:
            self.user_design_parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.user_design_parameter_options[name]['dynamic'] = dynamic

        if lower is not _unspecified:
            self.user_design_parameter_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.user_design_parameter_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.user_design_parameter_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.user_design_parameter_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.user_design_parameter_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.user_design_parameter_options[name]['ref'] = ref

    def add_input_parameter(self, name, val=_unspecified, units=_unspecified, targets=_unspecified,
                            desc=_unspecified, shape=_unspecified, dynamic=_unspecified):
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
        desc : str
            A description of the input parameter
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        shape : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        """
        self.check_parameter(name)

        if name not in self.user_input_parameter_options:
            self.user_input_parameter_options[name] = {'name': name}

        if units is not _unspecified:
            self.user_input_parameter_options[name]['units'] = units

        if val is not _unspecified:
            self.user_input_parameter_options[name]['val'] = val

        if desc is not _unspecified:
            self.user_input_parameter_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.user_input_parameter_options[name]['targets'] = (targets,)
            else:
                self.user_input_parameter_options[name]['targets'] = targets

        if shape is not _unspecified:
            self.user_input_parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.user_input_parameter_options[name]['dynamic'] = dynamic

    def add_traj_parameter(self, name, val=_unspecified, units=_unspecified, targets=_unspecified,
                           desc=_unspecified, shape=_unspecified, dynamic=_unspecified):
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
        desc : str
            A description of the input parameter
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        shape : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
        """
        self.check_parameter(name)

        if name not in self.user_traj_parameter_options:
            self.user_traj_parameter_options[name] = {'name': name}

        if units is not _unspecified:
            self.user_traj_parameter_options[name]['units'] = units

        if val is not _unspecified:
            self.user_traj_parameter_options[name]['val'] = val

        if desc is not _unspecified:
            self.user_traj_parameter_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, string_types):
                self.user_traj_parameter_options[name]['targets'] = (targets,)
            else:
                self.user_traj_parameter_options[name]['targets'] = targets

        if shape is not _unspecified:
            self.user_traj_parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.user_traj_parameter_options[name]['dynamic'] = dynamic

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

        self.add_timeseries_output(name, output_name=constraint_name, units=units, shape=shape)

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
        self.add_timeseries_output(name, output_name=constraint_name, units=units, shape=shape)

    def add_timeseries_output(self, name, output_name=None, units=None, shape=(1,), timeseries='timeseries'):
        r"""
        Add a variable to the timeseries outputs of the phase.

        Parameters
        ----------
        name : string
            The name of the variable to be used as a timeseries output.  Must be one of
            'time', 'time_phase', one of the states, controls, control rates, or parameters,
            in the phase, or the path to an output variable in the ODE.
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
        timeseries : str or None
            The name of the timeseries to which the output is being added.
        """
        if output_name is None:
            output_name = name.split('.')[-1]

        if timeseries not in self._timeseries:
            raise ValueError('Timeseries {0} does not exist in phase {1}'.format(timeseries, self.pathname))

        if name not in self._timeseries[timeseries]['outputs']:
            self._timeseries[timeseries]['outputs'][name] = {}
            self._timeseries[timeseries]['outputs'][name]['output_name'] = output_name

        self._timeseries[timeseries]['outputs'][name]['units'] = units
        self._timeseries[timeseries]['outputs'][name]['shape'] = shape

    def add_timeseries(self, name, transcription, subset='all'):
        r"""
        Adds a new timeseries output upon which outputs can be provided.

        Parameters
        ----------
        name : str
            A name for the timeseries output path.
        transcription : str
            A transcription object which provides a grid upon which the outputs of the timeseries
            are provided.
        subset : str
            The name of the node subset in the given transcription at which outputs
            are to be provided.
        """
        self._timeseries[name] = {'transcription': transcription,
                                  'subset': subset,
                                  'outputs': {}}

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

    def classify_var(self, var):
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

    def _check_ode(self):
        """
        Check that the provided ODE class meets minimum requirements.

        * The ode_class must be a class, not an instance.
        * The ode_class must derive from openmdao.core.System

        Raises
        ------
        ValueError
            ValueError is raised if the ODE does not meet one of the the requirements above.

        """
        ode_class = self.options['ode_class']
        if not inspect.isclass(ode_class):
            raise ValueError('ode_class must be a class, not an instance.')
        elif not issubclass(ode_class, System):
            raise ValueError('ode_class must be derived from openmdao.core.System.')

    def setup(self):
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        self.finalize_variables()

        transcription = self.options['transcription']

        transcription.setup_time(self)

        # The control interpolation comp to which we'll connect controls
        if self.control_options:
            transcription.setup_controls(self)

        if self.polynomial_control_options:
            transcription.setup_polynomial_controls(self)

        if self.design_parameter_options:
            transcription.setup_design_parameters(self)

        if self.input_parameter_options:
            transcription.setup_input_parameters(self)

        if self.traj_parameter_options:
            transcription.setup_traj_parameters(self)

        transcription.setup_states(self)
        self._check_ode()
        transcription.setup_ode(self)
        transcription.setup_defects(self)

        transcription.setup_boundary_constraints('initial', self)
        transcription.setup_boundary_constraints('final', self)
        transcription.setup_path_constraints(self)
        transcription.setup_endpoint_conditions(self)
        transcription.setup_objective(self)

        transcription.setup_timeseries_outputs(self)

        transcription.setup_solvers(self)

    def check_time_options(self):
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
        Check that control options are valid and issue warnings if invalid options are provided.

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
        Check that design parameter options are valid and issue warnings if invalid
        options are provided.

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

    def check_config(self, logger):
        """
        Print warnings associated with the Phase if check is enabled during setup.

        Parameters
        ----------
        logger
            The logger object to which warnings and errors will be sent.
        """
        self.options['transcription'].check_config(self, logger)

    def interpolate(self, xs=None, ys=None, nodes='all', kind='linear', axis=0):
        """
        Return an array of values on interpolated to the given node subset of the phase.

        Parameters
        ----------
        xs :  ndarray or Sequence or None
            Array of integration variable values.
        ys :  ndarray or Sequence or None
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

        gd = self.options['transcription'].grid_data
        node_locations = gd.node_ptau[gd.subset_node_indices[nodes]]
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
        from ..transcriptions import SolveIVP

        t = self.options['transcription']

        sim_phase = dm.Phase(from_phase=self,
                             transcription=SolveIVP(grid_data=t.grid_data,
                                                    method=method,
                                                    atol=atol,
                                                    rtol=rtol,
                                                    output_nodes_per_seg=times_per_seg))

        return sim_phase

    def initialize_values_from_phase(self, prob, from_phase):
        """
        Initializes values in the Phase using the phase from which it was created.

        Parameters
        ----------
        prob : Problem
            The problem instance under used to set values in this phase instance.
        from_phase : Phase
            The Phase instance from which the values in this phase are being initialized.
        """
        phs = from_phase

        op_dict = dict([(name, options) for (name, options) in phs.list_outputs(units=True,
                                                                                out_stream=None)])
        ip_dict = dict([(name, options) for (name, options) in phs.list_inputs(units=True,
                                                                               out_stream=None)])

        phs_path = phs.pathname + '.' if phs.pathname else ''

        if self.pathname.split('.')[0] == self.name:
            self_path = self.name + '.'
        else:
            self_path = self.pathname.split('.')[0] + '.' + self.name + '.'

        # Set the integration times
        op = op_dict['{0}timeseries.time'.format(phs_path)]
        prob.set_val('{0}t_initial'.format(self_path), op['value'][0, ...])
        prob.set_val('{0}t_duration'.format(self_path), op['value'][-1, ...] - op['value'][0, ...])

        # Assign initial state values
        for name in phs.state_options:
            op = op_dict['{0}timeseries.states:{1}'.format(phs_path, name)]
            prob['{0}initial_states:{1}'.format(self_path, name)][...] = op['value'][0, ...]

        # Assign control values
        for name, options in iteritems(phs.control_options):
            if options['opt']:
                op = op_dict['{0}control_group.indep_controls.controls:{1}'.format(phs_path, name)]
                prob['{0}controls:{1}'.format(self_path, name)][...] = op['value']
            else:
                ip = ip_dict['{0}control_group.control_interp_comp.controls:{1}'.format(phs_path, name)]
                prob['{0}controls:{1}'.format(self_path, name)][...] = ip['value']

        # Assign polynomial control values
        for name, options in iteritems(phs.polynomial_control_options):
            if options['opt']:
                op = op_dict['{0}polynomial_control_group.indep_polynomial_controls.'
                             'polynomial_controls:{1}'.format(phs_path, name)]
                prob['{0}polynomial_controls:{1}'.format(self_path, name)][...] = op['value']
            else:
                ip = ip_dict['{0}polynomial_control_group.interp_comp.'
                             'polynomial_controls:{1}'.format(phs_path, name)]
                prob['{0}polynomial_controls:{1}'.format(self_path, name)][...] = ip['value']

        # Assign design parameter values
        for name in phs.design_parameter_options:
            op = op_dict['{0}design_params.design_parameters:{1}'.format(phs_path, name)]
            prob['{0}design_parameters:{1}'.format(self_path, name)][...] = op['value']

        # Assign input parameter values
        for name in phs.input_parameter_options:
            op = op_dict['{0}input_params.input_parameters:{1}_out'.format(phs_path, name)]
            prob['{0}input_parameters:{1}'.format(self_path, name)][...] = op['value']

        # Assign traj parameter values
        for name in phs.traj_parameter_options:
            op = op_dict['{0}traj_params.traj_parameters:{1}_out'.format(phs_path, name)]
            prob['{0}traj_parameters:{1}'.format(self_path, name)][...] = op['value']

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

        sim_prob = om.Problem(model=om.Group())

        sim_phase = self.get_simulation_phase(times_per_seg, method=method, atol=atol, rtol=rtol)

        sim_prob.model.add_subsystem(self.name, sim_phase)

        if record_file is not None:
            rec = om.SqliteRecorder(record_file)
            sim_prob.model.recording_options['includes'] = ['*.timeseries.*']

            sim_prob.model.add_recorder(rec)

        sim_prob.setup(check=True)
        sim_phase.initialize_values_from_phase(sim_prob, self)

        print('\nSimulating phase {0}'.format(self.pathname))
        sim_prob.run_model()
        print('Done simulating phase {0}'.format(self.pathname))

        sim_prob.cleanup()

        return sim_prob
