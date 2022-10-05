import openmdao.api as om

from .phase import Phase
from ..transcriptions import Analytic
from .options import StateOptionsDictionary

from ..utils.misc import _unspecified


class AnalyticPhase(Phase):
    """
    The AnalyticPhase object in Dymos.

    The AnalyticPhase object in dymos inherits from PhaseBase but is used to override some base methods with ones
    that will warn about certain options or methods being invalid for the AnalyticPhase.

    Parameters
    ----------
    from_phase : <Phase> or None
        A phase instance from which the initialized phase should copy its data.
    **kwargs : dict
        Dictionary of optional phase arguments.
    """

    def __init__(self, from_phase=None, **kwargs):
        super().__init__(from_phase=from_phase, **kwargs)
        self.simulate_options = None

    def initialize(self):
        """
        Declare instantiation options for the phase.
        """
        super().initialize()
        self.options.declare('num_nodes', types=int, default=2,
                             desc='Number of points in time at which to evaluate the solution to the ODE.')

    def add_state(self, name, state_name=None, units=_unspecified, shape=_unspecified,
                  rate_source=_unspecified, targets=_unspecified,
                  val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                  lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                  ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                  defect_ref=_unspecified, solve_segments=_unspecified, connected_initial=_unspecified,
                  input_initial=_unspecified, initial_targets=_unspecified, opt=_unspecified,
                  initial_bounds=_unspecified, final_bounds=_unspecified):
        """
        Add a state variable to be integrated by the phase.

        Parameters
        ----------
        name : str
            Path to use as the source of the state variable.
        state_name : str
            Name of the state variable, if the last element of the source path is ambiguous.
        units : str or None
            Units in which the state variable is defined.  Internally components may use different
            units for the state variable, but the IndepVarComp which provides its value will provide
            it in these units, and collocation defects will use these units.  If units is not
            specified here then the unit will be determined from the rate_source.
        shape : tuple of int
            The shape of the state variable.  For instance, a 3D cartesian position vector would have
            a shape of (3,).  This only needs to be specified if the rate_source target points to
            a control or state whose shape isn't known in time.
        rate_source : str
            The path to the ODE output which provides the rate of this state variable.
        targets : str or Sequence of str
            The path to the targets of the state variable in the ODE system.  If given
            this will override the value given by the @declare_state decorator on the ODE.
            In the future, if left _unspecified (the default), the phase variable will try to connect to an ODE input
            of the same name. Set targets to None to prevent this.
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
            The unit-reference value of the state at the nodes of the phase.
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
            source. Deprecated - use input_initial.
        input_initial : bool
            If True, then the initial value for this state comes is an input.
        initial_targets : str
            The path to the ODE inputs to which the initial value of this state should be connected.
        opt : bool
            If True, state values are fixed at the input values and the optimizer resolves defect constraints by
            varying the other design variables in the phase.
        initial_bounds : tuple
            The bounds (lower, upper) of the state variable at the initial point in the phase.
        final_bounds : tuple
            The bounds (lower, upper) of the state variable at the final point in the phase.
        """
        _state_name = name.split('.')[-1] if state_name is None else state_name

        if name not in self.state_options:
            self.state_options[_state_name] = StateOptionsDictionary()
            self.state_options[_state_name]['name'] = _state_name

        self.set_state_options(name=name, state_name=state_name, units=units, shape=shape, rate_source=rate_source,
                               targets=targets, val=val, fix_initial=fix_initial,
                               fix_final=fix_final, lower=lower, upper=upper, scaler=scaler,
                               adder=adder, ref0=ref0, ref=ref, defect_scaler=defect_scaler,
                               defect_ref=defect_ref, solve_segments=solve_segments,
                               connected_initial=connected_initial, input_initial=input_initial,
                               initial_targets=initial_targets, opt=opt, initial_bounds=initial_bounds,
                               final_bounds=final_bounds)

    def set_state_options(self, name, state_name=None, units=_unspecified, shape=_unspecified,
                          rate_source=_unspecified, targets=_unspecified,
                          val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                          lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                          ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                          defect_ref=_unspecified, solve_segments=_unspecified, connected_initial=_unspecified,
                          input_initial=_unspecified, initial_targets=_unspecified, opt=_unspecified,
                          initial_bounds=_unspecified, final_bounds=_unspecified):
        """
        Set options that apply the EOM state variable of the given name.

        Parameters
        ----------
        name : str
            Path to use as the source of the state variable.
        state_name : str
            Name of the state variable, if the last element of the source path is ambiguous.
        units : str or None
            Units in which the state variable is defined.  Internally components may use different
            units for the state variable, but the IndepVarComp which provides its value will provide
            it in these units, and collocation defects will use these units.  If units is not
            specified here then the unit will be determined from the rate_source.
        shape : tuple of int
            The shape of the state variable.  For instance, a 3D cartesian position vector would have
            a shape of (3,).  This only needs to be specified if the rate_source target points to
            a control or state whose shape isn't known in time.
        rate_source : str
            The path to the ODE output which provides the rate of this state variable.
        targets : str or Sequence of str
            The path to the targets of the state variable in the ODE system.  If given
            this will override the value given by the @declare_state decorator on the ODE.
            In the future, if left _unspecified (the default), the phase variable will try to connect to an ODE input
            of the same name. Set targets to None to prevent this.
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
            The unit-reference value of the state at the nodes of the phase.
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
            source. Deprecated - use input_initial.
        input_initial : bool
            If True, then the initial value for this state comes is an input.
        initial_targets : str or Sequence of str
            The path to the ODE inputs to which the initial value of this state should be connected.
        opt : bool
            If True, state values are fixed at the input values and the optimizer resolves defect constraints by
            varying the other design variables in the phase.
        initial_bounds : tuple
            The bounds (lower, upper) of the state variable at the initial point in the phase.
        final_bounds : tuple
            The bounds (lower, upper) of the state variable at the final point in the phase.
        """
        source = name
        _state_name = name.split('.')[-1] if state_name is None else state_name

        if name not in self.state_options:
            self.state_options[_state_name] = StateOptionsDictionary()
            self.state_options[_state_name]['name'] = _state_name

        if units is not _unspecified:
            self.state_options[_state_name]['units'] = units

        if shape is not _unspecified:
            self.state_options[_state_name]['shape'] = shape

        if source is not _unspecified:
            self.state_options[_state_name]['source'] = source

        if rate_source is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `rate_source` is not a valid option for states in AnalyticPhase.')

        if targets is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `targets` is not a valid option for states in AnalyticPhase.')

        if val is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `val` is not a valid option for states in AnalyticPhase.')

        if fix_initial is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `fix_initial` is not a valid option for states in AnalyticPhase.')

        if fix_final is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `fix_final` is not a valid option for states in AnalyticPhase.')

        if lower is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `lower` is not a valid option for states in AnalyticPhase.')

        if upper is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `upper` is not a valid option for states in AnalyticPhase.')

        if scaler is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `scaler` is not a valid option for states in AnalyticPhase.')

        if adder is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `adder` is not a valid option for states in AnalyticPhase.')

        if ref0 is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `ref0` is not a valid option for states in AnalyticPhase.')

        if ref is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `ref` is not a valid option for states in AnalyticPhase.')

        if defect_scaler is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `defect_scaler` is not a valid option for states in AnalyticPhase.')

        if defect_ref is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `defect_ref` is not a valid option for states in AnalyticPhase.')

        if solve_segments is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `solve_segments` is not a valid option for states in AnalyticPhase.')

        if connected_initial is not _unspecified:
            self.state_options[_state_name]['connected_initial'] = connected_initial
            om.issue_warning(f'{self.pathname}: State option `connected_initial` is deprecated. Use input_initial',
                             om.OMDeprecationWarning)
            self.state_options[_state_name]['input_initial'] = connected_initial

        if input_initial is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `input_initial` is not a valid option for states in AnalyticPhase.')

        if opt is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `opt` is not a valid option for states in AnalyticPhase.')

        if initial_bounds is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `initial_bounds` is not a valid option for states in AnalyticPhase.')

        if final_bounds is not _unspecified:
            raise NotImplementedError('States in AnalyticPhase are strictly outputs of the ODE solution system. '
                                      'Option `final_bounds` is not a valid option for states in AnalyticPhase.')

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
            In the future, if left _unspecified (the default), the phase control will try to connect to an ODE input
            of the same name. Set targets to None to prevent this.
        rate_targets : Sequence of str or None
            The targets in the ODE to which the control rate is connected.
        rate2_targets : Sequence of str or None
            The parameter in the ODE to which the control 2nd derivative is connected.
        val : float
            The default value of the control variable at the control input nodes.
        shape : Sequence of int
            The shape of the control variable at each point in time. Only needed for controls that don't
            have a target in the ode.
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
        raise NotImplementedError('AnalyticPhase does not support controls.')

    def set_control_options(self, name, units=_unspecified, desc=_unspecified, opt=_unspecified,
                            fix_initial=_unspecified, fix_final=_unspecified, targets=_unspecified,
                            rate_targets=_unspecified, rate2_targets=_unspecified, val=_unspecified,
                            shape=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                            adder=_unspecified, ref0=_unspecified, ref=_unspecified, continuity=_unspecified,
                            continuity_scaler=_unspecified, rate_continuity=_unspecified,
                            rate_continuity_scaler=_unspecified, rate2_continuity=_unspecified,
                            rate2_continuity_scaler=_unspecified):
        """
        Set options on an existing dynamic control variable in the phase.

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
            In the future, if left _unspecified (the default), the phase control will try to connect to an ODE input
            of the same name. Set targets to None to prevent this.
        rate_targets : Sequence of str or None
            The targets in the ODE to which the control rate is connected.
        rate2_targets : Sequence of str or None
            The parameter in the ODE to which the control 2nd derivative is connected.
        val : float
            The default value of the control variable at the control input nodes.
        shape : Sequence of int
            The shape of the control variable at each point in time. Only needed for controls that don't
            have a target in the ode.
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
        raise NotImplementedError('AnalyticPhase does not support controls.')

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
            The order of the interpolating polynomial used to represent the control value in
            phase tau space.
        desc : str
            A description of the polynomial control.
        val : float or ndarray
            Default value of the control at all nodes.  If val scalar and the control
            is dynamic it will be broadcast.
        units : str or None or 0
            Units in which the control variable is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True (default) the value(s) of this control will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False, the values of this control will exist as input controls:{name}.
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
            The unit-reference value of the control at the nodes of the phase.
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
        raise NotImplementedError('AnalyticPhase does not support polynomial controls.')

    def set_polynomial_control_options(self, name, order=_unspecified, desc=_unspecified, val=_unspecified,
                                       units=_unspecified, opt=_unspecified, fix_initial=_unspecified,
                                       fix_final=_unspecified, lower=_unspecified, upper=_unspecified,
                                       scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                                       ref=_unspecified, targets=_unspecified, rate_targets=_unspecified,
                                       rate2_targets=_unspecified, shape=_unspecified):
        """
        Set options on an existing polynomial control variable in the phase.

        Parameters
        ----------
        name : str
            Name of the controllable parameter in the ODE.
        order : int
            The order of the interpolating polynomial used to represent the control value in
            phase tau space.
        desc : str
            A description of the polynomial control.
        val : float or ndarray
            Default value of the control at all nodes.  If val scalar and the control
            is dynamic it will be broadcast.
        units : str or None or 0
            Units in which the control variable is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True (default) the value(s) of this control will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False, the values of this control will exist as input controls:{name}.
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
            The unit-reference value of the control at the nodes of the phase.
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
        raise NotImplementedError('AnalyticPhase does not support polynomial controls.')

    def setup(self):
        """
        Build the model hierarchy for a Dymos AnalyticPhase.
        """
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        transcription = self.options['transcription'] = Analytic(order=self.options['num_nodes'])
        transcription.setup_time(self)

        if self.control_options:
            transcription.setup_controls(self)

        if self.polynomial_control_options:
            transcription.setup_polynomial_controls(self)

        if self.parameter_options:
            transcription.setup_parameters(self)

        transcription.setup_states(self)
        self._check_ode()
        transcription.setup_ode(self)

        transcription.setup_timeseries_outputs(self)
        transcription.setup_defects(self)
        transcription.setup_solvers(self)

    def simulate(self, times_per_seg=10, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                 first_step=_unspecified, max_step=_unspecified, record_file=None):
        """
        Stub to make sure users are informed that simulate cannot be done on AnalyticPhase.

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
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.
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
        raise NotImplementedError('Method `simulate` is not available for AnalyticPhase.')

    def set_simulate_options(self, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                             first_step=_unspecified, max_step=_unspecified):
        """
        Stub to make sure users are informed that simulate cannot be done on AnalyticPhase.

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
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.
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
        raise NotImplementedError('Method set_simulate_options is not available for AnalyticPhase.')
