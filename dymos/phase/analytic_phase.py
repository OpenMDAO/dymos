import openmdao.api as om

from .phase import Phase
from ..transcriptions import Analytic
from .options import StateOptionsDictionary

from ..utils.misc import _unspecified


class AnalyticPhase(Phase):

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
        raise NotImplementedError('AnalyticPhase does not support controls.')

    def set_control_options(self, name, units=_unspecified, desc=_unspecified, opt=_unspecified,
                            fix_initial=_unspecified, fix_final=_unspecified, targets=_unspecified,
                            rate_targets=_unspecified, rate2_targets=_unspecified, val=_unspecified,
                            shape=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                            adder=_unspecified, ref0=_unspecified, ref=_unspecified, continuity=_unspecified,
                            continuity_scaler=_unspecified, rate_continuity=_unspecified,
                            rate_continuity_scaler=_unspecified, rate2_continuity=_unspecified,
                            rate2_continuity_scaler=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support controls.')

    def add_polynomial_control(self, name, order, desc=_unspecified, val=_unspecified, units=_unspecified,
                               opt=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                               lower=_unspecified, upper=_unspecified,
                               scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                               ref=_unspecified, targets=_unspecified, rate_targets=_unspecified,
                               rate2_targets=_unspecified, shape=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support polynomial controls.')

    def set_polynomial_control_options(self, name, order=_unspecified, desc=_unspecified, val=_unspecified,
                                       units=_unspecified, opt=_unspecified, fix_initial=_unspecified,
                                       fix_final=_unspecified, lower=_unspecified, upper=_unspecified,
                                       scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                                       ref=_unspecified, targets=_unspecified, rate_targets=_unspecified,
                                       rate2_targets=_unspecified, shape=_unspecified):
        raise NotImplementedError('AnalyticPhase does not support polynomial controls.')

    def setup(self):
        """
        Build the model hierarchy for a Dymos AnalyticPhase.
        """
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        transcription = self.options['transcription'] =  Analytic(order=self.options['num_nodes'])
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
