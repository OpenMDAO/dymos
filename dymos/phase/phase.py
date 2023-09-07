from collections.abc import Iterable, Callable
import inspect
import warnings

import numpy as np

from scipy import interpolate

import openmdao
import openmdao.api as om
from openmdao.utils.mpi import MPI
from openmdao.utils.om_warnings import issue_warning
from openmdao.core.system import System
from openmdao.recorders.case import Case

import dymos as dm

from .options import ControlOptionsDictionary, ParameterOptionsDictionary, \
    StateOptionsDictionary, TimeOptionsDictionary, ConstraintOptionsDictionary, \
    PolynomialControlOptionsDictionary, GridRefinementOptionsDictionary, SimulateOptionsDictionary, \
    TimeseriesOutputOptionsDictionary, PhaseTimeseriesOptionsDictionary

from ..transcriptions.transcription_base import TranscriptionBase
from ..transcriptions.grid_data import GaussLobattoGrid, RadauGrid, UniformGrid
from ..transcriptions import ExplicitShooting, GaussLobatto, Radau
from ..utils.indexing import get_constraint_flat_idxs
from ..utils.introspection import configure_time_introspection, _configure_constraint_introspection, \
    configure_controls_introspection, configure_parameters_introspection, \
    configure_timeseries_output_introspection, classify_var, configure_timeseries_expr_introspection
from ..utils.misc import _unspecified
from ..utils.lgl import lgl


om_dev_version = openmdao.__version__.endswith('dev')
om_version = tuple(int(s) for s in openmdao.__version__.split('-')[0].split('.'))


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
    from_phase : <Phase> or None
        A phase instance from which the initialized phase should copy its data.
    **kwargs : dict
        Dictionary of optional phase arguments.
    """
    def __init__(self, from_phase=None, **kwargs):
        _kwargs = kwargs.copy()

        # These are the options which will be set at setup time.
        # Prior to setup, the options are placed into the user_*_options dictionaries.
        self.time_options = TimeOptionsDictionary()
        self.state_options = {}
        self.control_options = {}
        self.polynomial_control_options = {}
        self.parameter_options = {}
        self.refine_options = GridRefinementOptionsDictionary()
        self.simulate_options = SimulateOptionsDictionary()
        self.timeseries_ec_vars = {}
        self.timeseries_options = PhaseTimeseriesOptionsDictionary()

        # Dictionaries of variable options that are set by the user via the API
        # These will be applied over any defaults specified by decorators on the ODE
        if from_phase is None:
            self._initial_boundary_constraints = []
            self._final_boundary_constraints = []
            self._path_constraints = []
            self._timeseries = {'timeseries': {'transcription': None,
                                               'subset': 'all',
                                               'outputs': {}}}
            self._objectives = {}
        else:
            self.time_options.update(from_phase.time_options)
            self.state_options = from_phase.state_options.copy()
            self.control_options = from_phase.control_options.copy()
            self.polynomial_control_options = from_phase.polynomial_control_options.copy()
            self.parameter_options = from_phase.parameter_options.copy()

            self.refine_options.update(from_phase.refine_options)
            self.simulate_options.update(from_phase.simulate_options)

            self._initial_boundary_constraints = from_phase._initial_boundary_constraints.copy()
            self._final_boundary_constraints = from_phase._final_boundary_constraints.copy()
            self._path_constraints = from_phase._path_constraints.copy()
            self._timeseries = from_phase._timeseries.copy()
            self._objectives = from_phase._objectives.copy()

            _kwargs['ode_class'] = from_phase.options['ode_class']
            _kwargs['ode_init_kwargs'] = from_phase.options['ode_init_kwargs']

        super(Phase, self).__init__(**_kwargs)

    def initialize(self):
        """
        Declare instantiation options for the phase.
        """
        self.options.declare('ode_class', default=None,
                             desc='System defining the ODE',
                             recordable=False)
        self.options.declare('ode_init_kwargs', types=dict, default={},
                             desc='Keyword arguments provided when initializing the ODE System')
        self.options.declare('transcription', types=TranscriptionBase,
                             desc='Transcription technique of the optimal control problem.')

    def add_state(self, name, units=_unspecified, shape=_unspecified,
                  rate_source=_unspecified, targets=_unspecified,
                  val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                  lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                  ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                  defect_ref=_unspecified, continuity_scaler=_unspecified, continuity_ref=_unspecified,
                  solve_segments=_unspecified, connected_initial=_unspecified,
                  source=_unspecified, input_initial=_unspecified, initial_targets=_unspecified,
                  opt=_unspecified, initial_bounds=_unspecified, final_bounds=_unspecified):
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
        continuity_scaler : float
            Constraint scaler used to enforce the continuity mismatch defect between segments when transcription
            is not compressed.
        continuity_ref : float
            Reference unit value of the continuity mismatch defect between segments when transcription is not
            compressed. Used in place of scaler.
        solve_segments : bool(False)
            If True, a solver will be used to converge the collocation defects within a segment.
            Note that the state continuity defects between segements will still be
            handled by the optimizer.
        connected_initial : bool
            If True, then the initial value for this state comes from an externally connected
            source. Deprecated - use input_initial.
        source : str
            The path to the ODE output which provides the solution for this state variable when using an
            Analytic transcription.
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
        if name not in self.state_options:
            self.state_options[name] = StateOptionsDictionary()
            self.state_options[name]['name'] = name

        self.set_state_options(name=name, units=units, shape=shape, rate_source=rate_source,
                               targets=targets, val=val, fix_initial=fix_initial,
                               fix_final=fix_final, lower=lower, upper=upper, scaler=scaler,
                               adder=adder, ref0=ref0, ref=ref, defect_scaler=defect_scaler,
                               defect_ref=defect_ref, continuity_scaler=continuity_scaler,
                               continuity_ref=continuity_ref, solve_segments=solve_segments,
                               connected_initial=connected_initial, source=source, input_initial=input_initial,
                               initial_targets=initial_targets, opt=opt, initial_bounds=initial_bounds,
                               final_bounds=final_bounds)

    def set_state_options(self, name, units=_unspecified, shape=_unspecified,
                          rate_source=_unspecified, targets=_unspecified,
                          val=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                          lower=_unspecified, upper=_unspecified, scaler=_unspecified, adder=_unspecified,
                          ref0=_unspecified, ref=_unspecified, defect_scaler=_unspecified,
                          defect_ref=_unspecified, continuity_scaler=_unspecified, continuity_ref=_unspecified,
                          solve_segments=_unspecified, connected_initial=_unspecified,
                          source=_unspecified, input_initial=_unspecified, initial_targets=_unspecified,
                          opt=_unspecified, initial_bounds=_unspecified, final_bounds=_unspecified):
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
        continuity_scaler : float
            Constraint scaler used to enforce the continuity mismatch defect between segments when transcription
            is not compressed.
        continuity_ref : float
            Reference unit value of the continuity mismatch defect between segments when transcription is not
            compressed. Used in place of scaler.
        solve_segments : bool(False)
            If True, a solver will be used to converge the collocation defects within a segment.
            Note that the state continuity defects between segements will still be
            handled by the optimizer.
        connected_initial : bool
            If True, then the initial value for this state comes from an externally connected
            source. Deprecated - use input_initial.
        source : str
            The path to the ODE output which provides the solution for this state variable when using an
            Analytic transcription.
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
        if name not in self.state_options:
            # This state option will be picked up automatically from tags.
            self.state_options[name] = StateOptionsDictionary()
            self.state_options[name]['name'] = name

        if units is not _unspecified:
            self.state_options[name]['units'] = units

        if shape is not _unspecified:
            self.state_options[name]['shape'] = shape

        if rate_source is not _unspecified:
            self.state_options[name]['rate_source'] = rate_source

        if targets is not _unspecified:
            if isinstance(targets, str):
                self.state_options[name]['targets'] = (targets,)
            else:
                self.state_options[name]['targets'] = targets

        if val is not _unspecified:
            self.state_options[name]['val'] = val

        if fix_initial is not _unspecified:
            self.state_options[name]['fix_initial'] = fix_initial

        if fix_final is not _unspecified:
            self.state_options[name]['fix_final'] = fix_final

        if lower is not _unspecified:
            self.state_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.state_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.state_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.state_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.state_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.state_options[name]['ref'] = ref

        if defect_scaler is not _unspecified:
            self.state_options[name]['defect_scaler'] = defect_scaler

        if defect_ref is not _unspecified:
            self.state_options[name]['defect_ref'] = defect_ref

        if continuity_scaler is not _unspecified:
            self.state_options[name]['continuity_scaler'] = continuity_scaler

        if continuity_ref is not _unspecified:
            self.state_options[name]['continuity_ref'] = continuity_ref

        if solve_segments is not _unspecified:
            self.state_options[name]['solve_segments'] = solve_segments

        if connected_initial is not _unspecified:
            self.state_options[name]['connected_initial'] = connected_initial
            om.issue_warning(f'{self.pathname}: State option `connected_initial` is deprecated. Use input_initial',
                             om.OMDeprecationWarning)
            self.state_options[name]['input_initial'] = connected_initial

        if source is not _unspecified:
            self.state_options[name]['source'] = source

        if input_initial is not _unspecified:
            self.state_options[name]['input_initial'] = input_initial

        if opt is not _unspecified:
            self.state_options[name]['opt'] = opt

        if initial_bounds is not _unspecified:
            self.state_options[name]['initial_bounds'] = initial_bounds

        if final_bounds is not _unspecified:
            self.state_options[name]['final_bounds'] = final_bounds

    def check_parameter(self, name):
        """
        Checks that the parameter of the given name is valid.

        First name is checked against all existing states, controls, input parameters, and
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
        if name in ['time', 'time_phase', 't_initial', 't_duration']:
            raise ValueError(f'The name {name} is reserved for the independent variable of integration'
                             ' in Dymos and may not be used as a state, control, or parameter name')
        elif name in self.state_options:
            raise ValueError(f'{name} has already been added as a state.')
        elif name in self.control_options:
            raise ValueError(f'{name} has already been added as a control.')
        elif name in self.parameter_options:
            raise ValueError(f'{name} has already been added as a parameter.')
        elif name in self.polynomial_control_options:
            raise ValueError(f'{name} has already been added as a polynomial control.')

    def add_control(self, name, units=_unspecified, desc=_unspecified, opt=_unspecified,
                    fix_initial=_unspecified, fix_final=_unspecified, targets=_unspecified,
                    rate_targets=_unspecified, rate2_targets=_unspecified, val=_unspecified,
                    shape=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                    adder=_unspecified, ref0=_unspecified, ref=_unspecified, continuity=_unspecified,
                    continuity_scaler=_unspecified, continuity_ref=_unspecified, rate_continuity=_unspecified,
                    rate_continuity_scaler=_unspecified, rate_continuity_ref=_unspecified,
                    rate2_continuity=_unspecified, rate2_continuity_scaler=_unspecified,
                    rate2_continuity_ref=_unspecified):
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
        continuity_ref : bool
            Reference unit value to be used in place of continuity scaler.
        rate_continuity : bool
            Enforce continuity of control first derivatives  (in dimensionless time) at
            segment boundaries.
            This option is invalid if opt=False.
        rate_continuity_scaler : float
            Scaler of the rate continuity constraint at segment boundaries.
            This option is invalid if opt=False.
        rate_continuity_ref : float or None
            Reference unit value of the rate continuity constraint at segment boundaries, for use in
            place of rate_continuity_scaler.
        rate2_continuity : bool or None
            Enforce continuity of control second derivatives at segment boundaries.
            This option is invalid if opt=False.
        rate2_continuity_scaler : float or None
            Scaler of the dimensionless rate continuity constraint at segment boundaries.
            This option is invalid if opt=False.
        rate2_continuity_ref : float or None
            Reference unit value of the rate2 continuity constraint at segment boundaries, for use in
            place of rate_continuity_scaler.

        Notes
        -----
        rate and rate2 continuity are not enforced for input controls.
        """
        if name not in self.control_options:
            self.check_parameter(name)
            self.control_options[name] = ControlOptionsDictionary()
            self.control_options[name]['name'] = name

        self.set_control_options(name, units=units, desc=desc, opt=opt, fix_initial=fix_initial,
                                 fix_final=fix_final, targets=targets, rate_targets=rate_targets,
                                 rate2_targets=rate2_targets, val=val, shape=shape, lower=lower,
                                 upper=upper, scaler=scaler, adder=adder, ref0=ref0, ref=ref,
                                 continuity=continuity, continuity_scaler=continuity_scaler,
                                 continuity_ref=continuity_ref, rate_continuity=rate_continuity,
                                 rate_continuity_scaler=rate_continuity_scaler,
                                 rate_continuity_ref=rate_continuity_ref,
                                 rate2_continuity=rate2_continuity,
                                 rate2_continuity_scaler=rate2_continuity_scaler,
                                 rate2_continuity_ref=rate2_continuity_ref)

    def set_control_options(self, name, units=_unspecified, desc=_unspecified, opt=_unspecified,
                            fix_initial=_unspecified, fix_final=_unspecified, targets=_unspecified,
                            rate_targets=_unspecified, rate2_targets=_unspecified, val=_unspecified,
                            shape=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                            adder=_unspecified, ref0=_unspecified, ref=_unspecified, continuity=_unspecified,
                            continuity_scaler=_unspecified, continuity_ref=_unspecified,
                            rate_continuity=_unspecified, rate_continuity_scaler=_unspecified,
                            rate_continuity_ref=_unspecified, rate2_continuity=_unspecified,
                            rate2_continuity_scaler=_unspecified, rate2_continuity_ref=_unspecified):
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
        continuity_ref : bool
            Reference unit value to be used in place of continuity scaler.
        rate_continuity : bool
            Enforce continuity of control first derivatives  (in dimensionless time) at
            segment boundaries.
            This option is invalid if opt=False.
        rate_continuity_scaler : float
            Scaler of the rate continuity constraint at segment boundaries.
            This option is invalid if opt=False.
        rate_continuity_ref : float or None
            Reference unit value of the rate continuity constraint at segment boundaries, for use in
            place of rate_continuity_scaler.
        rate2_continuity : bool or None
            Enforce continuity of control second derivatives at segment boundaries.
            This option is invalid if opt=False.
        rate2_continuity_scaler : float or None
            Scaler of the dimensionless rate continuity constraint at segment boundaries.
            This option is invalid if opt=False.
        rate2_continuity_ref : float or None
            Reference unit value of the rate2 continuity constraint at segment boundaries, for use in
            place of rate_continuity_scaler.

        Notes
        -----
        rate and rate2 continuity are not enforced for input controls.
        """
        if units is not _unspecified:
            self.control_options[name]['units'] = units

        if opt is not _unspecified:
            self.control_options[name]['opt'] = opt

        if desc is not _unspecified:
            self.control_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, str):
                self.control_options[name]['targets'] = (targets,)
            else:
                self.control_options[name]['targets'] = targets

        if rate_targets is not _unspecified:
            if isinstance(rate_targets, str):
                self.control_options[name]['rate_targets'] = (rate_targets,)
            else:
                self.control_options[name]['rate_targets'] = rate_targets

        if rate2_targets is not _unspecified:
            if isinstance(rate2_targets, str):
                self.control_options[name]['rate2_targets'] = (rate2_targets,)
            else:
                self.control_options[name]['rate2_targets'] = rate2_targets

        if val is not _unspecified:
            self.control_options[name]['val'] = val

        if shape is not _unspecified:
            self.control_options[name]['shape'] = shape

        if fix_initial is not _unspecified:
            self.control_options[name]['fix_initial'] = fix_initial

        if fix_final is not _unspecified:
            self.control_options[name]['fix_final'] = fix_final

        if lower is not _unspecified:
            self.control_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.control_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.control_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.control_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.control_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.control_options[name]['ref'] = ref

        if continuity is not _unspecified:
            self.control_options[name]['continuity'] = continuity

        if continuity_scaler is not _unspecified:
            self.control_options[name]['continuity_scaler'] = continuity_scaler

        if continuity_ref is not _unspecified:
            self.control_options[name]['continuity_ref'] = continuity_ref

        if rate_continuity is not _unspecified:
            self.control_options[name]['rate_continuity'] = rate_continuity

        if rate_continuity_scaler is not _unspecified:
            self.control_options[name]['rate_continuity_scaler'] = rate_continuity_scaler

        if rate_continuity_ref is not _unspecified:
            self.control_options[name]['rate_continuity_ref'] = rate_continuity_ref

        if rate2_continuity is not _unspecified:
            self.control_options[name]['rate2_continuity'] = rate2_continuity

        if rate2_continuity_scaler is not _unspecified:
            self.control_options[name]['rate2_continuity_scaler'] = rate2_continuity_scaler

        if rate2_continuity_ref is not _unspecified:
            self.control_options[name]['rate2_continuity_ref'] = rate2_continuity_ref

    def add_polynomial_control(self, name, order, desc=_unspecified, val=_unspecified, units=_unspecified,
                               opt=_unspecified, fix_initial=_unspecified, fix_final=_unspecified,
                               lower=_unspecified, upper=_unspecified,
                               scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                               ref=_unspecified, targets=_unspecified, rate_targets=_unspecified,
                               rate2_targets=_unspecified, shape=_unspecified):
        """
        Adds a polynomial control variable to be tied to a parameter in the ODE.

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
        self.check_parameter(name)

        if name not in self.polynomial_control_options:
            self.polynomial_control_options[name] = PolynomialControlOptionsDictionary()
            self.polynomial_control_options[name]['name'] = name
            self.polynomial_control_options[name]['order'] = order

        self.set_polynomial_control_options(name, order, desc, val, units, opt,
                                            fix_initial, fix_final, lower, upper,
                                            scaler, adder, ref0, ref,
                                            targets, rate_targets, rate2_targets, shape)

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
        if order is not _unspecified:
            self.polynomial_control_options[name]['order'] = order

        if units is not _unspecified:
            self.polynomial_control_options[name]['units'] = units

        if opt is not _unspecified:
            self.polynomial_control_options[name]['opt'] = opt

        if desc is not _unspecified:
            self.polynomial_control_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, str):
                self.polynomial_control_options[name]['targets'] = (targets,)
            else:
                self.polynomial_control_options[name]['targets'] = targets

        if rate_targets is not _unspecified:
            if isinstance(rate_targets, str):
                self.polynomial_control_options[name]['rate_targets'] = (rate_targets,)
            else:
                self.polynomial_control_options[name]['rate_targets'] = rate_targets

        if rate2_targets is not _unspecified:
            if isinstance(rate2_targets, str):
                self.polynomial_control_options[name]['rate2_targets'] = (rate2_targets,)
            else:
                self.polynomial_control_options[name]['rate2_targets'] = rate2_targets

        if val is not _unspecified:
            self.polynomial_control_options[name]['val'] = val

        if shape is not _unspecified:
            self.polynomial_control_options[name]['shape'] = shape

        if fix_initial is not _unspecified:
            self.polynomial_control_options[name]['fix_initial'] = fix_initial

        if fix_final is not _unspecified:
            self.polynomial_control_options[name]['fix_final'] = fix_final

        if lower is not _unspecified:
            self.polynomial_control_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.polynomial_control_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.polynomial_control_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.polynomial_control_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.polynomial_control_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.polynomial_control_options[name]['ref'] = ref

    def add_parameter(self, name, val=_unspecified, units=_unspecified, opt=False,
                      desc=_unspecified, lower=_unspecified, upper=_unspecified, scaler=_unspecified,
                      adder=_unspecified, ref0=_unspecified, ref=_unspecified, targets=_unspecified,
                      shape=_unspecified, dynamic=_unspecified, static_target=_unspecified,
                      include_timeseries=_unspecified, static_targets=_unspecified):
        """
        Add a parameter (static control variable) to the phase.

        Parameters
        ----------
        name : str
            Name of the parameter.
        val : float or ndarray
            Default value of the parameter at all nodes.
        units : str or None or 0
            Units in which the parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True, the value(s) of this parameter will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False (default), the this parameter will still be owned by an IndepVarComp in the phase,
            but it will not be a design variable in the optimization.
        desc : str
            A description of the parameter.
        lower : float or ndarray
            The lower bound of the parameter value.
        upper : float or ndarray
            The upper bound of the parameter value.
        scaler : float or ndarray
            The scaler of the parameter value for the optimizer.
        adder : float or ndarray
            The adder of the parameter value for the optimizer.
        ref0 : float or ndarray
            The zero-reference value of the parameter for the optimizer.
        ref : float or ndarray
            The unit-reference value of the parameter for the optimizer.
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
            In the future, if left _unspecified (the default), the phase parameter will try to connect to an ODE input
            of the same name. Set targets to None to prevent this.
        shape : Sequence of int
            The shape of the parameter.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.
        static_target : bool or _unspecified
            True if the targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  Otherwise False.
        include_timeseries : bool
            True if the static parameters should be included in output timeseries, else False.
        static_targets : bool or Sequence or _unspecified
            True if ALL targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  If False, ALL targets are
            expected to be shaped with the first dimension as the number of nodes. If given
            as a Sequence, it provides those targets not shaped with num_nodes. If left _unspecified,
            static targets will be determined automatically.
        """
        self.check_parameter(name)

        if name not in self.parameter_options:
            self.parameter_options[name] = ParameterOptionsDictionary()
            self.parameter_options[name]['name'] = name

        self.set_parameter_options(name, val=val, units=units, opt=opt, desc=desc,
                                   lower=lower, upper=upper, scaler=scaler, adder=adder,
                                   ref0=ref0, ref=ref, targets=targets, shape=shape, dynamic=dynamic,
                                   static_target=static_target, static_targets=static_targets,
                                   include_timeseries=include_timeseries)

    def set_parameter_options(self, name, val=_unspecified, units=_unspecified, opt=False,
                              desc=_unspecified, lower=_unspecified, upper=_unspecified,
                              scaler=_unspecified, adder=_unspecified, ref0=_unspecified,
                              ref=_unspecified, targets=_unspecified, shape=_unspecified,
                              dynamic=_unspecified, static_target=_unspecified,
                              include_timeseries=_unspecified, static_targets=_unspecified):
        """
        Set options for an existing parameter (static control variable) in the phase.

        Parameters
        ----------
        name : str
            Name of the parameter.
        val : float or ndarray
            Default value of the parameter at all nodes.
        units : str or None or 0
            Units in which the parameter is defined.  If 0, use the units declared
            for the parameter in the ODE.
        opt : bool
            If True the value(s) of this parameter will be design variables in
            the optimization problem, in the path 'phase_name.indep_controls.controls:control_name'.
            If False (default), the this parameter will still be owned by an IndepVarComp in the phase,
            but it will not be a design variable in the optimization.
        desc : str
            A description of the parameter.
        lower : float or ndarray
            The lower bound of the parameter value.
        upper : float or ndarray
            The upper bound of the parameter value.
        scaler : float or ndarray
            The scaler of the parameter value for the optimizer.
        adder : float or ndarray
            The adder of the parameter value for the optimizer.
        ref0 : float or ndarray
            The zero-reference value of the parameter for the optimizer.
        ref : float or ndarray
            The unit-reference value of the parameter for the optimizer.
        targets : Sequence of str or None
            Targets in the ODE to which this parameter is connected.
            In the future, if left _unspecified (the default), the phase parameter will try to connect to an ODE input
            of the same name. Set targets to None to prevent this.
        shape : Sequence of int
            The shape of the parameter.
        dynamic : bool
            True if the targets in the ODE may be dynamic (if the inputs are sized to the number
            of nodes) else False.  This option is deprecated.
        static_target : bool or _unspecified
            True if the targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  Otherwise False.
        include_timeseries : bool
            True if the static parameters should be included in output timeseries, else False.
        static_targets : bool or Sequence or _unspecified
            True if ALL targets in the ODE are not shaped with num_nodes as the first dimension
            (meaning they cannot have a unique value at each node).  If False, ALL targets are
            expected to be shaped with the first dimension as the number of nodes. If given
            as a Sequence, it provides those targets not shaped with num_nodes. If left _unspecified,
            static targets will be determined automatically.
        """
        if units is not _unspecified:
            self.parameter_options[name]['units'] = units

        self.parameter_options[name]['opt'] = opt

        if desc is not _unspecified:
            self.parameter_options[name]['desc'] = desc

        if targets is not _unspecified:
            if isinstance(targets, str):
                self.parameter_options[name]['targets'] = (targets,)
            else:
                self.parameter_options[name]['targets'] = targets

        if val is not _unspecified:
            self.parameter_options[name]['val'] = val

        if shape is not _unspecified:
            if np.isscalar(shape):
                self.parameter_options[name]['shape'] = (shape,)
            elif isinstance(shape, list):
                self.parameter_options[name]['shape'] = tuple(shape)
            else:
                self.parameter_options[name]['shape'] = shape

        if dynamic is not _unspecified:
            self.parameter_options[name]['static_targets'] = not dynamic

        if static_target is not _unspecified:
            self.parameter_options[name]['static_targets'] = static_target

        if static_targets is not _unspecified:
            self.parameter_options[name]['static_targets'] = static_targets

        if dynamic is not _unspecified and static_target is not _unspecified:
            raise ValueError("Both the deprecated 'dynamic' option and option 'static_target' were\n"
                             f"specified for parameter '{name}'. Going forward, please use only\n"
                             "option static_targets. Options 'dynamic' and 'static_target'\n"
                             "will be removed in Dymos 2.0.0.")

        if dynamic is not _unspecified and static_targets is not _unspecified:
            raise ValueError("Both the deprecated 'dynamic' option and option 'static_targets' were "
                             f"specified for parameter '{name}'. Going forward, please use only "
                             "option static_targets.  Option 'dynamic' will be removed in "
                             "Dymos 2.0.0.")

        if lower is not _unspecified:
            self.parameter_options[name]['lower'] = lower

        if upper is not _unspecified:
            self.parameter_options[name]['upper'] = upper

        if scaler is not _unspecified:
            self.parameter_options[name]['scaler'] = scaler

        if adder is not _unspecified:
            self.parameter_options[name]['adder'] = adder

        if ref0 is not _unspecified:
            self.parameter_options[name]['ref0'] = ref0

        if ref is not _unspecified:
            self.parameter_options[name]['ref'] = ref

        if include_timeseries is not _unspecified:
            self.parameter_options[name]['include_timeseries'] = include_timeseries

    def add_boundary_constraint(self, name, loc, constraint_name=None, units=None,
                                shape=None, indices=None, lower=None, upper=None, equals=None,
                                scaler=None, adder=None, ref=None, ref0=None, linear=False, flat_indices=False):
        r"""
        Add a boundary constraint to a variable in the phase.

        Parameters
        ----------
        name : str
            Name of the variable to constrain. May also provide an expression to be evaluated and constrained.
            If a single variable and the name is not a state, control, or 'time',
            then this is assumed to be the path of the variable to be constrained in the ODE.
            If an expression, it must be provided in the form of an equation with a left- and right-hand side.
        loc : str
            The location of the boundary constraint ('initial' or 'final').
        constraint_name : str or None
            The name of the boundary constraint. By default, this is 'var_constraint' if name is a single variable,
             or the left-hand side of the equation if name is an expression.
        units : str or None
            The units in which the boundary constraint is to be applied.  If None, use the
            units associated with the constrained output.  If provided, must be compatible with
            the variables units.
        shape : tuple, list, ndarray, or None
            The shape of the variable being boundary-constrained.  This can be inferred
            automatically for time, states, controls, and parameters, but is required
            if the constrained variable is an output of the ODE system.
        indices : tuple, list, ndarray, slice, or None
            The indices of the output variable to be boundary constrained at either the initial or final time in the
            phase. When the variable is multi-dimensional, this should be a list of lists, one for each dimension,
            containing the indices to be constrained.  Note the behavior of indices changes depending on the value
            of the flat_indices option.
        lower : float or ndarray, optional
            Lower boundary for the variable.
        upper : float or ndarray, optional
            Upper boundary for the variable.
        equals : float or ndarray, optional
            Equality constraint value for the variable.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        linear : bool
            Set to True if constraint is linear. Setting this to True when the constraint is not a linear function
            of the design variables will result in a failure of the optimization.
        flat_indices : bool
            If True, treat indices as flattened C-ordered indices of elements to constrain. Otherwise,
            indices should be a tuple or list giving the elements to constrain at each point in time.
        """
        if loc not in ['initial', 'final']:
            raise ValueError(f'Invalid boundary constraint location "{loc}". Must be '
                             '"initial" or "final".')

        expr_operators = ['(', '+', '-', '/', '*', '&', '%', '@']
        if '=' in name:
            is_expr = True
        elif '=' not in name and any(opr in name for opr in expr_operators):
            raise ValueError(f'The expression provided `{name}` has invalid format. '
                             'Expression may be a single variable or an equation '
                             'of the form `constraint_name = func(vars)`')
        else:
            is_expr = False

        if is_expr:
            constraint_name = name.split('=')[0].strip()
        elif constraint_name is None:
            constraint_name = name.rpartition('.')[-1]

        bc_list = self._initial_boundary_constraints if loc == 'initial' else self._final_boundary_constraints

        existing_bc = [bc for bc in bc_list if bc['name'] == name and bc['indices'] is None and indices is None]

        if existing_bc:
            raise ValueError(f'Cannot add new {loc} boundary constraint for variable `{name}` and indices {indices}. '
                             f'One already exists.')

        existing_bc_name = [bc for bc in bc_list if bc['name'] == constraint_name and
                            bc['indices'] is None and indices is None]

        if existing_bc_name:
            raise ValueError(f'Cannot add new {loc} boundary constraint named `{constraint_name}`'
                             f' and indices {indices}. The name `{constraint_name}` is already in use'
                             f' as a {loc} boundary constraint')

        bc = ConstraintOptionsDictionary()
        bc_list.append(bc)

        bc['name'] = name
        bc['constraint_name'] = constraint_name
        bc['lower'] = lower
        bc['upper'] = upper
        bc['equals'] = equals
        bc['scaler'] = scaler
        bc['adder'] = adder
        bc['ref0'] = ref0
        bc['ref'] = ref
        bc['indices'] = indices
        bc['shape'] = shape
        bc['linear'] = linear
        bc['units'] = units
        bc['flat_indices'] = flat_indices
        bc['is_expr'] = is_expr

        # Automatically add the requested variable to the timeseries outputs if it's an ODE output.
        var_type = self.classify_var(name)
        if var_type == 'ode':
            if constraint_name not in self._timeseries['timeseries']['outputs']:
                self.add_timeseries_output(name, output_name=constraint_name, units=units, shape=shape)

    def add_path_constraint(self, name, constraint_name=None, units=None, shape=None, indices=None,
                            lower=None, upper=None, equals=None, scaler=None, adder=None, ref=None,
                            ref0=None, linear=False, flat_indices=False):
        r"""
        Add a path constraint to a variable in the phase.

        Parameters
        ----------
        name : str
            Name of the variable to constrain. May also provide an expression to be evaluated and constrained.
            If a single variable and the name is not a state, control, or 'time',
            then this is assumed to be the path of the variable to be constrained in the ODE.
            If an expression, it must be provided in the form of an equation with a left- and right-hand side.
        constraint_name : str or None
            The name of the path constraint. By default, this is 'var_constraint' if name is a single variable,
             or the left-hand side of the equation if name is an expression.
        units : str or None
            The units in which the boundary constraint is to be applied.  If None, use the
            units associated with the constrained output.  If provided, must be compatible with
            the variables units.
        shape : tuple, list, ndarray, or None
            The shape of the variable being boundary-constrained.  This can be inferred
            automatically for time, states, controls, and parameters, but is required
            if the constrained variable is an output of the ODE system.
        indices : tuple, list, ndarray, or None
            The indices of the output variable to be constrained at each point in time in the phase.
            When the variable is multi-dimensional, this should be a list of lists, one for each dimension,
            containing the indices to be constrained.  Note the behavior of indices changes depending on the value
            of the flat_indices option.
        lower : float or ndarray, optional
            Lower boundary for the variable.
        upper : float or ndarray, optional
            Upper boundary for the variable.
        equals : float or ndarray, optional
            Equality constraint value for the variable.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        linear : bool
            Set to True if constraint is linear. If set to True and the constrained output is not a linear function
            of the design variables, the optimization will fail.
        flat_indices : bool
            If True, treat indices as flattened C-ordered indices of elements to constrain at each given point in time.
            Otherwise, indices should be a tuple or list giving the elements to constrain at each point in time.
        """
        expr_operators = ['(', '+', '-', '/', '*', '&', '%', '@']
        if '=' in name:
            is_expr = True
        elif '=' not in name and any(opr in name for opr in expr_operators):
            raise ValueError(f'The expression provided `{name}` has invalid format. '
                             'Expression may be a single variable or an equation '
                             'of the form `constraint_name = func(vars)`')
        else:
            is_expr = False

        if is_expr:
            constraint_name = name.split('=')[0].strip()
        elif constraint_name is None:
            constraint_name = name.rpartition('.')[-1]

        existing_pc = [pc for pc in self._path_constraints
                       if pc['name'] == name and pc['indices'] == indices and pc['flat_indices'] == flat_indices]

        if existing_pc:
            raise ValueError(f'Cannot add new path constraint for variable `{name}` and indices {indices}. '
                             f'One already exists.')

        existing_bc_name = [pc for pc in self._path_constraints
                            if pc['name'] == constraint_name and pc['indices'] == indices and
                            pc['flat_indices'] == flat_indices]

        if existing_bc_name:
            raise ValueError(f'Cannot add new path constraint named `{constraint_name}` and indices {indices}.'
                             f' The name `{constraint_name}` is already in use as a path constraint')

        pc = ConstraintOptionsDictionary()
        self._path_constraints.append(pc)

        pc['name'] = name
        pc['constraint_name'] = constraint_name
        pc['lower'] = lower
        pc['upper'] = upper
        pc['equals'] = equals
        pc['scaler'] = scaler
        pc['adder'] = adder
        pc['ref0'] = ref0
        pc['ref'] = ref
        pc['indices'] = indices
        pc['shape'] = shape
        pc['linear'] = linear
        pc['units'] = units
        pc['flat_indices'] = flat_indices
        pc['is_expr'] = is_expr

        # Automatically add the requested variable to the timeseries outputs if it's an ODE output.
        var_type = self.classify_var(name)
        if var_type == 'ode':
            if constraint_name not in self._timeseries['timeseries']['outputs']:
                self.add_timeseries_output(name, output_name=constraint_name, units=units, shape=shape)

    def add_timeseries_output(self, name, output_name=None, units=_unspecified, shape=_unspecified,
                              timeseries='timeseries', **kwargs):
        r"""
        Add a variable to the timeseries outputs of the phase.

        If name is given as an expression, this expression will be passed to an OpenMDAO ExecComp and the result
        computed and stored in the timeseries output under the variable name to the left of the equal sign.

        Parameters
        ----------
        name : str, or list of str
            The name(s) of the variable to be used as a timeseries output, or a mathematical expression to be used
            as a timeseries output. If a name, it must be one of the integration variable, the phase-relative value
            of the integration variable (e.g. 'time_phase', one of the states, controls, control rates, or parameters,
            in the phase, the path to an output variable in the ODE, or a glob pattern matching some outputs in the ODE.
        output_name : str or None or list or dict
            The name of the variable as listed in the phase timeseries outputs. By
            default this is the last element in `name` when split by dots.  The user may
            override the constraint name if splitting the path causes name collisions.
        units : str or None or _unspecified
            The units to express the timeseries output.  If None, use the
            units associated with the target.  If provided, must be compatible with
            the target units.
            If a list of names is provided, units can be a matching list or dictionary.
        shape : tuple or _unspecified
            The shape of the timeseries output variable.  This must be provided (if not scalar)
            since Dymos doesn't necessarily know the shape of ODE outputs until setup time.
        timeseries : str or None
            The name of the timeseries to which the output is being added.
        **kwargs
            Additional arguments passed to the exec comp.
        """
        if type(name) is list:
            for i, name_i in enumerate(name):
                expr = True if '=' in name_i else False
                if type(units) is dict:  # accept dict for units when using array of name
                    unit = units.get(name_i, None)
                elif type(units) is list:  # allow matching list for units
                    unit = units[i]
                else:
                    unit = units

                oname = self._add_timeseries_output(name_i, output_name=output_name,
                                                    units=unit,
                                                    shape=shape,
                                                    timeseries=timeseries,
                                                    rate=False,
                                                    expr=expr)

                # Handle specific units for wildcard names.
                if oname is not None and '*' in name_i:
                    self._timeseries[timeseries]['outputs'][oname]['wildcard_units'] = units

        else:
            expr = True if '=' in name else False
            self._add_timeseries_output(name, output_name=output_name,
                                        units=units,
                                        shape=shape,
                                        timeseries=timeseries,
                                        rate=False,
                                        expr=expr,
                                        expr_kwargs=kwargs)

    def add_timeseries_rate_output(self, name, output_name=None, units=_unspecified, shape=_unspecified,
                                   timeseries='timeseries'):
        r"""
        Add the rate of a variable to the timeseries outputs of the phase.

        Parameters
        ----------
        name : str, or list of str
            The name(s) of the variable to be used as a timeseries output.  Must be one of
            the integration variable, 't_phase', one of the states, controls, control rates, or parameters,
            in the phase, the path to an output variable in the ODE, or a glob pattern
            matching some outputs in the ODE.
        output_name : str or None or list or dict
            The name of the variable as listed in the phase timeseries outputs.  By
            default this is the last element in `name` when split by dots.  The user may
            override the constraint name if splitting the path causes name collisions.
        units : str or None or _unspecified
            The units to express the timeseries output.  If None, use the
            units associated with the target.  If provided, must be compatible with
            the target units.
            If a list of names is provided, units can be a matching list or dictionary.
        shape : tuple or _unspecified
            The shape of the timeseries output variable.  This must be provided (if not scalar)
            since Dymos doesn't necessarily know the shape of ODE outputs until setup time.
        timeseries : str or None
            The name of the timeseries to which the output is being added.
        """
        if type(name) is list:
            for i, name_i in enumerate(name):
                expr = True if '=' in name_i else False
                if type(units) is dict:  # accept dict for units when using array of name
                    unit = units.get(name_i, None)
                elif type(units) is list:  # allow matching list for units
                    unit = units[i]
                else:
                    unit = units

                oname = self._add_timeseries_output(name_i, output_name=output_name,
                                                    units=unit,
                                                    shape=shape,
                                                    timeseries=timeseries,
                                                    rate=True,
                                                    expr=expr)

                # Handle specific units for wildcard names.
                if oname is not None and '*' in name_i:
                    self._timeseries[timeseries]['outputs'][oname]['wildcard_units'] = units

        else:
            self._add_timeseries_output(name, output_name=output_name,
                                        units=units,
                                        shape=shape,
                                        timeseries=timeseries,
                                        rate=True)

    def _add_timeseries_output(self, name, output_name=None, units=_unspecified, shape=_unspecified,
                               timeseries='timeseries', rate=False, expr=False, expr_kwargs=None):
        r"""
        Add a single variable or rate to the timeseries outputs of the phase.

        This is called by add_timeseries_output or add_timeseries_rate_output for each variable or rate
        that is added.

        Parameters
        ----------
        name : str
            The name of the variable to be used as a timeseries output.  Must be one of
            the integration variable, 't_phase', one of the states, controls, control rates, or parameters,
            in the phase, or the path to an output variable in the ODE.
        output_name : str or None
            The name of the variable as listed in the phase timeseries outputs.  By
            default this is the last element in `name` when split by dots.  The user may
            override the constraint name if splitting the path causes name collisions.  If rate
            is True, the rate name will be this name + _rate.
        units : str or None
            The units to express the timeseries output.  If None, use the
            units associated with the target.  If provided, must be compatible with
            the target units.
        shape : tuple
            The shape of the timeseries output variable.  This must be provided (if not scalar)
            since Dymos doesn't necessarily know the shape of ODE outputs until setup time.
        timeseries : str or None
            The name of the timeseries to which the output is being added.
        rate : bool
            If True, add the rate of change of the named variable to the timeseries outputs of the
            phase.  The rate variable will be named f'{name}_rate'.  Defaults to False.
        expr :

        Returns
        -------
        str or None
           Name of output that was added to the timeseries or None if nothing was added.
        """
        if timeseries not in self._timeseries:
            raise ValueError(f'Timeseries {timeseries} does not exist in phase {self.pathname}')

        if output_name is None:
            if expr:
                output_name = name.split('=')[0].strip()
            elif '*' in name:
                output_name = name
            elif output_name is None:
                output_name = name.rpartition('.')[-1]

            if rate:
                output_name = output_name + '_rate'

        if output_name not in self._timeseries[timeseries]['outputs']:
            ts_output = TimeseriesOutputOptionsDictionary()
            ts_output['name'] = name
            ts_output['output_name'] = output_name
            ts_output['wildcard_units'] = {}
            ts_output['units'] = units
            ts_output['shape'] = shape
            ts_output['is_rate'] = rate
            ts_output['is_expr'] = expr
            ts_output['expr_kwargs'] = expr_kwargs

            self._timeseries[timeseries]['outputs'][output_name] = ts_output

            return output_name

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

    def add_objective(self, name, loc='final', index=None, shape=(1,), units=None, ref=None, ref0=None,
                      adder=None, scaler=None, parallel_deriv_color=None):
        """
        Add an objective in the phase.

        If name is not a state, control, control rate, or 'time', then this is assumed to be the
        path of the variable to be constrained in the RHS.

        Parameters
        ----------
        name : str
            Name of the objective variable.  This should be one of the integration variable, a state or control
            variable, the path to an output from the top level of the RHS, or an expression to be evaluated.
            If an expression, it must be provided in the form of an equation with a left- and right-hand side.
        loc : str
            Where in the phase the objective is to be evaluated.  Valid
            options are 'initial' and 'final'.  The default is 'final'.
        index : int, optional
            If variable is an array at each point in time, this indicates which index is to be
            used as the objective, assuming C-ordered flattening.
        shape : int, optional
            The shape of the objective variable, at a point in time.
        units : str, optional
            The units of the objective function.  If None, use the units associated with the target.
            If provided, must be compatible with the target units.
        ref : float or ndarray, optional
            Value of response variable that scales to 1.0 in the driver.
        ref0 : float or ndarray, optional
            Value of response variable that scales to 0.0 in the driver.
        adder : float or ndarray, optional
            Value to add to the model value to get the scaled value. Adder
            is first in precedence.
        scaler : float or ndarray, optional
            Value to multiply the model value to get the scaled value. Scaler
            is second in precedence.
        parallel_deriv_color : str
            If specified, this design var will be grouped for parallel derivative
            calculations with other variables sharing the same parallel_deriv_color.
        """
        expr_operators = ['(', '+', '-', '/', '*', '&', '%', '@']
        if '=' in name:
            is_expr = True
        elif '=' not in name and any(opr in name for opr in expr_operators):
            raise ValueError(f'The expression provided `{name}` has invalid format. '
                             'Expression may be a single variable or an equation '
                             'of the form `constraint_name = func(vars)`')
        else:
            is_expr = False

        obj_name = name.split('=')[0].strip() if is_expr else name

        obj_dict = {'name': name,
                    'loc': loc,
                    'index': index,
                    'shape': shape,
                    'units': units,
                    'ref': ref,
                    'ref0': ref0,
                    'adder': adder,
                    'scaler': scaler,
                    'parallel_deriv_color': parallel_deriv_color,
                    'is_expr': is_expr}
        self._objectives[obj_name] = obj_dict
        if is_expr and obj_name not in self._timeseries['timeseries']['outputs']:
            self.add_timeseries_output(name, output_name=obj_name, units=units, shape=shape)

    def set_time_options(self, units=_unspecified, fix_initial=_unspecified,
                         fix_duration=_unspecified, input_initial=_unspecified,
                         input_duration=_unspecified, initial_val=_unspecified,
                         initial_bounds=_unspecified, initial_scaler=_unspecified,
                         initial_adder=_unspecified, initial_ref0=_unspecified,
                         initial_ref=_unspecified, duration_val=_unspecified,
                         duration_bounds=_unspecified, duration_scaler=_unspecified,
                         duration_adder=_unspecified, duration_ref0=_unspecified,
                         duration_ref=_unspecified, targets=_unspecified,
                         time_phase_targets=_unspecified, t_initial_targets=_unspecified,
                         t_duration_targets=_unspecified, name=_unspecified):
        """
        Sets options for time in the phase.

        Only those options which are specified in the arguments will be updated.

        Parameters
        ----------
        units : str
            The default units for time variables in the phase.  Default is 's'.
        fix_initial : bool
            If True, the initial time of the phase is not treated as a design variable for the
            optimization problem.
        fix_duration : bool
            If True, the duration of the phase is not treated as a design variable for the
            optimization problem.
        input_initial : bool
            If True, the user is expected to link phase.t_initial to an external output source.
            Providing input_initial=True makes all initial time optimization settings irrelevant.
        input_duration : bool
            If True, the user is expected to link phase.t_duration to an external output source.
            Providing input_duration=True makes all time duration optimization settings irrelevant.
        initial_val : float
            Default value of the time at the start of the phase.
        initial_bounds : iterable of (float, float)
            The bounds (lower, upper) for time at the start of the phase.
        initial_scaler : float
            Scalar for the initial value of time.
        initial_adder : float
            Adder for the initial value of time.
        initial_ref0 : float
            Zero-reference for the initial value of time.
        initial_ref : float
            Unit-reference for the initial value of time.
        duration_val : float
            Default value for the time duration of the phase.
        duration_bounds : iterable of (float, float)
            The bounds (lower, upper) for the time duration of the phase.
        duration_scaler : float
            Scaler for phase time duration.
        duration_adder : float
            Adder for phase time duration.
        duration_ref0 : float
            Zero-reference for phase time duration.
        duration_ref : float
            Unit-reference for phase time duration.
        targets : iterable of str
            Targets in the ODE for the value of current time.
        time_phase_targets : iterable of str
            Targets in the ODE for the value of current phase elapsed time.
        t_initial_targets : iterable of str
            Targets in the ODE for the value of phase initial time.
        t_duration_targets :  iterable of str
            Targets in the ODE for the value of phase time duration.
        name : str
            Name of the integration variable for this phase. Default is 'time'.
        """
        if units is not _unspecified:
            self.time_options['units'] = units

        if fix_initial is not _unspecified:
            self.time_options['fix_initial'] = fix_initial

        if fix_duration is not _unspecified:
            self.time_options['fix_duration'] = fix_duration

        if input_initial is not _unspecified:
            self.time_options['input_initial'] = input_initial

        if input_duration is not _unspecified:
            self.time_options['input_duration'] = input_duration

        if initial_val is not _unspecified:
            self.time_options['initial_val'] = initial_val

        if initial_bounds is not _unspecified:
            self.time_options['initial_bounds'] = initial_bounds

        if initial_scaler is not _unspecified:
            self.time_options['initial_scaler'] = initial_scaler

        if initial_adder is not _unspecified:
            self.time_options['initial_adder'] = initial_adder

        if initial_ref0 is not _unspecified:
            self.time_options['initial_ref0'] = initial_ref0

        if initial_ref is not _unspecified:
            self.time_options['initial_ref'] = initial_ref

        if duration_val is not _unspecified:
            self.time_options['duration_val'] = duration_val

        if duration_bounds is not _unspecified:
            self.time_options['duration_bounds'] = duration_bounds

        if duration_scaler is not _unspecified:
            self.time_options['duration_scaler'] = duration_scaler

        if duration_adder is not _unspecified:
            self.time_options['duration_adder'] = duration_adder

        if duration_ref0 is not _unspecified:
            self.time_options['duration_ref0'] = duration_ref0

        if duration_ref is not _unspecified:
            self.time_options['duration_ref'] = duration_ref

        if targets is not _unspecified:
            if isinstance(targets, str):
                self.time_options['targets'] = (targets,)
            else:
                self.time_options['targets'] = targets

        if time_phase_targets is not _unspecified:
            if isinstance(time_phase_targets, str):
                self.time_options['time_phase_targets'] = (time_phase_targets,)
            else:
                self.time_options['time_phase_targets'] = time_phase_targets

        if t_initial_targets is not _unspecified:
            if isinstance(t_initial_targets, str):
                self.time_options['t_initial_targets'] = (t_initial_targets,)
            else:
                self.time_options['t_initial_targets'] = t_initial_targets

        if t_duration_targets is not _unspecified:
            if isinstance(t_duration_targets, str):
                self.time_options['t_duration_targets'] = (t_duration_targets,)
            else:
                self.time_options['t_duration_targets'] = t_duration_targets

        if name is not _unspecified:
            self.time_options['name'] = name

    def set_duration_balance(self, name, val=0.0, index=None, units=None, mult_val=None, normalize=False):
        """
        Adds a condition for the duration of the phase. This is satisfied using a nonlinear solver.

        Parameters
        ----------
        name : str
            Name of the variable.  This should be a state or control variable, the path to an output
            from the top level of the RHS, or an expression to be evaluated. If an expression,
            it must be provided in the form of an equation with a left- and right-hand side.
        val :  float
            The value that the residual must equal at the end  point of the phase.
        index : int, optional
            If variable is an array at each point in time, this indicates which index is to be
            used as the objective, assuming C-ordered flattening.
        units : str, optional
            The units of the objective function.  If None, use the units associated with the target.
            If provided, must be compatible with the target units.
        mult_val : float, optional
            Default value for the LHS multiplier.
        normalize : bool, optional
            Specifies whether the resulting residual should be normalized by a quadratic
            function of the RHS.
        """
        if self.time_options['fix_duration']:
            raise ValueError('Cannot implicitly solve for phase duration when fix_duration is True')
        elif self.time_options['input_duration']:
            raise ValueError('Cannot implicitly solve for phase duration when input_duration is True')

        if isinstance(self.options['transcription'], ExplicitShooting):
            raise NotImplementedError('Transcription ExplicitShooting does not implement method setup_duration_balance')

        options = {'name': name,
                   'val': val,
                   'index': index,
                   'units': units,
                   'mult_val': mult_val,
                   'normalize': normalize}

        expr_operators = ['(', '+', '-', '/', '*', '&', '%', '@']
        if '=' in name:
            is_expr = True
        elif '=' not in name and any(opr in name for opr in expr_operators):
            raise ValueError(f'The expression provided `{name}` has invalid format. '
                             'Expression may be a single variable or an equation '
                             'of the form `constraint_name = func(vars)`')
        else:
            is_expr = False

        balance_name = name.split('=')[0].strip() if is_expr else name

        options['is_expr'] = is_expr
        options['balance_name'] = balance_name

        self.time_options['t_duration_balance_options'] = options

        var_type = self.classify_var(name)
        if var_type == 'ode':
            if balance_name not in self._timeseries['timeseries']['outputs']:
                self.add_timeseries_output(name, output_name=balance_name, units=units)

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
            't', 't_phase', 'state', 'control', 'control_rate',
            'control_rate2', 'polynomial_control',
            'polynomial_control_rate', 'polynomial_control_rate2', 'parameter',
            or 'ode'.
        """
        return classify_var(var, time_options=self.time_options, state_options=self.state_options,
                            parameter_options=self.parameter_options,
                            control_options=self.control_options,
                            polynomial_control_options=self.polynomial_control_options,
                            timeseries_options=self._timeseries)

    def _check_ode(self):
        """
        Check that the provided ODE class meets minimum requirements.

        * The ode_class must be provided as a class or a callable.
        * When given as a callable, ode_class must return an instance derived from openmdao.core.System.
        * When given as a class, ode_class must derive from openmdao.core.System

        Raises
        ------
        ValueError
            ValueError is raised if the ODE does not meet one of the the requirements above.

        """
        ode_class = self.options['ode_class'] or self.options['rhs_class']
        if not inspect.isclass(ode_class):
            if not isinstance(ode_class, Callable):
                raise ValueError('ode_class must be given as a callable object that returns an '
                                 'object derived from openmdao.core.System, or as a class derived '
                                 'from openmdao.core.System.')
            test_instance = ode_class(num_nodes=1, **self.options['ode_init_kwargs'])
            if not isinstance(test_instance, System):
                raise ValueError(f'When provided as a callable, ode_class must return an instance '
                                 f'of openmdao.core.System.  Got {type(test_instance)}')
        elif not issubclass(ode_class, System):
            raise ValueError('If given as a class, ode_class must be derived from openmdao.core.System.')

    def setup(self):
        """
        Build the model hierarchy for a Dymos phase.
        """
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        transcription = self.options['transcription']
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

        transcription.setup_duration_balance(self)

        transcription.setup_defects(self)
        transcription.setup_solvers(self)

    def configure(self):
        """
        Finalize connections after sizes are known.
        """
        # Finalize the variables if it hasn't happened already.
        # If this phase exists within a Trajectory, the trajectory will finalize them during setup.
        transcription = self.options['transcription']
        ode = transcription._get_ode(self)

        configure_time_introspection(self.time_options, ode)

        # The control interpolation comp to which we'll connect controls
        if self.control_options:
            configure_controls_introspection(self.control_options, ode,
                                             time_units=self.time_options['units'])

        if self.polynomial_control_options:
            configure_controls_introspection(self.polynomial_control_options, ode,
                                             time_units=self.time_options['units'])

        if self.parameter_options:
            try:
                configure_parameters_introspection(self.parameter_options, ode)
            except ValueError as e:
                raise ValueError(f'Invalid parameter in phase `{self.pathname}`.\n{str(e)}') from e

        transcription.configure_states_discovery(self)
        transcription.configure_states_introspection(self)
        transcription.configure_time(self)
        transcription.configure_controls(self)
        transcription.configure_polynomial_controls(self)
        transcription.configure_parameters(self)
        transcription.configure_states(self)

        transcription.configure_ode(self)

        transcription.configure_defects(self)

        _configure_constraint_introspection(self)

        configure_timeseries_expr_introspection(self)

        transcription.configure_boundary_constraints(self)

        transcription.configure_path_constraints(self)

        transcription.configure_objective(self)

        try:
            configure_timeseries_output_introspection(self)
        except RuntimeError as val_err:
            raise RuntimeError(f'Error during configure_timeseries_output_introspection in phase {self.pathname}.')\
                from val_err

        transcription.configure_timeseries_outputs(self)

        transcription.configure_duration_balance(self)

        transcription.configure_solvers(self)

    def check_time_options(self):
        """
        Check that time options are valid and issue warnings if invalid options are provided.

        Warns
        -----
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        phase_name = self.pathname

        if self.time_options['fix_initial'] or self.time_options['input_initial']:
            invalid_options = []
            init_bounds = self.time_options['initial_bounds']
            if init_bounds is not None and init_bounds != (None, None):
                invalid_options.append('initial_bounds')
            for opt in 'initial_scaler', 'initial_adder', 'initial_ref', 'initial_ref0':
                if self.time_options[opt] is not None:
                    invalid_options.append(opt)
            if invalid_options:
                str_invalid_opts = ', '.join(invalid_options)
                warnings.warn(f'Phase time options have no effect because fix_initial=True '
                              f'or input_initial=True for phase \'{phase_name}\': {str_invalid_opts}')

        if self.time_options['input_initial'] and self.time_options['fix_initial']:
            warnings.warn(f'Phase \'{self.name}\' initial time is an externally-connected input, '
                          'therefore fix_initial has no effect.', RuntimeWarning)

        if self.time_options['fix_duration'] or self.time_options['input_duration']:
            invalid_options = []
            duration_bounds = self.time_options['duration_bounds']
            if duration_bounds is not None and duration_bounds != (None, None):
                invalid_options.append('duration_bounds')
            for opt in 'duration_scaler', 'duration_adder', 'duration_ref', 'duration_ref0':
                if self.time_options[opt] is not None:
                    invalid_options.append(opt)
            if invalid_options:
                str_invalid_opts = ', '.join(invalid_options)
                warnings.warn(f'Phase time options have no effect because fix_duration=True '
                              f'or input_duration=True for phase \'{phase_name}\': {str_invalid_opts}')

        if self.time_options['input_duration'] and self.time_options['fix_duration']:
            warnings.warn(f'Phase \'{self.name}\' time duration is an externally-connected input, '
                          'therefore fix_duration has no effect.', RuntimeWarning)

    def _check_control_options(self):
        """
        Check that control options are valid and issue warnings if invalid options are provided.

        Warns
        -----
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        for name, options in self.control_options.items():
            if not options['opt']:
                invalid_options = []
                for opt in 'lower', 'upper', 'scaler', 'adder', 'ref', 'ref0':
                    if options[opt] is not None:
                        invalid_options.append(opt)
                if invalid_options:
                    warnings.warn(f"Invalid options for non-optimal control '{name}' in phase "
                                  f"'{self.name}': {', '.join(invalid_options)}",
                                  RuntimeWarning)

                # Do not enforce rate continuity/rate continuity for non-optimal controls
                self.control_options[name]['continuity'] = False
                self.control_options[name]['rate_continuity'] = False
                self.control_options[name]['rate2_continuity'] = False

    def _check_polynomial_control_options(self):
        """
        Check that polynomial control options are valid and issue warnings if invalid options are provided.

        Warns
        -----
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        for name, options in self.control_options.items():
            if not options['opt']:
                invalid_options = []
                for opt in 'lower', 'upper', 'scaler', 'adder', 'ref', 'ref0':
                    if options[opt] is not None:
                        invalid_options.append(opt)
                if invalid_options:
                    warnings.warn(f"Invalid options for non-optimal polynoimal control '{name}' in "
                                  f"phase '{self.name}': {', '.join(invalid_options)}",
                                  RuntimeWarning)

    def _check_parameter_options(self):
        """
        Check that parameter options are valid and issue warnings if invalid
        options are provided.

        Warns
        -----
        RuntimeWarning
            RuntimeWarning is issued in the case of one or more invalid time options.
        """
        for name, options in self.parameter_options.items():
            if not options['opt']:
                invalid_options = []
                for opt in 'lower', 'upper', 'scaler', 'adder', 'ref', 'ref0':
                    if options[opt] is not None:
                        invalid_options.append(opt)
                if invalid_options:
                    warnings.warn(f"Invalid options for non-optimal parameter '{name}' in "
                                  f"phase '{self.name}': {', '.join(invalid_options)}",
                                  RuntimeWarning)

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
            The name of the node subset.
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
        om.issue_warning('phase.interpolate has been deprecated and will be removed from Dymos '
                         '2.0.0. Use phase.interp instead, which uses a different order for the '
                         'arguments but is more terse and can interpolate polynomial control '
                         'values.', category=om.OMDeprecationWarning)

        if not isinstance(ys, Iterable):
            raise ValueError('ys must be provided as an Iterable of length at least 2.')
        if nodes not in ('col', 'all', 'state_disc', 'state_input', 'control_disc',
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

        if gd is None:
            raise RuntimeError('interpolate cannot be called until the associated '
                               'problem has been setup')

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

    def interp(self, name=None, ys=None, xs=None, nodes=None, kind='linear', axis=0):
        """
        Interpolate values onto the given subset of nodes in the phase.

        If specified, name will be used to determine the kind of variable being interpolated.

        Parameters
        ----------
        name : str or None
            If nodes is None, then use the name argument to determine which kind of variable is
            being interpolated.  If it is a state, assume nodes is 'state_input'.  If it is related
            to a control, assume nodes is 'control_input'.  If it is a polynomial control, assume
            the nodes are the input nodes for that polynomial control.  Any other type of variable
            will result in an error.
        ys :  ndarray or Sequence or None
            Array of control/state/parameter values.
        xs :  ndarray or Sequence or None
            Array of integration variable values.
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
        if nodes not in ('col', 'all', 'state_disc', 'state_input', 'control_disc',
                         'control_input', 'segment_ends', None):
            raise ValueError("nodes must be one of 'col', 'all', 'state_disc', "
                             "'state_input', 'control_disc', 'control_input', 'segment_ends', or "
                             "None.")

        if xs is None:
            if len(ys) != 2:
                raise ValueError('xs may only be unspecified when len(ys)=2')
            if kind != 'linear':
                raise ValueError('kind must be linear when xs is unspecified.')
            xs = [-1, 1]
        elif len(xs) != np.prod(np.asarray(xs).shape):
            raise ValueError('xs must be viewable as a 1D array')

        gd = self.options['transcription'].grid_data
        if nodes is None:
            if name is None:
                raise ValueError('nodes for interpolation were not specified but the name of the '
                                 'variable to be interpolated was not provided.\nPlease specify '
                                 'the name of the interpolated variable or a node subset.')
            elif name in self.state_options:
                # For states in explicit shooting phases, interp should just return the initial
                # value.
                if isinstance(self.options['transcription'], dm.ExplicitShooting):
                    node_locations = np.array([-1.0])
                else:
                    node_locations = gd.node_ptau[gd.subset_node_indices['state_input']]
            elif name in self.control_options:
                node_locations = gd.node_ptau[gd.subset_node_indices['control_input']]
            elif name in self.polynomial_control_options:
                node_locations, _ = lgl(self.polynomial_control_options[name]['order'] + 1)
            else:
                raise ValueError('Could not find a state, control, or polynomial control named '
                                 f'{name} to be interpolated.\nPlease explicitly specified the '
                                 f'node subset onto which this value should be interpolated.')
        else:
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

    def get_simulation_phase(self, times_per_seg=None, method=_unspecified, atol=_unspecified,
                             rtol=_unspecified, first_step=_unspecified, max_step=_unspecified,
                             reports=False):
        """
        Return a SimulationPhase instance that is essentially a copy of this Phase.

        This instance is initialized based on data from this Phase instance and
        the given simulation times.

        If left unspecified, options `method`, `atol`, `rtol`, `first_step`, and `max_step` will
        be taken from Phase.simulate_options.

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
        first_step : float
            Initial step size for the integration.
        max_step : float or _unspecified
            Maximum step size for the integration.
        reports : bool or None or str or Sequence
            The reports setting for the subproblem run under each simulation segment.

        Returns
        -------
        SimulationPhase
            An instance of SimulationPhase initialized based on data from this Phase and the given
            times.  This instance has not yet been setup.
        """
        from .simulation_phase import SimulationPhase
        sim_phase = SimulationPhase(from_phase=self, times_per_seg=times_per_seg, method=method,
                                    atol=atol, rtol=rtol, first_step=first_step, max_step=max_step,
                                    reports=reports)

        return sim_phase

    def initialize_values_from_phase(self, prob, from_phase, phase_path='', skip_params=None):
        """
        Initializes values in the Phase using the phase from which it was created.

        Parameters
        ----------
        prob : Problem
            The problem instance to set values taken from the from_phase instance.
        from_phase : Phase
            The Phase instance from which the values in this phase are being initialized.
        phase_path : str
            The pathname of the system in prob that contains the phases.
        skip_params : None or set
            Parameter names that will be skipped because they have already been initialized at the
            trajectory level (Deprecated).
        """
        phs = from_phase

        if skip_params is not None:
            om.issue_warning(f'{self.pathname}: Option `skip_params` to Phase.initialize_values_from_phase` is '
                             f'deprecated and will be removed dymos 2.0.0', category=om.OMDeprecationWarning)

        op_dict = dict([(name, options) for (name, options) in phs.list_outputs(units=True,
                                                                                list_autoivcs=True,
                                                                                out_stream=None)])
        ip_dict = dict([(name, options) for (name, options) in phs.list_inputs(units=True,
                                                                               out_stream=None)])

        phs_path = phs.pathname + '.' if phs.pathname else ''

        if self.pathname.partition('.')[0] == self.name:
            self_path = self.name + '.'
        else:
            self_path = self.pathname.partition('.')[0] + '.' + self.name + '.'

        if MPI:
            op_dict = MPI.COMM_WORLD.bcast(op_dict, root=0)

        # Set the integration times
        time_name = phs.time_options['name']
        op = op_dict[f'timeseries.timeseries_comp.{time_name}']
        prob.set_val(f'{self_path}t_initial', op['val'][0, ...])
        prob.set_val(f'{self_path}t_duration', op['val'][-1, ...] - op['val'][0, ...])

        # Assign initial state values
        for name in phs.state_options:
            op = op_dict[f'timeseries.timeseries_comp.states:{name}']
            prob[f'{self_path}initial_states:{name}'][...] = op['val'][0, ...]

        # Assign control values
        for name, options in phs.control_options.items():
            ip = ip_dict[f'control_group.control_interp_comp.controls:{name}']
            prob[f'{self_path}controls:{name}'][...] = ip['val']

        # Assign polynomial control values
        for name, options in phs.polynomial_control_options.items():
            ip = ip_dict[f'polynomial_control_group.interp_comp.'
                         f'polynomial_controls:{name}']
            prob[f'{self_path}polynomial_controls:{name}'][...] = ip['val']

        # Assign parameter values
        for name in phs.parameter_options:
            units = phs.parameter_options[name]['units']

            # We use this private function to grab the correctly sized variable from the
            # auto_ivc source.
            val = phs.get_val(f'parameters:{name}', units=units)

            if phase_path:
                prob_path = f'{phase_path}.{self.name}.parameters:{name}'
            else:
                prob_path = f'{self.name}.parameters:{name}'
            prob.set_val(prob_path, val)

    def simulate(self, times_per_seg=10, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                 first_step=_unspecified, max_step=_unspecified, record_file=None):
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
        sim_prob = om.Problem(model=om.Group())

        sim_phase = self.get_simulation_phase(times_per_seg, method=method, atol=atol, rtol=rtol,
                                              first_step=first_step, max_step=max_step)

        sim_prob.model.add_subsystem(self.name, sim_phase)

        if record_file is not None:
            rec = om.SqliteRecorder(record_file)
            sim_prob.add_recorder(rec)

        sim_prob.setup(check=True)
        sim_prob.final_setup()

        sim_phase.set_vals_from_phase(from_phase=self)

        print(f'\nSimulating phase {self.pathname}')
        sim_prob.run_model()
        print(f'Done simulating phase {self.pathname}')
        sim_prob.record('final')
        sim_prob.cleanup()

        return sim_prob

    def set_refine_options(self, refine=_unspecified, tol=_unspecified, min_order=_unspecified,
                           max_order=_unspecified, smoothness_factor=_unspecified):
        """
        Set the specified option(s) for grid refinement in the phase.

        Parameters
        ----------
        refine : bool
            If True, this Phase will undergo refinement during the grid refinement procedure.
        tol : float
            The error tolerance used by all grid-refinement algorithms.
        min_order : int
            The minimum allowable transcription order for segments in the phase.
            Used in hp and ph refinement methods.
        max_order : int
            The maximum allowable transcription order for segments in the phase.
            Used in hp and ph refinement methods.
        smoothness_factor : float
            The maximum allowable ratio of state second derivatives. If exceeded the segment must be split.
            Used in hp refinement method.
        """
        if refine is not _unspecified:
            self.refine_options['refine'] = refine
        if tol is not _unspecified:
            self.refine_options['tolerance'] = tol
        if min_order is not _unspecified:
            self.refine_options['min_order'] = min_order
        if max_order is not _unspecified:
            self.refine_options['max_order'] = max_order
        if smoothness_factor is not _unspecified:
            self.refine_options['smoothness_factor'] = smoothness_factor

    def set_simulate_options(self, method=_unspecified, atol=_unspecified, rtol=_unspecified,
                             first_step=_unspecified, max_step=_unspecified):
        """
        Set the specified option(s) for grid refinement in the phase.

        Parameters
        ----------
        method : str
            The scipy.integrate.solve_ivp integration method.
        atol : float
            Absolute convergence tolerance for the scipy.integrate.solve_ivp method.
        rtol : float
            Relative convergence tolerance for the scipy.integrate.solve_ivp method.
        first_step : float
            Initial step size for the integration.
        max_step : float
            Maximum step size for the integration.
        """
        if method is not _unspecified:
            self.simulate_options['method'] = method
        if atol is not _unspecified:
            self.simulate_options['atol'] = atol
        if rtol is not _unspecified:
            self.simulate_options['rtol'] = rtol
        if first_step is not _unspecified:
            self.simulate_options['first_step'] = first_step
        if max_step is not _unspecified:
            self.simulate_options['max_step'] = max_step

    def is_time_fixed(self, loc):
        """
        Test whether the initial or final time in the phase is guaranteed to be fixed.

        There are situations in which this can return False even if the final time is fixed
        in the problem.  If the initial time or duration are inputs, the phase knows nothing
        about their behavior upstream.

        Parameters
        ----------
        loc : str
            The location of time to be tested: either 'initial' or 'final'.

        Returns
        -------
        bool
            True if both the initial time and duration are not inputs and are fixed.
        """
        fix_initial = self.time_options['fix_initial']
        initial_bounds = self.time_options['initial_bounds']

        if loc == 'initial':
            res = fix_initial or (initial_bounds != (None, None) and np.diff(initial_bounds)[0] == 0.0)
        elif loc == 'final':
            fix_duration = self.time_options['fix_duration']
            duration_bounds = self.time_options['duration_bounds']
            initial_fixed = fix_initial or (initial_bounds != (None, None) and np.diff(initial_bounds)[0] == 0)
            duration_fixed = fix_duration or (duration_bounds != (None, None) and np.diff(duration_bounds)[0] == 0)
            res = initial_fixed and duration_fixed
        else:
            raise ValueError(f'Unknown value for argument "loc": must be either "initial" or '
                             f'"final" but got {loc}')
        return res

    def is_state_fixed(self, name, loc):
        """
        Test if the state of the given name is guaranteed to be fixed at the initial or final time.

        Parameters
        ----------
        name : str
            The name of the state to be tested.
        loc : str
            The location of time to be tested: either 'initial' or 'final'.

        Returns
        -------
        bool
            True if the state of the given name is guaranteed to be fixed at the given location.
        """
        if loc == 'initial':
            res = self.state_options[name]['fix_initial']
        elif loc == 'final':
            res = self.state_options[name]['fix_final']
        else:
            raise ValueError(f'Unknown value for argument "loc": must be either "initial" or '
                             f'"final" but got {loc}')
        return res

    def is_control_fixed(self, name, loc):
        """
        Test if the control of the given name is guaranteed to be fixed at the initial or final time.

        Parameters
        ----------
        name : str
            The name of the control to be tested.
        loc : str
            The location of time to be tested: either 'initial' or 'final'.

        Returns
        -------
        bool
            True if the state of the given name is guaranteed to be fixed at the given location.
        """
        control_opts = self.control_options[name]
        if loc == 'initial' and control_opts['fix_initial']:
            res = True
        elif loc == 'final' and control_opts['fix_final']:
            res = True
        else:
            res = not control_opts['opt']
        return res

    def is_control_rate_fixed(self, name, loc):
        """
        Test if the control rate of the given name is guaranteed to be fixed at the initial or final time.

        Parameters
        ----------
        name : str
            The name of the control to be tested.
        loc : str
            The location of time to be tested: either 'initial' or 'final'.

        Returns
        -------
        bool
            True if the state of the given name is guaranteed to be fixed at the given location.
        """
        if name.endswith('_rate') and self.control_options is not None and \
                name[:-5] in self.control_options:
            control_name = name[:-5]
        elif name.endswith('_rate2') and self.control_options is not None and \
                name[:-6] in self.control_options:
            control_name = name[:-6]
        return self.is_control_fixed(control_name, loc)

    def is_polynomial_control_fixed(self, name, loc):
        """
        Test if the polynomial control of the given name is guaranteed to be fixed at the initial or final time.

        Parameters
        ----------
        name : str
            The name of the polynomial control to be tested.
        loc : str
            The location of time to be tested: either 'initial' or 'final'.

        Returns
        -------
        bool
            True if the state of the given name is guaranteed to be fixed at the given location.
        """
        if loc == 'initial':
            res = self.polynomial_control_options[name]['fix_initial']
        elif loc == 'final':
            res = self.polynomial_control_options[name]['fix_final']
        else:
            raise ValueError(f'Unknown value for argument "loc": must be either "initial" or '
                             f'"final" but got {loc}')
        return res

    def is_polynomial_control_rate_fixed(self, name, loc):
        """
        Test if the polynomial control rate of the given name is guaranteed to be fixed at the initial or final time.

        Parameters
        ----------
        name : str
            The name of the control to be tested.
        loc : str
            The location of time to be tested: either 'initial' or 'final'.

        Returns
        -------
        bool
            True if the state of the given name is guaranteed to be fixed at the given location.
        """
        if name.endswith('_rate') and self.polynomial_control_options is not None and \
                name[:-5] in self.polynomial_control_options:
            control_name = name[:-5]
        elif name.endswith('_rate2') and self.polynomial_control_options is not None and \
                name[:-6] in self.options['polynomial_control_options']:
            control_name = name[:-6]
        return self.is_polynomial_control_fixed(control_name, loc)

    def _indices_in_constraints(self, name, loc):
        """
        Returns a set of the C-order flattened indices involving constraint of the given name at the given loc.

        Parameters
        ----------
        name : str
            The pathname of the constrained quantity.
        loc : str
            The type of constraint to search: 'initial', 'final', or 'path'.

        Returns
        -------
        all_flat_idxs : set
            A C-order flattened set of indices that apply to the constraint.
        """
        cons = {'initial': self._initial_boundary_constraints,
                'final': self._final_boundary_constraints,
                'path': self._path_constraints}

        all_flat_idxs = set()

        for con in cons[loc]:
            if con['name'] != name:
                continue

            flat_idxs = get_constraint_flat_idxs(con)
            duplicate_idxs = all_flat_idxs.intersection(flat_idxs)
            if duplicate_idxs:
                s = {'initial': 'initial boundary', 'final': 'final boundary', 'path': 'path'}
                raise ValueError(f'Duplicate constraint in phase {self.pathname}. '
                                 f'The following indices of `{name}` are used in '
                                 f'multiple {s[loc]} constraints:\n{duplicate_idxs}')

            all_flat_idxs.update(flat_idxs)

        return all_flat_idxs

    def _is_fixed(self, var_name, var_type, loc):
        """
        Determine whether a variable is fixed or not.

        Parameters
        ----------
        var_name : str
            Identifier of the variable as known to the phase.
        var_type : str
            The type of variable.
        loc : str
            Either 'initial' or 'final' for non-parameters.

        Returns
        -------
        bool
            True if the variable is fixed, otherwise False.
        """
        if var_type == 't':
            return self.is_time_fixed(loc)
        elif var_type == 'state':
            return self.is_state_fixed(var_name, loc)
        elif var_type in {'input_control', 'indep_control'}:
            return self.is_control_fixed(var_name, loc)
        elif var_type in {'input_polynomial_control', 'indep_polynomial_control'}:
            return self.is_polynomial_control_fixed(var_name, loc)
        elif var_type in {'control_rate', 'control_rate2'}:
            return self.is_control_rate_fixed(var_name, loc)
        elif var_type == 'parameter':
            return not self.parameter_options[var_name]['opt']

        return False  # No way to know so we allow these to go through

    def load_case(self, case):
        """
        Pull all input and output variables from a case into the Phase.

        Parameters
        ----------
        case : Case or dict
            A Case from a CaseReader, or a dictionary with key 'inputs' mapped to the
            output of problem.model.list_inputs and key 'outputs' mapped to the output
            of prob.model.list_outputs. Both list_inputs and list_outputs should be called
            with `units=True`, `prom_names=True` and `return_format='dict'`.
        """
        # allow old style arguments using a Case or OpenMDAO problem instead of dictionary
        assert (isinstance(case, Case) or isinstance(case, dict))
        if isinstance(case, Case):
            previous_solution = {
                'inputs': case.list_inputs(out_stream=None, return_format='dict',
                                           units=True, prom_name=True),
                'outputs': case.list_outputs(out_stream=None, return_format='dict',
                                             units=True, prom_name=True)
            }
        else:
            previous_solution = case

        prev_vars_abs2prom = {}
        prev_vars_abs2prom.update({k: v['prom_name'] for k, v in previous_solution['inputs'].items()})
        prev_vars_abs2prom.update({k: v['prom_name'] for k, v in previous_solution['outputs'].items()})
        prev_vars_prom2abs = {v: k for k, v in prev_vars_abs2prom.items()}

        prev_vars = {}
        prev_vars.update({v['prom_name']: {'val': v['val'], 'units': v['units'], 'abs_name': k}
                          for k, v in previous_solution['inputs'].items()})
        prev_vars.update({v['prom_name']: {'val': v['val'], 'units': v['units'], 'abs_name': k}
                          for k, v in previous_solution['outputs'].items()})

        phase_io = {'inputs': self.list_inputs(units=True, prom_name=True, out_stream=None),
                    'outputs': self.list_outputs(units=True, prom_name=True, out_stream=None)}

        phase_vars = {}
        phase_vars.update({f"{self.pathname}.{v['prom_name']}": {'val': v['val'], 'units': v['units'], 'abs_name': k}
                           for k, v in phase_io['inputs']})
        phase_vars.update({f"{self.pathname}.{v['prom_name']}": {'val': v['val'], 'units': v['units'], 'abs_name': k}
                           for k, v in phase_io['outputs']})

        phase_name = self.name

        # Get the initial time and duration from the previous result and set them into the new phase.
        integration_name = self.time_options['name']

        try:
            prev_time_path = prev_vars_abs2prom[f'{self.pathname}.timeseries.timeseries_comp.{integration_name}']
        except KeyError:
            om.issue_warning(f'load_case for phase {self.name} failed - phase not found in case data.')
            return

        prev_timeseries_prom_path, _, _ = prev_time_path.rpartition(f'.{integration_name}')
        prev_phase_prom_path, _, _ = prev_timeseries_prom_path.rpartition('.timeseries')

        prev_time_val = prev_vars[prev_time_path]['val']
        prev_time_val, unique_idxs = np.unique(prev_time_val, return_index=True)
        prev_time_units = prev_vars[prev_time_path]['units']

        t_initial = prev_time_val[0]
        t_duration = prev_time_val[-1] - prev_time_val[0]

        self.set_val('t_initial', t_initial, units=prev_time_units)
        self.set_val('t_duration', t_duration, units=prev_time_units)

        # Interpolate the timeseries state outputs from the previous solution onto the new grid.
        if not isinstance(self, dm.AnalyticPhase):
            for state_name, options in self.state_options.items():
                if f'{prev_timeseries_prom_path}.states:{state_name}' in prev_vars_prom2abs:
                    prev_state_path = f'{prev_timeseries_prom_path}.states:{state_name}'
                elif f'{prev_timeseries_prom_path}.{state_name}' in prev_vars_prom2abs:
                    prev_state_path = f'{prev_timeseries_prom_path}.{state_name}'
                else:
                    issue_warning(f'Unable to find state {state_name} in timeseries data from case being loaded.',
                                  om.OpenMDAOWarning)
                    continue

                prev_state_val = prev_vars[prev_state_path]['val']
                prev_state_units = prev_vars[prev_state_path]['units']
                interp_vals = self.interp(name=state_name,
                                          xs=prev_time_val,
                                          ys=prev_state_val[unique_idxs],
                                          kind='slinear')
                if options['lower'] is not None or options['upper'] is not None:
                    interp_vals = interp_vals.clip(options['lower'], options['upper'])
                self.set_val(f'states:{state_name}',
                             interp_vals,
                             units=prev_state_units)
                try:
                    self.set_val(f'initial_states:{state_name}', prev_state_val[0, ...], units=prev_state_units)
                except KeyError:
                    pass

                if options['fix_final']:
                    warning_message = f"{phase_name}.states:{state_name} specifies 'fix_final=True'. " \
                                      f"If the given restart file has a" \
                                      f" different final value this will overwrite the user-specified value"
                    issue_warning(warning_message)

            # Interpolate the timeseries control outputs from the previous solution onto the new grid.
            for control_name, options in self.control_options.items():
                if f'{prev_timeseries_prom_path}.controls:{control_name}' in prev_vars_prom2abs:
                    prev_control_path = f'{prev_timeseries_prom_path}.controls:{control_name}'
                elif f'{prev_timeseries_prom_path}.{control_name}' in prev_vars_prom2abs:
                    prev_control_path = f'{prev_timeseries_prom_path}.{control_name}'
                else:
                    issue_warning(f'Unable to find control {control_name} in timeseries data from case being loaded.',
                                  om.OpenMDAOWarning)
                    continue

                prev_control_val = prev_vars[prev_control_path]['val']
                prev_control_units = prev_vars[prev_control_path]['units']
                interp_vals = self.interp(name=control_name,
                                          xs=prev_time_val,
                                          ys=prev_control_val[unique_idxs],
                                          kind='slinear')
                if options['lower'] is not None or options['upper'] is not None:
                    interp_vals = interp_vals.clip(options['lower'], options['upper'])
                self.set_val(f'controls:{control_name}', interp_vals, units=prev_control_units)
                if options['fix_final']:
                    warning_message = f"{phase_name}.controls:{control_name} specifies 'fix_final=True'. " \
                                      f"If the given restart file has a" \
                                      f" different final value this will overwrite the user-specified value"
                    issue_warning(warning_message)

            # Set the output polynomial control outputs from the previous solution as the value
            for pc_name, options in self.polynomial_control_options.items():
                if f'{prev_timeseries_prom_path}.polynomial_controls:{pc_name}' in prev_vars_prom2abs:
                    prev_pc_path = f'{prev_timeseries_prom_path}.polynomial_controls:{pc_name}'
                elif f'{prev_timeseries_prom_path}.{pc_name}' in prev_vars_prom2abs:
                    prev_pc_path = f'{prev_timeseries_prom_path}.{pc_name}'
                else:
                    issue_warning(f'Unable to find polynomial control {pc_name} in timeseries data from case being '
                                  f'loaded.', om.OpenMDAOWarning)
                    continue

                prev_pc_val = prev_vars[prev_pc_path]['val']
                prev_pc_units = prev_vars[prev_pc_path]['units']
                interp_vals = self.interp(name=pc_name,
                                          xs=prev_time_val,
                                          ys=prev_pc_val[unique_idxs],
                                          kind='slinear')
                if options['lower'] is not None or options['upper'] is not None:
                    interp_vals = interp_vals.clip(options['lower'], options['upper'])
                self.set_val(f'polynomial_controls:{pc_name}',
                             interp_vals,
                             units=prev_pc_units)
                if options['fix_final']:
                    warning_message = f"{phase_name}.polynomial_controls:{pc_name} specifies 'fix_final=True'. " \
                                      f"If the given restart file has a" \
                                      f" different final value this will overwrite the user-specified value"
                    issue_warning(warning_message)

        # Set the timeseries parameter outputs from the previous solution as the parameter value
        for param_name in self.parameter_options:
            if f'{prev_phase_prom_path}.parameter_vals:{param_name}' in prev_vars:
                prev_param_val = prev_vars[f'{prev_phase_prom_path}.parameter_vals:{param_name}']['val']
                prev_param_units = prev_vars[f'{prev_phase_prom_path}.parameter_vals:{param_name}']['units']
                self.set_val(f'parameters:{param_name}', prev_param_val[0, ...], units=prev_param_units)
            else:
                issue_warning(f'Unable to find "{prev_phase_prom_path}.parameter_vals:{param_name}" '
                              f'in data from case being loaded.')
