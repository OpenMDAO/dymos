from copy import deepcopy

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

    def add_state(self, name, state_name=None, units=_unspecified, shape=_unspecified):
        """
        Add a state variable to be integrated by the phase.

        Parameters
        ----------
        name : str
            Path to use as the source of the state variable.
        state_name : str
            Name of the state variable, if the last element of the source path is ambiguous.
        units : str or None
            Units in which the state variable is defined.  If units is not
            specified here then the unit will be determined from the source.
        shape : tuple of int
            The shape of the state variable.  For instance, a 3D cartesian position vector would have
            a shape of (3,).
        """
        _state_name = name.split('.')[-1] if state_name is None else state_name

        if name not in self.state_options:
            self.state_options[_state_name] = StateOptionsDictionary()
            self.state_options[_state_name]['name'] = _state_name

        self.set_state_options(name=name, state_name=state_name, units=units, shape=shape)

    def set_state_options(self, name, state_name=None, units=_unspecified, shape=_unspecified):
        """
        Set options that apply the EOM state variable of the given name.

        Parameters
        ----------
        name : str
            Path to use as the source of the state variable.
        state_name : str
            Name of the state variable, if the last element of the source path is ambiguous.
        units : str or None
            Units in which the state variable is defined.  If units is not
            specified here then the unit will be determined from the source.
        shape : tuple of int
            The shape of the state variable.  For instance, a 3D cartesian position vector would have
            a shape of (3,).
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

        if self.parameter_options:
            transcription.setup_parameters(self)

        # Never allow state rate outputs for analytic phases
        self.timeseries_options['include_state_rates'] = False
        self.timeseries_options._dict['include_state_rates']['values'] = [False]

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

    def set_simulate_options(self, *args, **kwargs):
        """
        Stub to make sure users are informed that simulate cannot be done on AnalyticPhase.

        Parameters
        ----------
        *args
            Position arguments.
        **kwargs : float
            Keyword arguments.

        Raises
        ------
        NotImplementedError
            Simulation cannot be performed on AnalyticPhase.
        """
        raise NotImplementedError('Method set_simulate_options is not available for AnalyticPhase.')

    def duplicate(self, num_nodes=None, boundary_constraints=False, path_constraints=False, objectives=False,
                  fix_initial_time=False):
        """
        Create a copy of this phase where most options and attributes are deep copies of those in the original.

        By default, a deepcopy of the transcription in the original phase is used.
        Boundary constraints, path constraints, and objectives are _NOT_ copied by default, but the user may opt to do so.
        By default, initial time is not fixed, nor are the initial or final state values.
        These also can be overridden with the appropriate arguments.

        Parameters
        ----------
        num_nodes : int or None
            The number of nodes to use in the new phase, or None if it should use the same
            number as the phase being duplicated.
        boundary_constraints : bool
            If True, retain all boundary constraints from the phase to be copied.
        path_constraints : bool
            If True, retain all path constraints from the phase to be copied.
        objectives : bool
            If True, retain all objectives from the phase to be copied.
        fix_initial_time : bool
            If True, fix the initial time of the returned phase.

        Returns
        -------
        AnalyticPhase
            The new phase created by duplicating this one.
        """
        nn = num_nodes if num_nodes is not None else self.options['num_nodes']
        ode_class = self.options['ode_class']
        ode_init_kwargs = self.options['ode_init_kwargs']
        auto_solvers = self.options['auto_solvers']

        p = AnalyticPhase(num_nodes=nn, ode_class=ode_class, ode_init_kwargs=ode_init_kwargs,
                          auto_solvers=auto_solvers)

        p.time_options.update(deepcopy(self.time_options))
        p.time_options['fix_initial'] = fix_initial_time

        for state_name, state_options in self.state_options.items():
            p.state_options[state_name] = deepcopy(state_options)

        for param_name, param_options in self.parameter_options.items():
            p.parameter_options[param_name] = deepcopy(param_options)

        p._timeseries = deepcopy(self._timeseries)

        p.refine_options = deepcopy(self.refine_options)
        p.simulate_options = deepcopy(self.simulate_options)
        p.timeseries_options = deepcopy(self.timeseries_options)

        if boundary_constraints:
            p._initial_boundary_constraints = deepcopy(self._initial_boundary_constraints)
            p._final_boundary_constraints = deepcopy(self._final_boundary_constraints)

        if path_constraints:
            p._path_constraints = deepcopy(self._path_constraints)

        if objectives:
            p._objectives = deepcopy(self._objectives)

        return p
